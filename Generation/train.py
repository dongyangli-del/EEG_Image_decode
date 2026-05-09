"""
Unified training script for ATMS EEG encoder + Diffusion Prior.

Training strategy (two-phase):
  Phase 1 (first 1/4 epochs): train ATMS encoder only
    - Model selection & early stopping based on validation loss
  Phase 2 (remaining 3/4 epochs):
    --encoder_finetuning OFF (default): freeze encoder, train Prior only
      - Features extracted once; model selection by prior loss
    --encoder_finetuning ON: jointly train encoder + Prior
      - Features re-extracted each epoch; model selection by val loss
  - Validation set = stratified 1/10 split from training data
  - Test set is NEVER touched during training

All paths are passed via command-line arguments from benchmark.sh.
"""

import os
import sys
import csv
import random
import datetime
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eegdatasets import EEGDataset
from diffusion_prior import DiffusionPriorUNet, EmbeddingDataset, Pipe
from models.atms import ATMS, extract_id_from_string

# Shared encoder training utilities (generation loss mode: MSE + CLIP)
from encoder_utils import (
    train_encoder_epoch,
    evaluate_encoder as evaluate_val,
    stratified_condition_split,
)


# ─────────────────────────────────────────────────────────────────────────────
# Prior training helpers
# ─────────────────────────────────────────────────────────────────────────────
def train_prior_one_epoch(pipe, dataloader):
    pipe.diffusion_prior.train()
    device = pipe.device
    criterion = nn.MSELoss(reduction='none')
    num_train_timesteps = pipe.scheduler.config.num_train_timesteps
    loss_sum = 0

    for batch in dataloader:
        c_embeds = batch['c_embedding'].to(device) if 'c_embedding' in batch else None
        h_embeds = batch['h_embedding'].to(device)
        N = h_embeds.shape[0]

        if torch.rand(1) < 0.1:
            c_embeds = None

        noise = torch.randn_like(h_embeds)
        timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)
        perturbed = pipe.scheduler.add_noise(h_embeds, noise, timesteps)
        noise_pre = pipe.diffusion_prior(perturbed, timesteps, c_embeds)
        loss = criterion(noise_pre, noise).mean()

        pipe._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipe.diffusion_prior.parameters(), 1.0)
        pipe._lr_scheduler.step()
        pipe._optimizer.step()
        loss_sum += loss.item()

    return loss_sum / len(dataloader)


def setup_prior_optimizer(pipe, dataloader, num_epochs, learning_rate):
    from diffusers.optimization import get_cosine_schedule_with_warmup
    pipe._optimizer = optim.Adam(pipe.diffusion_prior.parameters(), lr=learning_rate)
    pipe._lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=pipe._optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prior-specific helper: extract ordered feature pairs for EmbeddingDataset
# ─────────────────────────────────────────────────────────────────────────────
def extract_features_ordered(sub, eeg_model, dataloader, device):
    """Extract paired (eeg_features, img_features) in dataset order (shuffle=False)."""
    eeg_model.eval()
    from models.atms import extract_id_from_string
    eeg_list, img_list = [], []
    subject_id = extract_id_from_string(sub)
    with torch.no_grad():
        for eeg_data, labels, text, text_features, img, img_features in dataloader:
            eeg_data = eeg_data.to(device)
            batch_size = eeg_data.size(0)
            subject_ids = torch.full((batch_size,), subject_id,
                                     dtype=torch.long, device=device)
            eeg_feat = eeg_model(eeg_data, subject_ids)
            eeg_list.append(eeg_feat.detach().cpu())
            img_list.append(img_features.cpu())
    return torch.cat(eeg_list, 0), torch.cat(img_list, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='ATMS + Diffusion Prior Joint Training')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--img_dir_training', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--img_dir_test', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--output_dir', type=str, default='./outputs/benchmark')
    parser.add_argument('--model_save_dir', type=str, default='./models/benchmark')
    parser.add_argument('--features_dir', type=str, default=None,
                        help='Directory for CLIP feature cache. '
                             'Defaults to EEG_Image_decode/features/ (shared with Retrieval). '
                             'Set explicitly only when you want a different cache location.')
    parser.add_argument('--subject', type=str, default='sub-08')
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--encoder_only_ratio', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_encoder', type=float, default=3e-4)
    parser.add_argument('--lr_prior', type=float, default=1e-3)
    parser.add_argument('--prior_epochs_per_step', type=int, default=1)
    parser.add_argument('--prior_batch_size', type=int, default=1024)
    parser.add_argument('--prior_dropout', type=float, default=0.1)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of training conditions held out for validation')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without val improvement)')
    parser.add_argument('--encoder_finetuning', action='store_true',
                        help='If set, encoder keeps training jointly with prior in Phase 2')
    parser.add_argument('--avg_trials', action='store_true',
                        help='Average the 4 trials per condition into one signal '
                             'before training (reduces noise, shrinks dataset 4x).')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    sub = args.subject
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    encoder_only_epochs = int(args.total_epochs * args.encoder_only_ratio)

    # ── Data ─────────────────────────────────────────────────────────────
    full_train_dataset = EEGDataset(args.data_path,
                                    img_dir_training=args.img_dir_training,
                                    img_dir_test=args.img_dir_test,
                                    features_dir=args.features_dir,
                                    subjects=[sub], train=True,
                                    avg_trials=args.avg_trials)
    img_features_all = full_train_dataset.img_features          # (16540, 1024)
    img_features_per_class = img_features_all[::10].clone()     # (1654, 1024)

    # Stratified split: 9 conditions → train, 1 condition → val (per class)
    tpc = 1 if args.avg_trials else 4
    train_indices, val_indices = stratified_condition_split(
        n_classes=1654, conditions_per_class=10,
        trials_per_condition=tpc, val_ratio=args.val_ratio, seed=args.seed,
    )
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    train_loader_ordered = DataLoader(train_subset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=0)
    # Full dataset loader for prior training (uses all 66160 samples, not just 90%)
    full_train_loader_ordered = DataLoader(full_train_dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    finetune = args.encoder_finetuning
    phase2_desc = "encoder + prior jointly" if finetune else "encoder frozen, prior only"

    print(f"=== Training Schedule ===")
    print(f"  Total epochs:    {args.total_epochs} (max)")
    print(f"  Phase 1 (encoder):  epoch 1 ~ {encoder_only_epochs}  [encoder trains, prior frozen]")
    print(f"  Phase 2:            epoch {encoder_only_epochs + 1} ~ {args.total_epochs}  [{phase2_desc}]")
    print(f"  Encoder finetuning: {finetune}")
    print(f"  Subject:         {sub}")
    print(f"  Train samples:   {len(train_indices)}  ({len(train_indices)/len(full_train_dataset)*100:.0f}%)")
    print(f"  Val samples:     {len(val_indices)}  ({len(val_indices)/len(full_train_dataset)*100:.0f}%)")
    print(f"  Early stopping:  patience = {args.patience} epochs")
    print(f"========================")

    # ── Models ───────────────────────────────────────────────────────────
    eeg_model = ATMS()
    eeg_model.to(device)
    encoder_optimizer = AdamW(eeg_model.parameters(), lr=args.lr_encoder)

    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=args.prior_dropout)
    pipe = Pipe(diffusion_prior, device=device)

    # ── Directories ──────────────────────────────────────────────────────
    encoder_save_dir = os.path.join(args.model_save_dir, 'encoder', sub, current_time)
    prior_save_dir = os.path.join(args.model_save_dir, 'prior', sub, current_time)
    results_dir = os.path.join(args.output_dir, sub, current_time)
    os.makedirs(encoder_save_dir, exist_ok=True)
    os.makedirs(prior_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────
    results = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_encoder_epoch = 0
    best_prior_loss = float('inf')
    best_prior_epoch = 0
    patience_counter = 0
    prior_initialized = False
    encoder_frozen = False
    encoder_done = False
    phase2_started = False

    for epoch in range(args.total_epochs):
        is_prior_phase = epoch >= encoder_only_epochs

        # Skip remaining phase 1 epochs if encoder already early-stopped
        if not is_prior_phase and encoder_done:
            continue

        # ── Phase transition ──
        if is_prior_phase and not phase2_started:
            best_enc_file = os.path.join(encoder_save_dir, 'best.pth')
            if os.path.exists(best_enc_file):
                eeg_model.load_state_dict(torch.load(best_enc_file, map_location=device))
                print(f"[INFO] Loaded best encoder weights from epoch {best_encoder_epoch}")

            if not finetune:
                for param in eeg_model.parameters():
                    param.requires_grad = False
                eeg_model.eval()
                encoder_frozen = True

            phase2_started = True
            patience_counter = 0
            print(f"\n{'='*55}")
            if finetune:
                print(f"Phase 2: Joint training (encoder + prior). "
                      f"Best Phase-1 encoder: epoch {best_encoder_epoch} "
                      f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
            else:
                print(f"Phase 2: Encoder frozen, prior-only. "
                      f"Best encoder: epoch {best_encoder_epoch} "
                      f"(val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
            print(f"{'='*55}\n")

        phase_str = ("Joint" if finetune else "Prior-only") if is_prior_phase else "Encoder-only"

        # 1. Train encoder (Phase 1, or Phase 2 with finetuning)
        train_loss, train_acc = None, None
        if not is_prior_phase or finetune:
            train_loss, train_acc = train_encoder_epoch(
                sub, eeg_model, train_loader, encoder_optimizer, device,
                img_features_per_class,
                loss_mode='generation', alpha=0.90,
            )

        # 2. Phase 2: extract features & train prior
        prior_loss = None
        if is_prior_phase:
            if not finetune and not prior_initialized:
                # Frozen encoder → features constant, extract once
                eeg_feats, img_feats = extract_features_ordered(
                    sub, eeg_model, full_train_loader_ordered, device)
                prior_dataset = EmbeddingDataset(c_embeddings=eeg_feats, h_embeddings=img_feats)
                prior_loader = DataLoader(prior_dataset, batch_size=args.prior_batch_size,
                                          shuffle=True, num_workers=0)
                remaining = args.total_epochs - encoder_only_epochs
                setup_prior_optimizer(pipe, prior_loader, remaining, args.lr_prior)
                prior_initialized = True
            elif finetune:
                # Finetuning → encoder changes, re-extract every epoch
                eeg_feats, img_feats = extract_features_ordered(
                    sub, eeg_model, full_train_loader_ordered, device)
                prior_dataset = EmbeddingDataset(c_embeddings=eeg_feats, h_embeddings=img_feats)
                prior_loader = DataLoader(prior_dataset, batch_size=args.prior_batch_size,
                                          shuffle=True, num_workers=0)
                if not prior_initialized:
                    remaining = args.total_epochs - encoder_only_epochs
                    setup_prior_optimizer(pipe, prior_loader, remaining, args.lr_prior)
                    prior_initialized = True

            for _ in range(args.prior_epochs_per_step):
                prior_loss = train_prior_one_epoch(pipe, prior_loader)

        # 3. Evaluate on VALIDATION set (never test set)
        val_loss, val_acc = evaluate_val(
            sub, eeg_model, val_loader, device, img_features_per_class,
            k=200, loss_mode='generation', alpha=0.99)

        # 4. Logging
        epoch_results = {
            "epoch": epoch + 1, "phase": phase_str,
            "train_loss": f"{train_loss:.4f}" if train_loss is not None else "N/A",
            "train_acc": f"{train_acc:.4f}" if train_acc is not None else "N/A",
            "val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.4f}",
            "prior_loss": f"{prior_loss:.4f}" if prior_loss is not None else "N/A",
        }
        results.append(epoch_results)

        if not is_prior_phase:
            print(f"[{phase_str}] Epoch {epoch+1}/{args.total_epochs} "
                  f"| Train L={train_loss:.4f} A={train_acc:.4f} "
                  f"| Val L={val_loss:.4f} A={val_acc:.4f}")
        elif finetune:
            print(f"[{phase_str}] Epoch {epoch+1}/{args.total_epochs} "
                  f"| Train L={train_loss:.4f} A={train_acc:.4f} "
                  f"| Prior L={prior_loss:.4f} "
                  f"| Val L={val_loss:.4f} A={val_acc:.4f}")
        else:
            print(f"[{phase_str}] Epoch {epoch+1}/{args.total_epochs} "
                  f"| Prior L={prior_loss:.4f} "
                  f"| Val L={val_loss:.4f} A={val_acc:.4f}")

        # 5. Model selection & early stopping
        if not is_prior_phase:
            # Phase 1: select best encoder by val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_encoder_epoch = epoch + 1
                patience_counter = 0
                torch.save(eeg_model.state_dict(),
                           os.path.join(encoder_save_dir, 'best.pth'))
                print(f"  ★ New best encoder: val_loss={best_val_loss:.4f} acc={best_val_acc:.4f}")
            else:
                patience_counter += 1
        elif finetune:
            # Phase 2 joint: select by val_loss, save both models
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_encoder_epoch = epoch + 1
                best_prior_epoch = epoch + 1
                patience_counter = 0
                torch.save(eeg_model.state_dict(),
                           os.path.join(encoder_save_dir, 'best.pth'))
                torch.save(pipe.diffusion_prior.state_dict(),
                           os.path.join(prior_save_dir, 'best.pth'))
                print(f"  ★ New best joint: val_loss={best_val_loss:.4f} acc={best_val_acc:.4f}")
            else:
                patience_counter += 1
        else:
            # Phase 2 frozen: select best prior by prior_loss
            if prior_loss < best_prior_loss:
                best_prior_loss = prior_loss
                best_prior_epoch = epoch + 1
                patience_counter = 0
                torch.save(pipe.diffusion_prior.state_dict(),
                           os.path.join(prior_save_dir, 'best.pth'))
                print(f"  ★ New best prior: loss={best_prior_loss:.4f}")
            else:
                patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            if is_prior_phase:
                torch.save(pipe.diffusion_prior.state_dict(),
                           os.path.join(prior_save_dir, f'{epoch+1}.pth'))
            if not encoder_frozen:
                torch.save(eeg_model.state_dict(),
                           os.path.join(encoder_save_dir, f'{epoch+1}.pth'))

        # Early stopping
        if patience_counter >= args.patience:
            if not is_prior_phase:
                print(f"\n[Early Stop] No encoder improvement for {args.patience} epochs. "
                      f"Best encoder epoch = {best_encoder_epoch}.")
                encoder_done = True
                print("[INFO] Skipping to Phase 2...")
            else:
                phase_name = "joint" if finetune else "prior"
                best_ep = best_encoder_epoch if finetune else best_prior_epoch
                print(f"\n[Early Stop] No {phase_name} improvement for {args.patience} epochs. "
                      f"Best epoch = {best_ep}.")
                break

    # ── Ensure best checkpoints exist ─────────────────────────────────
    best_prior_path = os.path.join(prior_save_dir, 'best.pth')
    if not os.path.exists(best_prior_path):
        torch.save(pipe.diffusion_prior.state_dict(), best_prior_path)

    best_encoder_path = os.path.join(encoder_save_dir, 'best.pth')
    if not os.path.exists(best_encoder_path):
        torch.save(eeg_model.state_dict(), best_encoder_path)

    print(f"\n{'='*55}")
    print(f"Training finished at epoch {epoch+1}")
    print(f"Best encoder: epoch {best_encoder_epoch}  val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.4f}")
    print(f"Best prior:   epoch {best_prior_epoch}  prior_loss={best_prior_loss:.4f}")
    print(f"  Encoder: {best_encoder_path}")
    print(f"  Prior:   {best_prior_path}")
    print(f"{'='*55}")

    # ── Save training log ────────────────────────────────────────────────
    results_file = os.path.join(results_dir, 'training_log.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Training log: {results_file}")

    # ── Write paths info for the evaluation script ───────────────────────
    info_path = os.path.join(results_dir, 'paths_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"encoder_path={best_encoder_path}\n")
        f.write(f"prior_path={best_prior_path}\n")
        f.write(f"subject={sub}\n")
        f.write(f"timestamp={current_time}\n")
        f.write(f"best_encoder_epoch={best_encoder_epoch}\n")
        f.write(f"best_val_loss={best_val_loss:.4f}\n")
        f.write(f"best_val_acc={best_val_acc:.4f}\n")
        f.write(f"best_prior_epoch={best_prior_epoch}\n")
        f.write(f"best_prior_loss={best_prior_loss:.4f}\n")
    print(f"Paths info:   {info_path}")


if __name__ == '__main__':
    main()
