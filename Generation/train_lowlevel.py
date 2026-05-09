"""
Low-level EEG-to-image encoder training.

The encoder_low_level model maps EEG signals directly to SDXL-VAE latent
space (4 × 64 × 64).  Training targets are pre-computed VAE latents stored in
  <latent_dir>/train_image_latent_512.pt   (key: 'image_latent')
  <latent_dir>/test_image_latent_512.pt

Trained with an MAE (L1) regression loss between predicted and true latents.
The frozen SDXL VAE is used only for visual logging (sample reconstruction).

Outputs
-------
  <output_dir>/<subject>/<run_id>/
      best_encoder.pth    -- encoder checkpoint (best val loss)
      last_encoder.pth    -- encoder checkpoint (last epoch)
      paths_info.txt      -- paths consumed by the evaluation script
"""

import os
import sys
import csv
import argparse
import datetime
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrained_paths import SDXL_BASE_DIR

from eegdatasets import EEGDataset
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


# ── Encoder architecture ──────────────────────────────────────────────────────

class encoder_low_level(nn.Module):
    """EEG → SDXL-VAE latent (4 × 64 × 64).

    A subject-wise linear layer followed by a CNN transpose upsampler that
    goes from a 1×1 bottleneck to 4×64×64.
    """

    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.subject_linear = nn.ModuleList(
            [nn.Linear(sequence_length, 128) for _ in range(num_subjects)])
        self.dropout = nn.Dropout(0.5)

        # Bottleneck: (B, 63*128, 1, 1) = (B, 8064, 1, 1)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(8064, 1024, 4, 2, 1),  # → (2,2)
            nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),   # → (4,4)
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),    # → (8,8)
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # → (16,16)
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),     # → (32,32)
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),      # → (64,64)
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 4, 1, 1, 0),       # → (4,64,64)
        )

    def forward(self, x):
        # x: (B, 63, 250)
        x = self.subject_linear[0](x[..., :250])  # (B, 63, 128)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1, 1, 1)        # (B, 8064, 1, 1)
        return self.upsampler(x)                   # (B, 4, 64, 64)


# ── Training / evaluation loops ───────────────────────────────────────────────

def _train_epoch(encoder, vae, loader, optimizer, device, save_dir, epoch):
    encoder.train()
    mae_fn = nn.L1Loss()
    total_loss = 0.0
    logged = False

    for batch_idx, (eeg, labels, _text, _tf, _img, img_latent) in enumerate(loader):
        eeg = eeg.to(device)
        img_latent = img_latent.to(device).float()

        optimizer.zero_grad()
        pred = encoder(eeg).float()
        loss = mae_fn(pred, img_latent)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Save one visual example per epoch for inspection
        if not logged and vae is not None:
            with torch.no_grad():
                x_pred = vae.decode(pred[:1]).sample
                x_true = vae.decode(img_latent[:1]).sample
                _save_vae_image(x_pred, os.path.join(save_dir, f'e{epoch}_pred.png'))
                _save_vae_image(x_true, os.path.join(save_dir, f'e{epoch}_true.png'))
            logged = True

        del eeg, img_latent, pred

    torch.cuda.empty_cache()
    return total_loss / (batch_idx + 1)


@torch.no_grad()
def _eval_epoch(encoder, loader, device):
    encoder.eval()
    mae_fn = nn.L1Loss()
    total_loss = 0.0

    for batch_idx, (eeg, _labels, _text, _tf, _img, img_latent) in enumerate(loader):
        eeg = eeg.to(device)
        img_latent = img_latent.to(device).float()
        pred = encoder(eeg).float()
        total_loss += mae_fn(pred, img_latent).item()
        del eeg, img_latent, pred

    torch.cuda.empty_cache()
    return total_loss / (batch_idx + 1)


def _save_vae_image(tensor, path):
    """Save a single-image VAE decode tensor as PNG."""
    from diffusers.image_processor import VaeImageProcessor
    processor = VaeImageProcessor()
    imgs = processor.postprocess(tensor, output_type='pil')
    imgs[0].save(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Low-level EEG encoder training')
    parser.add_argument('--data_path', type=str,
                        default='/root/autodl-tmp/THINGS/Preprocessed_data_250Hz')
    parser.add_argument('--img_dir_training', type=str, required=True)
    parser.add_argument('--img_dir_test', type=str, required=True)
    parser.add_argument('--latent_dir', type=str, required=True,
                        help='Directory containing train/test_image_latent_512.pt')
    parser.add_argument('--output_dir', type=str, default='./outputs/lowlevel')
    parser.add_argument('--subject', type=str, default='sub-08')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without val improvement)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of training data held out for validation')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--log_images', action='store_true',
                        help='Decode and save sample images each epoch (needs SDXL VAE)')
    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    run_id = datetime.datetime.now().strftime('%m-%d_%H-%M')
    sub = args.subject
    save_dir = os.path.join(args.output_dir, sub, run_id)
    os.makedirs(save_dir, exist_ok=True)
    img_log_dir = os.path.join(save_dir, 'img_log')
    if args.log_images:
        os.makedirs(img_log_dir, exist_ok=True)

    # ── Load datasets ──────────────────────────────────────────────────────
    common = dict(data_path=args.data_path,
                  img_dir_training=args.img_dir_training,
                  img_dir_test=args.img_dir_test,
                  feature_type='vae_latent',
                  latent_dir=args.latent_dir,
                  subjects=[sub])

    full_train = EEGDataset(**common, train=True)

    # Stratified 9:1 split by class
    n_total = len(full_train)
    n_cls = full_train.n_cls
    per_class = n_total // n_cls
    n_val_per_cls = max(1, int(per_class * args.val_ratio))
    val_idx, train_idx = [], []
    for c in range(n_cls):
        start = c * per_class
        end = start + per_class
        indices = list(range(start, end))
        val_idx.extend(indices[:n_val_per_cls])
        train_idx.extend(indices[n_val_per_cls:])

    from torch.utils.data import Subset
    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_train, val_idx)
    test_set = EEGDataset(**common, train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, drop_last=False)

    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    # ── Build model ────────────────────────────────────────────────────────
    encoder = encoder_low_level().to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                               eta_min=1e-6)

    # ── Optionally load SDXL VAE for visual logging ────────────────────────
    vae = None
    if args.log_images:
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl \
            import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            SDXL_BASE_DIR,
            torch_dtype=torch.float, variant='fp16',
            local_files_only=True)
        vae = pipe.vae.to(device)
        vae.requires_grad_(False)
        vae.eval()
        del pipe

    # ── Training loop with early stopping ─────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    results = []

    print(f"Training {sub}  epochs={args.epochs}  patience={args.patience}")
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(encoder, vae, train_loader, optimizer, device,
                                  img_log_dir if args.log_images else save_dir, epoch)
        val_loss = _eval_epoch(encoder, val_loader, device)
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(encoder.state_dict(), os.path.join(save_dir, 'best_encoder.pth'))
        else:
            patience_counter += 1

        results.append({'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best': improved})

        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}"
              + ('  *' if improved else ''))

        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs "
                  f"(no improvement for {args.patience} epochs).")
            break

    # Save last checkpoint and results CSV
    torch.save(encoder.state_dict(), os.path.join(save_dir, 'last_encoder.pth'))

    csv_path = os.path.join(save_dir, f'training_log_{sub}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Save paths_info.txt for downstream evaluation
    best_ckpt = os.path.join(save_dir, 'best_encoder.pth')
    with open(os.path.join(save_dir, 'paths_info.txt'), 'w') as f:
        f.write(f"encoder_path={best_ckpt}\n")
        f.write(f"subject={sub}\n")
        f.write(f"latent_dir={args.latent_dir}\n")
        f.write(f"img_dir_training={args.img_dir_training}\n")
        f.write(f"img_dir_test={args.img_dir_test}\n")
        f.write(f"data_path={args.data_path}\n")

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {best_ckpt}")
    print(f"Log: {csv_path}")


if __name__ == '__main__':
    main()
