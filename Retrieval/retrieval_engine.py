"""Shared training/evaluation engine for EEG retrieval experiments.

All train_model / evaluate_model / main_train_loop logic from train.py,
train_joint.py, train_contrast.py, and train_atme.py is merged here.

Supports two calling conventions:
  - ATMS-style: model(eeg_data, subject_ids), L2-normalized features
  - Generic-style: model(eeg_data), features used as-is

Controlled by ``use_subject_id`` and ``normalize_feats`` flags that are
set automatically by train_unified.py based on ``encoder_type``.
"""

import os
import csv
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA = 0.99   # image-loss weight in the combined img+text contrastive loss
EVAL_KS = [2, 4, 10, 50, 100, 200]  # k-way retrieval sizes evaluated each epoch


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------
class EMA:
    """Maintains shadow copies of all trainable parameters.

    Call ``update()`` after each optimizer step, ``apply_shadow()`` before
    evaluation, and ``restore()`` afterwards to resume training.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
LOGIT_SCALE_TYPES = ('exp', 'linear', 'softplus', 'fixed')


def _get_logit_scale(model, scale_type='exp'):
    """Return the effective logit scale from the model's raw parameter.

    The raw ``model.logit_scale`` is an ``nn.Parameter`` initialised to
    ``ln(1/0.07) ≈ 2.66``.  How it maps to the actual temperature multiplier
    depends on *scale_type*:

      exp       param.exp()        — OpenAI CLIP style (init ≈ 14.29)
      linear    param              — direct use (init ≈ 2.66)
      softplus  softplus(param)    — smooth, always positive (init ≈ 2.73)
      fixed     param.exp().detach() — non-learnable, stays at init value
    """
    raw = model.logit_scale
    if scale_type == 'exp':
        return raw.exp().clamp(max=100.0)
    if scale_type == 'linear':
        return raw
    if scale_type == 'softplus':
        return F.softplus(raw)
    if scale_type == 'fixed':
        return raw.exp().clamp(max=100.0).detach()
    raise ValueError(f"Unknown logit_scale_type {scale_type!r}. "
                     f"Choose from: {LOGIT_SCALE_TYPES}")


def _get(config, key, default=None):
    """Read a value from either an argparse Namespace or a plain dict."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _subject_id(sub):
    """Extract the integer subject index from a string like 'sub-03' -> 2."""
    import re
    m = re.search(r'\d+', sub)
    return int(m.group()) - 1 if m else 0


# ---------------------------------------------------------------------------
# Core training / evaluation functions
# ---------------------------------------------------------------------------
def train_epoch(sub, eeg_model, dataloader, optimizer, device,
                text_features_all, img_features_all, config,
                use_subject_id=False, normalize_feats=False,
                ema=None, logit_scale_type='exp'):
    """Train one epoch.

    Args:
        sub: subject string, e.g. 'sub-03'.
        use_subject_id: If True, passes integer subject id to the model.
        normalize_feats: If True, L2-normalises EEG / image / text features
            before computing the contrastive loss (required for ATMS).

    Returns:
        (avg_loss, accuracy, eeg_feature_tensor)
    """
    eeg_model.train()
    sid = _subject_id(sub) if use_subject_id else None

    img_pool = img_features_all.to(device).float()
    # Pool for fast per-batch accuracy estimate: full (normalised) vs subsampled
    if normalize_feats:
        img_pool_ref = F.normalize(img_pool, dim=-1)
    else:
        img_pool_ref = img_pool[::10]

    features_list = []
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (eeg_data, labels, _text, text_feats, _img, img_feats) in enumerate(dataloader):
        eeg_data  = eeg_data.to(device)
        img_feats = img_feats.to(device).float()
        txt_feats = text_feats.to(device).float()
        labels    = labels.to(device)
        bs        = eeg_data.size(0)

        optimizer.zero_grad()

        if use_subject_id:
            subject_ids = torch.full((bs,), sid, dtype=torch.long, device=device)
            eeg_out = eeg_model(eeg_data, subject_ids)
        else:
            eeg_out = eeg_model(eeg_data)

        # Some models return (clip_out, mse_out) – use only clip_out
        eeg_features = (eeg_out[0] if isinstance(eeg_out, tuple) else eeg_out).float()
        features_list.append(eeg_features.detach())

        logit_scale = _get_logit_scale(eeg_model, logit_scale_type)

        if normalize_feats:
            eeg_f = F.normalize(eeg_features, dim=-1)
            img_f = F.normalize(img_feats, dim=-1)
            txt_f = F.normalize(txt_feats, dim=-1)
        else:
            eeg_f, img_f, txt_f = eeg_features, img_feats, txt_feats

        img_loss  = eeg_model.loss_func(eeg_f, img_f, logit_scale)
        text_loss = eeg_model.loss_func(eeg_f, txt_f, logit_scale)
        loss = ALPHA * img_loss + (1 - ALPHA) * text_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if ema is not None:
            ema.update(eeg_model)

        # Full-pool accuracy (approximate via subsampled pool)
        logits    = logit_scale * eeg_f @ img_pool_ref.T
        predicted = torch.argmax(logits, dim=1)
        total    += bs
        correct  += (predicted == labels).sum().item()

        del eeg_data, eeg_features, img_feats, txt_feats

    n = batch_idx + 1
    return total_loss / n, correct / total, torch.cat(features_list, dim=0)


def evaluate(sub, eeg_model, dataloader, device,
             text_features_all, img_features_all, k, config,
             use_subject_id=False, normalize_feats=False,
             logit_scale_type='exp'):
    """k-way retrieval evaluation.

    For each test sample, k-1 distractors are sampled at random from all
    classes and the model must identify the correct class (top-1 and top-5).

    Returns:
        (avg_loss, top1_accuracy, top5_accuracy)
    """
    eeg_model.eval()
    sid = _subject_id(sub) if use_subject_id else None

    img_pool = img_features_all.to(device).float()
    txt_pool = text_features_all.to(device).float()
    img_pool_ref = F.normalize(img_pool, dim=-1) if normalize_feats else img_pool
    all_labels   = set(range(img_pool.size(0)))

    total_loss = 0.0
    correct = top5_correct = total = 0

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, _text, text_feats, _img, img_feats) in enumerate(dataloader):
            eeg_data  = eeg_data.to(device)
            img_feats = img_feats.to(device).float()
            txt_feats = text_feats.to(device).float()
            labels    = labels.to(device)
            bs        = eeg_data.size(0)

            if use_subject_id:
                subject_ids = torch.full((bs,), sid, dtype=torch.long, device=device)
                eeg_out = eeg_model(eeg_data, subject_ids)
            else:
                eeg_out = eeg_model(eeg_data)

            eeg_features = (eeg_out[0] if isinstance(eeg_out, tuple) else eeg_out).float()
            logit_scale  = _get_logit_scale(eeg_model, logit_scale_type)

            if normalize_feats:
                eeg_f = F.normalize(eeg_features, dim=-1)
                img_f = F.normalize(img_feats, dim=-1)
                txt_f = F.normalize(txt_feats, dim=-1)
            else:
                eeg_f, img_f, txt_f = eeg_features, img_feats, txt_feats

            img_loss  = eeg_model.loss_func(eeg_f, img_f, logit_scale)
            text_loss = eeg_model.loss_func(eeg_f, txt_f, logit_scale)
            total_loss += (ALPHA * img_loss + (1 - ALPHA) * text_loss).item()

            for i, label in enumerate(labels):
                possible = list(all_labels - {label.item()})
                selected = random.sample(possible, k - 1) + [label.item()]
                sel_feats = img_pool_ref[selected]
                logits    = logit_scale * eeg_f[i] @ sel_feats.T

                pred = selected[torch.argmax(logits).item()]
                if pred == label.item():
                    correct += 1
                if k >= 5:
                    _, top5_idx = torch.topk(logits, 5)
                    if label.item() in [selected[j] for j in top5_idx.tolist()]:
                        top5_correct += 1
                total += 1

            del eeg_data, eeg_features, img_feats, txt_feats

    n = batch_idx + 1
    top5_acc = top5_correct / total if total > 0 else 0.0
    return total_loss / n, correct / total, top5_acc


# ---------------------------------------------------------------------------
# Validation loss (same contrastive loss as training, computed in eval mode)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_val_loss(sub, eeg_model, val_dataloader, device,
                     use_subject_id=False, normalize_feats=False,
                     logit_scale_type='exp'):
    """Compute the average contrastive loss on a held-out validation set.

    Uses exactly the same loss formula as ``train_epoch`` (alpha-weighted
    image + text CLIP loss), but in eval mode with no gradient updates.
    The val_dataloader should yield batches that already carry per-sample
    image/text feature tensors (i.e. the same EEGDataset split as training).

    Returns:
        avg_val_loss (float)
    """
    eeg_model.eval()
    sid = _subject_id(sub) if use_subject_id else None
    total_loss = 0.0

    for batch_idx, (eeg_data, _labels, _text, text_feats, _img, img_feats) in enumerate(val_dataloader):
        eeg_data  = eeg_data.to(device)
        img_feats = img_feats.to(device).float()
        txt_feats = text_feats.to(device).float()
        bs        = eeg_data.size(0)

        if use_subject_id:
            subject_ids = torch.full((bs,), sid, dtype=torch.long, device=device)
            eeg_out = eeg_model(eeg_data, subject_ids)
        else:
            eeg_out = eeg_model(eeg_data)

        eeg_features = (eeg_out[0] if isinstance(eeg_out, tuple) else eeg_out).float()
        logit_scale  = _get_logit_scale(eeg_model, logit_scale_type)

        if normalize_feats:
            eeg_f = F.normalize(eeg_features, dim=-1)
            img_f = F.normalize(img_feats,    dim=-1)
            txt_f = F.normalize(txt_feats,    dim=-1)
        else:
            eeg_f, img_f, txt_f = eeg_features, img_feats, txt_feats

        img_loss  = eeg_model.loss_func(eeg_f, img_f, logit_scale)
        text_loss = eeg_model.loss_func(eeg_f, txt_f, logit_scale)
        total_loss += (ALPHA * img_loss + (1 - ALPHA) * text_loss).item()

        del eeg_data, eeg_features, img_feats, txt_feats

    return total_loss / (batch_idx + 1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------
def train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader,
               optimizer, device,
               text_features_train_all, text_features_test_all,
               img_features_train_all, img_features_test_all,
               config, logger=None,
               use_subject_id=False, normalize_feats=False,
               val_dataloader=None, ema_decay=0.0,
               logit_scale_type='exp'):
    """Run the full training loop with checkpointing, logging, and CSV export.

    Args:
        sub: subject identifier string (used for checkpoint/output paths).
        current_time: formatted timestamp string for run identification.
        config: argparse Namespace or dict with training hyperparameters.
            Reads ``epochs``, ``encoder_type``, ``mode``, and
            ``early_stopping_patience`` (0 = disabled).
        logger: boolean – whether to initialise a W&B logger.
        use_subject_id: passed through to train_epoch / evaluate / compute_val_loss.
        normalize_feats: passed through to train_epoch / evaluate / compute_val_loss.
        val_dataloader: DataLoader over a held-out subset of the **training** set
            (created via stratified or random split). When provided, val loss is
            computed with the same contrastive objective as training and is used
            for early stopping and printed each epoch. When None, early stopping
            falls back to the test-set loss (legacy behaviour).

    Returns:
        List of per-epoch result dicts.
    """
    from models.util import wandb_logger as _wandb_logger

    _logger = None
    if logger:
        _logger = _wandb_logger(config)
        _logger.watch(eeg_model, _logger)

    epochs       = _get(config, 'epochs', 40)
    encoder_type = _get(config, 'encoder_type', 'unknown')
    mode         = _get(config, 'mode', 'intra')
    patience     = _get(config, 'early_stopping_patience', 10)
    output_dir   = _get(config, 'output_dir', '.')

    use_proper_val = val_dataloader is not None

    # Checkpoint directory (created on first use)
    if mode == 'joint':
        ckpt_dir = os.path.join("./models/contrast/joint", encoder_type, current_time)
    elif mode == 'loso':
        ckpt_dir = os.path.join("./models/contrast/loso", encoder_type, current_time)
    else:
        ckpt_dir = os.path.join("./models/contrast", encoder_type, sub, current_time)
    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")

    train_losses, val_losses_log          = [], []
    train_accuracies                       = []
    test_losses, test_accuracies           = [], []
    v2_accs, v4_accs, v10_accs            = [], [], []

    best_accuracy    = 0.0
    best_val_loss    = float('inf')
    patience_counter = 0
    best_epoch_info  = {}
    results          = []

    if use_proper_val:
        print(f"  Val split: proper held-out val loader  "
              f"(early stopping on val contrastive loss, patience={patience})")
    else:
        print(f"  Val split: none — using test-set loss for early stopping "
              f"(patience={patience})")

    ema = EMA(eeg_model, decay=ema_decay) if ema_decay > 0 else None
    if ema is not None:
        print(f"  EMA enabled: decay={ema_decay}")
    init_scale = _get_logit_scale(eeg_model, logit_scale_type)
    print(f"  Logit scale: type={logit_scale_type}  init={init_scale.item():.4f}")

    for epoch in range(epochs):
        train_loss, train_accuracy, _ = train_epoch(
            sub, eeg_model, train_dataloader, optimizer, device,
            text_features_train_all, img_features_train_all, config,
            use_subject_id=use_subject_id, normalize_feats=normalize_feats,
            ema=ema, logit_scale_type=logit_scale_type)

        if (epoch + 1) % 5 == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{epoch + 1}.pth")
            torch.save(eeg_model.state_dict(), ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        if ema is not None:
            ema.apply_shadow(eeg_model)

        # Compute val loss on the proper held-out split (same objective as training)
        if use_proper_val:
            val_loss = compute_val_loss(
                sub, eeg_model, val_dataloader, device,
                use_subject_id=use_subject_id, normalize_feats=normalize_feats,
                logit_scale_type=logit_scale_type)
        else:
            val_loss = None   # filled after k-way eval below

        # k-way retrieval metrics on the test set
        eval_kwargs = dict(
            sub=sub, eeg_model=eeg_model, dataloader=test_dataloader,
            device=device,
            text_features_all=text_features_test_all,
            img_features_all=img_features_test_all,
            config=config,
            use_subject_id=use_subject_id, normalize_feats=normalize_feats,
            logit_scale_type=logit_scale_type,
        )
        test_loss, test_accuracy, top5_acc = evaluate(k=200, **eval_kwargs)
        _, v2_acc,  _           = evaluate(k=2,   **eval_kwargs)
        _, v4_acc,  _           = evaluate(k=4,   **eval_kwargs)
        _, v10_acc, _           = evaluate(k=10,  **eval_kwargs)
        _, v50_acc, v50_top5    = evaluate(k=50,  **eval_kwargs)
        _, v100_acc, v100_top5  = evaluate(k=100, **eval_kwargs)

        if val_loss is None:
            val_loss = test_loss   # fallback: use k=200 test loss

        val_losses_log.append(val_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)

        epoch_results = {
            "epoch":         epoch + 1,
            "train_loss":    train_loss,
            "val_loss":      val_loss,
            "test_loss":     test_loss,
            "test_accuracy": test_accuracy,
            "v2_acc":        v2_acc,
            "v4_acc":        v4_acc,
            "v10_acc":       v10_acc,
            "top5_acc":      top5_acc,
            "v50_acc":       v50_acc,
            "v100_acc":      v100_acc,
            "v50_top5_acc":  v50_top5,
            "v100_top5_acc": v100_top5,
        }
        results.append(epoch_results)

        if test_accuracy > best_accuracy:
            best_accuracy   = test_accuracy
            best_epoch_info = {
                "epoch":          epoch + 1,
                "train_loss":     train_loss,
                "val_loss":       val_loss,
                "train_accuracy": train_accuracy,
                "test_loss":      test_loss,
                "test_accuracy":  test_accuracy,
                "v2_acc":         v2_acc,
                "v4_acc":         v4_acc,
                "v10_acc":        v10_acc,
            }

        # ------------------------------------------------------------------
        # Early stopping: monitor val loss; stop if no improvement for
        # `patience` consecutive epochs (patience=0 disables this).
        # ------------------------------------------------------------------
        if patience > 0:
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                # Save best-model weights whenever val loss improves
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(eeg_model.state_dict(), best_ckpt_path)
            else:
                patience_counter += 1

        if ema is not None:
            ema.restore(eeg_model)

        if _logger:
            _logger.log({
                "Train Loss":      train_loss,
                "Val Loss":        val_loss,
                "Train Accuracy":  train_accuracy,
                "Test Loss":       test_loss,
                "Test Accuracy":   test_accuracy,
                "v2 Accuracy":     v2_acc,
                "v4 Accuracy":     v4_acc,
                "v10 Accuracy":    v10_acc,
                "Patience counter":patience_counter,
                "Epoch":           epoch,
            })

        es_tag = (f"  [ES {patience_counter}/{patience}]"
                  if patience > 0 else "")
        val_src = "" if use_proper_val else " (test)"
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Val Loss{val_src}: {val_loss:.4f} | "
              f"Train Acc: {train_accuracy:.4f}  Top5: {top5_acc:.4f}{es_tag}")
        print(f"           v2={v2_acc:.4f}  v4={v4_acc:.4f}  v10={v10_acc:.4f}  "
              f"v50={v50_acc:.4f}  v100={v100_acc:.4f}")

        if patience > 0 and patience_counter >= patience:
            print(f"\nEarly stopping triggered: val loss has not improved for "
                  f"{patience} epochs (best val loss = {best_val_loss:.6f}).")
            break

    if best_epoch_info:
        plot_dir  = os.path.join(output_dir, encoder_type, sub, current_time)
        os.makedirs(plot_dir, exist_ok=True)
        _plot_curves(
            train_losses, train_accuracies,
            val_losses_log, test_accuracies,
            v2_accs, v4_accs, v10_accs,
            best_epoch_info,
            save_path=os.path.join(plot_dir,
                                   f"curves_{encoder_type}_{sub}_{current_time}.png"),
        )

    # ------------------------------------------------------------------
    # Final evaluation: reload best model (lowest val loss) and report
    # ------------------------------------------------------------------
    if patience > 0 and os.path.exists(best_ckpt_path):
        ema_tag = " [EMA]" if ema is not None else ""
        print(f"\nLoading best model{ema_tag} from {best_ckpt_path} "
              f"(best val loss = {best_val_loss:.6f})")
        eeg_model.load_state_dict(torch.load(best_ckpt_path, map_location=device,
                                             weights_only=True))
    elif ema is not None:
        ema.apply_shadow(eeg_model)

    eval_kwargs_final = dict(
        sub=sub, eeg_model=eeg_model, dataloader=test_dataloader,
        device=device,
        text_features_all=text_features_test_all,
        img_features_all=img_features_test_all,
        config=config,
        use_subject_id=use_subject_id, normalize_feats=normalize_feats,
        logit_scale_type=logit_scale_type,
    )
    _, best_v200, best_top5   = evaluate(k=200, **eval_kwargs_final)
    _, best_v2,   _           = evaluate(k=2,   **eval_kwargs_final)
    _, best_v4,   _           = evaluate(k=4,   **eval_kwargs_final)
    _, best_v10,  _           = evaluate(k=10,  **eval_kwargs_final)
    _, best_v50,  best_v50t5  = evaluate(k=50,  **eval_kwargs_final)
    _, best_v100, best_v100t5 = evaluate(k=100, **eval_kwargs_final)

    ema_label = " EMA" if ema is not None else ""
    label = f"best-val-loss{ema_label} model" if (patience > 0 and os.path.exists(best_ckpt_path)) \
            else "last epoch model"
    print(f"\n{'='*60}")
    print(f"  Final Results ({label})  —  {sub}")
    print(f"{'='*60}")
    print(f"  Top-1 (v200): {best_v200:.4f}   Top-5 (v200): {best_top5:.4f}")
    print(f"  v2={best_v2:.4f}  v4={best_v4:.4f}  v10={best_v10:.4f}  "
          f"v50={best_v50:.4f}  v100={best_v100:.4f}")
    print(f"  v50-top5={best_v50t5:.4f}  v100-top5={best_v100t5:.4f}")
    print(f"{'='*60}\n")

    if _logger:
        _logger.finish()

    return results


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def _plot_curves(train_losses, train_accuracies, val_losses, test_accuracies,
                 v2_accs, v4_accs, v10_accs, best_epoch_info, save_path):
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(val_losses,   label='Val Loss')
    axs[0, 0].legend(); axs[0, 0].set_title("Loss Curve")

    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies,  label='Test Accuracy')
    axs[0, 1].legend(); axs[0, 1].set_title("Accuracy Curve")

    axs[1, 0].plot(v2_accs,  label='2-class Accuracy')
    axs[1, 0].legend(); axs[1, 0].set_title("2-Class Accuracy Curve")

    axs[1, 1].plot(v4_accs,  label='4-class Accuracy')
    axs[1, 1].legend(); axs[1, 1].set_title("4-Class Accuracy Curve")

    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend(); axs[2, 0].set_title("10-Class Accuracy Curve")

    info_text = (
        f"Best (Epoch {best_epoch_info['epoch']}):\n"
        f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
        f"Train Acc:  {best_epoch_info['train_accuracy']:.4f}\n"
        f"Test Loss:  {best_epoch_info['test_loss']:.4f}\n"
        f"Test Acc:   {best_epoch_info['test_accuracy']:.4f}\n"
        f"v2={best_epoch_info['v2_acc']:.4f}  "
        f"v4={best_epoch_info['v4_acc']:.4f}  "
        f"v10={best_epoch_info['v10_acc']:.4f}"
    )
    axs[2, 1].axis('off')
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10,
                   ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()
    plt.suptitle('EEG Retrieval Training Curves', fontsize=14, y=1.02)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature loading (for ATMS / external .pt files)
# ---------------------------------------------------------------------------
def load_features(feature_type, features_dir):
    """Load pre-extracted vision/text feature .pt files.

    Supported feature_type values:
        'ViT-H-14', 'UniLIP-2B', 'UniLIP-3B', 'InternVL3-2B',
        'EVA01_CLIP_g_14_plus'

    Returns dict with keys:
        img_features_train, text_features_train,
        img_features_test,  text_features_test,
        preloaded_train, preloaded_test
    """
    if features_dir is None:
        raise ValueError(
            "--features_dir must be set (path to pre-extracted .pt feature files).")

    def _pt(name):
        return os.path.join(features_dir, name)

    feature_paths = {
        'ViT-H-14': (
            _pt('ViT-H-14_features_train.pt'),
            _pt('ViT-H-14_features_test.pt'),
        ),
        'UniLIP-2B': (
            _pt('UniLIP-2B_vision_features_train.pt'),
            _pt('UniLIP-2B_vision_features_test.pt'),
        ),
        'UniLIP-3B': (
            _pt('UniLIP-3B_vision_features_train.pt'),
            _pt('UniLIP-3B_vision_features_test.pt'),
        ),
        'InternVL3-2B': (
            _pt('InternVL3-2B_vision_features_train.pt'),
            _pt('InternVL3-2B_vision_features_test.pt'),
        ),
        'EVA01_CLIP_g_14_plus': (
            _pt('training_images_eva_clip_embeddings.pt'),
            _pt('test_images_eva_clip_embeddings.pt'),
        ),
    }

    if feature_type not in feature_paths:
        raise ValueError(
            f"Unknown feature_type {feature_type!r}. "
            f"Choose from: {list(feature_paths)}")

    train_path, test_path = feature_paths[feature_type]
    print(f"Loading {feature_type} features:")

    loaded = []
    with tqdm.tqdm([train_path, test_path], desc=f"Loading {feature_type}", unit="file") as pbar:
        for path in pbar:
            pbar.set_postfix_str(os.path.basename(path))
            loaded.append(torch.load(path, map_location='cpu'))
    train_data, test_data = loaded

    if feature_type == 'ViT-H-14':
        text_train = train_data['text_features']
        img_train  = train_data['img_features']
        text_test  = test_data['text_features']
        img_test   = test_data['img_features']
    elif feature_type in ('UniLIP-2B', 'UniLIP-3B'):
        img_train  = train_data['features'][:, 0, :]
        img_test   = test_data['features'][:, 0, :]
        text_train = train_data.get('text_features', img_train.clone())
        text_test  = test_data.get('text_features', img_test.clone())
    elif feature_type == 'InternVL3-2B':
        img_train  = train_data['cls_features']
        img_test   = test_data['cls_features']
        text_train = train_data.get('text_features', img_train.clone())
        text_test  = test_data.get('text_features', img_test.clone())
    elif feature_type == 'EVA01_CLIP_g_14_plus':
        img_train  = train_data['embeddings']
        img_test   = test_data['embeddings']
        text_train = train_data.get('text_features', img_train.clone())
        text_test  = test_data.get('text_features', img_test.clone())

    print(f"  train: img {tuple(img_train.shape)}, text {tuple(text_train.shape)}")
    print(f"  test:  img {tuple(img_test.shape)},  text {tuple(text_test.shape)}")

    return {
        'img_features_train':  img_train,
        'text_features_train': text_train,
        'img_features_test':   img_test,
        'text_features_test':  text_test,
        'preloaded_train': {'img_features': img_train, 'text_features': text_train},
        'preloaded_test':  {'img_features': img_test,  'text_features': text_test},
    }


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def save_results(results, output_dir, encoder_type, sub, mode, current_time):
    """Write per-epoch result dicts to a CSV file.

    Returns:
        Path to the saved file.
    """
    results_dir = os.path.join(output_dir, encoder_type, sub, current_time)
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"{encoder_type}_{mode}_{sub}.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filepath}")
    return filepath
