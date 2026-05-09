"""Unified EEG retrieval training entry point.

Replaces the four separate scripts (train.py, train_joint.py,
train_contrast.py, train_atme.py) with a single configurable launcher.

Training modes (--mode):
  intra   – within-subject: train and test on the same subject's data
  loso    – leave-one-subject-out: train on N-1 subjects, test on the
             held-out subject (iterates over all subjects)
  joint   – joint training: all subjects' data combined into one model

Supported encoders (--encoder_type):
  ATMS, ATM_E, NICE, Projector, MetaEEG,
  EEGNetv4_Encoder, EEGConformer_Encoder, EEGITNet_Encoder,
  ShallowFBCSPNet_Encoder, ATCNet_Encoder

External vision features (--features_dir) are optional:
  When provided with a compatible encoder (e.g. ATMS), pre-extracted
  .pt files are loaded instead of using the features embedded in the
  dataset. Supported feature types: ViT-H-14, UniLIP-2B, UniLIP-3B,
  InternVL3-2B, EVA01_CLIP_g_14_plus.

Example calls (see run.sh for ready-to-use templates):

  # Intra-subject ATMS with external ViT-H-14 features
  python train_unified.py --mode intra --encoder_type ATMS \\
      --data_path /path/to/data --img_dir_training /path/to/train_imgs \\
      --img_dir_test /path/to/test_imgs --features_dir /path/to/features

  # LOSO with Projector (dataset-embedded features)
  python train_unified.py --mode loso --encoder_type Projector \\
      --data_path /path/to/data --img_dir_training ... --img_dir_test ...

  # Joint training with ATMS
  python train_unified.py --mode joint --encoder_type ATMS \\
      --data_path ... --img_dir_training ... --img_dir_test ... \\
      --features_dir ...
"""

import os
import sys
import datetime
import itertools
import random
import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, random_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # for eeg_encoders / retrieval_engine

# train_unified.py lives at  .../EEG_Image_decode/Retrieval/train_unified.py
# EEGDataset lives at        .../EEG_Image_decode/eegdatasets.py
# So we need  .../visual/  (three levels up) in sys.path for the
# "EEG_Image_decode.eegdatasets" package import to resolve correctly.
_this_dir   = os.path.dirname(os.path.abspath(__file__))  # .../Retrieval
_eeg_dir    = os.path.dirname(_this_dir)                   # .../EEG_Image_decode
_pkg_parent = os.path.dirname(_eeg_dir)                    # .../visual
sys.path.insert(0, _pkg_parent)   # enables:  from EEG_Image_decode.eegdatasets import ...
sys.path.insert(0, _eeg_dir)      # enables:  from encoder_utils import ...

from EEG_Image_decode.eegdatasets import EEGDataset
from encoder_utils import stratified_condition_split
from eeg_encoders import (
    build_encoder, ENCODER_REGISTRY, SUBJECT_ID_ENCODERS, NORMALIZE_FEAT_ENCODERS
)
from retrieval_engine import load_features, train_loop, save_results

# All available encoder names (including ATMS from models.atms)
ALL_ENCODER_TYPES = ['ATMS'] + list(ENCODER_REGISTRY)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Unified EEG retrieval training (intra / LOSO / joint)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required paths ---
    p.add_argument('--data_path', type=str, required=True,
                   help='Root directory of preprocessed EEG data (contains sub-*/)')
    p.add_argument('--img_dir_training', type=str, required=True,
                   help='Path to training image directory')
    p.add_argument('--img_dir_test', type=str, required=True,
                   help='Path to test image directory')

    # --- Mode & encoder ---
    p.add_argument('--mode', type=str, default='intra',
                   choices=['intra', 'loso', 'joint'],
                   help='Training mode: intra-subject | LOSO | joint all-subject')
    p.add_argument('--encoder_type', type=str, default='ATMS',
                   choices=ALL_ENCODER_TYPES,
                   help='EEG encoder architecture')

    # --- External feature files (optional, ATMS-style) ---
    p.add_argument('--features_dir', type=str, default=None,
                   help='Directory with pre-extracted vision .pt files for non-CLIP '
                        'encoders (UniLIP, EVA, …). When omitted, features are read '
                        'from the EEGDataset object.')
    p.add_argument('--clip_features_dir', type=str, default=None,
                   help='Override the shared CLIP feature cache directory. '
                        'Defaults to EEG_Image_decode/features/ (shared with Generation). '
                        'Set this only when you want a non-default cache location.')
    p.add_argument('--feature_type', type=str, default='ViT-H-14',
                   choices=['ViT-H-14', 'UniLIP-2B', 'UniLIP-3B',
                            'InternVL3-2B', 'EVA01_CLIP_g_14_plus'],
                   help='Vision encoder feature type (used when --features_dir is set)')

    # --- Subject selection ---
    p.add_argument('--subjects', nargs='+',
                   default=[f'sub-{i:02d}' for i in range(1, 11)],
                   help='Subject IDs to include')

    # --- Hyperparameters ---
    p.add_argument('--epochs',     type=int,   default=80)
    p.add_argument('--batch_size', type=int,   default=1024)
    p.add_argument('--lr',         type=float, default=3e-4)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--early_stopping_patience', type=int, default=10,
                   help='Stop training if val loss does not improve for this many '
                        'consecutive epochs. Set to 0 to disable.')
    p.add_argument('--ema_decay', type=float, default=0.999,
                   help='EMA decay rate for model parameters (0 = disable EMA). '
                        'Typical values: 0.999 or 0.9999.')
    p.add_argument('--logit_scale_type', type=str, default='exp',
                   choices=['exp', 'linear', 'softplus', 'fixed'],
                   help='How to derive the contrastive temperature from the '
                        'learnable logit_scale parameter.  '
                        'exp = param.exp() (CLIP-standard, init≈14.29); '
                        'linear = param directly (init≈2.66); '
                        'softplus = softplus(param) (init≈2.73); '
                        'fixed = param.exp() but frozen (non-learnable).')

    # --- Validation split (carved from training data, same as Generation) ---
    p.add_argument('--val_ratio', type=float, default=0.1,
                   help='Fraction of training data held out as a proper validation set '
                        '(used for early stopping and val-loss printing). '
                        '0 = no val split; early stopping then uses the test-set loss.')
    p.add_argument('--val_n_classes', type=int, default=1654,
                   help='Number of stimulus classes (THINGS-EEG default: 1654)')
    p.add_argument('--val_conditions_per_class', type=int, default=10,
                   help='Repetition conditions per class (THINGS-EEG default: 10)')
    p.add_argument('--val_trials_per_condition', type=int, default=4,
                   help='EEG trials per condition (THINGS-EEG default: 4)')
    p.add_argument('--avg_trials', action='store_true',
                   help='Average the 4 trials per condition into one signal '
                        'before training (reduces noise, shrinks dataset 4x).')

    # --- EEG signal shape ---
    p.add_argument('--n_chans', type=int, default=63,
                   help='Number of EEG channels')
    p.add_argument('--n_times', type=int, default=250,
                   help='Number of time samples per epoch')

    # --- Output & logging ---
    p.add_argument('--output_dir', type=str, default='./outputs/retrieval',
                   help='Root directory for CSV result files')
    p.add_argument('--gpu',    type=str, default='cuda:0')
    p.add_argument('--logger', action='store_true',
                   help='Enable W&B logging')
    p.add_argument('--project', type=str, default='eeg_retrieval',
                   help='W&B project name')
    p.add_argument('--entity',  type=str, default=None,
                   help='W&B entity (team/user)')
    p.add_argument('--name',    type=str, default=None,
                   help='W&B run name (defaults to encoder_mode)')
    return p


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def _make_datasets(args, subjects_train, subjects_test,
                   exclude_subject=None, preloaded=None):
    """Create train and test EEGDataset instances."""
    from EEG_Image_decode.eegdatasets import _DEFAULT_FEATURES_DIR
    # Explicit override via --clip_features_dir, otherwise use the shared default
    clip_dir = getattr(args, 'clip_features_dir', None) or _DEFAULT_FEATURES_DIR

    common = dict(
        img_dir_training=args.img_dir_training,
        img_dir_test=args.img_dir_test,
        avg_trials=getattr(args, 'avg_trials', False),
    )
    if preloaded is not None:
        # External pre-computed feature files (UniLIP, EVA, …) were provided;
        # EEGDataset does not need to load ViT-H-14 CLIP features in this path.
        train_ds = EEGDataset(
            args.data_path, subjects=subjects_train, train=True,
            exclude_subject=exclude_subject,
            feature_type=args.feature_type,
            features_dir=clip_dir,
            preloaded_features=preloaded['preloaded_train'],
            **common,
        )
        test_ds = EEGDataset(
            args.data_path, subjects=subjects_test, train=False,
            exclude_subject=exclude_subject,
            feature_type=args.feature_type,
            features_dir=clip_dir,
            preloaded_features=preloaded['preloaded_test'],
            **common,
        )
    else:
        # No external features → EEGDataset loads/caches ViT-H-14 CLIP features
        # in the shared directory so Generation and Retrieval reuse the same files.
        train_ds = EEGDataset(
            args.data_path, subjects=subjects_train, train=True,
            exclude_subject=exclude_subject,
            features_dir=clip_dir,
            **common,
        )
        test_ds = EEGDataset(
            args.data_path, subjects=subjects_test, train=False,
            exclude_subject=exclude_subject,
            features_dir=clip_dir,
            **common,
        )
    return train_ds, test_ds


def _make_loaders(train_ds, test_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=True, num_workers=0, drop_last=True)
    return train_loader, test_loader


def _make_val_split(train_ds, args, mode):
    """Carve a validation subset from *train_ds*.

    intra mode: stratified split by condition (same as Generation/train.py),
                uses args.val_n_classes / val_conditions_per_class / val_trials_per_condition.
    loso / joint: simple random split (dataset spans multiple subjects,
                  stratified single-subject indexing doesn't apply directly).

    Returns (train_subset, val_subset) or (train_ds, None) when val_ratio=0.
    """
    val_ratio = getattr(args, 'val_ratio', 0.0)
    if val_ratio <= 0:
        return train_ds, None

    if mode == 'intra':
        tpc = 1 if getattr(args, 'avg_trials', False) else args.val_trials_per_condition
        train_indices, val_indices = stratified_condition_split(
            n_classes=args.val_n_classes,
            conditions_per_class=args.val_conditions_per_class,
            trials_per_condition=tpc,
            val_ratio=val_ratio,
            seed=args.seed,
        )
        # Guard: clamp indices to actual dataset length (dataset might differ from defaults)
        ds_len = len(train_ds)
        train_indices = [i for i in train_indices if i < ds_len]
        val_indices   = [i for i in val_indices   if i < ds_len]
        train_sub = Subset(train_ds, train_indices)
        val_sub   = Subset(train_ds, val_indices)
    else:
        n_val   = max(1, int(len(train_ds) * val_ratio))
        n_train = len(train_ds) - n_val
        train_sub, val_sub = random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
    return train_sub, val_sub


def _get_features(ds_train, ds_test, preloaded):
    """Return (text_train, text_test, img_train, img_test) tensors."""
    if preloaded is not None:
        return (preloaded['text_features_train'],
                preloaded['text_features_test'],
                preloaded['img_features_train'],
                preloaded['img_features_test'])
    return (ds_train.text_features, ds_test.text_features,
            ds_train.img_features,  ds_test.img_features)


# ---------------------------------------------------------------------------
# Per-run training helper
# ---------------------------------------------------------------------------
def _run_one(sub, args, device, current_time, preloaded):
    """Build dataset, model, optimiser and call train_loop for one run."""
    mode = args.mode
    subjects = args.subjects

    use_subject_id  = args.encoder_type in SUBJECT_ID_ENCODERS
    normalize_feats = args.encoder_type in NORMALIZE_FEAT_ENCODERS

    # --- Full training dataset (before val split) ---
    if mode == 'intra':
        train_ds, test_ds = _make_datasets(
            args, subjects_train=[sub], subjects_test=[sub],
            preloaded=preloaded)
    elif mode == 'loso':
        train_ds, test_ds = _make_datasets(
            args, subjects_train=subjects, subjects_test=[sub],
            exclude_subject=sub, preloaded=preloaded)
    else:  # joint
        train_ds, test_ds = _make_datasets(
            args, subjects_train=subjects, subjects_test=subjects,
            preloaded=preloaded)

    # --- Validation split carved from training data ---
    train_subset, val_subset = _make_val_split(train_ds, args, mode)

    n_train = len(train_subset)
    n_val   = len(val_subset) if val_subset is not None else 0
    n_test  = len(test_ds)
    print(f"  Dataset: {n_train} train  |  {n_val} val  |  {n_test} test samples")

    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=1,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader   = (DataLoader(val_subset, batch_size=args.batch_size,
                               shuffle=False, num_workers=0)
                    if val_subset is not None else None)

    txt_tr, txt_te, img_tr, img_te = _get_features(train_ds, test_ds, preloaded)

    # --- Model ---
    joint_train = (mode == 'joint')
    model = build_encoder(
        args.encoder_type,
        n_chans=args.n_chans,
        n_times=args.n_times,
        joint_train=joint_train,
    )
    model.to(device)
    optimizer = AdamW(itertools.chain(model.parameters()), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {args.encoder_type} — {n_params:,} parameters")

    # Attach mode to config so train_loop can use it for checkpoint paths
    args.mode = mode

    results = train_loop(
        sub=sub,
        current_time=current_time,
        eeg_model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        device=device,
        text_features_train_all=txt_tr,
        text_features_test_all=txt_te,
        img_features_train_all=img_tr,
        img_features_test_all=img_te,
        config=args,
        logger=args.logger,
        use_subject_id=use_subject_id,
        normalize_feats=normalize_feats,
        val_dataloader=val_loader,
        ema_decay=args.ema_decay,
        logit_scale_type=args.logit_scale_type,
    )

    save_results(results, args.output_dir, args.encoder_type,
                 sub, mode, current_time)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Default W&B run name
    if args.name is None:
        args.name = f"{args.encoder_type}_{args.mode}"

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mode:   {args.mode}  |  Encoder: {args.encoder_type}")
    print(f"Subjects: {args.subjects}")

    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    # Load external feature files once (if provided)
    preloaded = None
    if args.features_dir is not None:
        preloaded = load_features(args.feature_type, args.features_dir)

    # -----------------------------------------------------------------------
    # Training loops
    # -----------------------------------------------------------------------
    if args.mode == 'joint':
        # Single model trained on all subjects simultaneously
        print(f"\n{'='*60}")
        print("Joint training over all subjects")
        print(f"{'='*60}")
        _run_one('joint', args, device, current_time, preloaded)

    else:
        # Intra-subject or LOSO: one model per subject
        for sub in args.subjects:
            print(f"\n{'='*60}")
            print(f"Subject: {sub}  (mode={args.mode})")
            print(f"{'='*60}")
            _run_one(sub, args, device, current_time, preloaded)


if __name__ == '__main__':
    main()
