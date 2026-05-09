#!/usr/bin/env bash
# =============================================================================
# run.sh  –  EEG Retrieval experiment launcher
#
# Usage:
#   bash run.sh                   # run with all defaults (intra-subject ATMS)
#   ENCODER=NICE bash run.sh      # override encoder via environment
#   MODE=loso bash run.sh         # override mode via environment
#
# All key variables are grouped at the top and can be overridden by setting
# the corresponding environment variable before calling this script, e.g.:
#   MODE=joint ENCODER=Projector EPOCHS=60 bash run.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# ① Data paths  (edit for your environment)
# ---------------------------------------------------------------------------
DATA_PATH="${DATA_PATH:-/vePFS-0x0d/visual/dataset/THINGS_EEG/Preprocessed_data_250Hz}"
IMG_DIR_TRAINING="${IMG_DIR_TRAINING:-/vePFS-0x0d/visual/dataset/THINGS_EEG/images_set/training_images}"
IMG_DIR_TEST="${IMG_DIR_TEST:-/vePFS-0x0d/visual/dataset/THINGS_EEG/images_set/test_images}"

# Directory that contains pre-extracted .pt feature files for ATMS.
# Set to "" to use the features embedded in the EEGDataset instead.
FEATURES_DIR="${FEATURES_DIR:-}"

# ---------------------------------------------------------------------------
# ② Encoder selection
#    Choose from:
#      ATMS | ATM_E | NICE | Projector | MetaEEG
#      EEGNetv4_Encoder | EEGConformer_Encoder | EEGITNet_Encoder
#      ShallowFBCSPNet_Encoder | ATCNet_Encoder
# ---------------------------------------------------------------------------
ENCODER="${ENCODER:-ATMS}"

# Feature type for ATMS external feature loading (only used when FEATURES_DIR is set)
#   ViT-H-14 | UniLIP-2B | UniLIP-3B | InternVL3-2B | EVA01_CLIP_g_14_plus
FEATURE_TYPE="${FEATURE_TYPE:-ViT-H-14}"

# ---------------------------------------------------------------------------
# ③ Training mode
#    intra  – within-subject (one model per subject)
#    loso   – leave-one-subject-out (train on N-1, test on held-out)
#    joint  – joint training on all subjects at once
# ---------------------------------------------------------------------------
MODE="${MODE:-intra}"

# ---------------------------------------------------------------------------
# ④ Subject list (space-separated)
# ---------------------------------------------------------------------------
SUBJECTS="${SUBJECTS:-sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10}"

# ---------------------------------------------------------------------------
# ⑤ Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-3e-4}"
SEED="${SEED:-42}"
# Early stopping: 0 = disabled
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
# EMA decay rate: 0 = disabled, typical: 0.999 or 0.9999
EMA_DECAY="${EMA_DECAY:-0.999}"
# Logit scale type: how the learnable temperature parameter is applied
#   exp      – param.exp()    (CLIP-standard, init ≈ 14.29)
#   linear   – param directly (init ≈ 2.66, original code behavior)
#   softplus – softplus(param)(init ≈ 2.73, smooth & always positive)
#   fixed    – param.exp() but frozen (non-learnable, stays at init)
LOGIT_SCALE_TYPE="${LOGIT_SCALE_TYPE:-exp}"
# Average the 4 EEG trials per condition into one signal before training
# true = average (16540 samples/subject), false = keep all trials (66160 samples/subject)
AVG_SIGNAL_TRAINING="${AVG_SIGNAL_TRAINING:-true}"
# Fraction of training data held out as val set (same as Generation/benchmark.sh)
# Use 0 to disable the val split (early stopping will then use test-set loss)
VAL_RATIO="${VAL_RATIO:-0.1}"

# EEG signal shape
N_CHANS="${N_CHANS:-63}"
N_TIMES="${N_TIMES:-250}"

# ---------------------------------------------------------------------------
# ⑥ Hardware
# ---------------------------------------------------------------------------
GPU="${GPU:-cuda:0}"

# ---------------------------------------------------------------------------
# ⑦ Output & logging
# ---------------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/retrieval}"
PROJECT="${PROJECT:-eeg_retrieval}"
ENTITY="${ENTITY:-}"
WANDB_LOGGING="${WANDB_LOGGING:-false}"   # set to "true" to enable W&B

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
echo "========================================"
echo "  EEG Retrieval Experiment"
echo "========================================"
echo "  Encoder    : ${ENCODER}"
echo "  Mode       : ${MODE}"
echo "  Feature    : ${FEATURE_TYPE}"
echo "  Features dir: ${FEATURES_DIR:-<dataset>}"
echo "  Subjects   : ${SUBJECTS}"
echo "  Epochs     : ${EPOCHS}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  LR         : ${LR}"
echo "  ES patience: ${EARLY_STOPPING_PATIENCE} (0=off)"
echo "  EMA decay  : ${EMA_DECAY} (0=off)"
echo "  Scale type : ${LOGIT_SCALE_TYPE}"
echo "  Avg trials : ${AVG_SIGNAL_TRAINING}"
echo "  Val ratio  : ${VAL_RATIO} (0=no val split)"
echo "  GPU        : ${GPU}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "========================================"

# ---------------------------------------------------------------------------
# Build the python command
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

CMD=(
    "${PYTHON}" "${SCRIPT_DIR}/train_unified.py"
    --data_path        "${DATA_PATH}"
    --img_dir_training "${IMG_DIR_TRAINING}"
    --img_dir_test     "${IMG_DIR_TEST}"
    --mode             "${MODE}"
    --encoder_type     "${ENCODER}"
    --feature_type     "${FEATURE_TYPE}"
    --subjects         ${SUBJECTS}
    --epochs           "${EPOCHS}"
    --batch_size       "${BATCH_SIZE}"
    --lr               "${LR}"
    --seed             "${SEED}"
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}"
    --ema_decay               "${EMA_DECAY}"
    --logit_scale_type        "${LOGIT_SCALE_TYPE}"
    --val_ratio               "${VAL_RATIO}"
    --n_chans          "${N_CHANS}"
    --n_times          "${N_TIMES}"
    --gpu              "${GPU}"
    --output_dir       "${OUTPUT_DIR}"
    --project          "${PROJECT}"
)

# Optional: external feature directory
if [[ -n "${FEATURES_DIR}" ]]; then
    CMD+=(--features_dir "${FEATURES_DIR}")
fi

# Optional: W&B entity
if [[ -n "${ENTITY}" ]]; then
    CMD+=(--entity "${ENTITY}")
fi

# Optional: W&B logging flag
if [[ "${WANDB_LOGGING}" == "true" ]]; then
    CMD+=(--logger)
fi

# Optional: average trials per condition
if [[ "${AVG_SIGNAL_TRAINING}" == "true" ]]; then
    CMD+=(--avg_trials)
fi

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
echo ""
echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"
