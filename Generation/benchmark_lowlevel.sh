#!/bin/bash
###############################################################################
# benchmark_lowlevel.sh
#
# Full benchmark for the low-level EEG-to-VAE-latent encoder.
#
# The encoder_low_level model maps EEG signals directly to SDXL-VAE latent
# space (4 × 64 × 64), supervised by pre-computed VAE latents of the training
# images.  Decoded latents are evaluated against GT test images.
#
# Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
#   Pre-compute VAE latents before running:
#     python extract_vae_latents.py \
#         --img_dir_training <TRAINING_IMAGES> \
#         --img_dir_test <TEST_IMAGES> \
#         --output_dir <LATENT_DIR>
#   which produces:
#     <LATENT_DIR>/train_image_latent_512.pt
#     <LATENT_DIR>/test_image_latent_512.pt
#
# Pipeline per subject
# ─────────────────────────────────────────────────────────────────────────────
#   STEP 1 – Training (train_lowlevel.py)
#     - Train encoder_low_level on train split (9:1 val hold-out)
#     - MAE loss vs. pre-computed SDXL-VAE latents
#     - Early stopping with patience PATIENCE on val MAE
#     - Best checkpoint saved as best_encoder.pth
#
#   STEP 2 – Evaluation (evaluate_lowlevel.py)
#     - Predict VAE latents for all test EEG samples
#     - Decode with frozen SDXL VAE → images
#     - Compute 7 reconstruction metrics vs. GT test images
#
#   STEP 3 – Cross-subject summary
#     - Aggregate per-subject CSVs, print mean ± std
#
# Usage:
#   bash benchmark_lowlevel.sh
#   SUBJECTS="sub-01 sub-08" bash benchmark_lowlevel.sh
#   GPU=cuda:0 bash benchmark_lowlevel.sh
###############################################################################

set -e

#==============================================================================
# [DATA PATHS] - Modify these to match your data location
#==============================================================================
DATA_PATH="${DATA_PATH:-/path/to/preprocessed_data}"
IMG_DIR_TRAINING="${IMG_DIR_TRAINING:-/path/to/image_set/training_images}"
IMG_DIR_TEST="${IMG_DIR_TEST:-/path/to/image_set/test_images}"
# Directory containing pre-computed train/test_image_latent_512.pt files
LATENT_DIR="/vePFS-0x0d/visual/Features/THINGS-EEG2/VAE_latents"
VISION_MODELS_DIR="${VISION_MODELS_DIR:-./vision_models}"

#==============================================================================
# [MODEL SAVE / OUTPUT PATHS]
#==============================================================================
MODEL_SAVE_DIR="./models/benchmark_lowlevel"
OUTPUT_DIR="./outputs/benchmark_lowlevel"

#==============================================================================
# [SUBJECT LIST]
#==============================================================================
SUBJECTS="${SUBJECTS:-sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10}"

#==============================================================================
# [TRAINING HYPERPARAMETERS]
#==============================================================================
EPOCHS=200
LR=1e-3
BATCH_SIZE=30
PATIENCE=20          # Early stopping patience (epochs without val improvement)
VAL_RATIO=0.1        # Fraction of training data held out for validation
SEED=42

#==============================================================================
# [EVALUATION HYPERPARAMETERS]
#==============================================================================
NUM_GEN_PER_CLASS=10   # Copies of each decoded image (all identical; sets eval rounds)

#==============================================================================
# [GPU SETTINGS]
#==============================================================================
GPU="${GPU:-cuda:1}"

#==============================================================================
# [MISC]
#==============================================================================
export WANDB_MODE="offline"
mkdir -p "${VISION_MODELS_DIR}"

###############################################################################
# DO NOT MODIFY BELOW THIS LINE
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Low-level EEG Encoder Benchmark"
echo "============================================================"
echo "  Subjects:      ${SUBJECTS}"
echo "  Data path:     ${DATA_PATH}"
echo "  Latent dir:    ${LATENT_DIR}"
echo "  Epochs:        ${EPOCHS}  |  Patience: ${PATIENCE}  |  Val ratio: ${VAL_RATIO}"
echo "  GPU:           ${GPU}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "============================================================"
echo ""

COMPLETED_SUBJECTS=()
LL_CSV_FILES=()

set +e

for SUBJECT in ${SUBJECTS}; do

    echo ""
    echo "############################################################"
    echo "  Subject: ${SUBJECT}"
    echo "############################################################"
    echo ""

    # ──────────────────────────────────────────────────────────────
    # STEP 1: Training
    # ──────────────────────────────────────────────────────────────
    echo "  [STEP 1] Training encoder_low_level ..."
    echo ""

    python train_lowlevel.py \
        --data_path "${DATA_PATH}" \
        --img_dir_training "${IMG_DIR_TRAINING}" \
        --img_dir_test "${IMG_DIR_TEST}" \
        --latent_dir "${LATENT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --subject "${SUBJECT}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --batch_size "${BATCH_SIZE}" \
        --patience "${PATIENCE}" \
        --val_ratio "${VAL_RATIO}" \
        --gpu "${GPU}"

    TRAIN_STATUS=$?
    if [ ${TRAIN_STATUS} -ne 0 ]; then
        echo "[WARN] Training failed for ${SUBJECT} (exit ${TRAIN_STATUS}). Skipping."
        continue
    fi

    # Locate paths_info.txt written by train_lowlevel.py
    PATHS_INFO=$(find "${OUTPUT_DIR}/${SUBJECT}" -name "paths_info.txt" -type f | sort | tail -1)
    if [ -z "${PATHS_INFO}" ]; then
        echo "[WARN] paths_info.txt not found for ${SUBJECT}. Skipping evaluation."
        continue
    fi

    ENCODER_PATH=$(grep "encoder_path=" "${PATHS_INFO}" | cut -d= -f2)
    RUN_DIR=$(dirname "${PATHS_INFO}")

    echo ""
    echo "  [INFO] Training complete for ${SUBJECT}."
    echo "    Encoder checkpoint: ${ENCODER_PATH}"
    echo ""

    # ──────────────────────────────────────────────────────────────
    # STEP 2: Evaluation
    # ──────────────────────────────────────────────────────────────
    echo "  [STEP 2] Evaluating on test set ..."
    echo ""

    python evaluate_lowlevel.py \
        --data_path "${DATA_PATH}" \
        --img_directory_test "${IMG_DIR_TEST}" \
        --img_dir_training "${IMG_DIR_TRAINING}" \
        --latent_dir "${LATENT_DIR}" \
        --encoder_path "${ENCODER_PATH}" \
        --output_dir "${RUN_DIR}" \
        --subject "${SUBJECT}" \
        --batch_size 64 \
        --num_gen_per_class "${NUM_GEN_PER_CLASS}" \
        --gpu "${GPU}" \
        --seed "${SEED}"

    EVAL_STATUS=$?
    if [ ${EVAL_STATUS} -ne 0 ]; then
        echo "[WARN] Evaluation failed for ${SUBJECT} (exit ${EVAL_STATUS}). Skipping."
        continue
    fi

    COMPLETED_SUBJECTS+=("${SUBJECT}")
    LL_CSV="${RUN_DIR}/reconstruction_metrics_${SUBJECT}_lowlevel.csv"
    LL_CSV_FILES+=("${LL_CSV}")

    echo ""
    echo "  [INFO] ${SUBJECT} complete."
    echo "    Metrics: ${LL_CSV}"

done  # end per-subject loop

set -e

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Cross-subject summary
# ──────────────────────────────────────────────────────────────────────────────
N_DONE=${#COMPLETED_SUBJECTS[@]}

if [ ${N_DONE} -eq 0 ]; then
    echo ""
    echo "[ERROR] No subjects completed successfully."
    exit 1
fi

echo ""
echo "============================================================"
echo "  STEP 3: Cross-Subject Summary  (${N_DONE} subjects)"
echo "============================================================"
echo "  Completed: ${COMPLETED_SUBJECTS[*]}"
echo ""

LL_LIST="${LL_CSV_FILES[*]}"
SUMMARY_CSV="${OUTPUT_DIR}/summary_lowlevel_$(date +%m-%d_%H-%M).csv"

python3 - "${LL_LIST}" "${SUMMARY_CSV}" "${COMPLETED_SUBJECTS[*]}" <<'PYEOF'
import sys, os, csv, numpy as np
from collections import defaultdict

def read_metric_csv(path):
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = row.get('Metric', '').strip()
            val = row.get('Mean', '').strip()
            if key and val:
                try:
                    result[key] = float(val)
                except ValueError:
                    pass
    return result

csv_files = sys.argv[1].split() if sys.argv[1] else []
summary_path = sys.argv[2]
subjects = sys.argv[3].split()

per_metric = defaultdict(list)
valid_subjects = []
for path, sub in zip(csv_files, subjects):
    if not os.path.isfile(path):
        print(f"  [WARN] Missing CSV for {sub}: {path}")
        continue
    d = read_metric_csv(path)
    if not d:
        print(f"  [WARN] Empty CSV for {sub}: {path}")
        continue
    valid_subjects.append(sub)
    for m, v in d.items():
        per_metric[m].append(v)

if not per_metric:
    print("  No valid data found.")
    sys.exit(0)

metrics = list(per_metric.keys())
col_w = 14
n = len(valid_subjects)

header = f"  {'Metric':<{col_w}}"
for sub in valid_subjects:
    header += f"  {sub:>10}"
header += f"  {'Mean':>10}  {'Std':>8}"

print(f"\n  Low-level Encoder  ({n} subjects)")
print("  " + "=" * (len(header) - 2))
print(header)
print("  " + "-" * (col_w + (10 + 2) * n + 24))

rows_out = []
for m in metrics:
    vals = per_metric[m]
    row = f"  {m:<{col_w}}"
    for v in vals:
        row += f"  {v:>10.4f}"
    mu, sd = np.mean(vals), np.std(vals)
    row += f"  {mu:>10.4f}  {sd:>8.4f}"
    print(row)
    rows_out.append({'Metric': m,
                     **{sub: f"{v:.4f}" for sub, v in zip(valid_subjects, vals)},
                     'Mean': f"{mu:.4f}", 'Std': f"{sd:.4f}"})

print("  " + "=" * (len(header) - 2))

if rows_out:
    fieldnames = ['Metric'] + valid_subjects + ['Mean', 'Std']
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n  Summary saved to: {summary_path}")
PYEOF

echo ""
echo "============================================================"
echo "  Low-level Benchmark Complete!"
echo "============================================================"
echo "  Subjects:       ${COMPLETED_SUBJECTS[*]}"
echo "  Summary:        ${SUMMARY_CSV}"
echo "  Per-subject outputs under: ${OUTPUT_DIR}/<subject>/<run_id>/"
echo "============================================================"
