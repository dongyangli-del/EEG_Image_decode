#!/bin/bash
###############################################################################
# benchmark.sh
#
# Full benchmark for EEG-to-Image reconstruction: ATMS encoder +
# Diffusion Prior + IP-Adapter + SDXL-Turbo.
#
# Pipeline per subject
# ──────────────────────────────────────────────────────────────────────────
#   STEP 1 – Training (train.py)
#     Phase 1  (first ENCODER_ONLY_RATIO of TOTAL_EPOCHS):
#       - Train ATMS encoder on train split (val split held out)
#       - Model selection by val_loss; early stopping with patience PATIENCE
#     Phase 2  (remaining epochs):
#       ENCODER_FINETUNING=false → freeze encoder, train Diffusion Prior only
#       ENCODER_FINETUNING=true  → jointly fine-tune encoder + prior
#       - Model selection by prior_loss (frozen) or val_loss (joint)
#       - Early stopping with the same PATIENCE
#
#   STEP 2 – Evaluation (evaluate.py)
#     - Extract EEG features → Diffusion Prior → IP-Adapter + SDXL-Turbo
#     - Compute 8 reconstruction metrics on the test set
#     - When EVAL_ENCODER_RECON=true (default):
#         Stage 1: images generated directly from encoder embeddings (no prior)
#         Stage 2: images generated via encoder → prior → SDXL
#         Side-by-side comparison table is printed and saved
#
#   STEP 3 – Cross-subject summary
#     - Aggregate per-subject metric CSVs
#     - Print mean ± std across subjects for both Stage 1 and Stage 2
#
# Usage:
#   bash benchmark.sh                                      # all subjects
#   SUBJECTS="sub-01 sub-08" bash benchmark.sh              # override from env
#   GPU=cuda:0 bash benchmark.sh
#   RESUME=05-09_12-49 SUBJECTS=sub-01 bash benchmark.sh   # eval only (skip training)
###############################################################################

set -e

#==============================================================================
# [DATA PATHS] - Modify these to match your data location
#==============================================================================
DATA_PATH="${DATA_PATH:-/vePFS-0x0d/visual/dataset/THINGS_EEG/Preprocessed_data_250Hz}"
IMG_DIR_TRAINING="${IMG_DIR_TRAINING:-/vePFS-0x0d/visual/dataset/THINGS_EEG/images_set/training_images}"
IMG_DIR_TEST="${IMG_DIR_TEST:-/vePFS-0x0d/visual/dataset/THINGS_EEG/images_set/test_images}"
# FEATURES_DIR: CLIP feature cache directory.
# Leave empty (default) → EEGDataset uses EEG_Image_decode/features/ (shared with Retrieval).
# Set explicitly only to override the cache location, e.g. a fast local SSD.
FEATURES_DIR="${FEATURES_DIR:-}"
VISION_MODELS_DIR="${VISION_MODELS_DIR:-./vision_models}"

#==============================================================================
# [MODEL SAVE / OUTPUT PATHS]
#==============================================================================
MODEL_SAVE_DIR="./models/benchmark"
OUTPUT_DIR="./outputs/benchmark"

#==============================================================================
# [SUBJECT LIST]
# Override at runtime: SUBJECTS="sub-01 sub-08" bash benchmark.sh
#==============================================================================
SUBJECTS="${SUBJECTS:-sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10}"

#==============================================================================
# [RESUME] - Skip training, load models from a previous run timestamp
# Set to a timestamp string (e.g. "05-09_12-49") to skip STEP 1 and jump
# directly to evaluation using the encoder/prior saved under that timestamp.
#   RESUME=05-09_12-49 SUBJECTS=sub-01 bash benchmark.sh
# Leave empty (default) to run full training + evaluation.
#==============================================================================
RESUME="${RESUME:-}"

#==============================================================================
# [TRAINING HYPERPARAMETERS]
#==============================================================================
TOTAL_EPOCHS=500
ENCODER_ONLY_RATIO=0.2    # Fraction of epochs for Phase 1 (encoder-only)
ENCODER_FINETUNING=false  # true → jointly train encoder + prior in Phase 2
BATCH_SIZE=64             # Encoder training batch size
LR_ENCODER=3e-4
LR_PRIOR=1e-3
PRIOR_EPOCHS_PER_STEP=1   # Prior gradient steps per encoder epoch (joint mode)
PRIOR_BATCH_SIZE=1024
PRIOR_DROPOUT=0.1
SAVE_INTERVAL=10          # Save a periodic checkpoint every N epochs
SEED=42
AVG_SIGNAL_TRAINING="${AVG_SIGNAL_TRAINING:-true}"  # Average 4 trials/condition into 1 signal
VAL_RATIO=0.1             # Fraction of training conditions held out for validation
PATIENCE=50               # Early-stopping patience (both Phase 1 and Phase 2)

#==============================================================================
# [EVALUATION HYPERPARAMETERS]
#==============================================================================
NUM_GEN_PER_CLASS=3      # Generated images per test class (= evaluation rounds)
PRIOR_INFERENCE_STEPS=50  # Diffusion Prior denoising steps
GUIDANCE_SCALE=5.0        # Classifier-free guidance scale
SDXL_INFERENCE_STEPS=4    # SDXL-Turbo denoising steps
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"  # SDXL generation batch size (images per forward pass)
#
# EVAL_ENCODER_RECON=true:
#   Also run Stage-1 generation (encoder embeddings → SDXL, bypassing the prior)
#   and print a Stage-1 vs Stage-2 comparison table.  Doubles SDXL render time.
EVAL_ENCODER_RECON="${EVAL_ENCODER_RECON:-true}"

#==============================================================================
# [GPU SETTINGS]
#==============================================================================
GPU="${GPU:-cuda:1}"
# export CUDA_VISIBLE_DEVICES=0  # Uncomment to restrict GPU

#==============================================================================
# [CONDA ENVIRONMENT]
#==============================================================================
# eval "$(conda shell.bash hook)"
# conda activate meg2speech

#==============================================================================
# [MISC]
#==============================================================================
export WANDB_MODE="offline"
mkdir -p "${VISION_MODELS_DIR}"
export OPEN_CLIP_CACHE_DIR="${VISION_MODELS_DIR}"

###############################################################################
# DO NOT MODIFY BELOW THIS LINE (unless you know what you're doing)
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENCODER_ONLY_EPOCHS=$(awk "BEGIN {printf \"%d\", ${TOTAL_EPOCHS} * ${ENCODER_ONLY_RATIO}}")

echo "============================================================"
echo "  ATMS EEG-to-Image Reconstruction Benchmark"
echo "============================================================"
echo "  Subjects:        ${SUBJECTS}"
echo "  Data path:       ${DATA_PATH}"
echo "  Total epochs:    ${TOTAL_EPOCHS}  (encoder-only: first ${ENCODER_ONLY_EPOCHS})"
echo "  Avg trials:      ${AVG_SIGNAL_TRAINING}"
echo "  Val ratio:       ${VAL_RATIO}  |  Early-stop patience: ${PATIENCE}"
echo "  Encoder finetune:${ENCODER_FINETUNING}"
if [ -n "${RESUME}" ]; then
echo "  RESUME timestamp:${RESUME}  (skip training)"
fi
echo "  Eval enc recon:  ${EVAL_ENCODER_RECON}"
echo "  Gen batch size:  ${GEN_BATCH_SIZE}"
echo "  GPU:             ${GPU}"
echo "  Output dir:      ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# Per-subject result bookkeeping
COMPLETED_SUBJECTS=()
STAGE2_CSV_FILES=()
STAGE1_CSV_FILES=()

# Per-subject failures are logged but do not abort the whole run
set +e

for SUBJECT in ${SUBJECTS}; do

    echo ""
    echo "############################################################"
    echo "  Subject: ${SUBJECT}"
    echo "############################################################"
    echo ""

    if [ -n "${RESUME}" ]; then
        # ──────────────────────────────────────────────────────────
        # RESUME MODE: skip training, load models from timestamp
        # ──────────────────────────────────────────────────────────
        echo "  [STEP 1] Skipped (RESUME=${RESUME})"
        echo ""

        TIMESTAMP="${RESUME}"
        EVAL_OUTPUT_DIR="${OUTPUT_DIR}/${SUBJECT}/${TIMESTAMP}"
        PATHS_INFO="${EVAL_OUTPUT_DIR}/paths_info.txt"

        if [ -f "${PATHS_INFO}" ]; then
            ENCODER_PATH=$(grep "encoder_path=" "${PATHS_INFO}" | cut -d= -f2)
            PRIOR_PATH=$(grep "prior_path="   "${PATHS_INFO}" | cut -d= -f2)
        else
            ENCODER_PATH="${MODEL_SAVE_DIR}/encoder/${SUBJECT}/${TIMESTAMP}/best.pth"
            PRIOR_PATH="${MODEL_SAVE_DIR}/prior/${SUBJECT}/${TIMESTAMP}/best.pth"
        fi

        if [ ! -f "${ENCODER_PATH}" ]; then
            echo "[WARN] Encoder not found: ${ENCODER_PATH}. Skipping ${SUBJECT}."
            continue
        fi
        if [ ! -f "${PRIOR_PATH}" ]; then
            echo "[WARN] Prior not found: ${PRIOR_PATH}. Skipping ${SUBJECT}."
            continue
        fi

        echo "  [INFO] Resuming from timestamp ${TIMESTAMP} for ${SUBJECT}."
        echo "    Encoder: ${ENCODER_PATH}"
        echo "    Prior:   ${PRIOR_PATH}"
        echo ""
    else
        # ──────────────────────────────────────────────────────────
        # STEP 1: Training
        # ──────────────────────────────────────────────────────────
        echo "  [STEP 1] Training (encoder + prior) ..."
        echo ""

        TRAIN_CMD="python train.py \
            --data_path ${DATA_PATH} \
            --img_dir_training ${IMG_DIR_TRAINING} \
            --img_dir_test ${IMG_DIR_TEST} \
            --output_dir ${OUTPUT_DIR} \
            --model_save_dir ${MODEL_SAVE_DIR} \
            --subject ${SUBJECT} \
            --total_epochs ${TOTAL_EPOCHS} \
            --encoder_only_ratio ${ENCODER_ONLY_RATIO} \
            --batch_size ${BATCH_SIZE} \
            --lr_encoder ${LR_ENCODER} \
            --lr_prior ${LR_PRIOR} \
            --prior_epochs_per_step ${PRIOR_EPOCHS_PER_STEP} \
            --prior_batch_size ${PRIOR_BATCH_SIZE} \
            --prior_dropout ${PRIOR_DROPOUT} \
            --gpu ${GPU} \
            --seed ${SEED} \
            --save_interval ${SAVE_INTERVAL} \
            --val_ratio ${VAL_RATIO} \
            --patience ${PATIENCE}"

        # Only override the shared feature cache when an explicit path is given
        if [ -n "${FEATURES_DIR}" ]; then
            TRAIN_CMD="${TRAIN_CMD} --features_dir ${FEATURES_DIR}"
        fi

        if [ "${ENCODER_FINETUNING}" = true ]; then
            TRAIN_CMD="${TRAIN_CMD} --encoder_finetuning"
        fi

        if [ "${AVG_SIGNAL_TRAINING}" = true ]; then
            TRAIN_CMD="${TRAIN_CMD} --avg_trials"
        fi

        eval ${TRAIN_CMD}
        TRAIN_STATUS=$?

        if [ ${TRAIN_STATUS} -ne 0 ]; then
            echo ""
            echo "[WARN] Training failed for ${SUBJECT} (exit ${TRAIN_STATUS}). Skipping to next subject."
            continue
        fi

        # Locate paths_info.txt written by the training script
        PATHS_INFO=$(find "${OUTPUT_DIR}/${SUBJECT}" -name "paths_info.txt" -type f | sort | tail -1)
        if [ -z "${PATHS_INFO}" ]; then
            echo "[WARN] paths_info.txt not found for ${SUBJECT}. Skipping evaluation."
            continue
        fi

        ENCODER_PATH=$(grep "encoder_path=" "${PATHS_INFO}" | cut -d= -f2)
        PRIOR_PATH=$(grep "prior_path="   "${PATHS_INFO}" | cut -d= -f2)
        TIMESTAMP=$(grep  "timestamp="    "${PATHS_INFO}" | cut -d= -f2)
        EVAL_OUTPUT_DIR="${OUTPUT_DIR}/${SUBJECT}/${TIMESTAMP}"

        echo ""
        echo "  [INFO] Training complete for ${SUBJECT}."
        echo "    Encoder: ${ENCODER_PATH}"
        echo "    Prior:   ${PRIOR_PATH}"
        echo ""
    fi

    # ──────────────────────────────────────────────────────────────
    # STEP 2: Evaluation
    # ──────────────────────────────────────────────────────────────
    echo "  [STEP 2] Evaluation on test set ..."
    echo ""

    EVAL_CMD="python evaluate.py \
        --data_path \"${DATA_PATH}\" \
        --img_directory_test \"${IMG_DIR_TEST}\" \
        --img_dir_training \"${IMG_DIR_TRAINING}\" \
        --output_dir \"${EVAL_OUTPUT_DIR}\" \
        --encoder_path \"${ENCODER_PATH}\" \
        --prior_path \"${PRIOR_PATH}\" \
        --subject \"${SUBJECT}\" \
        --batch_size ${PRIOR_BATCH_SIZE} \
        --num_gen_per_class ${NUM_GEN_PER_CLASS} \
        --prior_steps ${PRIOR_INFERENCE_STEPS} \
        --guidance_scale ${GUIDANCE_SCALE} \
        --sdxl_steps ${SDXL_INFERENCE_STEPS} \
        --gen_batch_size ${GEN_BATCH_SIZE} \
        --prior_dropout ${PRIOR_DROPOUT} \
        --gpu \"${GPU}\" \
        --seed ${SEED}"

    # Only override the shared feature cache when an explicit path is given
    if [ -n "${FEATURES_DIR}" ]; then
        EVAL_CMD="${EVAL_CMD} --features_dir \"${FEATURES_DIR}\""
    fi

    if [ "${EVAL_ENCODER_RECON}" = true ]; then
        EVAL_CMD="${EVAL_CMD} --eval_encoder_recon"
    fi

    eval ${EVAL_CMD}
    EVAL_STATUS=$?

    if [ ${EVAL_STATUS} -ne 0 ]; then
        echo ""
        echo "[WARN] Evaluation failed for ${SUBJECT} (exit ${EVAL_STATUS}). Skipping to next subject."
        continue
    fi

    # Record successful subject
    COMPLETED_SUBJECTS+=("${SUBJECT}")
    STAGE2_CSV="${EVAL_OUTPUT_DIR}/reconstruction_metrics_${SUBJECT}.csv"
    STAGE2_CSV_FILES+=("${STAGE2_CSV}")

    if [ "${EVAL_ENCODER_RECON}" = true ]; then
        STAGE1_CSV="${EVAL_OUTPUT_DIR}/reconstruction_metrics_${SUBJECT}_encoder_only.csv"
        if [ -f "${STAGE1_CSV}" ]; then
            STAGE1_CSV_FILES+=("${STAGE1_CSV}")
        fi
    fi

    echo ""
    echo "  [INFO] ${SUBJECT} complete."
    echo "    Stage-2 metrics: ${STAGE2_CSV}"

done  # end per-subject loop

set -e

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Cross-subject summary
# ──────────────────────────────────────────────────────────────────────────────
N_DONE=${#COMPLETED_SUBJECTS[@]}

if [ ${N_DONE} -eq 0 ]; then
    echo ""
    echo "[ERROR] No subjects completed successfully. No summary to print."
    exit 1
fi

echo ""
echo "============================================================"
echo "  STEP 3: Cross-Subject Summary  (${N_DONE} subjects)"
echo "============================================================"
echo "  Completed: ${COMPLETED_SUBJECTS[*]}"
echo ""

# Build space-separated file list strings for Python
STAGE2_LIST="${STAGE2_CSV_FILES[*]}"
STAGE1_LIST="${STAGE1_CSV_FILES[*]}"

# Write per-subject Stage-2 summary CSV
SUMMARY_CSV="${OUTPUT_DIR}/summary_stage2_$(date +%m-%d_%H-%M).csv"
SUMMARY_S1_CSV="${OUTPUT_DIR}/summary_stage1_$(date +%m-%d_%H-%M).csv"

python3 - "${STAGE2_LIST}" "${STAGE1_LIST}" "${SUMMARY_CSV}" "${SUMMARY_S1_CSV}" \
          "${COMPLETED_SUBJECTS[*]}" <<'PYEOF'
import sys, os, csv, numpy as np
from collections import defaultdict

def read_metric_csv(path):
    """Return {metric: mean_value} from a tab-separated reconstruction_metrics CSV."""
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = row.get('Metric', '').strip()
            val = row.get('Mean', row.get('Value', '')).strip()
            if key and val:
                try:
                    result[key] = float(val)
                except ValueError:
                    pass
    return result

def summarise(csv_files, label, subjects, summary_path):
    if not csv_files or not csv_files[0]:
        return
    per_metric = defaultdict(list)
    valid_subjects = []
    for path, sub in zip(csv_files, subjects):
        if not os.path.isfile(path):
            print(f"  [WARN] Missing {label} CSV for {sub}: {path}")
            continue
        d = read_metric_csv(path)
        if not d:
            print(f"  [WARN] Empty {label} CSV for {sub}: {path}")
            continue
        valid_subjects.append(sub)
        for m, v in d.items():
            per_metric[m].append(v)

    if not per_metric:
        print(f"  No valid {label} data found.")
        return

    metrics = list(per_metric.keys())
    col_w = 14
    n = len(valid_subjects)

    # Header row
    header = f"  {'Metric':<{col_w}}"
    for sub in valid_subjects:
        header += f"  {sub:>10}"
    header += f"  {'Mean':>10}  {'Std':>8}"
    sep = "  " + "-" * (col_w + (10 + 2) * n + 24)

    print(f"\n  {label}  ({n} subjects)")
    print("  " + "=" * (len(header) - 2))
    print(header)
    print(sep)

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

    # Save summary CSV
    if rows_out:
        fieldnames = ['Metric'] + valid_subjects + ['Mean', 'Std']
        with open(summary_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n  Summary saved to: {summary_path}")

stage2_arg = sys.argv[1].split() if sys.argv[1] else []
stage1_arg = sys.argv[2].split() if sys.argv[2] else []
summary_s2  = sys.argv[3]
summary_s1  = sys.argv[4]
subjects    = sys.argv[5].split()

summarise(stage2_arg, "Stage 2 (+Prior, full pipeline)", subjects, summary_s2)
if stage1_arg:
    summarise(stage1_arg, "Stage 1 (Encoder-only, no prior)", subjects, summary_s1)
    print("")
PYEOF

echo ""
echo "============================================================"
echo "  Benchmark Complete!"
echo "============================================================"
echo "  Subjects run:    ${COMPLETED_SUBJECTS[*]}"
echo "  Stage-2 summary: ${SUMMARY_CSV}"
if [ "${EVAL_ENCODER_RECON}" = true ]; then
    echo "  Stage-1 summary: ${SUMMARY_S1_CSV}"
fi
echo "  Per-subject outputs under: ${OUTPUT_DIR}/<subject>/<timestamp>/"
echo "============================================================"
