#!/bin/bash
###############################################################################
# benchmark_mixed.sh
#
# Mixed-ratio EEG-to-Image reconstruction benchmark.
#
# Requires pre-trained checkpoints from BOTH:
#   1. benchmark.sh      → ATMS encoder + Diffusion Prior  (high-level, HL)
#   2. benchmark_lowlevel.sh  → encoder_low_level               (low-level,  LL)
#
# What it does
# ─────────────────────────────────────────────────────────────────────────────
#   For each subject, evaluate_mixed.py:
#     a) Generates HL images:  EEG → ATMS → Prior → IP-Adapter + SDXL
#     b) Generates LL images:  EEG → encoder_low_level → SDXL-VAE decode
#     c) Blends HL and LL at each α in ALPHAS:
#          blended = α × image_HL + (1−α) × image_LL
#     d) Computes 7 reconstruction metrics at every α
#     e) Prints a comparison table across α values
#
# Then aggregates across subjects for each α.
#
# Usage:
#   bash benchmark_mixed.sh
#   SUBJECTS="sub-01 sub-08" bash benchmark_mixed.sh
#   GPU=cuda:0 bash benchmark_mixed.sh
#
# Tip: set SKIP_GENERATION=true to re-use already generated HL/LL images and
# only redo blending + metric computation.
###############################################################################

set -e

#==============================================================================
# [DATA PATHS] - Modify these to match your data location
#==============================================================================
DATA_PATH="${DATA_PATH:-/path/to/preprocessed_data}"
IMG_DIR_TRAINING="${IMG_DIR_TRAINING:-/path/to/image_set/training_images}"
IMG_DIR_TEST="${IMG_DIR_TEST:-/path/to/image_set/test_images}"
FEATURES_DIR="${FEATURES_DIR:-/path/to/clip_features}"
LATENT_DIR="/vePFS-0x0d/visual/Features/THINGS-EEG2/VAE_latents"
VISION_MODELS_DIR="${VISION_MODELS_DIR:-./vision_models}"

#==============================================================================
# [PRE-TRAINED CHECKPOINTS]
# These paths MUST point to valid checkpoints produced by the respective
# benchmark scripts.  The easiest way is to set them from paths_info.txt.
#
# Alternatively, pass overrides from the environment, e.g.:
#   ENCODER_HL_BASE_DIR=./models/benchmark \
#   ENCODER_LL_BASE_DIR=./outputs/benchmark_lowlevel \
#   bash benchmark_mixed.sh
#
# For each SUBJECT the script will search for the latest paths_info.txt under
# the given base dirs.
#==============================================================================
ENCODER_HL_BASE_DIR="${ENCODER_HL_BASE_DIR:-./outputs/benchmark}"
ENCODER_LL_BASE_DIR="${ENCODER_LL_BASE_DIR:-./outputs/benchmark_lowlevel}"

#==============================================================================
# [OUTPUT PATH]
#==============================================================================
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/benchmark_mixed}"

#==============================================================================
# [SUBJECT LIST]
#==============================================================================
SUBJECTS="${SUBJECTS:-sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10}"

#==============================================================================
# [BLENDING RATIOS]
# α=0 → pure low-level (LL)   α=1 → pure high-level (HL)
#==============================================================================
ALPHAS="${ALPHAS:-0.0 0.25 0.5 0.75 1.0}"

#==============================================================================
# [GENERATION / EVALUATION HYPERPARAMETERS]
#==============================================================================
NUM_GEN_PER_CLASS=10
PRIOR_INFERENCE_STEPS=50
GUIDANCE_SCALE=5.0
SDXL_INFERENCE_STEPS=4
PRIOR_DROPOUT=0.1
BATCH_SIZE=64
SEED=42

#==============================================================================
# [GPU SETTINGS]
#==============================================================================
GPU="${GPU:-cuda:1}"

#==============================================================================
# [MISC]
#==============================================================================
export WANDB_MODE="offline"
mkdir -p "${OUTPUT_DIR}" "${VISION_MODELS_DIR}"
export OPEN_CLIP_CACHE_DIR="${VISION_MODELS_DIR}"

# Set to "true" to skip image generation and only redo blending + metrics
SKIP_GENERATION="${SKIP_GENERATION:-false}"

###############################################################################
# DO NOT MODIFY BELOW THIS LINE
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Mixed HL+LL EEG Reconstruction Benchmark"
echo "============================================================"
echo "  Subjects:       ${SUBJECTS}"
echo "  Alphas:         ${ALPHAS}"
echo "  HL base dir:    ${ENCODER_HL_BASE_DIR}"
echo "  LL base dir:    ${ENCODER_LL_BASE_DIR}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "  Skip gen:       ${SKIP_GENERATION}"
echo "  GPU:            ${GPU}"
echo "============================================================"
echo ""

COMPLETED_SUBJECTS=()
declare -A ALPHA_CSV_MAP   # key: "alpha", value: space-separated CSV paths

# Initialize alpha arrays
for ALPHA in ${ALPHAS}; do
    ALPHA_CSV_MAP["${ALPHA}"]=""
done

set +e

for SUBJECT in ${SUBJECTS}; do

    echo ""
    echo "############################################################"
    echo "  Subject: ${SUBJECT}"
    echo "############################################################"
    echo ""

    # ── Locate HL checkpoint (ATMS encoder + Prior) ────────────────
    HL_INFO=$(find "${ENCODER_HL_BASE_DIR}/${SUBJECT}" -name "paths_info.txt" -type f 2>/dev/null \
              | sort | tail -1)
    if [ -z "${HL_INFO}" ]; then
        echo "[WARN] HL paths_info.txt not found for ${SUBJECT} under ${ENCODER_HL_BASE_DIR}/${SUBJECT}."
        echo "       Run benchmark.sh first or set ENCODER_HL_BASE_DIR."
        continue
    fi
    ENCODER_HL=$(grep "encoder_path=" "${HL_INFO}" | cut -d= -f2)
    PRIOR_PATH=$(grep "prior_path="   "${HL_INFO}" | cut -d= -f2)

    # ── Locate LL checkpoint (encoder_low_level) ──────────────────
    LL_INFO=$(find "${ENCODER_LL_BASE_DIR}/${SUBJECT}" -name "paths_info.txt" -type f 2>/dev/null \
              | sort | tail -1)
    if [ -z "${LL_INFO}" ]; then
        echo "[WARN] LL paths_info.txt not found for ${SUBJECT} under ${ENCODER_LL_BASE_DIR}/${SUBJECT}."
        echo "       Run benchmark_lowlevel.sh first or set ENCODER_LL_BASE_DIR."
        continue
    fi
    ENCODER_LL=$(grep "encoder_path=" "${LL_INFO}" | cut -d= -f2)

    echo "  [INFO] Checkpoints found for ${SUBJECT}:"
    echo "    HL encoder:  ${ENCODER_HL}"
    echo "    Prior:       ${PRIOR_PATH}"
    echo "    LL encoder:  ${ENCODER_LL}"
    echo ""

    SUB_OUTPUT="${OUTPUT_DIR}/${SUBJECT}"
    mkdir -p "${SUB_OUTPUT}"

    # ── Build evaluation command ───────────────────────────────────
    EVAL_CMD="python evaluate_mixed.py \
        --data_path \"${DATA_PATH}\" \
        --img_directory_test \"${IMG_DIR_TEST}\" \
        --img_dir_training \"${IMG_DIR_TRAINING}\" \
        --features_dir \"${FEATURES_DIR}\" \
        --latent_dir \"${LATENT_DIR}\" \
        --encoder_hl_path \"${ENCODER_HL}\" \
        --prior_path \"${PRIOR_PATH}\" \
        --encoder_ll_path \"${ENCODER_LL}\" \
        --output_dir \"${SUB_OUTPUT}\" \
        --subject \"${SUBJECT}\" \
        --alphas ${ALPHAS} \
        --num_gen_per_class ${NUM_GEN_PER_CLASS} \
        --prior_steps ${PRIOR_INFERENCE_STEPS} \
        --guidance_scale ${GUIDANCE_SCALE} \
        --sdxl_steps ${SDXL_INFERENCE_STEPS} \
        --prior_dropout ${PRIOR_DROPOUT} \
        --batch_size ${BATCH_SIZE} \
        --gpu \"${GPU}\" \
        --seed ${SEED}"

    if [ "${SKIP_GENERATION}" = "true" ]; then
        EVAL_CMD="${EVAL_CMD} --skip_generation"
    fi

    eval ${EVAL_CMD}
    EVAL_STATUS=$?

    if [ ${EVAL_STATUS} -ne 0 ]; then
        echo "[WARN] Evaluation failed for ${SUBJECT} (exit ${EVAL_STATUS}). Skipping."
        continue
    fi

    COMPLETED_SUBJECTS+=("${SUBJECT}")

    # Collect per-alpha CSV paths
    for ALPHA in ${ALPHAS}; do
        CSV_PATH="${SUB_OUTPUT}/reconstruction_metrics_${SUBJECT}_mixed_a${ALPHA}.csv"
        if [ -f "${CSV_PATH}" ]; then
            ALPHA_CSV_MAP["${ALPHA}"]="${ALPHA_CSV_MAP[${ALPHA}]} ${CSV_PATH}"
        fi
    done

    echo ""
    echo "  [INFO] ${SUBJECT} complete."

done  # end per-subject loop

set -e

# ──────────────────────────────────────────────────────────────────────────────
# Cross-subject summary for each α
# ──────────────────────────────────────────────────────────────────────────────
N_DONE=${#COMPLETED_SUBJECTS[@]}

if [ ${N_DONE} -eq 0 ]; then
    echo ""
    echo "[ERROR] No subjects completed successfully."
    exit 1
fi

echo ""
echo "============================================================"
echo "  Cross-Subject Summary  (${N_DONE} subjects)"
echo "  Completed: ${COMPLETED_SUBJECTS[*]}"
echo "============================================================"

RUN_TS=$(date +%m-%d_%H-%M)
SUMMARY_CSV="${OUTPUT_DIR}/summary_mixed_${RUN_TS}.csv"

# Build a JSON-style map of alpha→csv_list and pass to Python
ALPHA_ARGS=""
for ALPHA in ${ALPHAS}; do
    CSV_LIST="${ALPHA_CSV_MAP[${ALPHA}]}"
    ALPHA_ARGS="${ALPHA_ARGS}|${ALPHA}:${CSV_LIST}"
done

python3 - "${ALPHA_ARGS}" "${SUMMARY_CSV}" "${COMPLETED_SUBJECTS[*]}" "${ALPHAS}" <<'PYEOF'
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

alpha_arg  = sys.argv[1]       # pipe-separated "alpha:csv1 csv2 ..." entries
summary_path = sys.argv[2]
subjects   = sys.argv[3].split()
alphas     = sys.argv[4].split()

# Parse alpha → list of CSV paths
alpha_csvs = {}
for entry in alpha_arg.split('|'):
    if ':' not in entry:
        continue
    a, rest = entry.split(':', 1)
    a = a.strip()
    paths = [p for p in rest.split() if p]
    if a and paths:
        alpha_csvs[a] = paths

# For each alpha, compute mean across subjects for each metric
alpha_means = {}   # alpha → {metric: mean}
alpha_stds  = {}

for a in alphas:
    per_metric = defaultdict(list)
    for path in alpha_csvs.get(a, []):
        if not os.path.isfile(path):
            continue
        d = read_metric_csv(path)
        for m, v in d.items():
            per_metric[m].append(v)
    if per_metric:
        alpha_means[a] = {m: np.mean(v) for m, v in per_metric.items()}
        alpha_stds[a]  = {m: np.std(v)  for m, v in per_metric.items()}

if not alpha_means:
    print("  No valid data found.")
    sys.exit(0)

metrics = list(next(iter(alpha_means.values())).keys())
valid_alphas = [a for a in alphas if a in alpha_means]

col_w = 14
header = f"  {'Metric':<{col_w}}"
for a in valid_alphas:
    lbl = f"α={a}"
    header += f"  {lbl:>9}"
print()
print("  Mixed Reconstruction Summary  (mean over subjects, ± std)")
print("  " + "=" * (len(header) - 2))
print(header)
print("  " + "-" * (len(header) - 2))

rows_out = []
for m in metrics:
    row = f"  {m:<{col_w}}"
    row_data = {'Metric': m}
    for a in valid_alphas:
        mu = alpha_means[a][m]
        sd = alpha_stds[a][m]
        row += f"  {mu:>6.4f}±{sd:.3f}"[:11].rjust(11)
        row_data[f'alpha_{a}_mean'] = f"{mu:.4f}"
        row_data[f'alpha_{a}_std']  = f"{sd:.4f}"
    print(row)
    rows_out.append(row_data)

print("  " + "=" * (len(header) - 2))

if rows_out:
    fieldnames = ['Metric'] + [f'alpha_{a}_mean' for a in valid_alphas] + \
                               [f'alpha_{a}_std'  for a in valid_alphas]
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n  Summary saved to: {summary_path}")
PYEOF

echo ""
echo "============================================================"
echo "  Mixed Benchmark Complete!"
echo "============================================================"
echo "  Subjects:  ${COMPLETED_SUBJECTS[*]}"
echo "  Alphas:    ${ALPHAS}"
echo "  Summary:   ${SUMMARY_CSV}"
echo "  Per-subject outputs under: ${OUTPUT_DIR}/<subject>/"
echo "============================================================"
