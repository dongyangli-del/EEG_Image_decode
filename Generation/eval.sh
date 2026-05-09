#!/bin/bash
###############################################################################
# eval.sh
#
# Standalone evaluation script: loads the best encoder + prior checkpoints
# from a completed training run, then on the TEST set:
#   1. Extracts EEG embeddings → Prior Diffusion → IP-Adapter+SDXL-Turbo
#   2. Generates reconstructed images (200 classes × N images each)
#   3. Computes 8 reconstruction metrics and prints a summary table
#
# Usage:
#   bash eval.sh
#
# Customize the variables below to match your setup.
###############################################################################

set -e

#==============================================================================
# [DATA PATHS] - Must match training config
#==============================================================================
DATA_PATH="${DATA_PATH:-/path/to/preprocessed_data}"
IMG_DIR_TEST="${IMG_DIR_TEST:-/path/to/image_set/test_images}"
FEATURES_DIR="${FEATURES_DIR:-/path/to/clip_features}"
VISION_MODELS_DIR="${VISION_MODELS_DIR:-./vision_models}"

#==============================================================================
# [TRAINED MODEL PATHS] - Point to your best checkpoints
#==============================================================================
ENCODER_PATH="${ENCODER_PATH:-./models/benchmark/encoder/sub-08/best.pth}"
PRIOR_PATH="${PRIOR_PATH:-./models/benchmark/prior/sub-08/best.pth}"

#==============================================================================
# [SUBJECT]
#==============================================================================
SUBJECT="sub-08"

#==============================================================================
# [EVALUATION HYPERPARAMETERS]
#==============================================================================
NUM_GEN_PER_CLASS=5          # Generated images per test class
PRIOR_INFERENCE_STEPS=50      # Diffusion prior denoising steps
GUIDANCE_SCALE=5.0            # Classifier-free guidance scale
SDXL_INFERENCE_STEPS=4        # SDXL-Turbo steps
PRIOR_DROPOUT=0.1             # Must match training setting
BATCH_SIZE=1024
SEED=42

#==============================================================================
# [OUTPUT]
#==============================================================================
OUTPUT_DIR="./outputs/eval"

#==============================================================================
# [GPU]
#==============================================================================
GPU="cuda:0"

#==============================================================================
# [SKIP OPTIONS]
#   - Set SKIP_GENERATION=true to only recompute metrics on existing images
#   - Set GENERATED_IMGS_DIR to the folder with pre-generated images
#==============================================================================
SKIP_GENERATION=true
GENERATED_IMGS_DIR="${GENERATED_IMGS_DIR:-}"

###############################################################################
# DO NOT MODIFY BELOW THIS LINE
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Environment
mkdir -p "${VISION_MODELS_DIR}"
export OPEN_CLIP_CACHE_DIR="${VISION_MODELS_DIR}"
export WANDB_MODE="offline"

echo "============================================================"
echo "  ATMS Test-Set Evaluation"
echo "============================================================"
echo "  Subject:       ${SUBJECT}"
echo "  Encoder:       ${ENCODER_PATH}"
echo "  Prior:         ${PRIOR_PATH}"
echo "  GT images:     ${IMG_DIR_TEST}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Skip gen:      ${SKIP_GENERATION}"
echo "============================================================"
echo ""

# Build command
CMD="python evaluate.py \
    --data_path ${DATA_PATH} \
    --img_directory_test ${IMG_DIR_TEST} \
    --features_dir ${FEATURES_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --encoder_path ${ENCODER_PATH} \
    --prior_path ${PRIOR_PATH} \
    --subject ${SUBJECT} \
    --batch_size ${BATCH_SIZE} \
    --num_gen_per_class ${NUM_GEN_PER_CLASS} \
    --prior_steps ${PRIOR_INFERENCE_STEPS} \
    --guidance_scale ${GUIDANCE_SCALE} \
    --sdxl_steps ${SDXL_INFERENCE_STEPS} \
    --prior_dropout ${PRIOR_DROPOUT} \
    --gpu ${GPU} \
    --seed ${SEED}"

if [ "${SKIP_GENERATION}" = true ]; then
    CMD="${CMD} --skip_generation"
    if [ -n "${GENERATED_IMGS_DIR}" ]; then
        CMD="${CMD} --generated_imgs_dir ${GENERATED_IMGS_DIR}"
    fi
fi

eval ${CMD}

echo ""
echo "============================================================"
echo "  Evaluation Complete!"
echo "============================================================"
echo "  Metrics file:     ${OUTPUT_DIR}/reconstruction_metrics_${SUBJECT}.csv"
echo "  Generated images: ${OUTPUT_DIR}/generated_imgs/${SUBJECT}/"
echo "============================================================"
