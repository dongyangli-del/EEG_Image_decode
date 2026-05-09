"""Centralised local paths for all pretrained models.

All scripts in Generation/ and Retrieval/ should import from here instead
of hard-coding HuggingFace model IDs or absolute paths.

Override any path at runtime by setting the corresponding environment variable,
e.g.  PRETRAINED_ROOT=/mnt/fast_ssd/pretrained python train.py
"""

import os

# Root of the local pretrained-model store
PRETRAINED_ROOT = os.environ.get(
    'PRETRAINED_ROOT',
    '/vePFS-0x0d/visual/dataset/pretrained',
)

def _p(*parts):
    return os.path.join(PRETRAINED_ROOT, *parts)


# ── Diffusion models ──────────────────────────────────────────────────────────
SDXL_TURBO_DIR      = _p('sdxl-turbo')
SDXL_BASE_DIR       = _p('stable-diffusion-xl-base-1.0')

# ── IP-Adapter ────────────────────────────────────────────────────────────────
IP_ADAPTER_DIR      = _p('ip-adapter')          # root; subdirs: sdxl_models/

# ── Vision encoders ───────────────────────────────────────────────────────────
EVA_CLIP_DIR        = _p('EVA-CLIP')
INTERN_VIT_DIR      = _p('InternViT-6B-448px-V1-5')
SIGLIP2_DIR         = _p('siglip2-giant-opt-patch16-256')
PE_CORE_DIR         = _p('PE-Core-G14-448')
DFN_DIR             = _p('DFN-public')
DINOV2_DIR          = _p('dinov2')

# ── Misc ──────────────────────────────────────────────────────────────────────
PRIORS_DIR          = _p('priors')
