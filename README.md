<div align="center">

<h1>Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion</h1>

<p>
  <a href="https://arxiv.org/abs/2403.07721"><img src="https://img.shields.io/badge/arXiv-2403.07721-B31B1B.svg" alt="arXiv"></a>
  <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf"><img src="https://img.shields.io/badge/NeurIPS-2024-4b44ce.svg" alt="NeurIPS 2024"></a>
  <a href="https://huggingface.co/datasets/LidongYang/EEG_Image_decode"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-blue" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
</p>

<p>
  <b><a href="https://dongyangli-del.github.io/">Dongyang Li</a> · <a href="https://hedges0-0.github.io/">Chen Wei</a> · <a href="https://github.com/1275673085/"> Shiying Li</a> · <a href="https://phyever.github.io/">Jiachen Zou</a> · <a href="https://faculty.sustech.edu.cn/liuqy/en/">Quanying Liu</a></b><br>
  Southern University of Science and Technology
</p>

<img src="imgs/fig-framework.png" alt="Framework" style="max-width: 100%; height: auto;"/>
Framework of our proposed method.

<!--  -->
<img src="imgs/fig-genexample.png" alt="fig-genexample" style="max-width: 90%; height: auto;"/>  

Some examples of using EEG to reconstruct stimulus images.


</div>

---

## 🚀 Quick Start

### ⚙️ Environment

```bash
. setup.sh && conda activate BCI
```

### 📦 Data

Download preprocessed EEG data and THINGS stimulus images from [Hugging Face](https://huggingface.co/datasets/LidongYang/EEG_Image_decode).

---

## 🔍 Task 1 — Image Retrieval

Entry point: **`Retrieval/run.sh`**

Train an ATMS encoder with CLIP contrastive loss for zero-shot 200-way image retrieval.

```bash
cd Retrieval/
bash run.sh
```

All configurations are controlled by environment variables at the top of the script:

| Variable     | Default            | Description                                                                     |
|:-------------|:-------------------|:--------------------------------------------------------------------------------|
| `ENCODER`    | `ATMS`             | Encoder architecture (`ATMS` / `EEGNetv4_Encoder` / `EEGConformer_Encoder` / …) |
| `MODE`       | `intra`            | Training mode: `intra` (within-subject) / `loso` (leave-one-out) / `joint`      |
| `SUBJECTS`   | `sub-01 … sub-10`  | Space-separated subject list                                                    |
| `EPOCHS`     | `500`              | Training epochs                                                                 |
| `BATCH_SIZE` | `1024`             | Batch size                                                                      |
| `DATA_PATH`  | —                  | Path to preprocessed EEG data                                                   |

Override any variable inline:

```bash
ENCODER=EEGNetv4_Encoder MODE=loso EPOCHS=100 bash run.sh
```

---

## 🎨 Task 2 — Image Reconstruction

Three benchmark scripts cover the full reconstruction pipeline.

### 🧠 2.1 High-Level Reconstruction

Entry point: **`Generation/benchmark.sh`**

ATMS encoder → Diffusion Prior → IP-Adapter + SDXL-Turbo.

```bash
cd Generation/
bash benchmark.sh
```

Per-subject pipeline: (1) train encoder with early stopping, (2) train diffusion prior, (3) generate and evaluate with 7 metrics, (4) aggregate cross-subject summary.

```bash
# Override subjects or resume from a checkpoint
SUBJECTS="sub-01 sub-08" bash benchmark.sh
RESUME=05-09_12-49 SUBJECTS=sub-01 bash benchmark.sh   # eval only
```

### 🖼️ 2.2 Low-Level Reconstruction

Entry point: **`Generation/benchmark_lowlevel.sh`**

EEG → SDXL-VAE latent encoder for pixel-level reconstruction.

```bash
cd Generation/
bash benchmark_lowlevel.sh
```

Prerequisite — pre-extract VAE latents:

```bash
python extract_vae_latents.py \
    --img_dir_training <TRAINING_IMAGES> \
    --img_dir_test <TEST_IMAGES> \
    --output_dir <LATENT_DIR>
```

### 🔀 2.3 Mixed High/Low-Level Reconstruction

Entry point: **`Generation/benchmark_mixed.sh`**

Blends high-level and low-level reconstructions at multiple α ratios.

```bash
cd Generation/
bash benchmark_mixed.sh
```

Requires checkpoints from both `benchmark.sh` and `benchmark_lowlevel.sh`.

```bash
ALPHAS="0.0 0.25 0.5 0.75 1.0" bash benchmark_mixed.sh
```

---

## 📁 Project Structure

```
EEG_Image_decode/
├── Retrieval/
│   ├── run.sh              ← unified launcher
│   ├── train_unified.py
│   ├── eeg_encoders.py
│   └── retrieval_engine.py
├── Generation/
│   ├── benchmark.sh        ← high-level pipeline
│   ├── benchmark_lowlevel.sh
│   ├── benchmark_mixed.sh
│   ├── train.py / train_lowlevel.py / train_encoder.py
│   ├── evaluate.py / evaluate_lowlevel.py / evaluate_mixed.py
│   ├── pipeline.py / pipeline_lowlevel.py
│   └── diffusion_prior.py
├── EEG-preprocessing/
├── MEG-preprocessing/
├── models/                  # ATMS architecture + utilities
├── eegdatasets.py           # unified dataset loader
├── encoder_utils.py         # shared training utilities
└── pretrained_paths.py
```

---

## 📝 Related Citations

```bibtex
@inproceedings{li2024visual,
  author    = {Li, Dongyang and Wei, Chen and Li, Shiying and Zou, Jiachen and Liu, Quanying},
  title     = {Visual Decoding and Reconstruction via {EEG} Embeddings with Guided Diffusion},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {102822--102864},
  year      = {2024},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf}
}
@inproceedings{li2025brainflora,
  title={BrainFLORA: Uncovering Brain Concept Representation via Multimodal Neural Embeddings},
  author={Li, Dongyang and Qin, Haoyang and Wu, Mingyang and Wei, Chen and Liu, Quanying},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={5577--5586},
  year={2025}
}
@inproceedings{li2026mindpilot,
  title={MindPilot: Closed-loop Visual Stimulation Optimization for Brain Modulation with {EEG}-guided Diffusion},
  author={Dongyang Li and Kunpeng Xie and Mingyang Wu and Yiwei Kong and Jiahua Tang and Haoyang Qin and Chen Wei and Quanying Liu},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=7jdmXx869Q}
}
```