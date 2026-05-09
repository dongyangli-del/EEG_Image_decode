"""
Mixed-ratio EEG-to-Image reconstruction evaluation.

Combines a trained high-level ATMS+Prior model (semantic, CLIP space) with a
trained low-level encoder_low_level (perceptual, VAE latent space) and sweeps
a blending ratio α across a configurable grid.

Blending modes
--------------
  pixel_blend   α * image_hl + (1−α) * image_ll
                Pure pixel-level linear interpolation.  Simple and always valid.

  latent_blend  α * sdxl_gen + (1−α) * vae_decode
                Same as pixel_blend but makes the low-level VAE latent the
                "floor" of the reconstruction.  Identical to pixel_blend here
                since we blend after rendering.

The final generated images are evaluated with the standard seven reconstruction
metrics (PixCorr, SSIM, AlexNet(2/5), InceptionV3, CLIP, EffNet-B, SwAV) at
each α.  Results are printed as a comparison table and saved as CSV files.

Usage (see also benchmark_mixed.sh)
------
  python evaluate_mixed.py \\
      --data_path /path/to/THINGS/Preprocessed_data_250Hz \\
      --img_directory_test /path/to/test_images \\
      --img_dir_training /path/to/training_images \\
      --features_dir /path/to/clip_features \\
      --latent_dir /path/to/vae_latents \\
      --encoder_hl_path /path/to/atms_encoder.pth \\
      --prior_path /path/to/diffusion_prior.pth \\
      --encoder_ll_path /path/to/low_level_encoder.pth \\
      --output_dir ./outputs/mixed \\
      --subject sub-01 \\
      --alphas 0.0 0.25 0.5 0.75 1.0
"""

import os
import sys
import argparse
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrained_paths import SDXL_BASE_DIR

from eegdatasets import EEGDataset
from diffusion_prior import DiffusionPriorUNet, Pipe
from models.atms import ATMS, extract_id_from_string
from pipeline import Generator4Embeds
from train_lowlevel import encoder_low_level

from evaluate import (
    extract_eeg_features,
    load_test_texts,
    load_gt_images_as_tensor,
    load_generated_images_grouped,
    compute_metrics,
)


# ── Image generation helpers ──────────────────────────────────────────────────

@torch.no_grad()
def generate_hl_images(eeg_features, pipe, generator, texts, output_dir, sub,
                        num_gen_per_class=10, prior_steps=50, guidance_scale=5.0,
                        device='cuda'):
    """Generate high-level (Prior + IP-Adapter) images for each test sample."""
    gen_dir = os.path.join(output_dir, 'gen_hl', sub)
    os.makedirs(gen_dir, exist_ok=True)

    print(f"\n[HL] Generating high-level images for {len(eeg_features)} samples...")
    for k in tqdm(range(len(eeg_features)), desc='HL generation'):
        embed = eeg_features[k:k + 1].to(device)
        h = pipe.generate(c_embeds=embed, num_inference_steps=prior_steps,
                          guidance_scale=guidance_scale)
        text_label = texts[k] if k < len(texts) else f"class_{k}"
        cls_dir = os.path.join(gen_dir, text_label)
        os.makedirs(cls_dir, exist_ok=True)
        for j in range(num_gen_per_class):
            image = generator.generate(h.to(dtype=torch.float16))
            image.save(os.path.join(cls_dir, f'{j}.png'))

    return gen_dir


@torch.no_grad()
def generate_ll_images(encoder, vae, loader, texts, output_dir, sub,
                        num_gen_per_class=10, device='cuda'):
    """Generate low-level (VAE decode) images for each test sample."""
    gen_dir = os.path.join(output_dir, 'gen_ll', sub)
    os.makedirs(gen_dir, exist_ok=True)

    encoder.eval()
    decoded = {}

    for eeg_data, labels, _t, _tf, _img, _imf in tqdm(loader, desc='LL decode'):
        eeg_data = eeg_data.to(device)
        pred_latent = encoder(eeg_data).float()
        x_rec = vae.decode(pred_latent / vae.config.scaling_factor).sample
        x_rec = (x_rec / 2 + 0.5).clamp(0, 1)
        for i, label in enumerate(labels.tolist()):
            img_np = (x_rec[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            decoded[label] = Image.fromarray(img_np)

    for label, img_pil in sorted(decoded.items()):
        text_label = texts[label] if label < len(texts) else f"class_{label}"
        cls_dir = os.path.join(gen_dir, text_label)
        os.makedirs(cls_dir, exist_ok=True)
        for j in range(num_gen_per_class):
            img_pil.save(os.path.join(cls_dir, f'{j}.png'))

    return gen_dir


def blend_and_save(hl_gen_dir, ll_gen_dir, alpha, output_dir, sub,
                   target_size=(512, 512)):
    """Pixel-blend HL and LL images at ratio α, save to disk.

    Returns path to the blended image directory.
    """
    blend_dir = os.path.join(output_dir, f'gen_mixed_a{alpha:.2f}', sub)
    os.makedirs(blend_dir, exist_ok=True)

    resize = transforms.Resize(target_size)

    folders = sorted([d for d in os.listdir(hl_gen_dir)
                      if os.path.isdir(os.path.join(hl_gen_dir, d))])
    for folder in folders:
        hl_cls = os.path.join(hl_gen_dir, folder)
        ll_cls = os.path.join(ll_gen_dir, folder)
        out_cls = os.path.join(blend_dir, folder)
        os.makedirs(out_cls, exist_ok=True)

        hl_files = sorted(f for f in os.listdir(hl_cls) if f.endswith('.png'))
        ll_files = sorted(f for f in os.listdir(ll_cls) if f.endswith('.png'))

        for hf, lf in zip(hl_files, ll_files):
            hl_img = np.array(Image.open(os.path.join(hl_cls, hf)).convert('RGB').resize(
                target_size, Image.BILINEAR)).astype(np.float32) / 255.0
            ll_img = np.array(Image.open(os.path.join(ll_cls, lf)).convert('RGB').resize(
                target_size, Image.BILINEAR)).astype(np.float32) / 255.0

            blended = alpha * hl_img + (1.0 - alpha) * ll_img
            blended = (blended * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(blended).save(os.path.join(out_cls, hf))

    return blend_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Mixed HL+LL EEG reconstruction evaluation')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--img_directory_test', type=str, required=True)
    parser.add_argument('--img_dir_training', type=str, required=True)
    parser.add_argument('--features_dir', type=str, default='.',
                        help='Directory with pre-extracted CLIP feature .pt files')
    parser.add_argument('--latent_dir', type=str, required=True,
                        help='Directory with pre-computed VAE latent .pt files')
    parser.add_argument('--encoder_hl_path', type=str, required=True,
                        help='ATMS encoder checkpoint (.pth)')
    parser.add_argument('--prior_path', type=str, required=True,
                        help='Diffusion Prior checkpoint (.pth)')
    parser.add_argument('--encoder_ll_path', type=str, required=True,
                        help='encoder_low_level checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--subject', type=str, default='sub-08')
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help='Blending weights α (0=pure LL, 1=pure HL)')
    parser.add_argument('--num_gen_per_class', type=int, default=10)
    parser.add_argument('--prior_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--sdxl_steps', type=int, default=4)
    parser.add_argument('--prior_dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_generation', action='store_true',
                        help='Assume HL/LL images already exist, only do blending + metrics')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    sub = args.subject
    os.makedirs(args.output_dir, exist_ok=True)

    texts = load_test_texts(args.img_directory_test)

    hl_gen_dir = os.path.join(args.output_dir, 'gen_hl', sub)
    ll_gen_dir = os.path.join(args.output_dir, 'gen_ll', sub)

    # ── Step 1: Generate HL and LL images ────────────────────────────────────

    if not args.skip_generation:
        # --- High-level: ATMS encoder + Prior + IP-Adapter ---
        print("Loading test dataset (CLIP mode for high-level model)...")
        clip_test = EEGDataset(
            args.data_path,
            img_dir_training=args.img_dir_training,
            img_dir_test=args.img_directory_test,
            feature_type='clip',
            features_dir=args.features_dir,
            subjects=[sub], train=False)
        clip_loader = DataLoader(clip_test, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)

        print("Loading ATMS encoder...")
        atms = ATMS()
        atms.load_state_dict(torch.load(args.encoder_hl_path, map_location=device))
        atms = atms.to(device).eval()
        eeg_features = extract_eeg_features(sub, atms, clip_loader, device)
        del atms, clip_test, clip_loader
        torch.cuda.empty_cache()

        print("Loading Diffusion Prior...")
        dp = DiffusionPriorUNet(cond_dim=1024, dropout=args.prior_dropout)
        dp.load_state_dict(torch.load(args.prior_path, map_location=device))
        pipe = Pipe(dp, device=device)

        print("Loading IP-Adapter + SDXL-Turbo...")
        generator = Generator4Embeds(num_inference_steps=args.sdxl_steps, device=str(device))

        hl_gen_dir = generate_hl_images(
            eeg_features, pipe, generator, texts, args.output_dir, sub,
            num_gen_per_class=args.num_gen_per_class,
            prior_steps=args.prior_steps,
            guidance_scale=args.guidance_scale,
            device=device)
        del pipe, generator, dp
        torch.cuda.empty_cache()

        # --- Low-level: encoder_low_level + SDXL VAE ---
        print("\nLoading test dataset (VAE latent mode for low-level model)...")
        lat_test = EEGDataset(
            args.data_path,
            img_dir_training=args.img_dir_training,
            img_dir_test=args.img_directory_test,
            feature_type='vae_latent',
            latent_dir=args.latent_dir,
            subjects=[sub], train=False)
        lat_loader = DataLoader(lat_test, batch_size=args.batch_size,
                                shuffle=False, num_workers=0)

        print("Loading encoder_low_level...")
        ll_enc = encoder_low_level()
        ll_enc.load_state_dict(torch.load(args.encoder_ll_path, map_location=device))
        ll_enc = ll_enc.to(device).eval()

        print("Loading SDXL VAE for low-level decoding...")
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl \
            import DiffusionPipeline
        sdxl_pipe = DiffusionPipeline.from_pretrained(
            SDXL_BASE_DIR,
            torch_dtype=torch.float, variant='fp16',
            local_files_only=True)
        vae = sdxl_pipe.vae.to(device)
        vae.requires_grad_(False)
        vae.eval()
        del sdxl_pipe

        ll_gen_dir = generate_ll_images(
            ll_enc, vae, lat_loader, texts, args.output_dir, sub,
            num_gen_per_class=args.num_gen_per_class,
            device=device)
        del ll_enc, vae, lat_test, lat_loader
        torch.cuda.empty_cache()
    else:
        print(f"Skipping generation. Using:\n  HL: {hl_gen_dir}\n  LL: {ll_gen_dir}")

    # ── Step 2: Blend at each α and evaluate ─────────────────────────────────

    gt_images = load_gt_images_as_tensor(args.img_directory_test, device)

    all_results = {}  # alpha → {metric: mean}
    all_stds = {}

    for alpha in args.alphas:
        print(f"\n{'='*60}")
        print(f"  α = {alpha:.2f}  "
              f"({'pure LL' if alpha == 0 else 'pure HL' if alpha == 1.0 else f'blend {alpha:.0%} HL + {1-alpha:.0%} LL'})")
        print('='*60)

        if alpha == 0.0:
            blend_dir = ll_gen_dir
        elif alpha == 1.0:
            blend_dir = hl_gen_dir
        else:
            blend_dir = blend_and_save(hl_gen_dir, ll_gen_dir, alpha,
                                       args.output_dir, sub)

        recons_grouped, n_per_class = load_generated_images_grouped(blend_dir, device)
        metrics, stds = compute_metrics(recons_grouped, gt_images, device,
                                        n_per_class=n_per_class)
        all_results[alpha] = metrics
        all_stds[alpha] = stds

        # Save per-alpha CSV
        df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Mean': [f"{v:.4f}" for v in metrics.values()],
            'Std': [f"{stds[k]:.4f}" for k in metrics.keys()],
        })
        csv_path = os.path.join(args.output_dir,
                                f'reconstruction_metrics_{sub}_mixed_a{alpha:.2f}.csv')
        df.to_csv(csv_path, sep='\t', index=False)
        print(f"Saved: {csv_path}")

    # ── Step 3: Summary table ─────────────────────────────────────────────────

    metrics_keys = list(next(iter(all_results.values())).keys())
    print("\n" + "=" * 80)
    print("  MIXED RECONSTRUCTION METRICS ACROSS α VALUES")
    print("  (α=0: pure low-level VAE,  α=1: pure high-level ATMS+Prior+SDXL)")
    print("=" * 80)
    header = f"  {'Metric':<14}" + "".join(f"  α={a:.2f}" for a in args.alphas)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric in metrics_keys:
        row = f"  {metric:<14}"
        for a in args.alphas:
            row += f"  {all_results[a][metric]:>6.4f}"
        print(row)
    print("=" * 80)

    # Save full summary
    summary_rows = []
    for metric in metrics_keys:
        row = {'Metric': metric}
        for a in args.alphas:
            row[f'alpha_{a:.2f}_mean'] = f"{all_results[a][metric]:.4f}"
            row[f'alpha_{a:.2f}_std'] = f"{all_stds[a][metric]:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir,
                                f'summary_mixed_{sub}.csv')
    summary_df.to_csv(summary_path, sep='\t', index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
