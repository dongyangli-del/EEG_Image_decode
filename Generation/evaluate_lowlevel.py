"""
Evaluation script for the low-level EEG-to-VAE-latent encoder.

Given a trained encoder_low_level checkpoint:
  1. Load test EEG signals
  2. Predict SDXL-VAE latents for each test sample
  3. Decode predicted latents directly with the frozen SDXL VAE
  4. Compute reconstruction metrics (PixCorr, SSIM, AlexNet, InceptionV3, CLIP, EffNet-B, SwAV)

All paths are supplied via command-line arguments from benchmark_lowlevel.sh.
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
import scipy as sp
import pandas as pd
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_func

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrained_paths import SDXL_BASE_DIR

from eegdatasets import EEGDataset
from train_lowlevel import encoder_low_level

# Re-use helper functions from evaluate.py
from evaluate import (
    load_gt_images_as_tensor,
    load_generated_images_grouped,
    two_way_identification,
    compute_metrics,
    load_test_texts,
)


def generate_lowlevel_images(encoder, vae, loader, texts, output_dir, sub,
                              num_gen_per_class=10, device='cuda'):
    """Decode EEG → predicted VAE latent → image, saving results to disk.

    Because the low-level encoder is deterministic (no diffusion), each test
    sample produces the same image regardless of the round.  We tile the same
    decoded image `num_gen_per_class` times so the round-averaged metric
    computation matches the high-level evaluation format.
    """
    gen_dir = os.path.join(output_dir, 'generated_imgs_lowlevel', sub)
    os.makedirs(gen_dir, exist_ok=True)

    encoder.eval()

    # Collect (label, predicted_image) for all test samples
    decoded_images = {}  # label → PIL Image

    print(f"\n[Low-level] Decoding EEG features for {sub}...")
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, _text, _tf, _img, _img_feat) in enumerate(
                tqdm(loader, desc='Low-level decode')):
            eeg_data = eeg_data.to(device)
            pred_latent = encoder(eeg_data).float()  # (B, 4, 64, 64)

            # Decode with the frozen SDXL VAE
            # SDXL latents are stored scaled by vae.config.scaling_factor
            x_rec = vae.decode(pred_latent / vae.config.scaling_factor).sample
            x_rec = (x_rec / 2 + 0.5).clamp(0, 1)

            for i, label in enumerate(labels.tolist()):
                img_np = (x_rec[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                decoded_images[label] = Image.fromarray(img_np)

    print(f"Decoded {len(decoded_images)} test samples.")

    # Save images grouped by class label (using text label as folder name)
    for idx, (label, img_pil) in enumerate(sorted(decoded_images.items())):
        text_label = texts[label] if label < len(texts) else f"class_{label}"
        class_dir = os.path.join(gen_dir, text_label)
        os.makedirs(class_dir, exist_ok=True)
        # Write the same image `num_gen_per_class` times to match evaluation format
        for j in range(num_gen_per_class):
            img_pil.save(os.path.join(class_dir, f'{j}.png'))

    return gen_dir


def main():
    parser = argparse.ArgumentParser(description='Low-level EEG encoder evaluation')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--img_directory_test', type=str, required=True,
                        help='Path to ground truth test images directory')
    parser.add_argument('--img_dir_training', type=str, required=True)
    parser.add_argument('--latent_dir', type=str, required=True,
                        help='Directory containing train/test_image_latent_512.pt')
    parser.add_argument('--encoder_path', type=str, required=True,
                        help='Path to trained encoder_low_level checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--subject', type=str, default='sub-08')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_gen_per_class', type=int, default=10,
                        help='Number of image copies saved per class (all identical for deterministic decoder)')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip decoding, compute metrics on existing images')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    sub = args.subject

    # --- Load test dataset (VAE latent mode) ---
    print("Loading test dataset (vae_latent mode)...")
    test_dataset = EEGDataset(
        args.data_path,
        img_dir_training=args.img_dir_training,
        img_dir_test=args.img_directory_test,
        feature_type='vae_latent',
        latent_dir=args.latent_dir,
        subjects=[sub], train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    texts = load_test_texts(args.img_directory_test)

    gen_dir = os.path.join(args.output_dir, 'generated_imgs_lowlevel', sub)

    if not args.skip_generation:
        # --- Load low-level encoder ---
        print("Loading encoder_low_level...")
        encoder = encoder_low_level()
        encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
        encoder = encoder.to(device)
        encoder.eval()

        # --- Load frozen SDXL VAE for decoding ---
        print("Loading SDXL VAE for decoding...")
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl \
            import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            SDXL_BASE_DIR,
            torch_dtype=torch.float, variant='fp16',
            local_files_only=True)
        vae = pipe.vae.to(device)
        vae.requires_grad_(False)
        vae.eval()
        del pipe
        torch.cuda.empty_cache()

        gen_dir = generate_lowlevel_images(
            encoder, vae, test_loader, texts,
            args.output_dir, sub,
            num_gen_per_class=args.num_gen_per_class,
            device=device,
        )

        del encoder, vae
        torch.cuda.empty_cache()
    else:
        print(f"Skipping generation, loading images from: {gen_dir}")

    # --- Load ground-truth images ---
    gt_images = load_gt_images_as_tensor(args.img_directory_test, device)

    # --- Load generated images ---
    print("\nLoading generated images for metric computation...")
    recons_grouped, n_per_class = load_generated_images_grouped(gen_dir, device)
    print(f"  Reconstructed: {recons_grouped.shape}")
    print(f"  Ground truth:  {gt_images.shape}")

    # --- Compute metrics ---
    print("\n" + "=" * 60)
    print(f"[Low-level] Reconstruction Metrics ({n_per_class} rounds x {gt_images.shape[0]} classes)")
    print("=" * 60)
    metrics, stds = compute_metrics(recons_grouped, gt_images, device, n_per_class=n_per_class)

    print("\n" + "=" * 50)
    print("RECONSTRUCTION METRICS SUMMARY")
    print("=" * 50)
    df_print = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Mean": [f"{v:.4f}" for v in metrics.values()],
        "Std": [f"{stds[k]:.4f}" for k in metrics.keys()],
    })
    print(df_print.to_string(index=False))
    print("=" * 50)

    # --- Save results ---
    results_path = os.path.join(args.output_dir, f'reconstruction_metrics_{sub}_lowlevel.csv')
    df_print.to_csv(results_path, sep='\t', index=False)
    print(f"\nLow-level metrics saved to: {results_path}")


if __name__ == '__main__':
    main()
