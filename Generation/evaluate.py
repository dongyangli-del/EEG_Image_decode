"""
Evaluation script for EEG-to-Image reconstruction.

Given a trained ATMS encoder + Diffusion Prior:
  1. Extract EEG features from test set
  2. Generate image embeddings via Diffusion Prior
  3. Generate images via IP-Adapter + SDXL-Turbo
  4. Compute reconstruction metrics (PixCorr, SSIM, AlexNet, InceptionV3, CLIP, EffNet, SwAV)

All paths are passed via command-line arguments from benchmark.sh.
"""

import os
import sys
import re
import shutil
import argparse
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import scipy as sp
import pandas as pd
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim_func

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eegdatasets import EEGDataset
from diffusion_prior import DiffusionPriorUNet, Pipe
from models.atms import ATMS, extract_id_from_string
from pipeline import Generator4Embeds


def extract_eeg_features(sub, eeg_model, dataloader, device):
    """Extract EEG features from the model for all test samples."""
    eeg_model.eval()
    features_list = []
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            batch_size = eeg_data.size(0)
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            eeg_features = eeg_model(eeg_data, subject_ids)
            features_list.append(eeg_features.detach().cpu())
    return torch.cat(features_list, dim=0)


def load_test_texts(img_directory_test):
    """Load test class text labels from the test image directory."""
    dirnames = [d for d in os.listdir(img_directory_test) if os.path.isdir(os.path.join(img_directory_test, d))]
    dirnames.sort()
    texts = []
    for dir_name in dirnames:
        try:
            idx = dir_name.index('_')
            description = dir_name[idx + 1:]
        except ValueError:
            description = dir_name
        texts.append(description)
    return texts


def generate_images_encoder_only(eeg_features_test, generator, texts, output_dir, sub,
                                 num_gen_per_class=10, device='cuda',
                                 gen_batch_size=1):
    """Generate images directly from EEG encoder output, skipping the diffusion prior (Stage 1)."""
    gen_dir = os.path.join(output_dir, 'generated_imgs_encoder_only', sub)
    if os.path.isdir(gen_dir):
        shutil.rmtree(gen_dir)
    os.makedirs(gen_dir)

    n_samples = len(eeg_features_test)
    total = n_samples * num_gen_per_class

    for k in range(n_samples):
        text_label = texts[k] if k < len(texts) else f"class_{k}"
        os.makedirs(os.path.join(gen_dir, text_label), exist_ok=True)

    all_embeds = eeg_features_test.repeat_interleave(num_gen_per_class, dim=0)

    print(f"\n[Stage 1 / Encoder-only] Generating {total} images (batch_size={gen_batch_size})...")
    for i in tqdm(range(0, total, gen_batch_size), desc="Generating (encoder-only)"):
        batch = all_embeds[i:i + gen_batch_size].to(device, dtype=torch.float16)
        images = generator.generate_batch(batch)
        for bi, image in enumerate(images):
            flat_idx = i + bi
            k = flat_idx // num_gen_per_class
            j = flat_idx % num_gen_per_class
            text_label = texts[k] if k < len(texts) else f"class_{k}"
            image.save(os.path.join(gen_dir, text_label, f'{j}.png'))

    return gen_dir


def generate_images(eeg_features_test, pipe, generator, texts, output_dir, sub,
                    num_gen_per_class=10, prior_steps=50, guidance_scale=5.0,
                    device='cuda', gen_batch_size=1, prior_batch_size=1024):
    """Generate images from EEG test features using prior + IP-Adapter."""
    gen_dir = os.path.join(output_dir, 'generated_imgs', sub)
    if os.path.isdir(gen_dir):
        shutil.rmtree(gen_dir)
    os.makedirs(gen_dir)

    n_samples = len(eeg_features_test)
    total = n_samples * num_gen_per_class

    for k in range(n_samples):
        text_label = texts[k] if k < len(texts) else f"class_{k}"
        os.makedirs(os.path.join(gen_dir, text_label), exist_ok=True)

    # Step 1: run diffusion prior in batches
    print(f"\nRunning Diffusion Prior for {n_samples} test samples (batch_size={prior_batch_size})...")
    prior_embeds = []
    for i in range(0, n_samples, prior_batch_size):
        batch = eeg_features_test[i:i + prior_batch_size].to(device)
        h = pipe.generate(c_embeds=batch, num_inference_steps=prior_steps,
                          guidance_scale=guidance_scale)
        prior_embeds.append(h.cpu())
    prior_embeds = torch.cat(prior_embeds, dim=0)

    # Step 2: replicate each embedding num_gen_per_class times and generate in batches
    all_embeds = prior_embeds.repeat_interleave(num_gen_per_class, dim=0)

    print(f"\nGenerating {total} images in batches of {gen_batch_size}...")
    for i in tqdm(range(0, total, gen_batch_size), desc="Generating"):
        batch = all_embeds[i:i + gen_batch_size].to(device, dtype=torch.float16)
        images = generator.generate_batch(batch)
        for bi, image in enumerate(images):
            flat_idx = i + bi
            k = flat_idx // num_gen_per_class
            j = flat_idx % num_gen_per_class
            text_label = texts[k] if k < len(texts) else f"class_{k}"
            image.save(os.path.join(gen_dir, text_label, f'{j}.png'))

    return gen_dir


def load_generated_images_grouped(gen_dir, device):
    """Load generated images grouped by class.

    Returns:
        recons: (n_classes, n_per_class, C, H, W)
        n_per_class: int
    """
    folders = sorted([d for d in os.listdir(gen_dir) if os.path.isdir(os.path.join(gen_dir, d))])
    all_classes = []
    n_per_class = 0
    for folder_name in folders:
        folder_path = os.path.join(gen_dir, folder_name)
        images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not images:
            continue
        n_per_class = len(images)
        class_tensors = []
        for img_name in images:
            with Image.open(os.path.join(folder_path, img_name)) as img:
                t = torch.tensor(np.array(img.convert('RGB'))).permute(2, 0, 1).float() / 255.0
                class_tensors.append(t)
        all_classes.append(torch.stack(class_tensors, dim=0))
    recons = torch.stack(all_classes, dim=0).to(device)
    print(f"  Loaded {recons.shape[0]} classes x {recons.shape[1]} images per class")
    return recons, n_per_class


def load_gt_images_as_tensor(img_directory_test, device):
    """Load ground truth test images. Returns (n_classes, C, H, W)."""
    tensor_list = []
    folders = sorted([d for d in os.listdir(img_directory_test)
                      if os.path.isdir(os.path.join(img_directory_test, d))])
    for folder_name in folders:
        folder_path = os.path.join(img_directory_test, folder_name)
        images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if images:
            with Image.open(os.path.join(folder_path, images[0])) as img:
                t = torch.tensor(np.array(img.convert('RGB'))).permute(2, 0, 1).float() / 255.0
                tensor_list.append(t.unsqueeze(0))
    return torch.cat(tensor_list, dim=0).to(device)


@torch.no_grad()
def two_way_identification(recons_round, gt_images, model, preprocess,
                           feature_layer=None):
    """Standard 200-way identification for ONE round (one generated image per class).

    recons_round: (n_classes, C, H, W)
    gt_images:    (n_classes, C, H, W)
    Returns: accuracy (float)
    """
    device = gt_images.device
    preds = model(torch.stack([preprocess(r) for r in recons_round], dim=0).to(device))
    reals = model(torch.stack([preprocess(g) for g in gt_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    n_classes = len(gt_images)
    r = np.corrcoef(reals, preds)
    r = r[:n_classes, n_classes:]
    congruents = np.diag(r)
    success = r < congruents
    return np.mean(np.sum(success, 0)) / (n_classes - 1)


def compute_metrics(recons_grouped, gt_images, device, n_per_class=1):
    """Compute reconstruction metrics using round-based evaluation.

    For each of n_per_class rounds, select the j-th generated image per class,
    compute all metrics against the GT images, then average across rounds.
    This ensures no cross-contamination between generated images of the same class.

    recons_grouped: (n_classes, n_per_class, C, H, W)
    gt_images:      (n_classes, C, H, W)
    """
    from torchvision.models.feature_extraction import create_feature_extractor

    n_classes = gt_images.shape[0]
    imsize = 256
    gt_images = transforms.Resize((imsize, imsize))(gt_images)
    recons_grouped = transforms.Resize((imsize, imsize))(
        recons_grouped.view(-1, *recons_grouped.shape[2:])
    ).view(n_classes, n_per_class, 3, imsize, imsize).clamp(0, 1).to(gt_images.dtype)

    results = {k: [] for k in ['PixCorr', 'SSIM', 'AlexNet(2)', 'AlexNet(5)',
                                'InceptionV3', 'CLIP', 'EffNet-B', 'SwAV']}

    # Pre-compute pixel-level preprocessed GT (shared across rounds)
    preprocess_pix = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    gt_pix = preprocess_pix(gt_images).reshape(n_classes, -1).cpu()
    gt_gray_arr = rgb2gray(
        preprocess_pix(gt_images).permute(0, 2, 3, 1).cpu().numpy())

    # Load all feature extraction models once
    from torchvision.models import alexnet, AlexNet_Weights
    alex_model = create_feature_extractor(
        alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
        return_nodes=['features.4', 'features.11']
    ).to(device).eval().requires_grad_(False)
    preprocess_alex = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from torchvision.models import inception_v3, Inception_V3_Weights
    inception_model = create_feature_extractor(
        inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes=['avgpool']
    ).to(device).eval().requires_grad_(False)
    preprocess_inc = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    import clip
    clip_model, _ = clip.load("ViT-L/14", device=device)
    preprocess_clip = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
    eff_model = create_feature_extractor(
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes=['avgpool']
    ).to(device).eval().requires_grad_(False)
    preprocess_eff = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    swav_model = create_feature_extractor(
        swav_model, return_nodes=['avgpool']
    ).to(device).eval().requires_grad_(False)
    preprocess_swav = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Pre-extract GT features for distance metrics (shared across rounds)
    gt_eff = eff_model(preprocess_eff(gt_images))['avgpool'].reshape(n_classes, -1).cpu().numpy()
    gt_swav = swav_model(preprocess_swav(gt_images))['avgpool'].reshape(n_classes, -1).cpu().numpy()

    print(f"\nRunning {n_per_class} evaluation rounds ({n_classes} classes per round)...")
    for j in range(n_per_class):
        recons_j = recons_grouped[:, j]  # (n_classes, C, H, W)

        # --- PixCorr ---
        rec_pix = preprocess_pix(recons_j).reshape(n_classes, -1).cpu()
        pix_scores = [np.corrcoef(gt_pix[i].numpy(), rec_pix[i].numpy())[0][1]
                      for i in range(n_classes)]
        results['PixCorr'].append(np.mean(pix_scores))

        # --- SSIM ---
        rec_gray = rgb2gray(preprocess_pix(recons_j).permute(0, 2, 3, 1).cpu().numpy())
        ssim_scores = [ssim_func(rec_gray[i], gt_gray_arr[i],
                                 multichannel=True, gaussian_weights=True,
                                 sigma=1.5, use_sample_covariance=False, data_range=1.0)
                       for i in range(n_classes)]
        results['SSIM'].append(np.mean(ssim_scores))

        # --- AlexNet 2-way ---
        results['AlexNet(2)'].append(two_way_identification(
            recons_j.to(device).float(), gt_images, alex_model, preprocess_alex, 'features.4'))
        results['AlexNet(5)'].append(two_way_identification(
            recons_j.to(device).float(), gt_images, alex_model, preprocess_alex, 'features.11'))

        # --- InceptionV3 ---
        results['InceptionV3'].append(two_way_identification(
            recons_j, gt_images, inception_model, preprocess_inc, 'avgpool'))

        # --- CLIP ---
        results['CLIP'].append(two_way_identification(
            recons_j, gt_images, clip_model.encode_image, preprocess_clip, None))

        # --- EffNet-B ---
        fake_eff = eff_model(preprocess_eff(recons_j))['avgpool'].reshape(n_classes, -1).cpu().numpy()
        results['EffNet-B'].append(np.mean(
            [sp.spatial.distance.correlation(gt_eff[i], fake_eff[i]) for i in range(n_classes)]))

        # --- SwAV ---
        fake_swav = swav_model(preprocess_swav(recons_j))['avgpool'].reshape(n_classes, -1).cpu().numpy()
        results['SwAV'].append(np.mean(
            [sp.spatial.distance.correlation(gt_swav[i], fake_swav[i]) for i in range(n_classes)]))

        print(f"  Round {j+1}/{n_per_class}: " +
              "  ".join(f"{k}={results[k][-1]:.4f}" for k in results))

    del alex_model, inception_model, clip_model, eff_model, swav_model
    torch.cuda.empty_cache()

    final = {k: np.mean(v) for k, v in results.items()}
    return final, {k: np.std(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description='EEG-to-Image Reconstruction Evaluation')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--img_directory_test', type=str, required=True,
                        help='Path to ground truth test images directory')
    parser.add_argument('--img_dir_training', type=str, default=None,
                        help='Path to training images directory (needed by EEGDataset init)')
    parser.add_argument('--features_dir', type=str, default=None,
                        help='Directory containing pre-extracted CLIP features. '
                             'Defaults to EEG_Image_decode/features/ (shared cache).')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--encoder_path', type=str, required=True)
    parser.add_argument('--prior_path', type=str, required=True)
    parser.add_argument('--sdxl_model_path', type=str, default=None,
                        help='Local path to SDXL-Turbo model directory. '
                             'Defaults to /vePFS-0x0d/visual/dataset/pretrained/sdxl-turbo')
    parser.add_argument('--ip_adapter_path', type=str, default=None,
                        help='Local path to IP-Adapter root directory (contains sdxl_models/). '
                             'Defaults to /vePFS-0x0d/visual/dataset/pretrained/ip-adapter')
    parser.add_argument('--subject', type=str, default='sub-08')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_gen_per_class', type=int, default=10)
    parser.add_argument('--prior_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--sdxl_steps', type=int, default=4)
    parser.add_argument('--gen_batch_size', type=int, default=1,
                        help='SDXL generation batch size (images per forward pass). '
                             'Higher values use more VRAM but generate faster.')
    parser.add_argument('--prior_dropout', type=float, default=0.1)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip image generation, only compute metrics on existing images')
    parser.add_argument('--generated_imgs_dir', type=str, default=None,
                        help='Path to pre-generated images (used with --skip_generation)')
    parser.add_argument('--eval_encoder_recon', action='store_true',
                        help='Also evaluate encoder-only reconstruction (Stage 1, no prior) '
                             'and print a side-by-side comparison with the full pipeline (Stage 2)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    sub = args.subject

    # --- Load test data ---
    print("Loading test dataset...")
    test_dataset = EEGDataset(args.data_path,
                              img_dir_training=args.img_dir_training,
                              img_dir_test=args.img_directory_test,
                              features_dir=args.features_dir,
                              subjects=[sub], train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    img_features_test_all = test_dataset.img_features

    texts = load_test_texts(args.img_directory_test)

    enc_gen_dir = None  # encoder-only generated images directory

    if not args.skip_generation:
        # --- Load encoder ---
        print("Loading ATMS encoder...")
        eeg_model = ATMS()
        eeg_model.load_state_dict(torch.load(args.encoder_path, map_location=device))
        eeg_model = eeg_model.to(device)
        eeg_model.eval()

        # --- Extract EEG features ---
        print("Extracting EEG features from test set...")
        eeg_features_test = extract_eeg_features(sub, eeg_model, test_loader, device)
        print(f"  EEG features shape: {eeg_features_test.shape}")
        del eeg_model
        torch.cuda.empty_cache()

        # --- Load prior ---
        print("Loading Diffusion Prior...")
        diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=args.prior_dropout)
        diffusion_prior.load_state_dict(torch.load(args.prior_path, map_location=device))
        pipe = Pipe(diffusion_prior, device=device)

        # --- Load generator (shared by both stages) ---
        print("Loading IP-Adapter + SDXL-Turbo...")
        generator = Generator4Embeds(
            num_inference_steps=args.sdxl_steps,
            device=str(device),
            sdxl_model_path=args.sdxl_model_path,
            ip_adapter_path=args.ip_adapter_path,
        )

        # --- Stage 1: encoder-only generation (optional) ---
        if args.eval_encoder_recon:
            enc_gen_dir = generate_images_encoder_only(
                eeg_features_test, generator, texts,
                args.output_dir, sub,
                num_gen_per_class=args.num_gen_per_class,
                device=device,
                gen_batch_size=args.gen_batch_size,
            )

        # --- Stage 2: full pipeline generation (encoder → prior → SDXL) ---
        print("\n[Stage 2 / Full pipeline] Generating images via Diffusion Prior...")
        gen_dir = generate_images(
            eeg_features_test, pipe, generator, texts,
            args.output_dir, sub,
            num_gen_per_class=args.num_gen_per_class,
            prior_steps=args.prior_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            gen_batch_size=args.gen_batch_size,
            prior_batch_size=args.batch_size,
        )
        del pipe, generator
        torch.cuda.empty_cache()
    else:
        gen_dir = args.generated_imgs_dir or os.path.join(args.output_dir, 'generated_imgs', sub)
        print(f"Skipping generation, using images from: {gen_dir}")
        if args.eval_encoder_recon:
            enc_gen_dir = os.path.join(args.output_dir, 'generated_imgs_encoder_only', sub)
            if not os.path.isdir(enc_gen_dir):
                print(f"[WARN] --eval_encoder_recon requested but encoder-only image dir not found: {enc_gen_dir}")
                enc_gen_dir = None

    # --- Load ground-truth images (shared) ---
    gt_images = load_gt_images_as_tensor(args.img_directory_test, device)

    # --- Stage 1 metrics (encoder-only) ---
    enc_metrics = None
    enc_stds = None
    if enc_gen_dir is not None and os.path.isdir(enc_gen_dir):
        print("\nLoading encoder-only images for metric computation...")
        enc_recons_grouped, enc_n_per_class = load_generated_images_grouped(enc_gen_dir, device)
        print(f"  Encoder-only: {enc_recons_grouped.shape}  |  GT: {gt_images.shape}")
        print("\n" + "=" * 60)
        print(f"[Stage 1] Encoder-only Metrics ({enc_n_per_class} rounds x {gt_images.shape[0]} classes)")
        print("=" * 60)
        enc_metrics, enc_stds = compute_metrics(enc_recons_grouped, gt_images, device,
                                                n_per_class=enc_n_per_class)

    # --- Stage 2 metrics (full pipeline) ---
    print("\nLoading full-pipeline images for metric computation...")
    recons_grouped, n_per_class = load_generated_images_grouped(gen_dir, device)
    print(f"  Reconstructed: {recons_grouped.shape}  (classes, images_per_class, C, H, W)")
    print(f"  Ground truth:  {gt_images.shape}  (classes, C, H, W)")

    print("\n" + "=" * 60)
    print(f"[Stage 2] Full Pipeline Metrics ({n_per_class} rounds x {gt_images.shape[0]} classes)")
    print("=" * 60)
    metrics, stds = compute_metrics(recons_grouped, gt_images, device, n_per_class=n_per_class)

    # --- Print results table ---
    if enc_metrics is not None:
        # Two-stage comparison table
        print("\n" + "=" * 70)
        print("  RECONSTRUCTION METRICS: Encoder-only (Stage 1) vs. +Prior (Stage 2)")
        print("=" * 70)
        header = f"  {'Metric':<14}  {'Enc-only':>10}  {'+ Prior':>10}  {'Δ (Prior gain)':>14}"
        print(header)
        print("  " + "-" * 66)
        for k in metrics:
            enc_val = enc_metrics[k]
            full_val = metrics[k]
            delta = full_val - enc_val
            sign = "+" if delta >= 0 else ""
            print(f"  {k:<14}  {enc_val:>10.4f}  {full_val:>10.4f}  {sign}{delta:>13.4f}")
        print("=" * 70)

        # Save encoder-only metrics
        enc_df = pd.DataFrame({
            "Metric": list(enc_metrics.keys()),
            "Mean": [f"{v:.4f}" for v in enc_metrics.values()],
            "Std": [f"{enc_stds[k]:.4f}" for k in enc_metrics.keys()],
        })
        enc_results_path = os.path.join(args.output_dir, f'reconstruction_metrics_{sub}_encoder_only.csv')
        enc_df.to_csv(enc_results_path, sep='\t', index=False)
        print(f"\nEncoder-only metrics saved to: {enc_results_path}")
    else:
        print("\n" + "=" * 50)
        print("RECONSTRUCTION METRICS SUMMARY")
        print("=" * 50)
        df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Mean": [f"{v:.4f}" for v in metrics.values()],
            "Std": [f"{stds[k]:.4f}" for k in metrics.keys()],
        })
        print(df.to_string(index=False))
        print("=" * 50)

    # --- Save full-pipeline results ---
    df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Mean": [f"{v:.4f}" for v in metrics.values()],
        "Std": [f"{stds[k]:.4f}" for k in metrics.keys()],
    })
    results_path = os.path.join(args.output_dir, f'reconstruction_metrics_{sub}.csv')
    df.to_csv(results_path, sep='\t', index=False)
    print(f"\nFull-pipeline metrics saved to: {results_path}")


if __name__ == '__main__':
    main()
