"""
extract_vae_latents.py

Pre-extracts SDXL-VAE latent encodings for all training and test images and
saves them as .pt files.  Run this once before low-level training.

Expected output layout:
    <output_dir>/train_image_latent_512.pt   {'image_latent': Tensor(N, 4, 64, 64)}
    <output_dir>/test_image_latent_512.pt    {'image_latent': Tensor(M, 4, 64, 64)}

Usage:
    python extract_vae_latents.py \\
        --img_dir_training /path/to/image_set/training_images \\
        --img_dir_test     /path/to/image_set/test_images \\
        --output_dir       /path/to/vae_latents \\
        --image_size       512 \\
        --batch_size       64 \\
        --gpu              cuda:0
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL


def build_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def collect_image_paths(root_dir: str) -> list[str]:
    """Return sorted list of image file paths under root_dir (all sub-directories)."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in extensions:
                paths.append(os.path.join(dirpath, fname))
    paths.sort()
    return paths


@torch.no_grad()
def encode_images(image_paths: list[str], vae: AutoencoderKL,
                  transform: T.Compose, batch_size: int,
                  device: torch.device) -> torch.Tensor:
    """Encode images to VAE latents.  Returns Tensor of shape (N, 4, H/8, W/8)."""
    all_latents = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='Encoding', unit='batch'):
        batch_paths = image_paths[i: i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
            except Exception as e:
                print(f'  [WARN] Failed to load {p}: {e}')
                continue
            imgs.append(transform(img))
        if not imgs:
            continue
        pixel_values = torch.stack(imgs).to(device)
        latents = vae.encode(pixel_values).latent_dist.mean
        all_latents.append(latents.cpu())
    return torch.cat(all_latents, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-extract SDXL-VAE latents from THINGS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--img_dir_training', type=str, required=True,
                        help='Root directory of training images')
    parser.add_argument('--img_dir_test', type=str, required=True,
                        help='Root directory of test images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where latent .pt files will be saved')
    parser.add_argument('--vae_model', type=str,
                        default='stabilityai/sdxl-turbo',
                        help='HuggingFace model ID for the SDXL VAE')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Resize/crop target resolution before VAE encoding')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for VAE encoding')
    parser.add_argument('--gpu', type=str, default='cuda:0',
                        help='GPU device string')
    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading VAE from {args.vae_model} …')
    vae = AutoencoderKL.from_pretrained(args.vae_model, subfolder='vae').to(device)
    vae.eval()

    transform = build_transform(args.image_size)

    for split_name, img_dir in [('train', args.img_dir_training),
                                 ('test',  args.img_dir_test)]:
        print(f'\n[{split_name}] Collecting images from {img_dir}')
        paths = collect_image_paths(img_dir)
        if not paths:
            print(f'  [WARN] No images found in {img_dir}')
            continue
        print(f'  Found {len(paths)} images')

        latents = encode_images(paths, vae, transform, args.batch_size, device)
        print(f'  Latent shape: {tuple(latents.shape)}')

        out_file = os.path.join(args.output_dir,
                                f'{split_name}_image_latent_{args.image_size}.pt')
        torch.save({'image_latent': latents, 'image_paths': paths}, out_file)
        print(f'  Saved → {out_file}')

    print('\nDone.')


if __name__ == '__main__':
    main()
