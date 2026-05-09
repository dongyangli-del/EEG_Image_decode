import os
import sys
import csv
import random
import datetime
import itertools
import math
import re
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ATMS model is defined once in models/atms.py and shared with Retrieval
from models.atms import (
    Config, iTransformer, PatchEmbedding, ResidualAdd, FlattenHead,
    Enc_eeg, Proj_eeg, ATMS, extract_id_from_string,
)
from models.loss import ClipLoss
from models.util import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
from eegdatasets import EEGDataset

# Shared encoder training utilities
from encoder_utils import (
    train_encoder_epoch as _train_encoder_epoch,
    evaluate_encoder    as _evaluate_encoder,
    stratified_condition_split,
)


def train_model(sub, eeg_model, dataloader, optimizer, device,
                text_features_all, img_features_all, config):
    """Train ATMS encoder for one epoch. Returns (avg_loss, accuracy, features)."""
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.90
    features_list = []
    mse_loss_fn = nn.MSELoss()

    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        batch_size = eeg_data.size(0)
        subject_id = extract_id_from_string(sub)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
        eeg_features = eeg_model(eeg_data, subject_ids).float()

        features_list.append(eeg_features)
        logit_scale = eeg_model.logit_scale

        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        regress_loss = mse_loss_fn(eeg_features, img_features)
        loss = alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        logits_img = logit_scale * eeg_features @ img_features_all.T
        predicted = torch.argmax(logits_img, dim=1)
        total += batch_size
        correct += (predicted == labels).sum().item()

        del eeg_data, eeg_features, img_features

    return total_loss / (batch_idx + 1), correct / total, torch.cat(features_list, dim=0)


def evaluate_model(sub, eeg_model, dataloader, device,
                   text_features_all, img_features_all, k, config):
    """Evaluate encoder with k-way retrieval accuracy. Returns (avg_loss, accuracy, top5_acc)."""
    eeg_model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    top5_correct_count = 0
    alpha = 0.99
    all_labels = set(range(text_features_all.size(0)))
    mse_loss_fn = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            batch_size = eeg_data.size(0)
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            eeg_features = eeg_model(eeg_data, subject_ids)

            logit_scale = eeg_model.logit_scale
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            regress_loss = mse_loss_fn(eeg_features, img_features)
            loss = alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10
            total_loss += loss.item()

            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]

                logits = logit_scale * eeg_features[idx] @ selected_img_features.T
                predicted_label = selected_classes[torch.argmax(logits).item()]
                if predicted_label == label.item():
                    correct += 1

                if k >= 50:
                    _, top5_indices = torch.topk(logits, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1

                total += 1

            del eeg_data, eeg_features, img_features

    top5_acc = top5_correct_count / total if total > 0 else 0.0
    return total_loss / (batch_idx + 1), correct / total, top5_acc


def main():
    parser = argparse.ArgumentParser(description='ATMS EEG Encoder Training')
    parser.add_argument('--data_path', type=str,
                        default="/root/autodl-tmp/THINGS/Preprocessed_data_250Hz")
    parser.add_argument('--img_dir_training', type=str, required=True,
                        help='Path to training images directory')
    parser.add_argument('--img_dir_test', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--features_dir', type=str, default='.',
                        help='Directory for pre-extracted CLIP features')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast')
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep")
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci")
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--logger', type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--insubject', type=bool, default=True)
    parser.add_argument('--encoder_type', type=str, default='ATMS')
    parser.add_argument('--subjects', nargs='+',
                        default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
                                 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'])
    args = parser.parse_args()

    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    subjects = args.subjects
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in subjects:
        eeg_model = globals()[args.encoder_type]()
        eeg_model.to(device)
        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=args.lr)

        if args.insubject:
            train_dataset = EEGDataset(args.data_path,
                                       img_dir_training=args.img_dir_training,
                                       img_dir_test=args.img_dir_test,
                                       features_dir=args.features_dir,
                                       subjects=[sub], train=True)
            test_dataset = EEGDataset(args.data_path,
                                      img_dir_training=args.img_dir_training,
                                      img_dir_test=args.img_dir_test,
                                      features_dir=args.features_dir,
                                      subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(args.data_path,
                                       img_dir_training=args.img_dir_training,
                                       img_dir_test=args.img_dir_test,
                                       features_dir=args.features_dir,
                                       exclude_subject=sub, subjects=subjects, train=True)
            test_dataset = EEGDataset(args.data_path,
                                      img_dir_training=args.img_dir_training,
                                      img_dir_test=args.img_dir_test,
                                      features_dir=args.features_dir,
                                      exclude_subject=sub, subjects=subjects, train=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1,
                                 shuffle=True, num_workers=0, drop_last=True)

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []
        best_accuracy = 0.0
        results = []

        logger = wandb_logger(args) if args.logger else None
        if logger:
            logger.watch(eeg_model, logger)

        for epoch in range(args.epochs):
            train_loss, train_accuracy, _ = train_model(
                sub, eeg_model, train_loader, optimizer, device,
                text_features_train_all, img_features_train_all, config=args)

            if (epoch + 1) % 5 == 0:
                if args.insubject:
                    save_dir = f"./models/contrast/{args.encoder_type}/{sub}/{current_time}"
                else:
                    save_dir = f"./models/contrast/across/{args.encoder_type}/{current_time}"
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, f"{epoch + 1}.pth")
                torch.save(eeg_model.state_dict(), file_path)
                print(f"Model saved in {file_path}!")

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            test_loss, test_accuracy, top5_acc = evaluate_model(
                sub, eeg_model, test_loader, device,
                text_features_test_all, img_features_test_all, k=200, config=args)
            _, v2_acc, _ = evaluate_model(sub, eeg_model, test_loader, device,
                                          text_features_test_all, img_features_test_all, k=2, config=args)
            _, v4_acc, _ = evaluate_model(sub, eeg_model, test_loader, device,
                                          text_features_test_all, img_features_test_all, k=4, config=args)
            _, v10_acc, _ = evaluate_model(sub, eeg_model, test_loader, device,
                                           text_features_test_all, img_features_test_all, k=10, config=args)
            _, v50_acc, v50_top5_acc = evaluate_model(sub, eeg_model, test_loader, device,
                                                       text_features_test_all, img_features_test_all, k=50, config=args)
            _, v100_acc, v100_top5_acc = evaluate_model(sub, eeg_model, test_loader, device,
                                                         text_features_test_all, img_features_test_all, k=100, config=args)

            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            results.append({
                "epoch": epoch + 1,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc": v2_acc,
                "v4_acc": v4_acc,
                "v10_acc": v10_acc,
                "top5_acc": top5_acc,
                "v50_acc": v50_acc,
                "v100_acc": v100_acc,
                "v50_top5_acc": v50_top5_acc,
                "v100_top5_acc": v100_top5_acc,
            })

            if logger:
                logger.log({
                    "Train Loss": train_loss, "Train Accuracy": train_accuracy,
                    "Test Loss": test_loss, "Test Accuracy": test_accuracy,
                    "v2 Accuracy": v2_acc, "v4 Accuracy": v4_acc,
                    "v10 Accuracy": v10_acc, "Epoch": epoch,
                })

            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train L={train_loss:.4f} A={train_accuracy:.4f} | "
                  f"Test L={test_loss:.4f} A={test_accuracy:.4f} Top5={top5_acc:.4f}")
            print(f"  v2={v2_acc:.4f}  v4={v4_acc:.4f}  v10={v10_acc:.4f}  "
                  f"v50={v50_acc:.4f}  v100={v100_acc:.4f}")

        if logger:
            logger.finish()

        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)
        if args.insubject:
            results_file = f"{results_dir}/{args.encoder_type}_{sub}.csv"
        else:
            results_file = f"{results_dir}/{args.encoder_type}_cross_exclude_{sub}.csv"

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {results_file}')


if __name__ == '__main__':
    main()
