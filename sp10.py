
import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from cls_eegdatasets import EEGDataset
from eegencoder import eeg_encoder
from einops.layers.torch import Rearrange, Reduce
from lavis.models.clip_models.loss import ClipLoss
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from utils import wandb_logger

def make_block(h_c, h_l, dropout_rate=0.4):
    block = nn.Sequential(
        nn.LayerNorm(h_l),
        nn.Linear(h_l, h_l), 
        nn.GELU(),
        # nn.Dropout(dropout_rate),
        Rearrange('B C L->B L C'),
        nn.LayerNorm(h_c),
        nn.Linear(h_c, h_c), 
        nn.GELU(),
        # nn.Dropout(dropout_rate),
        Rearrange('B L C->B C L'),
    )
    return block

class Projector(nn.Module):

    def __init__(self, in_features, h_dim=(64, 512), n_hidden_layer=4, dropout_rate=0.4):
        # in_features: (c, l)
        super().__init__()
        c, l = in_features
        h_c, h_l = h_dim
        c_o, l_o = 1, 512

        self.input_layer = nn.Sequential(
            nn.LayerNorm(l),
            nn.Linear(l, h_l), 
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            Rearrange('B C L->B L C'),
            nn.LayerNorm(c),
            nn.Linear(c, h_c), 
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            Rearrange('B L C->B C L'),
        )
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(h_l),
            nn.Linear(h_l, l_o), 
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            Rearrange('B C L->B L C'),
            nn.LayerNorm(h_c),
            nn.Linear(h_c, c_o), 
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            Rearrange('B L C->B (C L)'),
        )
        
        self.blocks = nn.Sequential(*[
            make_block(h_c, h_l) for _ in range(n_hidden_layer)
        ])
        
        self.projector = nn.Sequential(*[
            self.input_layer,
            self.blocks,
            self.output_layer,
        ])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
    
    def forward(self, eeg_embeds):
        eeg_embeds = self.projector(eeg_embeds)
        eeg_features = F.normalize(eeg_embeds, dim=-1)
        return eeg_features

class eegEncoder(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.mae_encoder = eeg_encoder() 
        self.projector = Projector(in_features)
        self.freeze_mae()
        self.logit_scale = self.projector.logit_scale
        self.loss_func = self.projector.loss_func

    def freeze_mae(self):
        for param in self.mae_encoder.parameters():
            param.requires_grad = False

    def forward(self, eeg):
        eeg_embeds = self.mae_encoder(eeg)
        eeg_features = self.projector(eeg_embeds)
        return eeg_features


def train_model(model, dataloader, optimizer, device, text_features_all, img_features_all):
    model.train()
    text_features_all = text_features_all.to(device).float() # (n_cls, d)
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        eeg_features = model(eeg_data).float()
        logit_scale = model.logit_scale
        
        loss = model.loss_func(eeg_features, img_features, logit_scale)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        
        logits = logit_scale * eeg_features @ img_features_all.T # (n_batch, n_cls)
        predicted = torch.argmax(logits, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy

def evaluate_model(model, dataloader, device, text_features_all, img_features_all):
    model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
   
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            img_features = img_features.to(device).float()
            labels = labels.to(device)
        
            eeg_features = model(eeg_data).float()
            logit_scale = model.logit_scale 

            loss = model.loss_func(eeg_features, img_features, logit_scale)
         
            total_loss += loss.item()
            
            logits = logit_scale * eeg_features @ img_features_all.T # (n_batch, n_cls)
            predicted = torch.argmax(logits, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}
           
            batch_size = predicted.shape[0]
            total += batch_size
            correct += (predicted == labels).sum().item()
            print(predicted)
            print(labels)

    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy

def main_train_loop(model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(model,logger) 
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}

    for epoch in range(config['epochs']):
        # 训练模型
        train_loss, train_accuracy = train_model(model, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 评估模型
        test_loss, test_accuracy = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all)
 
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
      

        # 如果当前epoch的测试正确率是最好的，保存模型和相关信息
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_weights = model.state_dict().copy()
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)

    # 保存最佳模型
    # torch.save(model.state_dict(), 'best_model_lr=3e-4_img_maskeeg.pth')

# 创建5个子图
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # 损失图
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # 总体正确率图
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")


    # 构造你要注释的字符串信息
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
               )

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)


    # 添加大标题
    plt.suptitle('best_model_lr=3e-4_img_maskeeg', fontsize=16, y=1.05)
    plt.savefig('best_model_lr=3e-4_img_maskeeg.png')
    logger.log({"Plots": logger.Image(plt)})
    logger.finish()

def main():
    config = {
        'project': 'eeg_clf',
        'entity': 'vis-obj',
        'name': '特殊图片十分类', 
        'lr': 3e-3, 
        'epochs': 50, 
        'batch_size': 512, 
        'logger': True, 
    }

   # Instantiate the dataset and dataloader
    data_path = '/home/weichen/projects/shiyin/EEG-cm/data'  # Replace with the path to your data
    train_dataset = EEGDataset(data_path, train=True)
    test_dataset = EEGDataset(data_path, train=True, classes = [920, 1292, 1498, 694, 83, 439, 821, 164, 494, 1265], pictures=[4, 9, 9, 3, 6, 0, 1, 2, 0, 8])
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=0, drop_last=False)
    
    text_features_train_all = train_dataset.text_features
    text_features_test_all = test_dataset.text_features
    img_features_train_all = train_dataset.img_features
    img_features_test_all = test_dataset.img_features
    
    
    # 模型和优化器定义
    in_features = (17, 512)
    model = eegEncoder(in_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    
    print('number of parameters:', sum([p.numel() for p in model.parameters()]))

    # 确定使用的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    main_train_loop(model, train_loader, test_loader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=config['logger'])
    
if __name__ == '__main__':
    main()