import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset
from eegencoder import eeg_encoder
from einops.layers.torch import Rearrange, Reduce
from lavis.models.clip_models.loss import ClipLoss
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from utils import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
import itertools
import math
import datetime

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x




class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 5), (1, 1)),
            nn.AvgPool2d((1, 17), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 

class ATM_S_reconstruction_scale_0_300(nn.Module):    
    def __init__(self, num_channels=63, sequence_length=25, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(ATM_S_reconstruction_scale_0_300, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)   
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
         
    def forward(self, x):
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
         
        x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)
        # print(f'After enc_eeg shape: {eeg_embedding.shape}')
        out = self.proj_eeg(eeg_embedding)
        return out  
    
      

def train_model(eegmodel, imgmodel, dataloader, optimizer, device, text_features_all, img_features_all):
    eegmodel.train()
    text_features_all = text_features_all.to(device).float() # (n_cls, d)
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha=0.9
    features_list = []  # List to store features
    save_features= True
    ridge_lambda = 0.1
    mse_loss_fn = nn.MSELoss()
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        optimizer.zero_grad()
        eeg_features = eegmodel(eeg_data[:, :, :75]).float()
        # img_features_outputs = regression(eeg_features).float()
        features_list.append(eeg_features)
        logit_scale = eegmodel.logit_scale
        img_loss = eegmodel.loss_func(eeg_features, img_features, logit_scale)
        text_loss = eegmodel.loss_func(eeg_features, text_features, logit_scale)
        contrastive_loss = img_loss
        # print("text_loss", text_loss)
        # print("img_loss", img_loss)
        
        regress_loss =  mse_loss_fn(eeg_features, img_features)
        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = (regress_loss + ridge_lambda * l2_norm)       
        loss = (alpha * regress_loss *10 + (1 - alpha) * contrastive_loss*10)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()
        
        # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
        
        logits_img = logit_scale * eeg_features @ img_features_all.T
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        # logits_single = (logits_text + logits_img) / 2.0        
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy

def evaluate_model(eegmodel, imgmodel, dataloader, device, text_features_all, img_features_all, k):
    eegmodel.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha =0.9
    top5_correct = 0
    top5_correct_count = 0
    
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    ridge_lambda = 0.1
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            eeg_features = eegmodel(eeg_data[:, :, :75]).float()

            logit_scale = eegmodel.logit_scale 
                   
            regress_loss =  mse_loss_fn(eeg_features, img_features)
            # print("eeg_features", eeg_features.shape)
            # print(torch.std(eeg_features, dim=-1))
            # print(torch.std(img_features, dim=-1))
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = (regress_loss + ridge_lambda * l2_norm)       
            img_loss = eegmodel.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eegmodel.loss_func(eeg_features, text_features, logit_scale)
            contrastive_loss = img_loss
            # loss = img_loss + text_loss

            regress_loss =  mse_loss_fn(eeg_features, img_features)
            # print("text_loss", text_loss)
            # print("img_loss", img_loss)
            # print("regress_loss", regress_loss)            
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = (regress_loss + ridge_lambda * l2_norm)       
            loss = alpha * regress_loss *10 + (1 - alpha) * contrastive_loss*10
            # print("loss", loss)
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                if k==200:
                    
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) \in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1
                    
                    
                    
                    
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                           
                    
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                    
    print("total_loss", total_loss)
    print("batch_idx+1", batch_idx+1)                
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc

def main_train_loop(sub, eeg_model, img_model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model,logger) 
    logger.watch(img_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")  
    for epoch in range(config['epochs']):
        
        train_loss, train_accuracy = train_model(eeg_model, img_model, train_dataloader, optimizer, device, text_features_train_all, img_features_train_all)
        if (epoch +1) % 5 == 0:                    
            
            if config['insubject']==True:       
                os.makedirs(f"./models/contrast/{config['encoder_type']}/{current_time}/{sub}", exist_ok=True)             
                file_path = f"./models/contrast/{config['encoder_type']}/{current_time}/{sub}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)            
            else:                
                os.makedirs(f"./models/contrast/across/{config['encoder_type']}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/across/{config['encoder_type']}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        regression = None
        
        test_loss, test_accuracy, top5_acc = evaluate_model(eeg_model, regression, test_dataloader, device, text_features_test_all, img_features_test_all,k=200)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "top5_acc":top5_acc
        }
        results.append(epoch_results)
        
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_weights = eeg_model.state_dict().copy()
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

        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")        
  
    
    # model.load_state_dict(best_model_weights)

    
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')

    
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()
    return results


import datetime
import os
import torch
from torch.utils.data import DataLoader

def main():
    config = {
        "data_path": "/home/ldy/Workspace/THINGS/Preprocessed_data_250Hz",
        "project": "train_pos_img_text_rep",
        "entity": "sustech_rethinkingbci",
        "name": "lr=3e-4_img_pos_pro_eeg",
        "lr": 3e-4,
        "epochs": 40,
        "batch_size": 1024,
        "logger": True,
        "insubject": True,
        "encoder_type": 'ATM_S_reconstruction_scale_0_300',
        "img_encoder": 'Proj_img'
    }

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    data_path = config['data_path']
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']

    for sub in subjects:
        # Reinitialize the models and optimizer for each subject
        eeg_model = globals()[config['encoder_type']](63, 300//4)
        img_model = globals()[config['img_encoder']]()
        eeg_model.to(device)
        img_model.to(device)
        optimizer = torch.optim.AdamW(itertools.chain(eeg_model.parameters(), img_model.parameters()), lr=config['lr'])
        
        print(f'Processing {sub}: number of parameters:', sum([p.numel() for p in eeg_model.parameters()]) + sum([p.numel() for p in img_model.parameters()]))

        train_dataset = EEGDataset(data_path, subjects=[sub] if config['insubject'] else [], exclude_subject=sub if not config['insubject'] else None, train=True)
        test_dataset = EEGDataset(data_path, subjects=[sub] if config['insubject'] else [], exclude_subject=sub if not config['insubject'] else None, train=False)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, eeg_model, img_model, train_loader, test_loader, optimizer, device, 
                                  text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=config['logger'])

        # Save results to a CSV file
        results_dir = f"./outputs/{config['encoder_type']}/{sub}/"
        os.makedirs(results_dir, exist_ok=True)
        results_file = f"{results_dir}/{config['encoder_type']}_{('cross_exclude_' if not config['insubject'] else '')}{sub}.csv"

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {results_file}')

            
if __name__ == '__main__':
    main()