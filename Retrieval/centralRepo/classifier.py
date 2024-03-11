
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder without pooling
        self.encoder_layers = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        
        # Separate MaxPool layer with return_indices=True
        self.pool = nn.MaxPool1d(2, stride=2, return_indices=True)
        
        # Middle layers (if needed)
        # Currently it just acts as a placeholder
        self.middle = nn.Sequential()
        
        # Decoder without unpooling
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Separate MaxUnpool layer
        self.unpool = nn.MaxUnpool1d(2, stride=2)
        
    def forward(self, x):
        # If size is odd, pad the tensor
        if x.size(2) % 2 != 0:
            x = F.pad(x, (0, 1))  # Pad last dimension
        
        x_encoded = self.encoder_layers(x)
        x_pooled, indices = self.pool(x_encoded)
        
        x_middle = self.middle(x_pooled)
        
        x_unpooled = self.unpool(x_middle, indices)
        x_decoded = self.decoder_layers(x_unpooled)
        return x_decoded


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(59, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        flattened_size = 12000  # Calculated from torch.Size([1, 32, 375])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),  # Adjusted with the calculated flattened size
            nn.ReLU(True),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = x.reshape(1, x.shape[0], x.shape[1])
        x = self.classifier(x)
        return x
