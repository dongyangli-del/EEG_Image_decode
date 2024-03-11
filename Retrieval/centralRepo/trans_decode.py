# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
eeg_size = 750
patch_size = 25


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbedding(nn.Module):
    def __init__(self, num_channels, time_length, patch_length, hidden_size):
        super(PatchEmbedding, self).__init__()

        num_patches = time_length // patch_length

        self.num_channels = num_channels
        self.time_length = time_length
        self.hidden_size = hidden_size
        self.num_patches = num_patches

        self.projection = nn.Conv1d(num_channels, hidden_size, kernel_size=patch_length, stride=patch_length)

    def forward(self, pixel_values):
        """
            (batch_size, num_channels, time_length) -> (batch_size, seq_length, hidden_size)
        """
        x = self.projection(pixel_values)
        # embeddings = x.transpose(1, 2)
        embeddings = x.transpose(0, 1)
        # embeddings = self.projection(pixel_values).transpose(1, 2)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, num_channels, time_length, patch_length, hidden_size):
        super(Encoder, self).__init__()
        self.patch_embeddings = PatchEmbedding(num_channels, time_length, patch_length, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4,
                                                        dropout=.1, activation="gelu")
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.pos = torch.arange(0, time_length // patch_length, step=1, dtype=torch.float32)
        # self.pos_embed = nn.Parameter(torch.zeros(1, time_length // patch_length, hidden_size), requires_grad=False)
        self.pos_embed = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(embed_dim=hidden_size, pos=self.pos)).to(
            device)

    def forward(self, pixel_values):
        embeddings = self.patch_embeddings(pixel_values)
        embeddings += self.pos_embed
        # embeddings = self.patch_embeddings(pixel_values)
        output = self.encoder(embeddings)
        return output


class Decoder(nn.Module):
    def __init__(self, embed_dim, classes):
        # def __init__(self):
        super(Decoder, self).__init__()
        # self.mlp = nn.Sequential(nn.Linear(640*64, 4))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * (eeg_size // patch_size), 4),
            #  nn.Dropout(),
            #  nn.LayerNorm(1000),
            #  nn.ReLU(),
            nn.Dropout(p=0.5))

        #  nn.Linear(1000, classes)
        # )

    def forward(self, x):
        # print('x: ', x.shape)
        # x = torch.flatten(x, start_dim=1)
        x = torch.flatten(x)
        # print(x.shape)
        return self.mlp(x)


class Decode:
    def __init__(self):
        print("encoder")
        self.encoder = Encoder(num_channels=22, time_length=eeg_size, patch_length=patch_size, hidden_size=768).to(device)
        self.encoder.load_state_dict(torch.load('./model/encoder.pkl', map_location=device))
        print("decoder")
        self.decoder = Decoder(embed_dim=768, classes=4).to(device)
        self.decoder.load_state_dict(torch.load('./model/decoder.pkl', map_location=device))

        self.encoder.eval()
        self.decoder.eval()

    def decode(self, data):
        data = torch.from_numpy(data).float().to(device)
        x = self.encoder(data)
        return self.decoder(x).cpu().detach().numpy()
