"""
ATMS (Attention-based Time-series Model for EEG) encoder.

Shared by Generation and Retrieval pipelines.  Import with:

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.atms import ATMS, extract_id_from_string
"""

import re
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange

from models.subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from models.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.subject_layers.Embed import DataEmbedding
from models.loss import ClipLoss


# ── iTransformer config ───────────────────────────────────────────────────────

class Config:
    task_name      = 'classification'
    seq_len        = 250
    pred_len       = 250
    output_attention = False
    d_model        = 250
    embed          = 'timeF'
    freq           = 'h'
    dropout        = 0.25
    factor         = 1
    n_heads        = 4
    e_layers       = 1
    d_ff           = 256
    activation     = 'gelu'
    enc_in         = 63


# ── iTransformer backbone ─────────────────────────────────────────────────────

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout, joint_train=joint_train, num_subjects=num_subjects)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        # The subject token is prepended at index 0; skip it to keep the 63 channel tokens.
        # enc_out shape before slice: (B, 1+63, d_model)
        enc_out = enc_out[:, 1:, :]
        return enc_out


# ── Shallow CNN head ──────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Shallow temporal-spatial convolution adapted from ShallowConvNet."""

    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class FlattenHead(nn.Sequential):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40):
        super().__init__(PatchEmbedding(emb_size), FlattenHead())


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


# ── ATMS ──────────────────────────────────────────────────────────────────────

class ATMS(nn.Module):
    """
    EEG → 1024-dim embedding encoder.

    Architecture: iTransformer (channel tokens) → PatchEmbedding CNN → Proj_eeg MLP.
    The ``logit_scale`` parameter and ``loss_func`` (ClipLoss) are kept on the model
    so that checkpoints are fully self-contained.

    Parameters
    ----------
    joint_train : bool
        If True, enables cross-subject joint training mode in the DataEmbedding layer
        (shared subject token table).
    """

    def __init__(self, num_channels=63, sequence_length=250,
                 num_subjects=2, num_features=64, num_latents=1024, num_blocks=1,
                 joint_train=False):
        super().__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config, joint_train=joint_train,
                                    num_subjects=num_subjects)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(default_config.d_model, sequence_length)
             for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        return self.proj_eeg(eeg_embedding)


# ── Utility ───────────────────────────────────────────────────────────────────

def extract_id_from_string(s: str) -> int:
    """Extract the trailing integer from a subject ID, e.g. 'sub-08' → 8."""
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else None
