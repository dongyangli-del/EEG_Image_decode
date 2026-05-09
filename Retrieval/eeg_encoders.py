"""EEG encoder architectures for retrieval experiments.

Consolidates encoder classes that were previously scattered across
train_contrast.py and train_atme.py, alongside the ATMS import from
models.atms.  All encoders expose a common interface:

    model.logit_scale  – learnable temperature parameter
    model.loss_func    – ClipLoss instance
    model(eeg_data)    – returns a (B, D) feature tensor
                         OR (clip_features, mse_features) for MetaEEG

Encoders that additionally require a subject ID (ATMS, MetaEEG) are
tracked in SUBJECT_ID_ENCODERS and handled by train_unified.py /
retrieval_engine.py automatically.

Usage:
    from eeg_encoders import build_encoder, ENCODER_REGISTRY, SUBJECT_ID_ENCODERS
    model = build_encoder('ATMS')
    model = build_encoder('Projector', in_shape=(63, 250))
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from braindecode.models import ATCNet, EEGConformer, EEGITNet, EEGNetv4, ShallowFBCSPNet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.loss import ClipLoss


# ===========================================================================
# Shared building blocks
# ===========================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        return x + pe


class EEGAttention(nn.Module):
    """Single-layer Transformer encoder applied along the time axis."""
    def __init__(self, channel: int, d_model: int, nhead: int):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       batch_first=False),
            num_layers=1,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = src.permute(2, 0, 1)       # (T, B, C)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src)
        return out.permute(1, 2, 0)      # (B, C, T)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)


class FlattenHead(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.contiguous().view(x.size(0), -1)


# ===========================================================================
# NICE encoder  (ShallowConv + projection head)
# ===========================================================================

class _NICEPatchEmbedding(nn.Module):
    """Shallow-conv patch embedding from NICE (Ye et al.)."""
    def __init__(self, emb_size: int = 40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
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
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        return self.projection(x)


class _Proj_eeg(nn.Sequential):
    """Residual projection head shared by NICE and ATM_E."""
    def __init__(self, embedding_dim: int = 1440, proj_dim: int = 1024,
                 drop_proj: float = 0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class NICE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_eeg     = nn.Sequential(_NICEPatchEmbedding(), FlattenHead())
        self.proj_eeg    = _Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func   = ClipLoss()

    def forward(self, data: Tensor) -> Tensor:
        return self.proj_eeg(self.enc_eeg(data))


# ===========================================================================
# ATM_E encoder  (EEGAttention + EEGNetv4 backbone + projection head)
# ===========================================================================

class _ATMEPatchEmbedding(nn.Module):
    """EEGNetv4-based patch embedding used by ATM_E."""
    def __init__(self, n_chans: int = 63, n_times: int = 250):
        super().__init__()
        self.tsconv = EEGNetv4(
            in_chans=n_chans, n_classes=1440,
            input_window_samples=n_times,
            final_conv_length='auto', pool_mode='mean',
            F1=8, D=20, F2=160, kernel_length=4,
            third_kernel_size=(4, 2), drop_prob=0.25,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(3)
        return self.tsconv(x)


class ATM_E(nn.Module):
    def __init__(self, num_channels: int = 63, sequence_length: int = 250,
                 num_subjects: int = 1, num_latents: int = 1024):
        super().__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(sequence_length, sequence_length)
             for _ in range(num_subjects)]
        )
        self.enc_eeg     = nn.Sequential(_ATMEPatchEmbedding(num_channels, sequence_length),
                                         FlattenHead())
        self.proj_eeg    = _Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func   = ClipLoss()

    def forward(self, x: Tensor, subject_id: int = 0) -> Tensor:
        x = self.attention_model(x)
        x = self.subject_wise_linear[subject_id](x)
        return self.proj_eeg(self.enc_eeg(x))


# ===========================================================================
# Projector  (MLP mixer-style)
# ===========================================================================

def _make_mixer_block(h_c: int, h_l: int, dropout: float = 0.25) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(h_l),
        nn.Linear(h_l, h_l),
        nn.GELU(),
        nn.Dropout(dropout),
        Rearrange('B C L -> B L C'),
        nn.LayerNorm(h_c),
        nn.Linear(h_c, h_c),
        nn.GELU(),
        nn.Dropout(dropout),
        Rearrange('B L C -> B C L'),
    )


class Projector(nn.Module):
    def __init__(self, in_features=(63, 250), h_dim=(64, 1024),
                 n_hidden_layer: int = 2, dropout: float = 0.25):
        super().__init__()
        c, l    = in_features
        h_c, h_l = h_dim
        self.input_layer = nn.Sequential(
            nn.LayerNorm(l),
            nn.Linear(l, h_l), nn.GELU(), nn.Dropout(dropout),
            Rearrange('B C L -> B L C'),
            nn.LayerNorm(c),
            nn.Linear(c, h_c), nn.GELU(), nn.Dropout(dropout),
            Rearrange('B L C -> B C L'),
        )
        self.blocks = nn.Sequential(
            *[_make_mixer_block(h_c, h_l, dropout) for _ in range(n_hidden_layer)]
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(h_l),
            nn.Linear(h_l, 1024), nn.GELU(), nn.Dropout(dropout),
            Rearrange('B C L -> B L C'),
            nn.LayerNorm(h_c),
            nn.Linear(h_c, 1), nn.GELU(), nn.Dropout(dropout),
            Rearrange('B L C -> B (C L)'),
        )
        self.projector   = nn.Sequential(self.input_layer, self.blocks, self.output_layer)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self.loss_func   = ClipLoss()

    def forward(self, eeg_embeds: Tensor) -> Tensor:
        return F.normalize(self.projector(eeg_embeds), dim=-1)


# ===========================================================================
# MetaEEG  (conv-attention hybrid with dual output heads)
# ===========================================================================

class _ConvBlock(nn.Module):
    def __init__(self, num_channels: int, num_features: int):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(num_features, num_features, kernel_size=3, padding=1)
        # LayerNorm over last dim (= time); preserved from original to keep
        # identical behaviour (num_features == sequence_length in MetaEEG)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)
        self.residual_conv = nn.Conv1d(num_channels, num_features, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual_conv(x)
        x = self.norm1(F.gelu(self.conv1(x)))
        x = self.norm2(F.gelu(self.conv2(x)))
        x = self.norm3(F.gelu(self.conv3(x)))
        return x + residual


class _MLPHead(nn.Module):
    def __init__(self, in_features: int, num_latents: int, dropout: float = 0.25):
        super().__init__()
        self.layer = nn.Sequential(
            Rearrange('B C L -> B L C'),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_latents),
            nn.GELU(),
            nn.Dropout(dropout),
            Rearrange('B L C -> B (C L)'),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class MetaEEG(nn.Module):
    """Attention + conv-block encoder with separate CLIP and MSE heads.

    forward() returns (clip_features, mse_features).
    train_unified.py / retrieval_engine.py use only the clip_features tensor.
    """
    def __init__(self, num_channels: int = 63, sequence_length: int = 250,
                 num_subjects: int = 1, num_latents: int = 1024,
                 num_blocks: int = 1):
        super().__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(sequence_length, sequence_length)
             for _ in range(num_subjects)]
        )
        self.conv_blocks = nn.Sequential(
            *[_ConvBlock(num_channels, sequence_length) for _ in range(num_blocks)],
            Rearrange('B C L -> B L C'),
        )
        self.linear_projection = nn.Sequential(
            Rearrange('B L C -> B C L'),
            nn.Linear(sequence_length, num_latents),
            Rearrange('B C L -> B L C'),
        )
        self.temporal_aggregation = nn.Linear(sequence_length, 1)
        self.clip_head   = _MLPHead(num_latents, num_latents)
        self.mse_head    = _MLPHead(num_latents, num_latents)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.01))
        self.loss_func   = ClipLoss()

    def forward(self, x: Tensor, subject_id: int = 0):
        x = self.attention_model(x)
        x = self.subject_wise_linear[subject_id](x)
        x = self.conv_blocks(x)
        x = self.linear_projection(x)
        x = self.temporal_aggregation(x)
        return self.clip_head(x), self.mse_head(x)


# ===========================================================================
# BrainDecode wrapper encoders
# ===========================================================================

class EEGNetv4_Encoder(nn.Module):
    def __init__(self, n_chans: int = 63, n_times: int = 250):
        super().__init__()
        self.eegnet = EEGNetv4(
            in_chans=n_chans, n_classes=1024,
            input_window_samples=n_times,
            final_conv_length='auto', pool_mode='mean',
            F1=8, D=20, F2=160, kernel_length=4,
            third_kernel_size=(4, 2), drop_prob=0.25,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func   = ClipLoss()

    def forward(self, data: Tensor) -> Tensor:
        data = data.unsqueeze(0)
        data = data.reshape(data.shape[1], data.shape[2], data.shape[3], data.shape[0])
        return self.eegnet(data)


class EEGConformer_Encoder(nn.Module):
    def __init__(self, n_chans: int = 63, n_times: int = 250):
        super().__init__()
        self.eegConformer = EEGConformer(
            n_outputs=None, n_chans=n_chans,
            n_filters_time=40, filter_time_length=10,
            pool_time_length=25, pool_time_stride=5,
            drop_prob=0.25, att_depth=2, att_heads=1, att_drop_prob=0.5,
            final_fc_length=1760, return_features=False,
            n_times=None, chs_info=None, input_window_seconds=None,
            n_classes=1024, input_window_samples=n_times,
            add_log_softmax=True,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func   = ClipLoss()

    def forward(self, data: Tensor) -> Tensor:
        return self.eegConformer(data)


class EEGITNet_Encoder(nn.Module):
    def __init__(self, n_chans: int = 63, n_times: int = 250):
        super().__init__()
        self.eegEEGITNet = EEGITNet(
            n_outputs=1024, n_chans=n_chans, n_times=None,
            drop_prob=0.4, chs_info=None,
            input_window_seconds=1.0, sfreq=250,
            input_window_samples=n_times, add_log_softmax=True,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func   = ClipLoss()

    def forward(self, data: Tensor) -> Tensor:
        return self.eegEEGITNet(data)


class ShallowFBCSPNet_Encoder(nn.Module):
    def __init__(self, n_chans: int = 63, n_times: int = 250):
        super().__init__()
        self.model = ShallowFBCSPNet(
            n_chans=n_chans, n_outputs=1024, n_times=n_times,
            n_filters_time=20, filter_time_length=20,
            n_filters_spat=20, pool_time_length=25, pool_time_stride=5,
            final_conv_length='auto', pool_mode='mean',
            split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
            drop_prob=0.5, chs_info=None,
            input_window_seconds=1.0, sfreq=250, add_log_softmax=True,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func   = ClipLoss()

    def forward(self, data: Tensor) -> Tensor:
        return self.model(data)


class ATCNet_Encoder(nn.Module):
    def __init__(self, n_chans: int = 63, n_times: int = 250):
        super().__init__()
        self.model = ATCNet(
            n_chans=n_chans, n_outputs=1024,
            input_window_seconds=1.0, sfreq=250.,
            conv_block_n_filters=8,
            conv_block_kernel_length_1=32, conv_block_kernel_length_2=8,
            conv_block_pool_size_1=4, conv_block_pool_size_2=3,
            conv_block_depth_mult=2, conv_block_dropout=0.3,
            n_windows=5, att_head_dim=4, att_num_heads=2, att_dropout=0.5,
            tcn_depth=2, tcn_kernel_size=4, tcn_n_filters=16,
            tcn_dropout=0.3, tcn_activation=nn.ELU(),
            concat=False, max_norm_const=0.25,
            chs_info=None, n_times=None, n_channels=None,
            n_classes=None, input_size_s=None, add_log_softmax=True,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func   = ClipLoss()

    def forward(self, data: Tensor) -> Tensor:
        return self.model(data)


# ===========================================================================
# Registry and factory
# ===========================================================================

#: All locally-defined encoders (does not include ATMS which lives in models.atms)
ENCODER_REGISTRY = {
    'NICE':                    NICE,
    'ATM_E':                   ATM_E,
    'Projector':               Projector,
    'MetaEEG':                 MetaEEG,
    'EEGNetv4_Encoder':        EEGNetv4_Encoder,
    'EEGConformer_Encoder':    EEGConformer_Encoder,
    'EEGITNet_Encoder':        EEGITNet_Encoder,
    'ShallowFBCSPNet_Encoder': ShallowFBCSPNet_Encoder,
    'ATCNet_Encoder':          ATCNet_Encoder,
}

#: Encoders whose forward() accepts an integer subject_id as the second argument
SUBJECT_ID_ENCODERS = {'ATMS', 'MetaEEG'}

#: Encoders that benefit from L2 feature normalisation before ClipLoss
NORMALIZE_FEAT_ENCODERS = {'ATMS'}


def build_encoder(encoder_type: str, n_chans: int = 63, n_times: int = 250,
                  joint_train: bool = False, **kwargs) -> nn.Module:
    """Instantiate an encoder by name with sensible defaults.

    Args:
        encoder_type: One of the keys in ENCODER_REGISTRY or 'ATMS'.
        n_chans: Number of EEG channels (default 63).
        n_times: Number of time samples (default 250).
        joint_train: Passed to ATMS when True (enables joint-subject mode).
        **kwargs: Extra constructor arguments forwarded to the encoder class.

    Returns:
        Instantiated nn.Module.
    """
    if encoder_type == 'ATMS':
        from models.atms import ATMS
        return ATMS(joint_train=joint_train, **kwargs)

    cls = ENCODER_REGISTRY.get(encoder_type)
    if cls is None:
        raise ValueError(
            f"Unknown encoder_type {encoder_type!r}. "
            f"Available: {['ATMS'] + list(ENCODER_REGISTRY)}"
        )

    # Encoders that accept explicit shape parameters
    if encoder_type == 'Projector':
        return cls(in_features=(n_chans, n_times), **kwargs)
    if encoder_type in ('ATM_E', 'MetaEEG'):
        return cls(num_channels=n_chans, sequence_length=n_times, **kwargs)
    if encoder_type in ('EEGNetv4_Encoder', 'EEGConformer_Encoder',
                        'EEGITNet_Encoder', 'ShallowFBCSPNet_Encoder',
                        'ATCNet_Encoder'):
        return cls(n_chans=n_chans, n_times=n_times, **kwargs)

    return cls(**kwargs)
