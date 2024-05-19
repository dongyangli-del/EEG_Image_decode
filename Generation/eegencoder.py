import utils 
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F

class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, time_len=51, patch_size=3, in_chans=17, embed_dim=512):
        super().__init__()
        num_patches = time_len // patch_size
        self.patch_shape = patch_size
        self.time_len = time_len
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels

        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x  

class eeg_encoder(nn.Module):
    def __init__(self, time_len=80, patch_size=3, embed_dim=512, in_chans=17,
                 depth=24, num_heads=16, mlp_ratio=1., norm_layer=nn.LayerNorm, global_pool=False):
        super().__init__()
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)

        num_patches = time_len // patch_size

        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = utils.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # print(x.shape)
        # print(self.pos_embed[:, 1:, :].shape)
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # print(x.shape)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        # print(x.shape)
        x = self.norm(x)
        # print(x.shape)
        return x  

    def forward(self, eeg):
        if eeg.ndim == 2:
            eeg = torch.unsqueeze(eeg, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(eeg)
   
        return latent  # N, n_seq, embed_dim
    
    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        utils.interpolate_pos_embed(self, state_dict)
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        return 
    



if __name__ == '__main__':

    input = torch.randn(2,17,51)

    encoder = eeg_encoder()
    out = encoder(input)
    print(out.shape)
 