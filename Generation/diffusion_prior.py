import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset


class DiffusionPrior(nn.Module):
    """Simple MLP-based diffusion prior."""

    def __init__(self, embed_dim=1024, cond_dim=42, hidden_dim=1024,
                 layers_per_block=4, time_embed_dim=512, act_fn=nn.SiLU, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim

        # Time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        self.time_embedding = TimestepEmbedding(time_embed_dim, hidden_dim)

        # Conditional embedding
        self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
        )

        # Hidden layers (residual)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            )
            for _ in range(layers_per_block)
        ])

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, t, c=None):
        # x: (batch, embed_dim)  t: (batch,)  c: (batch, cond_dim)
        t = self.time_embedding(self.time_proj(t))           # (batch, hidden_dim)
        c = self.cond_embedding(c) if c is not None else 0   # (batch, hidden_dim) or 0
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = x + t + c
            x = layer(x) + x
        return self.output_layer(x)


class DiffusionPriorUNet(nn.Module):
    """U-Net shaped MLP diffusion prior with skip connections."""

    def __init__(self, embed_dim=1024, cond_dim=42,
                 hidden_dim=[1024, 512, 256, 128, 64],
                 time_embed_dim=512, act_fn=nn.SiLU, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        # Time sinusoidal projection (shared across encoder/decoder)
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # Input projection: embed_dim → hidden_dim[0]
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        # Encoder: hidden_dim[0] → ... → hidden_dim[-1]
        self.encode_time_embedding = nn.ModuleList([
            TimestepEmbedding(time_embed_dim, hidden_dim[i])
            for i in range(self.num_layers - 1)
        ])
        self.encode_cond_embedding = nn.ModuleList([
            nn.Linear(cond_dim, hidden_dim[i])
            for i in range(self.num_layers - 1)
        ])
        self.encode_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i + 1]),
                nn.LayerNorm(hidden_dim[i + 1]),
                act_fn(),
                nn.Dropout(dropout),
            )
            for i in range(self.num_layers - 1)
        ])

        # Decoder: hidden_dim[-1] → ... → hidden_dim[0] (with skip connections)
        self.decode_time_embedding = nn.ModuleList([
            TimestepEmbedding(time_embed_dim, hidden_dim[i])
            for i in range(self.num_layers - 1, 0, -1)
        ])
        self.decode_cond_embedding = nn.ModuleList([
            nn.Linear(cond_dim, hidden_dim[i])
            for i in range(self.num_layers - 1, 0, -1)
        ])
        self.decode_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim[i], hidden_dim[i - 1]),
                nn.LayerNorm(hidden_dim[i - 1]),
                act_fn(),
                nn.Dropout(dropout),
            )
            for i in range(self.num_layers - 1, 0, -1)
        ])

        # Output projection: hidden_dim[0] → embed_dim
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)

    def forward(self, x, t, c=None):
        # x: (batch, embed_dim)  t: (batch,)  c: (batch, cond_dim)
        t = self.time_proj(t)  # sinusoidal features, shared by all layers

        x = self.input_layer(x)

        # Encoder pass — save activations for skip connections
        hidden_activations = []
        for i in range(self.num_layers - 1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t)
            c_emb = self.encode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)

        # Decoder pass — add skip connections from encoder
        for i in range(self.num_layers - 1):
            t_emb = self.decode_time_embedding[i](t)
            c_emb = self.decode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x += hidden_activations[-1 - i]

        return self.output_layer(x)


class EmbeddingDataset(Dataset):
    """Paired (condition, target) embedding dataset for prior training."""

    def __init__(self, c_embeddings, h_embeddings):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings

    def __len__(self):
        return len(self.c_embeddings)

    def __getitem__(self, idx):
        return {
            "c_embedding": self.c_embeddings[idx],
            "h_embedding": self.h_embeddings[idx],
        }


class Pipe:
    """Inference/training wrapper around a DiffusionPrior model."""

    def __init__(self, diffusion_prior=None, scheduler=None, device='cuda'):
        self.diffusion_prior = diffusion_prior.to(device)

        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler()
        else:
            self.scheduler = scheduler

        self.device = device

    def train(self, dataloader, num_epochs=10, learning_rate=1e-4):
        self.diffusion_prior.train()
        device = self.device
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=len(dataloader) * num_epochs,
        )
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for epoch in range(num_epochs):
            loss_sum = 0
            for batch in dataloader:
                c_embeds = batch['c_embedding'].to(device) if 'c_embedding' in batch else None
                h_embeds = batch['h_embedding'].to(device)
                N = h_embeds.shape[0]

                # 10% classifier-free guidance dropout
                if torch.rand(1) < 0.1:
                    c_embeds = None

                noise = torch.randn_like(h_embeds)
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)
                perturbed_h_embeds = self.scheduler.add_noise(h_embeds, noise, timesteps)
                noise_pre = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
                loss = criterion(noise_pre, noise).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                lr_scheduler.step()
                optimizer.step()
                loss_sum += loss.item()

            print(f'epoch: {epoch}, loss: {loss_sum / len(dataloader)}')

    def generate(self, c_embeds=None, num_inference_steps=50, timesteps=None,
                 guidance_scale=5.0, generator=None):
        """Denoise from Gaussian noise to predicted image embedding.

        Uses classifier-free guidance when guidance_scale > 0 and c_embeds is provided.
        """
        self.diffusion_prior.eval()
        N = c_embeds.shape[0] if c_embeds is not None else 1

        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, timesteps)

        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)

        h_t = torch.randn(N, self.diffusion_prior.embed_dim,
                          generator=generator, device=self.device)

        for _, t in tqdm(enumerate(timesteps)):
            t_batch = torch.ones(N, dtype=torch.float, device=self.device) * t
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = self.diffusion_prior(h_t, t_batch)
            else:
                noise_pred_cond = self.diffusion_prior(h_t, t_batch, c_embeds)
                noise_pred_uncond = self.diffusion_prior(h_t, t_batch)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            h_t = self.scheduler.step(noise_pred, t.long().item(), h_t,
                                      generator=generator).prev_sample

        return h_t


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    prior = DiffusionPriorUNet(cond_dim=1024)
    x = torch.randn(2, 1024)
    t = torch.randint(0, 1000, (2,))
    c = torch.randn(2, 1024)
    y = prior(x, t, c)
    print(y.shape)
