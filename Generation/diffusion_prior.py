import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset


class DiffusionPrior(nn.Module):

    def __init__(
            self, 
            embed_dim=1024, 
            cond_dim=42,
            hidden_dim=1024,
            layers_per_block=4, 
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
        ):
        super().__init__()
        
        self.embed_dim = embed_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        self.time_embedding = TimestepEmbedding(
            time_embed_dim,
            hidden_dim,
        )

        # 2. conditional embedding 
        self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
        )

        # 3.2 hidden
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout),
                )
                for _ in range(layers_per_block)
            ]
        )

        # 3.3 output
        self.output_layer = nn.Linear(hidden_dim, embed_dim)
        

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t) # (batch_size, time_embed_dim)
        t = self.time_embedding(t) # (batch_size, hidden_dim)

        # 2. conditional embedding 
        c = self.cond_embedding(c) if c is not None else 0 # (batch_size, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x) 

        # 3.2 hidden
        for layer in self.hidden_layers:
            x = x + t + c
            x = layer(x) + x
            
        # 3.3 output
        x = self.output_layer(x)

        return x


class DiffusionPriorUNet(nn.Module):

    def __init__(
            self, 
            embed_dim=1024, 
            cond_dim=42,
            hidden_dim=[1024, 512, 256, 128, 64],
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
        ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # 2. conditional embedding 
        # to 3.2, 3,3

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        # 3.2 hidden encoder
        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers-1)]
        ) # d_0, ..., d_{n-1}
        self.encode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1)]
        )
        self.encode_layers = nn.ModuleList(
            [nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                    nn.LayerNorm(hidden_dim[i+1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) for i in range(self.num_layers-1)]
        )

        # 3.3 hidden decoder
        self.decode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers-1,0,-1)]
        ) # d_{n}, ..., d_1
        self.decode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1,0,-1)]
        )
        self.decode_layers = nn.ModuleList(
            [nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                    nn.LayerNorm(hidden_dim[i-1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) for i in range(self.num_layers-1,0,-1)]
        )

        # 3.4 output
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)
        

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t) # (batch_size, time_embed_dim)

        # 2. conditional embedding 
        # to 3.2, 3.3

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x) 
        # print("x.shape:", x.shape, "t:", t.shape, "c", c.shape)
        # 3.2 hidden encoder
        hidden_activations = []
        for i in range(self.num_layers-1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t) 
            if c is not None:
                c_emb = self.encode_cond_embedding[i](c)
            else:
                # Adjust the zero tensor size to match the expected dimensions for the current layer
                c_emb = torch.zeros(x.size(0), self.hidden_dim[i], device=x.device)

            # c_emb = self.decode_cond_embedding[i](c) if c is not None else torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            # print("x.shape:", x.shape, "t_emb.shape:", t_emb.shape, "c_emb.shape:", c_emb.shape)
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)
        
        # 3.3 hidden decoder
        for i in range(self.num_layers-1):
            t_emb = self.decode_time_embedding[i](t)
            if c is not None:
                c_emb = self.decode_cond_embedding[i](c)
            else:
                # Adjust the zero tensor size to match the expected dimensions for the current layer
                c_emb = torch.zeros_like(x)
            # c_emb = self.decode_cond_embedding[i](c) if c is not None else torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            # print("x.shape:", x.shape, "t_emb.shape:", t_emb.shape, "c_emb.shape:", c_emb.shape)
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x += hidden_activations[-1-i]
            
        # 3.4 output
        x = self.output_layer(x)

        return x
    
class EmbeddingDataset(Dataset):
    def __init__(self, c_embeddings=None, h_embeddings=None, h_embeds_uncond=None, cond_sampling_rate=0.5):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings
        self.h_embeds_uncond = h_embeds_uncond
        self.cond_sampling_rate = cond_sampling_rate

        
        self.N_cond = len(h_embeddings) if h_embeddings is not None else 0
        self.N_uncond = len(h_embeds_uncond) if h_embeds_uncond is not None else 0

        
        self.total_samples = self.N_cond + self.N_uncond

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < self.N_cond:
            
            return {
                "c_embedding": self.c_embeddings[idx] if self.c_embeddings is not None else torch.zeros(self.h_embeddings[idx].shape),
                "h_embedding": self.h_embeddings[idx],
            }
        else:
            
            idx_uncond = idx - self.N_cond
            return {
                "c_embedding": torch.zeros_like(self.h_embeddings[0]) if self.c_embeddings is not None else torch.zeros(self.h_embeds_uncond[idx_uncond].shape),
                "h_embedding": self.h_embeds_uncond[idx_uncond],
            }


class EmbeddingDatasetVICE(Dataset):
    def __init__(self, path_data):
        image_features_dict = torch.load(os.path.join(path_data, 'openclip_emb/image_features.pt'))
        self.embedding_vise = torch.load(os.path.join(path_data, 'variables/embedding_vise.pt'))
        self.image_features = image_features_dict['image_features']
        self.labels = image_features_dict['labels']
        self.label2index = image_features_dict['l2i']

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        idx_c = self.label2index[self.labels[idx]]
        return {
            "c_embedding": self.embedding_vise[idx_c],
            "h_embedding": self.image_features[idx]
        }
    

# Copied from diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler.add_noise
def add_noise_with_sigma(
    self,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timesteps: torch.FloatTensor,
) -> torch.FloatTensor:
    # Make sure sigmas and timesteps have the same device and dtype as original_samples
    sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
        # mps does not support float64
        schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
        timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    else:
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

    step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigma
    return noisy_samples, sigma


# diffusion pipe
class Pipe:
    
    def __init__(self, diffusion_prior=None, scheduler=None, device='cuda'):
        self.diffusion_prior = diffusion_prior.to(device)
        
        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler() 
            # self.scheduler.add_noise_with_sigma = add_noise_with_sigma.__get__(self.scheduler)
        else:
            self.scheduler = scheduler
            
        self.device = device
            
    def train(self, dataloader, dataloader_uncond, num_epochs=10, learning_rate=1e-4):
        self.diffusion_prior.train()
        device = self.device
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for epoch in range(num_epochs):
            loss_sum = 0
            count = 0
            for (batch_cond, batch_uncond) in zip(dataloader, dataloader_uncond):
                count+=1
                
                c_embeds_cond = batch_cond['c_embedding'].to(device)
                h_embeds_cond = batch_cond['h_embedding'].to(device)
                # print("c_embeds_cond", c_embeds_cond.shape)
                # print("h_embeds_cond", h_embeds_cond.shape)
                
                h_embeds_uncond = batch_uncond['h_embedding'].to(device)
                c_embeds_uncond = batch_uncond['c_embedding'].to(device)
                # print("c_embeds_uncond", c_embeds_uncond.shape)
                
                
                c_embeds = torch.cat([c_embeds_cond, c_embeds_uncond], dim=0)
                h_embeds = torch.cat([h_embeds_cond, h_embeds_uncond], dim=0)
                # print("c_embeds", c_embeds.shape)
                # print("h_embeds", h_embeds.shape)
                N = h_embeds.shape[0]

                
                noise = torch.randn_like(h_embeds)
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)
                perturbed_h_embeds = self.scheduler.add_noise(h_embeds, noise, timesteps)

                
                noise_pred = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)

                
                loss = criterion(noise_pred, noise).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                lr_scheduler.step()
                optimizer.step()

                loss_sum += loss.item()
            # print("count", count)
            loss_epoch = loss_sum / (len(dataloader) + len(dataloader_uncond))
            print(f'epoch: {epoch}, loss: {loss_epoch}')


    def generate(
            self, 
            c_embeds=None, 
            num_inference_steps=50, 
            timesteps=None,
            guidance_scale=5.0,
            generator=None
        ):
        # c_embeds (batch_size, cond_dim)
        self.diffusion_prior.eval()
        N = c_embeds.shape[0] if c_embeds is not None else 1

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare c_embeds
        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)

        # 3. Prepare noise
        h_t = torch.randn(N, self.diffusion_prior.embed_dim, generator=generator, device=self.device)

        # 4. denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t = torch.ones(h_t.shape[0], dtype=torch.float, device=self.device) * t
            # 4.1 noise prediction
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = self.diffusion_prior(h_t, t)
            else:
                noise_pred_cond = self.diffusion_prior(h_t, t, c_embeds)
                noise_pred_uncond = self.diffusion_prior(h_t, t)
                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 4.2 compute the previous noisy sample h_t -> h_{t-1}
            h_t = self.scheduler.step(noise_pred, t.long().item(), h_t, generator=generator).prev_sample
        
        return h_t

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 1. test prior
    prior = DiffusionPriorUNet(cond_dim=1024)
    x = torch.randn(2, 1024)
    t = torch.randint(0, 1000, (2,))
    c = torch.randn(2, 1024)
    y = prior(x, t, c)
    print(y.shape)



