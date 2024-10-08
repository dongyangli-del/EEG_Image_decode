import numpy as np
import math
import torch
import os
import sys
import time
from torch import inf
import wandb

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
        


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
        
def train_one_epoch(model, data_loader, optimizer, device, epoch, 
                        loss_scaler, log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                        img_feature_extractor=None, preprocess=None):
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_cor = []
    accum_iter = config.accum_iter
    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        # print(data_iter_step)
        # print(len(data_loader))
        
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = data_dcit['eeg']
        
        img_features = None
        valid_idx = None
        if img_feature_extractor is not None:
            images = data_dcit['image']
            valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(preprocess(images[valid_idx]).to(device))['layer2']
        samples = samples.to(device)
        # img_features = img_features.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = model(samples, img_features, valid_idx=valid_idx, mask_ratio=config.mask_ratio)
        # loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        # optimizer.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:
        # cal the cor
        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        # pred = pred.transpose(1,2) #model_without_ddp.unpatchify(pred)
        pred = model_without_ddp.unpatchify(pred)
            
        cor = torch.mean(torch.tensor([torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)],axis=0))[0,1] for p, s in zip(pred, samples)])).item()
        optimizer.zero_grad()

        total_loss.append(loss_value)
        total_cor.append(cor)
        if device == torch.device('cuda:0'):
            lr = optimizer.param_groups[0]["lr"]
            print('train_loss_step:', np.mean(total_loss), 'lr:', lr, 'cor', np.mean(total_cor))

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        log_writer.log('cor', np.mean(total_cor), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')

    return np.mean(total_cor)

def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=float)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
        # height (== width) for the new position embedding
        new_size = int(num_patches)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %d to %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size))
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs 
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


    

def load_model(config, model, checkpoint_path ):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Model loaded with {checkpoint_path}')
    

def patchify(imgs, patch_size):
    """
    imgs: (N, 1, num_voxels)
    x: (N, L, patch_size)
    """
    p = patch_size
    assert imgs.ndim == 3 and imgs.shape[2] % p == 0

    h = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], h, p))
    return x

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size)
    imgs: (N, 1, num_voxels)
    """
    p = patch_size
    h = x.shape[1]
    
    imgs = x.reshape(shape=(x.shape[0], 1, h * p))
    return imgs

class wandb_logger:
    def __init__(self, config):
        try:
            wandb.init(
                # Set the project where this run will be logged
                project=config['project'],
                name=config['name'],
                config=config,
                entity=config['entity'],            
                )
        except:
                wandb.init(
                # Set the project where this run will be logged
                project=config.project,
                name=config.name,
                config=config,
                entity=config.entity,            
                )

        self.config = config
        self.step = None
    
    def log(self, data, step=None):
        if step is None:
            wandb.log(data)
        else:
            wandb.log(data, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, figs):
        if self.step is None:
            wandb.log(figs)
        else:
            wandb.log(figs, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

    def load(self, net):
        path = os.path.join(self.config['path_data'], self.config['path_ckpt'], self.config['file_ckpt'])
        net.load_state_dict(torch.load(path))
        print(f'load {path}')

    def save(self, net, file_name=None):
        path_ckpt = os.path.join(self.config['path_data'], self.config['path_ckpt'])
        if not os.path.exists(path_ckpt):
            os.makedirs(path_ckpt)
            print(f'{path_ckpt} created!')

        path = os.path.join(path_ckpt, file_name)
        torch.save(net.state_dict(), path)

    def watch(self, model, log):
        wandb.watch(model, log)