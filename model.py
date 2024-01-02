import sys
sys.path.append('./')

# %matplotlib inline
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from torch.optim import Adam
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch import nn, einsum
import torch.nn.functional as F

from Unet1D import Unet1D
from networks import Unet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from Unet1D import Unet1D
import platform
from utils import * 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Diffusion():
    def __init__(self, betas, timesteps = 1000):
        self.channels = 1
        self.epochs = 10000
        self.epoch = 0
        self.optim = Adam(self.model.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.epochs, eta_min=1e-6)
        
        if platform.system().lower() == 'linux':
            self.path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
        elif platform.system().lower() == 'windows':
            self.path = 'H:/深度学习/checkpoint/'
        
        self.betas = betas
        self.timesteps = timesteps
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_noisy_image(self, x_start, t):
        # add noise
        x_noisy = self.q_sample(x_start, t=t)

        # turn back into PIL image
        noisy_image = x_noisy[0].cpu().numpy().transpose(1,2,0)

        return noisy_image

    def p_losses(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            loss = F.l1_loss(noise, predicted_noise) + F.mse_loss(noise, predicted_noise)

        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, shape, noise=None):
        device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        if noise is None:
            img = torch.randn(shape, device=device)
        else:
            img = noise

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, channels=1, noise=None):
        return self.p_sample_loop(shape=(batch_size, channels, 512), noise = noise)
    
    def save_checkpoint(self):
        state = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optim': self.optim.state_dict(),
            }
        name = 'net1D.pth'
        torch.save(state, os.path.join(self.path, name))
        
    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.path, 'net1D.pth'))
        self.model.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.epoch = checkpoint['epoch']
        # self.optim = self.optim
        # self.epoch = 0
        print("pretrained model loaded, iteration: ", self.epoch)
    
    def train(self, train_loader):
        pass

class GetDataset(Dataset):
    def __init__(self, data, arg=True):
        self.data = data
        self.arg = arg
        self.length = data.shape[0]

    def __getitem__(self, idx):
        return np.float32(np.expand_dims(self.data[idx,:,:], 0))

    def __len__(self):
        return self.length
    
class DiffusionAirfoil1D(Diffusion):
    def __init__(self, betas, timesteps=1000):
        self.model = Unet1D(
            dim=16,
            init_dim=16,
            out_dim=1,
            channels=1,
            self_condition=False,
            dim_mults=(1, 2, 4, 8), 
            learned_variance = True,
            learned_sinusoidal_cond = True,
            random_fourier_features = True,
        )
        self.model.to(device)
        super().__init__(betas, timesteps)
        
    def save_checkpoint(self):
        state = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optim': self.optim.state_dict(),
            }
        name = 'net1D.pth'
        torch.save(state, os.path.join(self.path, name))
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.path + 'net1D.pth')
        self.model.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.epoch = checkpoint['epoch']
        # self.optim = self.optim
        # self.epoch = 0
        print("pretrained model loaded, iteration: ", self.epoch)
        
    def train(self, train_loader):
        while self.epoch < self.epochs:
            losses = []
            for step, labels in enumerate(train_loader):
                labels = labels.to(device)
                labels = Variable(labels)
                batch_size = labels.shape[0]
                labels = labels.reshape(batch_size, 1, 512)
                self.optim.zero_grad()
                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
                loss = self.p_losses(labels, t, loss_type="l1+l2")
                losses.append(loss.item())
                loss.backward()
                self.optim.step()
            print("Epoch: ", self.epoch, "Loss:", np.array(losses).mean(), 'lr: ', self.optim.param_groups[0]['lr'])
            self.save_checkpoint()
            self.epoch += 1
            self.scheduler.step()
            
class DiffusionAirfoil(Diffusion):
    def __init__(self, betas, timesteps=1000):
        self.model = Unet(
            dim=16,
            init_dim=16,
            out_dim=1,
            channels=1,
            self_condition=False,
            dim_mults=(4, 8), 
            learned_variance = True,
            learned_sinusoidal_cond = True,
            random_fourier_features = True,
        )
        self.model.to(device)
        
        super().__init__(betas, timesteps)
        
    def save_checkpoint(self):
        state = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optim': self.optim.state_dict(),
            }
        name = 'net.pth'
        torch.save(state, os.path.join(self.path, name))
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.path + 'net.pth')
        self.model.load_state_dict(checkpoint['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.epoch = checkpoint['epoch']
        # self.optim = self.optim
        # self.epoch = 0
        print("pretrained model loaded, iteration: ", self.epoch)
        
    def train(self, train_loader):
        while self.epoch < self.epochs:
            losses = []
            for step, labels in enumerate(train_loader):
                labels = labels.to(device)
                labels = Variable(labels)
                batch_size = labels.shape[0]
                self.optim.zero_grad()
                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
                loss = self.p_losses(labels, t, loss_type="l1+l2")
                losses.append(loss.item())
                loss.backward()
                self.optim.step()
            print("Epoch: ", self.epoch, "Loss:", np.array(losses).mean(), 'lr: ', self.optim.param_groups[0]['lr'])
            self.save_checkpoint()
            self.epoch += 1
            self.scheduler.step()
            
    @torch.no_grad()
    def sample(self, batch_size=16, channels=1, noise=None):
        return self.p_sample_loop(shape=(batch_size, channels, 256, 2), noise = noise)