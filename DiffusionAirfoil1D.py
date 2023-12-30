import sys
sys.path.append('./')
import math
from inspect import isfunction
from functools import partial

# %matplotlib inline
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch import nn, einsum
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from Unet1D import Unet1D

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE = 64

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = 1000
# define beta schedule
betas = cosine_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = x_noisy[0].cpu().numpy().transpose(1,2,0)

  return noisy_image

# take time step

dataroot = 'data/airfoil_interp.npy'
data = np.load(dataroot)

t = torch.tensor([1]).to(device)
noisy = torch.from_numpy(data[0,:,:]).to(device)
noisy = noisy.unsqueeze(dim=-1).unsqueeze(dim=0).permute(0,3,1,2)
noisy = get_noisy_image(noisy, t)
noisy = noisy.reshape(256, 2)

fig, axs = plt.subplots(1, 1)
axs.plot(noisy[:,0], noisy[:,1])
axs.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()
plt.close()

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        loss = F.smooth_l1_loss(noise, predicted_noise) + F.mse_loss(noise, predicted_noise)

    return loss

class GetDataset(Dataset):
    def __init__(self, data, arg=True):
        self.data = data
        self.arg = arg
        self.length = data.shape[0]

    def __getitem__(self, idx):
        return np.float32(np.expand_dims(self.data[idx,:,:], 0))

    def __len__(self):
        return self.length

dataset = GetDataset(data)
train_loader = DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=True)
batch = next(iter(train_loader))
print(batch.shape)

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
    return img

@torch.no_grad()
def sample(model, batch_size=16, channels=1):
    return p_sample_loop(model, shape=(batch_size, channels, 256, 2))

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def Normalize(airfoil):
    r = np.maximum(airfoil[0,0], airfoil[-1,0])
    r = float(1.0/r)
    return airfoil * r

from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"


model = Unet1D(
    dim=64,
    init_dim=64,
    out_dim=1,
    channels=1,
    self_condition=False,
    dim_mults=(1, 2, 4, 8)
)
model.to(device)

def save_checkpoint(epoch, model, optim, path):
    state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optim.state_dict(),
        }
    name = 'net1D.pth'
    torch.save(state, os.path.join(path, name))
    
def load_checkpoint(path, model, optim, epoch):
    checkpoint = torch.load(os.path.join(path, 'net1D.pth'))
    model.load_state_dict(checkpoint['model'])
    # optim.load_state_dict(checkpoint['optim'])
    # epoch = checkpoint['epoch']
    optim = optim
    epoch = 0
    print("pretrained model loaded, iteration: ", epoch)
    return model, optim, epoch

channels = 1
epochs = 10000
epoch = 0
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
# try:
#     model, optimizer, epoch = load_checkpoint(path, model, optimizer, epoch)
# except:
#     pass 
# model, optimizer, epoch = load_checkpoint(path, model, optimizer, epoch)

while epoch < epochs:
    losses = []
    for step, labels in enumerate(train_loader):
        labels = labels.to(device)
        labels = Variable(labels)
        batch_size = labels.shape[0]
        labels = labels.reshape(batch_size, 1, 512)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, labels, t, loss_type="l1+l2")
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Epoch: ", epoch, "Loss:", np.array(losses).mean(), 'lr: ', optimizer.param_groups[0]['lr'])
    save_checkpoint(epoch, model, optimizer, path)
    epoch += 1
    scheduler.step()

# sample 64 images
samples = sample(model, batch_size=BATCHSIZE, channels=1)
samples = samples.reshape(BATCHSIZE, 256, 2)
np.save('sample.npy', samples.cpu().numpy())
fig, axs = plt.subplots(1, 1)
airfoil = samples[0,0,:,:].cpu().numpy()
airfoil = Normalize(airfoil)
axs.plot(airfoil[:,0], airfoil[:,1])
axs.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()
plt.savefig('sample.svg')
plt.close()