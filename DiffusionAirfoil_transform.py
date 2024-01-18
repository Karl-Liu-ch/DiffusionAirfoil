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
from networks import Unet
import platform

from utils import * 
from model import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def transform(tensor):
    tensor = tensor.clone().detach()
    tensor[:, :, :, 0] = tensor[:, :, :, 0] * 2.0 - 1.0
    tensor[:, :, :, 1] = tensor[:, :, :, 1] * 10.0
    return tensor

def reverse_transform(tensor):
    tensor = tensor.clone().detach()
    tensor[:, :, :, 0] = (tensor[:, :, :, 0] + 1.0) / 2.0
    tensor[:, :, :, 1] = tensor[:, :, :, 1] / 10.0
    return tensor

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE = 2**8

timesteps = 1000
# define beta schedule
betas = cosine_beta_schedule(timesteps=timesteps)
model = Unet(
    dim=64,
    init_dim=64,
    out_dim=1,
    channels=1,
    self_condition=False,
    dim_mults=(4, 4), 
    learned_variance = True,
    learned_sinusoidal_cond = True,
    random_fourier_features = True,
)
model.to(device)

class DiffusionAirfoilTransform(DiffusionAirfoil):
    def __init__(self, betas, timesteps=1000):
        super().__init__(betas, timesteps)
        self.model = model
        self.optim = Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.epochs, eta_min=1e-6)
        self.path = '/work3/s212645/DiffusionAirfoilTransform/checkpoint/'
        try: 
            os.makedirs(self.path)
        except:
            pass

    def train(self, train_loader):
        while self.epoch < self.epochs:
            losses = []
            for step, labels in enumerate(train_loader):
                labels = transform(labels)
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
        result = self.p_sample_loop(shape=(batch_size, channels, 256, 2), noise = noise)
        result = reverse_transform(result)
        return result

dataroot = 'data/airfoil_interp.npy'
data = np.load(dataroot)
dataset = GetDataset(data)
train_loader = DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=True)
Diff = DiffusionAirfoilTransform(betas)
Diff.load_checkpoint()
Diff.optim = Adam(Diff.model.parameters(), lr=5e-4)
Diff.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Diff.optim, Diff.epochs, eta_min=1e-6)
Diff.epoch = 0

# sample 64 images
if __name__ == '__main__':
    B = 2 ** 8
    try:
        os.mkdir('/work3/s212645/DiffusionAirfoilTransform/')
    except:
        pass

    if platform.system().lower() == 'linux':
        path = '/work3/s212645/DiffusionAirfoilTransform/checkpoint/'
    elif platform.system().lower() == 'windows':
        path = 'H:/深度学习/checkpoint/'
    
    Diff.train(train_loader)
    
    if platform.system().lower() == 'linux':
        airfoilpath = '/work3/s212645/DiffusionAirfoilTransform/Airfoils/'
    elif platform.system().lower() == 'windows':
        airfoilpath = 'H:/深度学习/Airfoils/'
    
    samples = Diff.sample(batch_size=B, channels=1)
    fig, axs = plt.subplots(1, 1)
    airfoil = samples[0,0,:,:].cpu().numpy()
    airfoil = Normalize(airfoil)
    axs.plot(airfoil[:,0], airfoil[:,1])
    axs.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()
    plt.savefig('sample.png')
    plt.close()

    try:
        os.makedirs('/work3/s212645/DiffusionAirfoilTransform/Airfoils/')
    except:
        pass
    
    for i in range(1000):
        num = str(i).zfill(3)
        samples = Diff.sample(batch_size=B, channels=1)
        samples = samples.reshape(B, 256, 2)
        airfoils = samples.cpu().numpy()
        np.save(airfoilpath+num+'.npy', airfoils)
        print(num + '.npy saved. ')
