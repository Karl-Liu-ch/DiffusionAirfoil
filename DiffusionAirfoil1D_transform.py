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

from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from Unet1D import Unet1D
import platform
from utils import * 
from model import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE = 2**8

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

timesteps = 1000
# define beta schedule
betas = cosine_beta_schedule(timesteps=timesteps)

dataroot = 'data/airfoil_interp.npy'
data = np.load(dataroot)
dataset = GetDataset(data)
train_loader = DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=True)

model = Unet1D(
    dim=32,
    init_dim=32,
    out_dim=1,
    channels=1,
    self_condition=False,
    dim_mults=(1, 2, 4, 8), 
    learned_variance = True,
    learned_sinusoidal_cond = True,
    random_fourier_features = True,
)
model.to(device)

class DiffusionAirfoil1DTransform(DiffusionAirfoil1D):
    def __init__(self, betas, timesteps=1000):
        super().__init__(betas, timesteps)
        self.model = model
        self.optim = Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.epochs, eta_min=1e-6)
        self.path = '/work3/s212645/DiffusionAirfoil1DTransform/checkpoint/'
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

    @torch.no_grad()
    def sample(self, batch_size=16, channels=1, noise=None):
        result = self.p_sample_loop(shape=(batch_size, channels, 512), noise = noise)
        result = result.reshape(batch_size, 1, 256, 2)
        result = reverse_transform(result)
        result = result.reshape(batch_size, 1, 512)
        return result

dataroot = 'data/airfoil_interp.npy'
data = np.load(dataroot)
dataset = GetDataset(data)
train_loader = DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=True)
Diff = DiffusionAirfoil1DTransform(betas)
Diff.load_checkpoint()


if __name__ == '__main__':
    B = 2 ** 10
    if platform.system().lower() == 'linux':
        path = '/work3/s212645/DiffusionAirfoil1DTransform/checkpoint/'
    elif platform.system().lower() == 'windows':
        path = 'H:/深度学习/checkpoint/'
    
    # Diff.train(train_loader)
    
    if platform.system().lower() == 'linux':
        airfoilpath = '/work3/s212645/DiffusionAirfoil1DTransform/Airfoils1D/'
    elif platform.system().lower() == 'windows':
        airfoilpath = 'H:/深度学习/Airfoils1D/'
        
    # samples = Diff.sample(batch_size=B, channels=1)
    # samples = samples.reshape(B, 256, 2)
    # fig, axs = plt.subplots(1, 1)
    # airfoil = samples[0,:,:].cpu().numpy()
    # airfoil = Normalize(airfoil)
    # axs.plot(airfoil[:,0], airfoil[:,1])
    # axs.set_aspect('equal', 'box')
    # fig.tight_layout()
    # plt.show()
    # plt.savefig('sample.svg')
    # plt.close()

    try:
        os.makedirs('/work3/s212645/DiffusionAirfoil1DTransform/Airfoils1D/')
    except:
        pass
        
    for i in range(1000):
        num = str(i).zfill(3)
        samples = Diff.sample(batch_size=B, channels=1)
        samples = samples.reshape(B, 256, 2)
        airfoils = samples.cpu().numpy()
        np.save(airfoilpath+num+'.npy', airfoils)
        print(num + '.npy saved. ')