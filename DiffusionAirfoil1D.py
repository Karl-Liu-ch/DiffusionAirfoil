import sys
sys.path.append('./')
import re
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

timesteps = 1000
# define beta schedule
betas = cosine_beta_schedule(timesteps=timesteps)

dataroot = 'data/airfoil_interp.npy'
data = np.load(dataroot)
dataset = GetDataset(data)
train_loader = DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=True)

model = Unet1D(
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
model.to(device)
Diff = DiffusionAirfoil1D(betas)
Diff.load_checkpoint()
Diff.optim = Adam(Diff.model.parameters(), lr=1e-5)
Diff.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Diff.optim, Diff.epochs, eta_min=1e-6)
Diff.epoch = 0

if __name__ == '__main__':
    if platform.system().lower() == 'linux':
        path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
    elif platform.system().lower() == 'windows':
        path = 'H:/深度学习/checkpoint/'
    
    # Diff.train(train_loader)
    
    if platform.system().lower() == 'linux':
        airfoilpath = '/work3/s212645/DiffusionAirfoil/Airfoils1D/'
    elif platform.system().lower() == 'windows':
        airfoilpath = 'H:/深度学习/Airfoils1D/'
        
    num = 0
    fileformat = re.compile('.npy')
    for path, dir, files in os.walk(airfoilpath):
        files.sort()
        for file in files:
            if fileformat.search(file) is not None:
                n = file.split('.')[0]
                n = int(n)
                if n > num:
                    num = n
    print(num)
    start_n = num
    B = 2 ** 10
    while 1:
        start_n += 1
        num = str(start_n).zfill(3)
        samples = Diff.sample(batch_size=B, channels=1)
        samples = samples.reshape(B, 256, 2)
        airfoils = samples.cpu().numpy()
        np.save(airfoilpath+num+'.npy', airfoils)
        print(num + '.npy saved. ')
# fig, axs = plt.subplots(1, 1)
# airfoil = samples[0,:,:].cpu().numpy()
# airfoil = Normalize(airfoil)
# axs.plot(airfoil[:,0], airfoil[:,1])
# axs.set_aspect('equal', 'box')
# fig.tight_layout()
# plt.show()
# plt.savefig('sample.svg')
# plt.close()