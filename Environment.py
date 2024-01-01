import torch
import numpy as np
import platform
if platform.system().lower() == 'windows':
    from simulation_win import evaluate
elif platform.system().lower() == 'linux':
    from simulation import evaluate
from main import derotate, Normalize
from scipy.signal import savgol_filter
from DiffusionAirfoil1D import sample, model, load_checkpoint, optimizer, epoch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if platform.system().lower() == 'linux':
    path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
elif platform.system().lower() == 'windows':
    path = 'H:/深度学习/checkpoint/'
model, optimizer, epoch = load_checkpoint(path, model, optimizer, epoch)

class OptimEnv():
    def __init__(self):
        self.cl = 0.65
        self.best_perf = 0
    
    def reset(self):
        self.noise = torch.randn([1, 1, 512]).to(device)
        self.airfoil = sample(model, batch_size=1, channels=1, noise = self.noise)
        self.state = torch.concat([self.noise, self.airfoil], dim=-1).squeeze(dim=1)
        return self.state.cpu().numpy()
    
    def step(self, action):
        self.noise += torch.from_numpy(action).reshape([1,1,512]).to(device)
        self.airfoil = sample(model, batch_size=1, channels=1, noise = self.noise)
        airfoil = self.airfoil.reshape(1, 256, 2)
        airfoil = airfoil.cpu().numpy()
        airfoil = airfoil[0]
        airfoil = derotate(airfoil)
        airfoil = Normalize(airfoil)
        xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
        airfoil[:,0] = xhat
        airfoil[:,1] = yhat
        perf = evaluate(airfoil, self.cl)
        print(perf)
        if perf == np.nan:
            reward = 0
        else:
            reward = 0.01 * perf
        if perf > self.best_perf:
            self.best_perf = perf
            np.savetxt('results/airfoilPPO.dat', airfoil)
        self.state = torch.concat([self.noise, self.airfoil], dim=-1).squeeze(dim=1)
        
        if perf > 50:
            done = True
            reward += 100
        else:
            done = False
        info = None
        return self.state.cpu().numpy(), reward, done, info