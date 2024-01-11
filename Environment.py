import torch
import numpy as np
import platform
if platform.system().lower() == 'windows':
    from simulation_win import evaluate
elif platform.system().lower() == 'linux':
    from simulation import evaluate
from main import derotate, Normalize
from scipy.signal import savgol_filter
from DiffusionAirfoil import Diff
from DiffusionAirfoil1D import Diff as Diff1D
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if platform.system().lower() == 'linux':
    path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
elif platform.system().lower() == 'windows':
    path = 'H:/深度学习/checkpoint/'
base_airfoil = np.loadtxt('BETTER/20150114-50 +2 d.dat', skiprows=1)
base_airfoil = interpolate(base_airfoil, 256, 3)

class OptimEnv():
    def __init__(self, base_airfoil = base_airfoil, cl = 0.65, thickness = 0.058, Re1 = 58000, Re2 = 400000, alpha=0.01, mode = '2d'):
        self.cl = cl
        self.R = 1
        self.base_airfoil = torch.from_numpy(base_airfoil).to(device)
        self.alpha = alpha
        self.thickness = thickness
        self.Re1 = Re1
        self.Re2 = Re2
        self.mode = mode
    
    def reset(self):
        # self.noise = torch.randn([1, 1, 512]).to(device)
        # self.airfoil = Diff1D.sample(batch_size=1, channels=1, noise = self.noise)
        # self.airfoil = self.airfoil * self.alpha + (1-self.alpha) * self.base_airfoil.reshape(1, 1, 512)
        self.airfoil = self.base_airfoil.reshape(1, 1, 512)
        self.state = self.airfoil.reshape(512)
        return self.state.cpu().numpy()
    
    def step(self, action):
        if self.mode == '1d':
            self.noise = torch.from_numpy(action).reshape([1,1,512]).to(device)
            self.airfoil = Diff1D.sample(batch_size=1, channels=1, noise = self.noise) * self.alpha \
                + (1-self.alpha) * self.airfoil
        elif self.mode == '2d':
            self.noise = torch.from_numpy(action).reshape([1,1,256,2]).to(device)
            self.airfoil = Diff.sample(batch_size=1, channels=1, noise = self.noise).reshape([1,1,512]) * self.alpha \
                + (1-self.alpha) * self.airfoil
        airfoil = self.airfoil.reshape(1, 256, 2)
        airfoil = airfoil.cpu().numpy()
        airfoil = airfoil[0]
        airfoil = derotate(airfoil)
        airfoil = Normalize(airfoil)
        xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
        airfoil[:,0] = xhat
        airfoil[:,1] = yhat
        thickness = cal_thickness(airfoil)
        perf, CD, af, R = evaluate(airfoil, self.cl, Re1 = self.Re1, Re2 = self.Re2, lamda=5, check_thickness=False)
        # print(f'perf: {perf}, R: {R}')
        if np.isnan(R):
            reward = -2
        else:
            reward = (0.042 - R) * 10 + thickness - self.thickness
        # print(reward)
        if R < self.R:
            self.R = R
            np.savetxt('results/airfoilPPO.dat', airfoil, header='airfoilPPO', comments="")
        self.state = self.airfoil.reshape(512)
        
        if R < 0.039 and perf > 38:
            done = True
            reward += 100
        else:
            done = False
        info = None
        return self.state.detach().cpu().numpy(), reward, done, info