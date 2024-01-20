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
from DiffusionAirfoil1D_transform import Diff as Diff1D_transform
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if platform.system().lower() == 'linux':
    path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
elif platform.system().lower() == 'windows':
    path = 'H:/深度学习/checkpoint/'
base_airfoil = np.loadtxt('BETTER/20150114-50 +2 d.dat', skiprows=1)
base_airfoil = interpolate(base_airfoil, 256, 3)

class OptimEnv():
    def __init__(self, base_airfoil = base_airfoil, cl = 0.65, thickness = 0.065, Re1 = 58000, Re2 = 400000, alpha=0.2, mode = '2d'):
        self.cl = cl
        self.base_airfoil = torch.from_numpy(base_airfoil).to(device)
        self.alpha = alpha
        self.thickness = thickness
        self.Re1 = Re1
        self.Re2 = Re2
        self.mode = mode
        self.thickness = thickness
    
    def reset(self):
        successful = False
        while not successful:
            try:
                self.airfoil = Diff1D_transform.sample(batch_size=1, channels=1).reshape(256, 2).cpu().numpy()
                self.airfoil[:,1] = self.airfoil[:,1] * self.thickness / cal_thickness(self.airfoil)
                self.airfoil = self.airfoil.reshape(1, 1, 512)
                self.airfoil = torch.from_numpy(self.airfoil).to(device)

                # self.airfoil = self.base_airfoil.reshape(1, 1, 512)
                self.state = self.airfoil.reshape(512)

                airfoil = self.airfoil.reshape(1, 256, 2)
                airfoil = airfoil.cpu().numpy()
                airfoil = airfoil[0]
                airfoil = derotate(airfoil)
                airfoil = Normalize(airfoil)
                perf, CD, af, R = evaluate(airfoil, self.cl, Re1 = self.Re1, Re2 = self.Re2, lamda=5, check_thickness=False)
                if R is not np.nan:
                    successful = True
            except:
                pass
        self.R_prev = R
        self.Rbl = R
        self.R = R
        return self.state.cpu().numpy()
    
    def step(self, action):
        if self.mode == '1d':
            self.noise = torch.from_numpy(action).reshape([1,1,512]).to(device)
            af = Diff1D.sample(batch_size=1, channels=1, noise = self.noise)
            af = af.reshape(256, 2).cpu().numpy()
            af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
            af = torch.from_numpy(af).to(device)
            af = af.reshape([1,1,512]).detach()
        elif self.mode == '2d':
            self.noise = torch.from_numpy(action).reshape([1,1,256,2]).to(device)
            af = Diff.sample(batch_size=1, channels=1, noise = self.noise)
            af = af.reshape(256, 2).cpu().numpy()
            af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
            af = torch.from_numpy(af).to(device)
            af = af.reshape([1,1,512]).detach()
        elif self.mode == '1d_t':
            self.noise = torch.from_numpy(action).reshape([1,1,512]).to(device)
            af = Diff1D_transform.sample(batch_size=1, channels=1, noise = self.noise)
            af = af.reshape(256, 2).cpu().numpy()
            af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
            af = torch.from_numpy(af).to(device)
            af = af.reshape([1,1,512]).detach()
        
        self.airfoil = af  * self.alpha + (1-self.alpha) * self.airfoil
        airfoil = self.airfoil.reshape(1, 256, 2)
        airfoil = airfoil.cpu().numpy()
        airfoil = airfoil[0]
        airfoil = derotate(airfoil)
        airfoil = Normalize(airfoil)
        perf, CD, af, R = evaluate(airfoil, self.cl, Re1 = self.Re1, Re2 = self.Re2, lamda=5, check_thickness=False)
        if np.isnan(R):
            reward = -1
            reward_final = -1
        else:
            reward_final = 1.0 / R
            reward = (self.R_prev - R) * 10
            self.R_prev = R
        # print(reward)
        if R < self.R:
            self.R = R
            np.savetxt('results/airfoilPPO.dat', airfoil, header='airfoilPPO', comments="")
        self.state = self.airfoil.reshape(512)
        
        if R < 0.04 and perf > 40:
            done = True
            reward += 100
        else:
            done = False
        info = None
        return self.state.detach().cpu().numpy(), reward, done, reward_final