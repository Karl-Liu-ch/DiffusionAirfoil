import sys
sys.path.append('./')
import numpy as np
import torch
from DiffusionAirfoil import Diff
from DiffusionAirfoil1D import Diff as Diff1D
from utils import *
from scipy.signal import savgol_filter
import platform
if platform.system().lower() == 'windows':
    from simulation_win import evaluate
elif platform.system().lower() == 'linux':
    from simulation import evaluate
import os
import torch
import argparse
parser = argparse.ArgumentParser(description="DiffusionAirfoil")
parser.add_argument('--method', type=str, default='2d')
opt = parser.parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE = 1024

if platform.system().lower() == 'linux':
    path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
elif platform.system().lower() == 'windows':
    path = 'H:/深度学习/checkpoint/'
    

LAMBDA = 5
perf_BL, R_BL = cal_baseline(lamda=LAMBDA)

def optimization(model, cl, R_BL, lamda):
    m = 0
    name = 'af2d'
    while True:
        samples = model.sample(batch_size=BATCHSIZE, channels=1)
        airfoils = np.squeeze(samples.cpu().numpy(), axis=1)
        for i in range(BATCHSIZE):
            airfoil = airfoils[i,:,:]
            airfoil = derotate(airfoil)
            airfoil = Normalize(airfoil)
            xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
            airfoil[:,0] = xhat
            airfoil[:,1] = yhat
            if cal_thickness(airfoil) > 0.08 or cal_thickness(airfoil) < 0.05:
                airfoil[:,1] = airfoil[:,1] * 0.06 / cal_thickness(airfoil)
            perf, CD, af, R = evaluate(airfoil, cl, lamda=lamda)
            if perf == np.nan:
                pass
            elif R < R_BL:
                mm = str(m).zfill(3)
                np.savetxt(f'samples/{name}_{mm}.dat', airfoil, header=f'{name}_{mm}', comments="")
                np.savetxt(f'samples/{name}_{mm}F.dat', af, header=f'{name}_{mm}F', comments="")
                f = open(f'results/{name}_simperf.log', 'a')
                f.write(f'perf: {perf}, R: {R}, m: {mm}, path: samples/{name}_{mm}.dat\n')
                f.close()
                m += 1
                del f

def optimization1D(model, cl, R_BL, lamda):
    m = 0
    name = 'af1d'
    while True:
        samples = model.sample(batch_size=BATCHSIZE, channels=1)
        samples = samples.reshape(BATCHSIZE, 256, 2)
        airfoils = samples.cpu().numpy()
        for i in range(BATCHSIZE):
            airfoil = airfoils[i,:,:]
            airfoil = derotate(airfoil)
            airfoil = Normalize(airfoil)
            xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
            airfoil[:,0] = xhat
            airfoil[:,1] = yhat
            if cal_thickness(airfoil) > 0.08 or cal_thickness(airfoil) < 0.05:
                airfoil[:,1] = airfoil[:,1] * 0.06 / cal_thickness(airfoil)
            perf, CD, af, R = evaluate(airfoil, cl, lamda=lamda)
            if perf == np.nan:
                pass
            elif R < R_BL:
                mm = str(m).zfill(3)
                np.savetxt(f'samples/{name}_{mm}.dat', airfoil, header=f'{name}_{mm}', comments="")
                np.savetxt(f'samples/{name}_{mm}F.dat', af, header=f'{name}_{mm}F', comments="")
                f = open(f'results/{name}_simperf.log', 'a')
                f.write(f'perf: {perf}, R: {R}, m: {mm}, path: samples/{name}_{mm}.dat\n')
                f.close()
                m += 1
                del f
                
if __name__ == '__main__':
    if opt.method == '2d':
        optimization(Diff, cl=0.65, R_BL=R_BL, lamda=LAMBDA)
    elif opt.method == '1d':
        optimization1D(Diff1D, cl=0.65, R_BL=R_BL, lamda=LAMBDA)