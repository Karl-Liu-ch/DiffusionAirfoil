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
import logging
import argparse
parser = argparse.ArgumentParser(description="DiffusionAirfoil")
parser.add_argument('--method', type=str, default='2d')
opt = parser.parse_args()
logging.basicConfig(filename='results/perf.log', encoding='utf-8', level=logging.DEBUG)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCHSIZE = 1024

if platform.system().lower() == 'linux':
    path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
elif platform.system().lower() == 'windows':
    path = 'H:/深度学习/checkpoint/'
    
def optimization(model, cl, best_perf = 0):
    best_airfoil = None
    n=0
    while best_perf < 50:
        samples = model.sample(batch_size=BATCHSIZE, channels=1)
        airfoils = np.squeeze(samples.cpu().numpy(), axis=1)
        for i in range(BATCHSIZE):
            airfoil = airfoils[i]
            airfoil = derotate(airfoil)
            airfoil = Normalize(airfoil)
            xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
            airfoil[:,0] = xhat
            airfoil[:,1] = yhat
            perf = evaluate(airfoil, cl)
            if perf == np.nan:
                pass
            elif perf > best_perf:
                best_perf = perf
                best_airfoil = airfoil
                np.savetxt('results/airfoil.dat', best_airfoil)
                logging.info(f'perf: {perf}, thickness: {yhat.max()-yhat.min()}')
            if perf > 35:
                nn = str(n).zfill(3)
                np.savetxt(f'samples/airfoil{nn}.dat', airfoil)
                logging.info(f'perf: {perf}, n: {nn}')
                n += 1

def optimization1D(model, cl, best_perf = 0):
    best_airfoil = None
    n=0
    while best_perf < 50:
        samples = model.sample(batch_size=BATCHSIZE, channels=1)
        samples = samples.reshape(BATCHSIZE, 256, 2)
        airfoils = samples.cpu().numpy()
        for i in range(BATCHSIZE):
            airfoil = airfoils[i]
            airfoil = derotate(airfoil)
            airfoil = Normalize(airfoil)
            xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
            airfoil[:,0] = xhat
            airfoil[:,1] = yhat
            perf = evaluate(airfoil, cl)
            if perf == np.nan:
                pass
            elif perf > best_perf:
                best_perf = perf
                best_airfoil = airfoil
                np.savetxt('results/airfoil1D.dat', best_airfoil)
                logging.info(f'perf: {perf}, thickness: {yhat.max()-yhat.min()}')
            if perf > 35:
                nn = str(n).zfill(3)
                np.savetxt(f'samples/airfoil1D{nn}.dat', airfoil)
                logging.info(f'perf: {perf}, n: {nn}')
                n += 1
                
if __name__ == '__main__':
    if opt.method == '2d':
        optimization(Diff, cl=0.67, best_perf=34.78824390025072)
    elif opt.method == '1d':
        optimization1D(Diff1D, cl=0.67, best_perf=34.78824390025072)