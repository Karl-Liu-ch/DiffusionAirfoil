import sys
sys.path.append('./')
import numpy as np
from networks import Unet
from DiffusionAirfoil import sample, load_checkpoint, epoch, optimizer, Normalize
from DiffusionAirfoil1D import sample as sample1D
from DiffusionAirfoil1D import model as model1D
from scipy.signal import savgol_filter
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
model = Unet(
    dim=16,
    init_dim=16,
    out_dim=1,
    channels=1,
    self_condition=False,
    dim_mults=(2, 4)
)
model.to(device)
path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
model, optimizer, epoch = load_checkpoint(path, model, optimizer, epoch)

def optimization(model, cl, best_perf = 0):
    best_airfoil = None
    while best_perf < 50:
        samples = sample(model, batch_size=BATCHSIZE, channels=1)
        airfoils = np.squeeze(samples.cpu().numpy(), axis=1)
        for i in range(BATCHSIZE):
            airfoil = airfoils[i]
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

def optimization1D(model, cl, best_perf = 0):
    best_airfoil = None
    while best_perf < 50:
        samples = sample1D(model, batch_size=BATCHSIZE, channels=1)
        samples = samples.reshape(BATCHSIZE, 256, 2)
        airfoils = samples.cpu().numpy()
        for i in range(BATCHSIZE):
            airfoil = airfoils[i]
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
                
if __name__ == '__main__':
    if opt.method == '2d':
        optimization(model, cl=0.65, best_perf=33)
    elif opt.method == '1d':
        optimization1D(model1D, cl=0.65, best_perf=33)