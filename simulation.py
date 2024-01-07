import numpy as np
import sys
import os
from xfoil import XFoil
from xfoil.model import Airfoil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import argparse
parser = argparse.ArgumentParser(description="DiffusionAirfoil")
parser.add_argument('--method', type=str, default='1d')
opt = parser.parse_args()
from utils import *

def evaluate(airfoil, cl = 0.65, Re1 = 5.8e4, Re2 = 4e5, lamda = 3, return_CL_CD=False, check_thickness = True):
        
    if detect_intersect(airfoil):
        # print('Unsuccessful: Self-intersecting!')
        perf = np.nan
        R = np.nan
        CD = np.nan
        
    elif (cal_thickness(airfoil) < 0.06 or cal_thickness(airfoil) > 0.09) and check_thickness:
        # print('Unsuccessful: Too thin!')
        perf = np.nan
        R = np.nan
        CD = np.nan
    
    elif np.abs(airfoil[0,0]-airfoil[-1,0]) > 0.01 or np.abs(airfoil[0,1]-airfoil[-1,1]) > 0.01:
        # print('Unsuccessful:', (airfoil[0,0],airfoil[-1,0]), (airfoil[0,1],airfoil[-1,1]))
        perf = np.nan
        R = np.nan
        CD = np.nan
        
    else:
        xf = XFoil()
        xf.print = 0
        airfoil = setupflap(airfoil, theta=-2)
        xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
        xf.Re = Re2
        xf.M = 0.11
        xf.max_iter = 200
        a, CL, CD, cm, cp = xf.aseq(-2, 2, 0.5)
        CD = CD[~np.isnan(CD)]
        try:
            CD = CD.min()
        except:
            CD = np.nan
            
        airfoil = setflap(airfoil, theta=2)
        xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
        xf.Re = Re1
        xf.M = 0.015
        xf.max_iter = 200
        a, cd, cm, cp = xf.cl(cl)
        perf = cl/cd
        R = cd + CD * lamda
        if perf < -100 or perf > 300 or cd < 1e-3:
            perf = np.nan
        elif not np.isnan(perf):
            print('Successful: CL/CD={:.4f}, R={}'.format(perf, R))
            
    if return_CL_CD:
        return perf, cl.max(), cd.min()
    else:
        return perf, CD, airfoil, R

if __name__ == "__main__":
    LAMBDA = 5
    perf_BL, R_BL = cal_baseline(lamda=LAMBDA)
    CD_BL = 0.004852138459682465
    cl = 0.65
    best_perf=perf_BL
    best_airfoil = None
    if opt.method == '2d':
        name = 'Airfoils2D'
        airfoilpath = '/work3/s212645/DiffusionAirfoil/Airfoils/'
    elif opt.method == '1d':
        name = 'Airfoils1D'
        airfoilpath = '/work3/s212645/DiffusionAirfoil/'+name+'/'
    elif opt.method == 'bezier':
        name = 'Airfoilsbezier'
        airfoilpath = '/work3/s212645/BezierGANPytorch/Airfoils/'

    try:
        log = np.loadtxt(f'results/{name}_simlog.txt')
        i = int(log[0])
        k = int(log[1])
        m = int(log[2])
    except:
        m = 0
        i = 0
        k = 0

    print(f'i: {i}, k: {k}, m: {m}')
    while i < 1000:
        f = open(f'results/{name}_simperf.log', 'a')
        f.write(f'files: {i}\n')
        f.close()
        num = str(i).zfill(3)
        airfoils = np.load(airfoilpath+num+'.npy')
        airfoils = delete_intersect(airfoils)
        while k < airfoils.shape[0]:
            airfoil = airfoils[k,:,:]
            airfoil = derotate(airfoil)
            airfoil = Normalize(airfoil)
            xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
            airfoil[:,0] = xhat
            airfoil[:,1] = yhat
            perf, CD, af, R = evaluate(airfoil, cl, lamda=LAMBDA)
            if perf == np.nan:
                pass
            elif R < R_BL:
                mm = str(m).zfill(3)
                np.savetxt(f'samples/{name}_{mm}.dat', airfoil, header=f'{name}_{mm}', comments="")
                np.savetxt(f'samples/{name}_{mm}F.dat', af, header=f'{name}_{mm}F', comments="")
                f = open(f'results/{name}_simperf.log', 'a')
                f.write(f'perf: {perf}, R: {R}, m: {mm}\n')
                f.close()
                m += 1
            k += 1
            log = np.array([i, k, m])
            np.savetxt(f'results/{name}_simlog.txt', log)
        k = 0
        i += 1