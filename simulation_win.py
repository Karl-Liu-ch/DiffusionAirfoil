#import os
from __future__ import division
import configparser
import platform
import gc
import numpy as np
from scipy.interpolate import interp1d
import logging
logging.basicConfig(filename='results/perfwin.log', encoding='utf-8', level=logging.DEBUG)
from scipy.signal import savgol_filter
from utils_win import *
import argparse
parser = argparse.ArgumentParser(description="DiffusionAirfoil")
parser.add_argument('--method', type=str, default='1d')
opt = parser.parse_args()

def evaluate(airfoil, cl = 0.65, Re1 = 5.8e4, Re2 = 4e5, lamda = 3, return_CL_CD=False, check_thickness = True):
        
    if detect_intersect(airfoil):
        perf = np.nan
        R = np.nan
        CD = np.nan
        
    elif (cal_thickness(airfoil) < 0.06 or cal_thickness(airfoil) > 0.09) and check_thickness:
        perf = np.nan
        R = np.nan
        CD = np.nan
        
    elif np.abs(airfoil[0,0]-airfoil[-1,0]) > 0.01 or np.abs(airfoil[0,1]-airfoil[-1,1]) > 0.01:
        perf = np.nan
        R = np.nan
        CD = np.nan
        
    else:
        airfoil = setupflap(airfoil, theta=-2)
        CD = evalpreset_win(airfoil, Re2)
        airfoil = setflap(airfoil, theta=2)
        perf, cd = evalperf_win(airfoil, cl = cl, Re = Re1)
        R = cd + CD * lamda
        
        if perf < -100 or perf > 300 or CD < 1e-3:
            perf = np.nan
        elif not np.isnan(perf):
            print('Successful: CL/CD={:.4f}, R={}'.format(perf, R))
    return perf, CD, R, airfoil

def evalpreset_win(airfoil, Re = 4e5):
    a = np.linspace(-1,1,5)
    CD = []
    for alfa in a:
        cl, cd = compute_coeff(airfoil, reynolds=Re, alpha=alfa, cl=False)
        CD.append(cd)
    i_nan = np.argwhere(np.isnan(CD))
    a = np.delete(a, i_nan)
    CD = np.delete(CD, i_nan)
    try:
        i_min = CD.argmin()
        CD = CD[i_min]
        a = a[i_min]
    except:
        CD = np.nan
    return CD

def evalperf_win(airfoil, cl = 0.65, Re = 5.8e4):
    cl, cd = compute_coeff(airfoil, reynolds=Re, alpha=cl, cl=True)
    perf = cl/cd
    return perf, cd

def lowestD(airfoil, cl = 0.65, Re1 = 5.8e4, Re2 = 4e5):
    if detect_intersect(airfoil):
        af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL = airfoil, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    elif abs(airfoil[:128,1] - np.flip(airfoil[128:,1])).max() < 0.055 or abs(airfoil[:128,1] - np.flip(airfoil[128:,1])).max() > 0.08:
        af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL = airfoil, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    elif np.abs(airfoil[0,0]-airfoil[-1,0]) > 0.01 or np.abs(airfoil[0,1]-airfoil[-1,1]) > 0.01:
        af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL = airfoil, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        alpha = np.linspace(-3,0,num=4)
        ail = np.linspace(0.6,0.7,num=3)
        R_BL = 10
        CD_BL = 10
        a_BL = -3
        b_BL = 0.6
        perfBL = 0
        cdbl = 10
        af_BL = airfoil
        for a in alpha:
            for b in ail:
                af = setupflap(airfoil, a, b)
                CD, aa = evalpreset_win(af, Re = Re2)
                afc = setflap(af, -a, b)
                perf, cd = evalperf_win(afc, cl=cl, Re = Re1)
                R = cd + CD * 3
                if R < R_BL:
                    R_BL = R
                    a_BL = a
                    b_BL = b
                    af_BL = af
                    perfBL = perf
                    cdbl = cd
                    CD_BL = CD
        print('perf: ', perfBL, 'R: ', R_BL)
    return af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL

if __name__ == "__main__":
    af = np.loadtxt('BETTER/20150114-50 +2 d.dat', skiprows=1)
    af = interpolate(af, 256, 3)
    cl = 0.65
    perf_BL, cd, R_BL, af = evaluate(af)
    R_BL = 0.031195905059576035
    perf_BL = 39.06369801476684
    CD_BL = 0.004852138459682465
    cl = 0.65
    best_perf=perf_BL
    best_airfoil = None
    
    if platform.system().lower() == 'windows':
        name = 'Airfoils'
        airfoilpath = 'H:/深度学习/Airfoils/'
    elif platform.system().lower() == 'linux':
        if opt.method == '2d':
            name = 'Airfoils2D'
            airfoilpath = '/work3/s212645/DiffusionAirfoil/Airfoils/'
        elif opt.method == '1d':
            name = 'Airfoils1D'
            airfoilpath = '/work3/s212645/DiffusionAirfoil/'+name+'/'
        elif opt.method == 'bezier':
            name = 'Airfoilsbezier'
            airfoilpath = '/work3/s212645/BezierGANPytorch/Airfoils/'
    best_airfoil = None
    try:
        log = np.loadtxt(f'results/{name}win_log.txt')
        i = int(log[0])
        k = int(log[1])
        m = int(log[2])
    except:
        m = 0
        i = 0
        k = 0

    print(f'i: {i}, k: {k}, m: {m}')
    while i < 100:
        f = open(f'results/{name}win_perf.log', 'a')
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
            perf, cd, R, af = evaluate(airfoil)
            a = -2
            b = 0.7
            if perf == np.nan:
                pass
            elif R < R_BL:
                mm = str(m).zfill(3)
                np.savetxt(f'BETTER/{name}_{mm}_{a}_{b}.dat', airfoil, header=f'{name}_{mm}_{a}_{b}', comments="")
                np.savetxt(f'BETTER/{name}_{mm}_{a}_{b}F.dat', af, header=f'{name}_{mm}_{a}_{b}F', comments="")
                f = open(f'results/{name}win_perf.log', 'a')
                f.write(f'perf: {perf}, R: {R}, m: {mm}, a: {a}, b: {b}\n')
                f.close()
                m += 1
            k += 1
            log = np.array([i, k, m])
            np.savetxt(f'results/{name}win_log.txt', log)
        k = 0
        i += 1