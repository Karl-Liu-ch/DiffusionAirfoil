import numpy as np
import sys
import os
from xfoil import XFoil
from xfoil.model import Airfoil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import logging
logging.basicConfig(filename='results/perf1D.log', encoding='utf-8', level=logging.DEBUG)
from utils import *

def evaluate(airfoil, cl, Re = 5e4, return_CL_CD=False):
        
    if detect_intersect(airfoil):
        print('Unsuccessful: Self-intersecting!')
        perf = np.nan
        CL = np.nan
        CD = np.nan
        
    elif abs(airfoil[:128,1] - np.flip(airfoil[128:,1])).max() < 0.055:
        print('Unsuccessful: Too thin!')
        perf = np.nan
        CL = np.nan
        CD = np.nan
    
    elif np.abs(airfoil[0,0]-airfoil[-1,0]) > 0.01 or np.abs(airfoil[0,1]-airfoil[-1,1]) > 0.01:
        print('Unsuccessful:', (airfoil[0,0],airfoil[-1,0]), (airfoil[0,1],airfoil[-1,1]))
        perf = np.nan
        CL = np.nan
        CD = np.nan
        
    else:
        xf = XFoil()
        airfoil = setupflap(airfoil, theta=-2)
        xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
        xf.Re = 4e6
        xf.max_iter = 2000
        a, CL, CD, cm, cp = xf.aseq(-2, 2, 0.5)
        CD = CD[~np.isnan(CD)]
        try:
            CD = CD.min()
        except:
            CD = np.nan
            
        airfoil = setflap(airfoil, theta=2)
        xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
        xf.Re = Re
        xf.max_iter = 2000
        # a, cl, cd, cm, cp = xf.aseq(2, 5, 0.5)
        a, cl, cd, cm, cp = xf.cseq(0.6, 0.9, 0.02)
        perf = cl/cd
        perf = perf[~np.isnan(perf)]
        try:
            perf = (perf).max()
        except:
            perf = np.nan
        
        # a, cd, cm, cp = xf.cl(cl)
        # perf = cl/cd
        
        if perf < -100 or perf > 300 or cd.min() < 1e-3:
            print('Unsuccessful:', cl.max(), cd.min(), perf)
            perf = np.nan
        # if perf < -100 or perf > 300 or cd < 1e-3:
        #     print('Unsuccessful:', cl, cd, perf)
        #     perf = np.nan
        elif not np.isnan(perf):
            print('Successful: CL/CD={:.4f}'.format(perf))
            
    if return_CL_CD:
        return perf, cl.max(), cd.min()
    else:
        return perf, CD

if __name__ == "__main__":    
    perf_BL = 35.981436463391404
    CD_BL = 0.004539919085800648
    cl = 0.65
    best_perf=35.90357900866657
    name = 'Airfoils1D'
    airfoilpath = '/work3/s212645/DiffusionAirfoil/'+name+'/'
    best_airfoil = None
    n = 0
    m = 0
    count = 0
    for i in range(100):
        logging.info(f'files: {i+count}')
        num = str(i+count).zfill(3)
        airfoils = np.load(airfoilpath+num+'.npy')
        airfoils = delete_intersect(airfoils)
        for k in range(airfoils.shape[0]):
            airfoil = airfoils[k,:,:]
            airfoil = derotate(airfoil)
            airfoil = Normalize(airfoil)
            xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
            airfoil[:,0] = xhat
            airfoil[:,1] = yhat
            perf, CD = evaluate(airfoil, cl)
            if perf == np.nan:
                pass
            elif perf > best_perf:
                best_perf = perf
                best_airfoil = airfoil
                np.savetxt('results/'+name+'.dat', best_airfoil)
                logging.info(f'perf: {perf}, cd: {CD}, thickness: {yhat.max()-yhat.min()}')
            if perf > 35:
                nn = str(n).zfill(3)
                np.savetxt('samples/'+name+nn+'.dat', airfoil)
                logging.info(f'perf: {perf}, cd: {CD}, n: {nn}')
                n += 1
            if perf > perf_BL and CD < CD_BL:
                mm = str(m).zfill(3)
                np.savetxt(f'BETTER/airfoil{mm}.dat', airfoil)
                logging.info(f'perf: {perf}, cd: {CD}, n: {mm}')
                m += 1