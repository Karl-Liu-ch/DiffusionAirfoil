import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import logging
logging.basicConfig(filename='results/perf.log', encoding='utf-8', level=logging.DEBUG)
from utils import interpolate, derotate, Normalize, delete_intersect, detect_intersect

def evaluate(airfoil, cl, return_CL_CD=False):
        
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
        xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
        xf.Re = 4.5e4
        xf.max_iter = 200
        # a, cl, cd, cm, cp = xf.aseq(2, 5, 0.5)
        # perf = (cl/cd).max()
        a, cd, cm, cp = xf.cl(cl)
        perf = cl/cd
        
        # if perf < -100 or perf > 300 or cd.min() < 1e-3:
        #     print('Unsuccessful:', cl.max(), cd.min(), perf)
        #     perf = np.nan
        if perf < -100 or perf > 300 or cd < 1e-3:
            print('Unsuccessful:', cl, cd, perf)
            perf = np.nan
        elif not np.isnan(perf):
            print('Successful: CL/CD={:.4f}'.format(perf))
            
    if return_CL_CD:
        return perf, cl.max(), cd.min()
    else:
        return perf

if __name__ == "__main__":
    cl = 0.65
    best_perf=34.78824390025072
    airfoilpath = '/work3/s212645/DiffusionAirfoil/Airfoils/'
    best_airfoil = None
    for i in range(100):
        num = str(i).zfill(3)
        airfoils = np.load(airfoilpath+num+'.npy')
        airfoils = delete_intersect(airfoils)
        for k in range(airfoils.shape[0]):
            airfoil = airfoils[k,:,:]
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