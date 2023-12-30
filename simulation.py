import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def detect_intersect(airfoil):
    # Get leading head
    lh_idx = np.argmin(airfoil[:,0])
    lh_x = airfoil[lh_idx, 0]
    # Get trailing head
    th_x = np.minimum(airfoil[0,0], airfoil[-1,0])
    # Interpolate
    f_up = interp1d(airfoil[:lh_idx+1,0], airfoil[:lh_idx+1,1])
    f_low = interp1d(airfoil[lh_idx:,0], airfoil[lh_idx:,1])
    xx = np.linspace(lh_x, th_x, num=1000)
    yy_up = f_up(xx)
    yy_low = f_low(xx)
    # Check if intersect or not
    if np.any(yy_up < yy_low):
        return True
    else:
        return False

def evaluate(airfoil, return_CL_CD=False):
        
    if detect_intersect(airfoil):
        print('Unsuccessful: Self-intersecting!')
        perf = np.nan
        CL = np.nan
        CD = np.nan
        
    elif airfoil[:,1].max() - airfoil[:,1].min() < 0.05:
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
        a, cl, cd, cm, cp = xf.aseq(0, 6, 0.5)
        # cl, cd, cm, cp = xf.a(2)
        perf = (cl/cd).max()
        
        if perf < -100 or perf > 300 or cd.min() < 1e-3:
            print('Unsuccessful:', cl.max(), cd.min(), perf)
            perf = np.nan
        elif not np.isnan(perf):
            print('Successful: CL/CD={:.4f}'.format(perf))
            
    if return_CL_CD:
        return perf, cl.max(), cd.min()
    else:
        return perf

if __name__ == "__main__":
    airfoil = np.load('data/airfoil_interp.npy')
    # airfoil = np.load('sample.npy')
    # airfoil = np.squeeze(airfoil, axis=1)
    airfoil = airfoil[2]
    perf = evaluate(airfoil)
    print(perf)
