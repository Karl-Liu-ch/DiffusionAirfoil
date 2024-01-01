#import os
from __future__ import division
import configparser
import wexpect
import gc
import numpy as np
from scipy.interpolate import interp1d
import logging
logging.basicConfig(filename='results/perf.log', encoding='utf-8', level=logging.DEBUG)

from scipy.signal import savgol_filter
from utils import safe_remove, create_dir

from utils import interpolate, derotate, Normalize, delete_intersect, detect_intersect

def compute_coeff(airfoil, reynolds=500000, mach=0, alpha=3, n_iter=200, tmp_dir='tmp'):
    
    create_dir(tmp_dir)
    
    gc.collect()
    safe_remove('{}/airfoil.log'.format(tmp_dir))
    fname = '{}/airfoil.dat'.format(tmp_dir)
    CL = alpha
    with open(fname, 'wb') as f:
        np.savetxt(f, airfoil)
    
    try:
        # Has error: Floating point exception (core dumped)
        # This is the "empty input file: 'tmp/airfoil.log'" warning in other approaches
        child = wexpect.spawn('xfoil')
        timeout = 10
        
        child.expect('XFOIL   c> ', timeout)
        child.sendline('load {}/airfoil.dat'.format(tmp_dir))
        child.expect('Enter airfoil name   s> ', timeout)
        child.sendline('af')
        child.expect('XFOIL   c> ', timeout)
        child.sendline('OPER')
        child.expect('.OPERi   c> ', timeout)
        child.sendline('VISC {}'.format(reynolds))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('ITER {}'.format(n_iter))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('MACH {}'.format(mach))
        child.expect('.OPERv   c> ', timeout)
        child.sendline('PACC')
        child.expect('Enter  polar save filename  OR  <return> for no file   s> ', timeout)
        child.sendline('{}/airfoil.log'.format(tmp_dir))
        child.expect('Enter  polar dump filename  OR  <return> for no file   s> ', timeout)
        child.sendline()
        child.expect('.OPERva   c> ', timeout)
        child.sendline('CL {}'.format(alpha))
        child.expect('.OPERva   c> ', timeout)
        child.sendline()
        child.expect('XFOIL   c> ', timeout)
        child.sendline('quit')
        
        child.close()
    
        res = np.loadtxt('{}/airfoil.log'.format(tmp_dir), skiprows=12)
        CD = res[2]
            
    except Exception as ex:
        print(ex)
        print('XFoil error!')
        CL = np.nan
        CD = np.nan
        
    safe_remove(':00.bl')
    
    return CL, CD

def read_config(config_fname):
    
    # Airfoil operating conditions
    Config = configparser.ConfigParser()
    Config.read(config_fname)
    reynolds = float(Config.get('OperatingConditions', 'Reynolds'))
    mach = float(Config.get('OperatingConditions', 'Mach'))
    alpha = float(Config.get('OperatingConditions', 'Alpha'))
    n_iter = int(Config.get('OperatingConditions', 'N_iter'))
    
    return reynolds, mach, alpha, n_iter


def evaluate(airfoil, cl, return_CL_CD=False):
    
    # Read airfoil operating conditions from a config file
    reynolds = 4.5e4
    mach = 0.01
    n_iter = 2000
            
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
        
        CL, CD = compute_coeff(airfoil, reynolds, mach, cl, n_iter)
        CL = cl
        perf = CL/CD
        
        if perf < -100 or perf > 300 or CD < 1e-3:
            print('Unsuccessful:', CL, CD, perf)
            perf = np.nan
        elif not np.isnan(perf):
            print('Successful: CL/CD={:.4f}'.format(perf))
            
    if return_CL_CD:
        return perf, CL, CD
    else:
        return perf
    
def derotate(airfoil):
    ptail = 0.5 * (airfoil[0,:]+airfoil[-1,:])
    ptails = np.expand_dims(ptail, axis=0)
    ptails = np.repeat(ptails, 256, axis=0)
    i = np.linalg.norm(airfoil - ptails, axis=1).argmax()
    phead = airfoil[i,:]
    theta = np.arctan2(-(airfoil[i,1] - ptail[1]), -(airfoil[i,0] - ptail[0]))
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    airfoil_R = airfoil
    airfoil_R -= np.repeat(np.expand_dims(phead, axis=0), 256, axis=0)
    airfoil_R = np.matmul(airfoil_R, R)
    return airfoil_R

def Normalize(airfoil):
    r = np.maximum(airfoil[0,0], airfoil[-1,0])
    r = float(1.0/r)
    return airfoil * r

if __name__ == "__main__":
    cl = 0.65
    best_perf=34.78824390025072
    airfoilpath = 'H:/深度学习/Airfoils/'
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
                np.savetxt('results/airfoil1D.dat', best_airfoil)
                logging.info(f'perf: {perf}, thickness: {yhat.max()-yhat.min()}')