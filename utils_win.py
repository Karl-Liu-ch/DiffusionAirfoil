import os
import itertools
import time
import torch
import math
from inspect import isfunction
from functools import partial
import re
from airfoil_process import *

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
# import tensorflow as tf

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import splev, splprep, interp1d
from scipy.integrate import cumtrapz

from scipy.signal import savgol_filter
import platform
if platform.system().lower() == 'windows':
    import wexpect
elif platform.system().lower() == 'linux':
    import pexpect
else:
    import pexpect
import gc

def compute_coeff(airfoil, reynolds=58000, mach=0, alpha=0, n_iter=2000, tmp_dir='tmp', hold_cl = True):
    create_dir(tmp_dir)
    gc.collect()
    safe_remove('{}/airfoil.log'.format(tmp_dir))
    fname = '{}/airfoil.dat'.format(tmp_dir)
    with open(fname, 'wb') as f:
        np.savetxt(f, airfoil)
    try:
        if platform.system().lower() == 'windows':
            child = wexpect.spawn('xfoil')
        if platform.system().lower() == 'linux':
            child = pexpect.spawn('xfoil')
        else:
            child = pexpect.spawn('xfoil')
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
        if hold_cl == True:
            child.sendline('CL {}'.format(alpha))
        else:
            child.sendline('ALFA {}'.format(alpha))
        child.expect('c> ', timeout)
        # child.expect('.OPERva   c> ', timeout)
        child.sendline()
        child.expect('XFOIL   c> ', timeout)
        child.sendline('quit')
        
        child.close()
    
        res = np.loadtxt('{}/airfoil.log'.format(tmp_dir), skiprows=12)
        CL = res[1]
        CD = res[2]
            
    except Exception as ex:
        print(ex)
        print('XFoil error!')
        CL = np.nan
        CD = np.nan
        
    safe_remove(':00.bl')
    
    return CL, CD

def cal_polar(path, reynolds=58000, mach=0, alpha_min=-1, alpha_max = 10, alpha_step = 0.5, n_iter=200, tmp_dir='polar'):
    create_dir(tmp_dir)
    gc.collect()
    safe_remove('{}/{}.log'.format(tmp_dir, str(reynolds)))
    try:
        if platform.system().lower() == 'windows':
            child = wexpect.spawn('xfoil')
        else:
            child = pexpect.spawn('xfoil')
        timeout = 100
        
        child.expect('XFOIL   c> ', timeout)
        child.sendline('load {}'.format(path))
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
        child.sendline('{}/{}.log'.format(tmp_dir, str(reynolds)))
        child.expect('Enter  polar dump filename  OR  <return> for no file   s> ', timeout)
        child.sendline()
        child.expect('.OPERva   c> ', timeout)
        child.sendline('aseq {} {} {}'.format(alpha_min, alpha_max, alpha_step))
        child.expect('.OPERva   c> ', timeout)
        child.sendline()
        child.expect('XFOIL   c> ', timeout)
        child.sendline('quit')
        
        child.close()
            
    except Exception as ex:
        print(ex)
        print('XFoil error!')
        
    safe_remove(':00.bl')

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def rewrite_polar(root):
    log = re.compile('.log')
    for path, dir, files in os.walk(root):
        for file in files:
            if log.search(file) is not None:
                lines=[]
                infile = '{}/{}'.format(path,file)
                with open(infile, 'r') as fin:
                    for line in fin:
                        line = line.replace('9.000  9.000', '9.000')
                        lines.append(line)
                fin.close()
                with open(infile, "w") as fout:
                    for line in lines:
                        fout.write(line)
                fout.close()

def get_dat(root = 'F3K_airfoils/'):
    tail = re.compile('.dat')
    airfoils = []
    for path, dir, files in os.walk(root):
        files.sort()
        for file in files:
            if tail.search(file) is not None:
                airfoils.append('{}{}'.format(path, file))
    return airfoils

def cal_afs_polar(re_min = 500, re_max = 400000, re_step = 500, path = 'F3K_airfoils/'):
    afs = get_dat(path)
    for af in afs:
        print(af)
    len = (re_max - re_min) // re_step + 1
    for af in afs:
        root = af.split('/')[-1].split('.dat')[0]
        for re in range(len):
            cal_polar(af, reynolds=re*re_step+re_min, tmp_dir = path + root)
        rewrite_polar(root)

def cal_afs_polar_re_list(reynolds: list, path = 'F3K_airfoils/'):
    afs = get_dat(path)
    for af in afs:
        print(af)
    for af in afs:
        root = af.split('/')[-1].split('.dat')[0]
        for re in reynolds:
            cal_polar(af, reynolds=re, tmp_dir = path + root)
        rewrite_polar(root)


if __name__ == '__main__':
    # afs = ['F3K_airfoils/Airfoils2D_049_0.6F_80_-3.dat', 
    #        'F3K_airfoils/Airfoils2D_049_0.6F_20_0.dat', 
    #        'F3K_airfoils/Airfoils1D_004F_0.dat', 
    #        'F3K_airfoils/Airfoils2D_049_0.6F_100_0.dat', 
    #        'F3K_airfoils/Airfoils2D_049_0.6F_80_0.dat', 
    #        'F3K_airfoils/Airfoils2D_049_0.6F_50_0.dat', 
    #        'F3K_airfoils/Airfoils2D_049_0.6F_50_-3.dat']
    # # afs = get_dat()
    # for af in afs:
    #     print(af)
    # for af in afs:
    #     root = af.split('/')[1].split('.dat')[0]+'/'
    #     print(root)
    #     for re in range(400000//500):
    #         cal_polar(af, reynolds=re*500+500, tmp_dir = root)
    #     rewrite_polar(root)
    root = 'airfoilP5B'
    rewrite_polar(root)