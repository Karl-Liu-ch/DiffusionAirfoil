"""
Author(s): Wei Chen (wchen459@umd.edu)
"""

import os
import itertools
import time
import torch
import math
from inspect import isfunction
from functools import partial

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
from xfoil import XFoil
from xfoil.model import Airfoil

def evalpreset(airfoil, Re = 4e5):
    xf = XFoil()
    xf.print = 0
    xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
    xf.Re = Re
    xf.max_iter = 200
    a = np.linspace(-2,2,5)
    CD = []
    for alfa in a:
        _, cd, _, _ = xf.a(alfa)
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
    return CD, a

def evalperf(airfoil, cl = 0.65, Re = 5.8e4):
    xf = XFoil()
    xf.print = 0
    xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
    xf.Re = Re
    xf.max_iter = 200
    a, cd, cm, cp = xf.cl(cl)
    perf = cl/cd
    return perf, a, cd

def lowestD(airfoil, cl = 0.65, Re1 = 5.8e4, Re2 = 4e5):
    if detect_intersect(airfoil):
        # print('Unsuccessful: Self-intersecting!')
        af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL = airfoil, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    elif abs(airfoil[:128,1] - np.flip(airfoil[128:,1])).max() < 0.055 or abs(airfoil[:128,1] - np.flip(airfoil[128:,1])).max() > 0.08:
        # print('Unsuccessful: Too thin!')
        af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL = airfoil, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    elif np.abs(airfoil[0,0]-airfoil[-1,0]) > 0.01 or np.abs(airfoil[0,1]-airfoil[-1,1]) > 0.01:
        # print('Unsuccessful:', (airfoil[0,0],airfoil[-1,0]), (airfoil[0,1],airfoil[-1,1]))
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
                CD, aa = evalpreset(af, Re = Re2)
                afc = setflap(af, -a, b)
                perf, aa, cd = evalperf(afc, cl=cl, Re = Re1)
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

def setflap(airfoil, theta = -2, pose = 0.7):
    airfoil = np.copy(airfoil)
    phead_i = airfoil[:,0].argmin()
    pflap_i_down = abs(airfoil[:phead_i,0] - pose).argmin()
    pflap_i_up = abs(airfoil[phead_i:,0] - pose).argmin() + phead_i
    theta = theta * np.pi / 180
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    if theta < 0:
        p_mid = airfoil[pflap_i_down,:]
    else:
        p_mid = airfoil[pflap_i_up,:]
    airfoil[pflap_i_up:,:] = np.matmul(airfoil[pflap_i_up:,:] - p_mid, R) + p_mid
    airfoil[:pflap_i_down,:] = np.matmul(airfoil[:pflap_i_down,:] - p_mid, R) + p_mid
    airfoil = interpolate(airfoil, 256, 3)
    airfoil = derotate(airfoil)
    airfoil = Normalize(airfoil)
    return airfoil

def setupflap(airfoil, theta = -2, pose = 0.7):
    airfoil = np.copy(airfoil)
    phead_i = airfoil[:,0].argmin()
    pflap_i_down = abs(airfoil[:phead_i,0] - pose).argmin()
    pflap_i_up = abs(airfoil[phead_i:,0] - pose).argmin() + phead_i
    theta = theta * np.pi / 180
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    if theta < 0:
        p_mid = airfoil[pflap_i_down,:]
        airfoil[:pflap_i_down,:] = np.matmul(airfoil[:pflap_i_down,:] - p_mid, R) + p_mid
        alpha = -np.arctan2((airfoil[0,1] - airfoil[phead_i,1]), (airfoil[0,0] - airfoil[phead_i,0]))
        c = np.cos(alpha)
        s = np.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        airfoil[phead_i:,:] = np.matmul(airfoil[phead_i:,:] - airfoil[phead_i,:], R) + airfoil[phead_i,:]
    else:
        p_mid = airfoil[pflap_i_up,:]
        airfoil[pflap_i_up:,:] = np.matmul(airfoil[pflap_i_up:,:] - p_mid, R) + p_mid
        alpha = -np.arctan2((airfoil[-1,1] - airfoil[phead_i,1] + 0.0005), (airfoil[-1,0] - airfoil[phead_i,0]))
        c = np.cos(alpha)
        s = np.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        airfoil[:phead_i,:] = np.matmul(airfoil[:phead_i,:] - airfoil[phead_i,:], R) + airfoil[phead_i,:]
    airfoil = interpolate(airfoil, 256, 3)
    airfoil = derotate(airfoil)
    airfoil = Normalize(airfoil)
    return airfoil

def delete_intersect(samples):
    indexs = []
    for i in range(samples.shape[0]):
        xhat, yhat = savgol_filter((samples[i,:,0], samples[i,:,1]), 10, 3)
        samples[i,:,0] = xhat
        samples[i,:,1] = yhat
        af = samples[i,:,:]
        if detect_intersect(af):
            indexs.append(i)
    for i in indexs:
        xhat, yhat = savgol_filter((samples[i,:,0], samples[i,:,1]), 10, 3)
        samples[i,:,0] = xhat
        samples[i,:,1] = yhat
        af = samples[i,:,:]
        point = 1.0
        while detect_intersect(af) and af.shape[0] > 200:
            indexs = []
            for index in range(af.shape[0]):
                if af[index,0] > point:
                    indexs.append(index)
            af = np.delete(af, indexs, axis=0)
            point -= 0.01
        af = interpolate(af, 256, 3)
        af = Normalize(af)
        samples[i,:,:] = af
    return samples

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

def interpolate(Q, N, k, D=20, resolution=1000):
    ''' Interpolate N points whose concentration is based on curvature. '''
    res, fp, ier, msg = splprep(Q.T, u=None, k=k, s=1e-6, per=0, full_output=1)
    tck, u = res
    uu = np.linspace(u.min(), u.max(), resolution)
    x, y = splev(uu, tck, der=0)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5 + D
    cv_int = cumtrapz(cv, uu, initial=0)
    fcv = interp1d(cv_int, uu)
    cv_int_samples = np.linspace(0, cv_int.max(), N)
    u_new = fcv(cv_int_samples)
    x_new, y_new = splev(u_new, tck, der=0)
    xy_new = np.vstack((x_new, y_new)).T
    return xy_new

def convert_sec(sec):
    if sec < 60:
        return "%.2f sec" % sec
    elif sec < (60 * 60):
        return "%.2f min" % (sec / 60)
    else:
        return "%.2f hr" % (sec / (60 * 60))

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed_time(self):
        return convert_sec(time.time() - self.start_time)
    
def gen_grid(d, points_per_axis, lb=0., rb=1.):
    ''' Generate a grid in a d-dimensional space 
        within the range [lb, rb] for each axis '''
    
    lincoords = []
    for i in range(0, d):
        lincoords.append(np.linspace(lb, rb, points_per_axis))
    coords = list(itertools.product(*lincoords))
    
    return np.array(coords)

def mean_err(metric_list):
    n = len(metric_list)
    mean = np.mean(metric_list)
    std = np.std(metric_list)
    err = 1.96*std/n**.5
    return mean, err

def visualize(X):
    
    X = X.reshape((X.shape[0], -1))
    pca = PCA(n_components=3)
    F = pca.fit_transform(X)
    
    # Reconstruction error
    X_rec = pca.inverse_transform(F)
    err = mean_squared_error(X, X_rec)
    print('Reconstruct error: {}'.format(err))
    
    # 3D Plot
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection = '3d')
    
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([F[:,0].max()-F[:,0].min(), F[:,1].max()-F[:,1].min(), F[:,2].max()-F[:,2].min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(F[:,0].max()+F[:,0].min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(F[:,1].max()+F[:,1].min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(F[:,2].max()+F[:,2].min())
    ax3d.scatter(Xb, Yb, Zb, c='white', alpha=0)
    
    ax3d.scatter(F[:,0], F[:,1], F[:,2])
    matplotlib.rcParams.update({'font.size': 22})
#    ax3d.set_xticks([])
#    ax3d.set_yticks([])
#    ax3d.set_zticks([])
    plt.show()
    
def safe_remove(filename):
    if os.path.exists(filename):
        os.remove(filename)

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
def train_test_plit(X, split=0.8):
    # Split training and test data
    N = X.shape[0]
    split = int(N*split)
    X_train = X[:split]
    X_test = X[split:]
    return X_train, X_test

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