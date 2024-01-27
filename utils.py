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
import gc

def check_af(airfoil):
    if detect_intersect(airfoil) or np.abs(airfoil[0,0]-airfoil[-1,0]) > 0.01 or np.abs(airfoil[0,1]-airfoil[-1,1]) > 0.01:
        return False
    else: 
        return True

def evalperfrange(airfoil, reynolds = 50000, clmin = 0.6, clmax = 0.7, step = 0.02):
    if check_af(airfoil):
        xf = XFoil()
        xf.print = 0
        xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
        xf.Re = reynolds
        a, cl, cd, cm, cp = xf.cseq(clmin, clmax, step)
        del xf
        perfs = cl/cd
        perfs = perfs[~np.isnan(perfs)]
        perf = perfs.mean()
    else:
        perf = np.nan
    return perf

def reynolds_pipe(velocity, diameter, density = 1.225, viscosity = 1.5*10**(-5)):
    return density * velocity * diameter / viscosity

def mode_cl(velocity, area, mass):
    g = 9.81
    density = 1.225
    cl = mass * g / (0.5 * area * density * (velocity ** 2))
    return cl

def type2_simu(af, mass, diameter, area):
    if check_af(af):
        g = 9.806
        density = 1.225
        viscosity = 1.5*10**(-5)
        re_sqrtcl = np.sqrt(2 * mass * g / density / area) * diameter * density / viscosity
        cl_list = np.linspace(0.6, 0.7, 6)
        perfs = []
        for cl in cl_list:
            re = re_sqrtcl / np.sqrt(cl)
            perf,_,_ = evalperf(af, cl, re)
            if not np.isnan(perf):
                perfs.append(perf)
        try:
            perfs = np.array(perfs).mean()
        except:
            perfs = np.nan
    else:
        perfs = np.nan
    return perfs

def hicks_henne(x_coord, y_coord , n, a, w, xM):
    y_deformed = np.array(y_coord)
    for i in range(n):
        ai = a[i]
        xMi = xM[i]
        wi = w[i]
        m = np.log(0.5)/np.log(xMi)
        f = np.sin(np.pi * np.array(x_coord) ** m ) ** wi
        y_deformed += ai * f
        x_1 = []
        for j in range(0,1001):
            x_1.append(j/1000)
        f_b = np.sin(np.pi * np.array(x_1) ** m ) ** wi
    return y_deformed

def split(af):
    half = af[:,0].argmin()
    return af[:half,0], af[half:,0],af[:half,1], af[half:,1]

def mute_airfoil(airfoil, a_up, a_low):
    airfoil = Normalize(airfoil)
    airfoil = derotate(airfoil)
    n = 15
    w = np.full(n,2) 
    xM = np.array([])
    for i in range(1,n+1):
        x_m =  0.5 * (1 - math.sin(math.pi * 2 * i / n))
        xM = np.append(xM,x_m)
    xM = np.sort(xM)
    x_up, x_low, y_up, y_low = split(airfoil)
    y_mod_up = hicks_henne(x_up, y_up, n, a_up, w, xM)
    y_mod_low = hicks_henne(x_low, y_low, n, a_low, w, xM)
    y_mod = np.concatenate((y_mod_up, y_mod_low))
    new_af = np.zeros_like(airfoil)
    new_af[:,0] = airfoil[:,0]
    new_af[:,1] = y_mod
    return new_af

def cal_flap_thickness(af):
    lh_idx = np.argmin(af[:,0])
    lh_x = af[lh_idx, 0]
    # Get trailing head
    th_x = np.minimum(af[0,0], af[-1,0])
    # Interpolate
    f_up = interp1d(af[:lh_idx+1,0], af[:lh_idx+1,1])
    f_low = interp1d(af[lh_idx:,0], af[lh_idx:,1])
    xx = np.linspace(lh_x, th_x, num=1000)
    yy_up = f_up(xx)
    yy_low = f_low(xx)
    flap = yy_up[900:] - yy_low[900:]
    return flap.mean()

def evalpreset(airfoil, Re = 4e5):
    if check_af(airfoil):
        Relist = np.linspace(Re - 1000,Re + 1000,3)
        alfas = np.linspace(-1,1,5)
        CD = []
        for alfa in alfas:
            xf = XFoil()
            xf.print = 0
            xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
            cds = []
            for re in Relist:
                xf.Re = re
                xf.max_iter = 1000
                _, cd, _, _ = xf.a(alfa)
                cds.append(cd)
            cds = np.array(cds)
            i_nan = np.argwhere(np.isnan(cds))
            cds = np.delete(cds, i_nan)
            try:
                cd = cds.max()
            except:
                cd = np.nan
            CD.append(cd)
            del xf
            gc.collect()
        i_nan = np.argwhere(np.isnan(CD))
        a = np.delete(alfas, i_nan)
        CD = np.delete(CD, i_nan)
        try:
            i_min = CD.argmin()
            CD = CD[i_min]
            a = a[i_min]
        except:
            CD = np.nan
    else:
        CD = np.nan
        a = np.nan
    return CD, a

def evalperf(airfoil, cl = 0.65, Re = 5.8e4):
    if check_af(airfoil):
        xf = XFoil()
        xf.print = 0
        xf.airfoil = Airfoil(airfoil[:,0], airfoil[:,1])
        xf.Re = Re
        xf.max_iter = 1000
        a, cd, cm, cp = xf.cl(cl)
        del xf
        gc.collect()
        perf = cl/cd
    else:
        perf = np.nan
        a = np.nan
        cd = np.nan
    return perf, a, cd

def lowestD(airfoil, mass = 0.22, diameter = 0.135, area = 0.194, Re2 = 4e5, lamda = 3, thickness = 0.058,  check_thickness = True, modify_thickness = False):
    if detect_intersect(airfoil):
        # print('Unsuccessful: Self-intersecting!')
        af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL = airfoil, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    elif (cal_thickness(airfoil) < 0.06 or cal_thickness(airfoil) > 0.09) and check_thickness:
        # print('Unsuccessful: Too thin!')
        af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL = airfoil, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    elif np.abs(airfoil[0,0]-airfoil[-1,0]) > 0.01 or np.abs(airfoil[0,1]-airfoil[-1,1]) > 0.01:
        # print('Unsuccessful:', (airfoil[0,0],airfoil[-1,0]), (airfoil[0,1],airfoil[-1,1]))
        af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL = airfoil, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        alpha = np.linspace(-3,0,num=4)
        ail = [0.6, 0.65, 0.7]
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
                if modify_thickness:
                    af[:,1] = af[:,1] * thickness / cal_thickness(af)
                af = interpolate(af, 300, 3)
                CD, _ = evalpreset(af, Re=Re2)
                i = 0
                while CD < 0.004 and (not np.isnan(CD)):
                    i += 1
                    print(not np.isnan(CD), CD)
                    af = interpolate(af, 300 + i * 100, 3)
                    CD, _ = evalpreset(af, Re=Re2)
                    print(CD)
                afc = setflap(af, -a, b)
                perf = type2_simu(afc, mass, diameter, area)
                cd = 0.65 / perf
                R = cd + CD * lamda
                if R < R_BL:
                    R_BL = R
                    a_BL = a
                    b_BL = b
                    af_BL = af
                    perfBL = perf
                    cdbl = cd
                    CD_BL = CD
        # print('perf: ', perfBL, 'R: ', R_BL)
    af_BL = Normalize(af_BL)
    af_BL = derotate(af_BL)
    return af_BL, R_BL, a_BL, b_BL, perfBL, cdbl, CD_BL

def cal_baseline(lamda = 3, mass = 0.22, diameter = 0.135, area = 0.194):
    af = np.loadtxt('BETTER/20150114-50 +2 d.dat', skiprows=1)
    af = interpolate(af, 256, 3)
    cl = 0.65
    perf = type2_simu(af, mass = mass, diameter=diameter, area = area)
    cdc = cl/perf
    af = setflap(af, theta=-2, pose = 0.7)
    cd, _ = evalpreset(af)
    perf_bl = perf
    R_bl = cdc + cd * lamda
    return perf_bl, R_bl

def check_backpoint(af):
    lh_idx = np.argmin(af[:,0])
    lh_x = af[lh_idx, 0]
    # Get trailing head
    th_x = np.minimum(af[0,0], af[-1,0])
    # Interpolate
    f_up = interp1d(af[:lh_idx+1,0], af[:lh_idx+1,1])
    f_low = interp1d(af[lh_idx:,0], af[lh_idx:,1])
    xx = np.linspace(lh_x, th_x, num=1000)
    yy_up = f_up(xx)
    yy_low = f_low(xx)
    back_i = 899 - (yy_up[100:] - yy_low[100:]).argmin()
    return back_i

def cal_thickness(airfoil):
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
    return (yy_up-yy_low).max()

def cal_camber(airfoil):
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
    center = yy_up + yy_low
    i = center.argmax()
    center[i]
    xx[i]
    v1 = np.array([xx[i], center[i]])
    v2 = np.array([xx[-1] - xx[i], center[-1] - center[i]])
    return np.arccos(np.matmul(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi

def cal_thickness_percent(airfoil):
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
    return (yy_up-yy_low).argmax() / 10.0

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
    airfoil = Normalize(airfoil)
    airfoil = derotate(airfoil)
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
    airfoil = interpolate(airfoil, 256, 3)
    return airfoil

def set_thickness_pose(airfoil, pose):
    airfoil = Normalize(airfoil)
    airfoil = derotate(airfoil)
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
    max_thickness_index = (yy_up-yy_low).argmax()
    p = pose*10/max_thickness_index
    xxx = np.copy(xx)
    xxx[:max_thickness_index] = xx[:max_thickness_index] * p
    xxx[max_thickness_index:] = (xx[max_thickness_index:] - xx[max_thickness_index]) * (1000 - max_thickness_index * p) / (1000 - max_thickness_index) + xx[max_thickness_index] * p
    f_up = interp1d(xxx, yy_up)
    f_low = interp1d(xxx, yy_low)
    xx = np.linspace(xxx[0], xxx[-1], num=128)
    yy_up = f_up(xx)
    yy_low = f_low(xx)
    airfoil[:128,0] = np.flip(xx)
    airfoil[:128,1] = np.flip(yy_up)
    airfoil[128:,0] = xx
    airfoil[128:,1] = yy_low
    xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
    airfoil[:,0] = xhat
    airfoil[:,1] = yhat
    airfoil = interpolate(airfoil, 256, 3)
    airfoil = derotate(airfoil)
    airfoil = Normalize(airfoil)
    airfoil = interpolate(airfoil, 256, 3)
    return airfoil

def set_camber(airfoil, angle=8):
    airfoil = derotate(airfoil)
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
    center = (yy_up + yy_low)/2
    center *= (1-angle / cal_camber(airfoil))
    yy_up -= center
    yy_low -= center
    f_up = interp1d(xx, yy_up)
    f_low = interp1d(xx, yy_low)
    xx = np.linspace(xx[0], xx[-1], num=128)
    yy_up = f_up(xx)
    yy_low = f_low(xx)
    airfoil[:128,0] = np.flip(xx)
    airfoil[:128,1] = np.flip(yy_up)
    airfoil[128:,0] = xx
    airfoil[128:,1] = yy_low
    xhat, yhat = savgol_filter((airfoil[:,0], airfoil[:,1]), 10, 3)
    airfoil[:,0] = xhat
    airfoil[:,1] = yhat
    airfoil = interpolate(airfoil, 256, 3)
    airfoil = derotate(airfoil)
    airfoil = Normalize(airfoil)
    airfoil = interpolate(airfoil, 256, 3)
    return airfoil

def setdownflap(airfoil, theta=2, pose=0.65):
    lh_idx = np.argmin(airfoil[:,0])
    theta = theta * np.pi / 180
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    af_down = airfoil[lh_idx:,:]
    _i = np.abs(af_down[:,0] - pose).argmin()
    flap = af_down[_i:,:]
    flap = smooth_line(flap, flap.shape[0], 3)
    flap_new = flap - flap[0,:]
    flap_new = np.matmul(flap_new, R) + flap[0,:]
    af_down_new = np.copy(af_down)
    af_down_new[_i:,:] = np.copy(flap_new)
    theta = np.arctan2(af_down_new[-1,1] - af_down[-1,1], 1)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    af_down_new = np.matmul(af_down_new, R)
    af = np.zeros_like(airfoil)
    af[:lh_idx,:] = airfoil[:lh_idx,:]
    af[lh_idx:,:] = af_down_new
    return af

def setupflap(airfoil, theta=-2, pose=0.65):
    lh_idx = np.argmin(airfoil[:,0])
    theta = theta * np.pi / 180
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    af_up = airfoil[:lh_idx,:]
    _i = np.abs(af_up[:,0] - pose).argmin()
    flap = af_up[:_i,:]
    flap = smooth_line(flap, flap.shape[0], 3)
    flap_new = flap - flap[-1,:]
    flap_new = np.matmul(flap_new, R) + flap[-1,:]
    af_up_new = np.copy(af_up)
    af_up_new[:_i,:] = np.copy(flap_new)
    theta = np.arctan2(af_up_new[0,1] - af_up[0,1], 1)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    af_up_new = np.matmul(af_up_new, R)
    af = np.zeros_like(airfoil)
    af[lh_idx:,:] = airfoil[lh_idx:,:]
    af[:lh_idx,:] = af_up_new
    return af

def smooth_line(Q, N, k, D=20, resolution=1000):
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

def show_airfoil(af):
    fig, axs = plt.subplots(1, 1)
    axs.plot(af[:,0], af[:,1])
    axs.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

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
    airfoil = Normalize(airfoil)
    ptail = 0.5 * (airfoil[0,:]+airfoil[-1,:])
    ptails = np.expand_dims(ptail, axis=0)
    ptails = np.repeat(ptails, airfoil.shape[0], axis=0)
    i = airfoil[:,0].argmin()
    phead = airfoil[i,:]
    theta = np.arctan2(-(airfoil[i,1] - ptail[1]), -(airfoil[i,0] - ptail[0]))
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    airfoil_R = airfoil
    airfoil_R -= np.repeat(np.expand_dims(phead, axis=0), airfoil_R.shape[0], axis=0)
    airfoil_R = np.matmul(airfoil_R, R)
    return airfoil_R

def Normalize(airfoil):
    airfoil[:,0] -= airfoil[:,0].min()
    r = np.maximum(airfoil[0,0], airfoil[-1,0])
    r = float(1.0/r)
    return airfoil * r

def interpolate(airfoil, points = 256, N = 3):
    af = np.copy(airfoil)
    lh_idx = np.argmin(af[:,0])
    lh_x = af[lh_idx, 0]
    th_x = np.minimum(af[0,0], af[-1,0])
    f_up = interp1d(airfoil[:lh_idx+1,0], airfoil[:lh_idx+1,1], kind='linear')
    f_low = interp1d(airfoil[lh_idx:,0], airfoil[lh_idx:,1], kind='linear')
    x = np.linspace(0,1,points//2)
    xx_down = ((np.cos(np.pi + x * np.pi) + 1) / 2) ** 1.2 * (th_x - lh_x - 1e-5) + lh_x
    x = np.linspace(0,1,points//2+1)
    xx_up = ((np.cos(np.pi + x * np.pi) + 1) / 2) ** 1.2 * (th_x - lh_x - 1e-5) + lh_x
    # xx = np.linspace(lh_x, th_x, num=1000)
    yy_low = f_low(xx_down)
    yy_up = f_up(xx_up)
    aff = np.zeros([points,2])
    aff[:points//2,0] = np.flip(xx_up[1:])
    aff[:points//2,1] = np.flip(yy_up[1:])
    aff[points//2:,0] = xx_down
    aff[points//2:,1] = yy_low
    return aff

# def interpolate(Q, N, k, D=20, resolution=1000):
#     Q = Normalize(Q)
#     ''' Interpolate N points whose concentration is based on curvature. '''
#     res, fp, ier, msg = splprep(Q.T, u=None, k=k, s=1e-6, per=0, full_output=1)
#     tck, u = res
#     uu = np.linspace(u.min(), u.max(), resolution)
#     x, y = splev(uu, tck, der=0)
#     dx, dy = splev(uu, tck, der=1)
#     ddx, ddy = splev(uu, tck, der=2)
#     cv = np.abs(ddx*dy - dx*ddy)/(dx*dx + dy*dy)**1.5 + D
#     cv_int = cumtrapz(cv, uu, initial=0)
#     fcv = interp1d(cv_int, uu)
#     cv_int_samples = np.linspace(0, cv_int.max(), N)
#     u_new = fcv(cv_int_samples)
#     x_new, y_new = splev(u_new, tck, der=0)
#     xy_new = np.vstack((x_new, y_new)).T
#     return xy_new

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