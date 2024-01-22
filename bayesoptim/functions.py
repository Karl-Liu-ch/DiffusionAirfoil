import sys
sys.path.append('./')
import numpy as np
from pyDOE import lhs
from scipy.optimize import minimize
from DiffusionAirfoil1D_transform import Diff as Difft
from DiffusionAirfoil_transform import Diff as Diff2Dt
from DiffusionAirfoil1D import Diff as Diff1d
from DiffusionAirfoil import Diff as Diff2d
from simulation import evaluate
import torch
from utils import *

class Airfoil(object):
    
    def __init__(self):
        self.y = None
        self.bounds = None
        self.dim = None
            
    def __call__(self, x):
        x = np.array(x, ndmin=2)
        y = - np.apply_along_axis(lambda x: evaluate(self.synthesize(x))[-1], 1, x)
        self.y = np.squeeze(y)
        return self.y
    
    def is_feasible(self, x):
        x = np.array(x, ndmin=2)
        if self.y is None:
            self.y = self.__call__(x)
        feasibility = np.logical_not(np.isnan(self.y))
        return feasibility
    
    def synthesize(self, x):
        pass
    
    def sample_design_variables(self, n_sample, method='random'):
        if method == 'lhs':
            x = lhs(self.dim, samples=n_sample, criterion='cm')
            x = x * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]
        else:
            # x = np.random.uniform(self.bounds[:,0], self.bounds[:,1], size=(n_sample, self.dim))
            x = np.random.normal(size=(n_sample, self.dim))
        return np.squeeze(x)
    
    def sample_airfoil(self, n_sample, method='random'):
        x = self.sample_design_variables(n_sample, method)
        airfoils = self.synthesize(x)
        return airfoils
    
class AirfoilDiffusion(Airfoil):
    def __init__(self, thickness = 0.065):
        super().__init__()
        self.thickness = thickness
        self.dim = 512
        self.alpha0 = np.random.normal(size=[self.dim])
        self.model = Diff1d
        self.bounds = np.array([[-1., 1.]])
        self.bounds = np.tile(self.bounds, [self.dim, 1])
        # self.bounds = np.array([-0.5, 0.5])
        # self.bounds = np.tile(self.bounds, [self.dim, 1])

    def synthesize(self, x):
        x = torch.from_numpy(x).cuda()
        x = x.reshape([1,1,512])
        x = x.to(torch.float32)
        af = self.model.sample(batch_size=1, channels=1, noise = x).reshape(256, 2).cpu().numpy()
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        xhat, yhat = savgol_filter((af[:,0], af[:,1]), 10, 3)
        af[:,0] = xhat
        af[:,1] = yhat
        return af    
    
class AirfoilDiffusionT(Airfoil):
    def __init__(self, thickness = 0.065):
        super().__init__()
        self.thickness = thickness
        self.dim = 512
        self.alpha0 = np.random.normal(size=[self.dim])
        self.model = Difft
        self.bounds = np.array([[-1., 1.]])
        self.bounds = np.tile(self.bounds, [self.dim, 1])
        # self.bounds = np.array([-0.5, 0.5])
        # self.bounds = np.tile(self.bounds, [self.dim, 1])

    def synthesize(self, x):
        x = torch.from_numpy(x).cuda()
        x = x.reshape([1,1,512])
        x = x.to(torch.float32)
        af = self.model.sample(batch_size=1, channels=1, noise = x).reshape(256, 2).cpu().numpy()
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        xhat, yhat = savgol_filter((af[:,0], af[:,1]), 10, 3)
        af[:,0] = xhat
        af[:,1] = yhat
        return af    
    
class AirfoilDiffusion2D(Airfoil):
    def __init__(self, thickness = 0.065):
        super().__init__()
        self.thickness = thickness
        self.dim = 512
        self.model = Diff2d
        self.bounds = np.array([[-1., 1.]])
        self.bounds = np.tile(self.bounds, [self.dim, 1])
        # self.bounds = np.array([-0.5, 0.5])
        # self.bounds = np.tile(self.bounds, [self.dim, 1])

    def synthesize(self, x):
        x = torch.from_numpy(x).cuda()
        x = x.reshape([1,1,256,2])
        x = x.to(torch.float32)
        af = self.model.sample(batch_size=1, channels=1, noise = x).reshape(256, 2).cpu().numpy()
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        xhat, yhat = savgol_filter((af[:,0], af[:,1]), 10, 3)
        af[:,0] = xhat
        af[:,1] = yhat
        return af
    
class AirfoilDiffusion2DT(Airfoil):
    def __init__(self, thickness = 0.065):
        super().__init__()
        self.thickness = thickness
        self.dim = 512
        self.model = Diff2Dt
        self.bounds = np.array([[-1., 1.]])
        self.bounds = np.tile(self.bounds, [self.dim, 1])
        # self.bounds = np.array([-0.5, 0.5])
        # self.bounds = np.tile(self.bounds, [self.dim, 1])

    def synthesize(self, x):
        x = torch.from_numpy(x).cuda()
        x = x.reshape([1,1,256,2])
        x = x.to(torch.float32)
        af = self.model.sample(batch_size=1, channels=1, noise = x).reshape(256, 2).cpu().numpy()
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        xhat, yhat = savgol_filter((af[:,0], af[:,1]), 10, 3)
        af[:,0] = xhat
        af[:,1] = yhat
        return af