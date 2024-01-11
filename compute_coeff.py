from __future__ import division
import configparser
import platform
if platform.system().lower() == 'windows':
    import wexpect
elif platform.system().lower() == 'linux':
    import pexpect
import gc
import numpy as np
from scipy.interpolate import interp1d
import logging
logging.basicConfig(filename='results/perfwin.log', encoding='utf-8', level=logging.DEBUG)
from scipy.signal import savgol_filter
from utils_win import *

tmp_dir='tmp'
if platform.system().lower() == 'windows':
    child = wexpect.spawn('xfoil')
if platform.system().lower() == 'linux':
    child = pexpect.spawn('xfoil')
timeout = 10

child.expect('XFOIL   c> ', timeout)
child.sendline('load {}/airfoil.dat'.format(tmp_dir))
child.expect('Enter airfoil name   s> ', timeout)
child.sendline('af')
child.expect('XFOIL   c> ', timeout)
child.sendline('OPER')
child.expect('.OPERi   c> ', timeout)
child.sendline('VISC {}'.format(5.8e4))
child.expect('.OPERv   c> ', timeout)
child.sendline('ITER {}'.format(2000))
child.expect('.OPERv   c> ', timeout)
child.sendline('MACH {}'.format(0))
child.expect('.OPERv   c> ', timeout)
child.sendline('PACC')
child.expect('Enter  polar save filename  OR  <return> for no file   s> ', timeout)
child.sendline('{}/airfoil.log'.format(tmp_dir))
child.expect('Enter  polar dump filename  OR  <return> for no file   s> ', timeout)
child.sendline()
child.expect('.OPERva   c> ', timeout)
child.sendline('alfa {}'.format(0.65))
child.expect('.OPERva   c> ', timeout)
child.sendline()
child.expect('XFOIL   c> ', timeout)
child.sendline('quit')

child.close()

res = np.loadtxt('{}/airfoil.log'.format(tmp_dir), skiprows=12)
CL = res[1]
CD = res[2]