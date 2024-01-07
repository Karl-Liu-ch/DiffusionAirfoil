from utils import *
import argparse
parser = argparse.ArgumentParser(description="DiffusionAirfoil")
parser.add_argument('--method', type=str, default='1d')
opt = parser.parse_args()

LAMBDA = 5
perf_BL, R_BL = cal_baseline(lamda=LAMBDA)
CD_BL = 0.004852138459682465
cl = 0.65
best_perf=perf_BL
best_airfoil = None
if opt.method == '2d':
    name = 'Airfoils2D'
    airfoilpath = '/work3/s212645/DiffusionAirfoil/Airfoils/'
elif opt.method == '1d':
    name = 'Airfoils1D'
    airfoilpath = '/work3/s212645/DiffusionAirfoil/'+name+'/'
elif opt.method == 'bezier':
    name = 'Airfoilsbezier'
    airfoilpath = '/work3/s212645/BezierGANPytorch/Airfoils/'

try:
    log = np.loadtxt(f'results/{name}_log.txt')
    i = int(log[0])
    k = int(log[1])
    m = int(log[2])
except:
    m = 0
    i = 0
    k = 0

print(f'i: {i}, k: {k}, m: {m}')
while i < 100:
    f = open(f'results/{name}_perf.log', 'a')
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
        af, R, a, b, perf, cd, CD_BL = lowestD(airfoil, lamda=LAMBDA)
        if perf == np.nan:
            pass
        elif R < R_BL:
            mm = str(m).zfill(3)
            np.savetxt(f'BETTER/{name}_{mm}_{a}_{b}.dat', airfoil, header=f'{name}_{mm}_{a}_{b}', comments="")
            np.savetxt(f'BETTER/{name}_{mm}_{a}_{b}F.dat', af, header=f'{name}_{mm}_{a}_{b}F', comments="")
            f = open(f'results/{name}_perf.log', 'a')
            f.write(f'perf: {perf}, R: {R}, m: {mm}, a: {a}, b: {b}\n')
            f.close()
            m += 1
        k += 1
        log = np.array([i, k, m])
        np.savetxt(f'results/{name}_log.txt', log)
    k = 0
    i += 1