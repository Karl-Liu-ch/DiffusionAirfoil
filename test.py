from utils import *
import re
import os
LAMBDA = 3
dat = re.compile('.dat')
root = 'samples/'

airfoils = []
for path, dir, files in os.walk(root):
    for file in files:
        if dat.search(file) is not None:
            airfoils.append(file)

Rbl = 1
files.sort()
for file in files:
    points = '{}{}'.format(root, file)
    airfoil = np.loadtxt(points, skiprows=1)
    af, R, a, b, perf, cd, CD_BL = lowestD(airfoil, Re2= 400000, lamda = LAMBDA, check_thickness=False, modify_thickness = True)
    if perf > 39.06369801476684 and CD_BL < 0.00485806493088603:
        name = points.split('/')[1].split('.')[0]+f'_{a}_{b}B'
        np.savetxt('BETTER/'+name+'.dat', af, header=name, comments="")
        print(f'R: {R}, angle: {a}, pose: {b}, perf: {perf}, cruise cd: {cd}, launch cd: {CD_BL}, intersection: {detect_intersect(af)}, thickness: {cal_thickness(af)}, thickness pose: {cal_thickness_percent(af)}, tail cross pose: {check_backpoint(af)}')
    os.remove(points)
    print('{} removed'.format(points))
    if R < Rbl:
        Rbl = R
        best_af = points.split('.')[0]+f'_{a}_{b}F'+'.dat'
        np.savetxt('results/best.dat', af, header='best', comments="")