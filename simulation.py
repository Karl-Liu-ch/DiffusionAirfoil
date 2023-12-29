from xfoil import XFoil
from xfoil.test import naca0012
import matplotlib.pyplot as plt
xf = XFoil()
xf.airfoil = naca0012
xf.Re = 1e6
xf.max_iter = 200
a, cd, cm = xf.cl(cl=1)
print(cd)