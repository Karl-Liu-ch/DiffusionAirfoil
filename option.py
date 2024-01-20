import argparse
parser = argparse.ArgumentParser(description="DiffusionAirfoil")
parser.add_argument('--method', type=str, default='1d')
parser.add_argument('--path', type=str, default='/work3/s212645/DiffusionAirfoil1DTransform/Airfoils1D/')
parser.add_argument('--agent', type=str, default='ppo')
parser.add_argument("--gpu_id", type=str, default='0', help='gpu id')
opt = parser.parse_args()