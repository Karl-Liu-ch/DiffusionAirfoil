import argparse
parser = argparse.ArgumentParser(description="DiffusionAirfoil")
parser.add_argument('--method', type=str, default='1d')
parser.add_argument('--path', type=str, default='/work3/s212645/DiffusionAirfoil1DTransform/Airfoils1D/')
parser.add_argument('--agent', type=str, default='ppo')
parser.add_argument("--gpu_id", type=str, default='0', help='gpu id')
parser.add_argument('--n_runs', type=int, default=10, help='number of runs')
parser.add_argument('--n_eval', type=int, default=1000, help='number of total evaluations per run')
opt = parser.parse_args()