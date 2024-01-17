from utils_win import cal_afs_polar
import argparse
parser = argparse.ArgumentParser(description="DiffusionAirfoil")
parser.add_argument('--min', type=int, default=500)
parser.add_argument('--max', type=int, default=400000)
parser.add_argument('--step', type=int, default=500)
parser.add_argument('--path', type=str, default='F3K_airfoils/')
args = parser.parse_args()

if __name__ == '__main__':
    min = args.min
    max = args.max
    step = args.step 
    path = args.path
    cal_afs_polar(re_min = min, re_max = max, re_step = step, path = path)