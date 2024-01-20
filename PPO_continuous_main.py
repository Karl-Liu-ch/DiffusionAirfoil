from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
import torch.nn as nn
from AgentNet import AttnBlock
import numpy as np
import platform
if platform.system().lower() == 'windows':
    from simulation_win import evaluate
elif platform.system().lower() == 'linux':
    from simulation import evaluate
from utils import derotate, Normalize
# from scipy.signal import savgol_filter
# from DiffusionAirfoil import Diff
# from DiffusionAirfoil1D import Diff as Diff1D
from DiffusionAirfoil1D_transform import Diff as Diff1D_transform
from utils import *
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if platform.system().lower() == 'linux':
    path = '/work3/s212645/DiffusionAirfoil/checkpoint/'
elif platform.system().lower() == 'windows':
    path = 'H:/深度学习/checkpoint/'
base_airfoil = np.loadtxt('BETTER/20150114-50 +2 d.dat', skiprows=1)
base_airfoil = interpolate(base_airfoil, 256, 3)

def normalize_af(af):
    af[:,0] -= af[:,0].min()
    af /= (af[:,0].max() - af[:,0].min())
    return af

class OptimEnv(gym.Env):
    def __init__(self, base_airfoil = base_airfoil, cl = 0.65, thickness = 0.065, maxsteps = 50, Re1 = 58000, Re2 = 400000, alpha=0.2, mode = '2d'):
        self.cl = cl
        self.base_airfoil = torch.from_numpy(base_airfoil).to(device)
        self.alpha = alpha
        self.thickness = thickness
        self.Re1 = Re1
        self.Re2 = Re2
        self.mode = mode
        self.thickness = thickness
        self.action_space = spaces.Box(high=1., low=-1., shape=(1,512), dtype=np.float32)
        self.observation_space = spaces.Box(high=1., low=-1., shape=(1,512), dtype=np.float32)
        self.steps = 0
        self.maxsteps = maxsteps
    
    def reset(self, seed=None, options=None):
        self.steps = 0
        successful = False
        while not successful:
            try:
                # self.airfoil = Diff1D_transform.sample(batch_size=1, channels=1).reshape(256, 2).cpu().numpy()
                # self.airfoil[:,1] = self.airfoil[:,1] * self.thickness / cal_thickness(self.airfoil)
                # self.airfoil[:,0] -= self.airfoil[:,0].min()
                # self.airfoil /= (self.airfoil[:,0].max() - self.airfoil[:,0].min())
                # self.airfoil = self.airfoil.reshape(1, 1, 512)
                # self.airfoil = torch.from_numpy(self.airfoil).to(device)

                self.airfoil = self.base_airfoil.reshape(1, 1, 512)
                self.state = self.airfoil.reshape(512)

                airfoil = self.airfoil.reshape(1, 256, 2)
                airfoil = airfoil.cpu().numpy()
                airfoil = airfoil[0]
                airfoil = derotate(airfoil)
                airfoil = normalize_af(airfoil)
                perf, CD, af, R = evaluate(airfoil, self.cl, Re1 = self.Re1, Re2 = self.Re2, lamda=5, check_thickness=False)
                if R is not np.nan:
                    successful = True
            except Exception as e:
                print(e)
        self.R_prev = R
        self.Rbl = R
        self.R = R
        info = {}
        return self.state.reshape(1,512).cpu().numpy(), info
    
    def step(self, action):
        self.steps += 1
        self.noise = torch.from_numpy(action).reshape([1,1,512]).to(device)
        af = Diff1D_transform.sample(batch_size=1, channels=1, noise = self.noise)
        af = af.reshape(256, 2).cpu().numpy()
        af[:,0] -= af[:,0].min()
        af /= (af[:,0].max() - af[:,0].min())
        af[:,1] = af[:,1] * self.thickness / cal_thickness(af)
        af = torch.from_numpy(af).to(device)
        af = af.reshape([1,1,512]).detach()
        
        self.airfoil = af  * self.alpha + (1-self.alpha) * self.airfoil
        airfoil = self.airfoil.reshape(1, 256, 2)
        airfoil = airfoil.cpu().numpy()
        airfoil = airfoil[0]
        airfoil = derotate(airfoil)
        airfoil = Normalize(airfoil)
        perf, CD, af, R = evaluate(airfoil, self.cl, Re1 = self.Re1, Re2 = self.Re2, lamda=5, check_thickness=False)
        if np.isnan(R):
            reward = -1
            reward_final = -1
        else:
            reward_final = 1.0 / R
            reward = (self.R_prev - R) * 10
            self.R_prev = R
        # print(reward)
        if R < self.R:
            self.R = R
            np.savetxt('results/airfoilPPO.dat', airfoil, header='airfoilPPO', comments="")
        self.state = self.airfoil.reshape(512)
        
        truncated = False
        done = False
        if R < 0.04 and perf > 40:
            done = True
            reward += 100
            truncated = False
        if self.steps > self.maxsteps:
            done = True
            truncated = True
            reward += reward_final
        reward_final = {'reward_final': reward_final}
        return self.state.reshape(1,512).detach().cpu().numpy(), reward, done, truncated, reward_final


class Net(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Net, self).__init__(observation_space, features_dim)
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Linear(n_input_channels, features_dim), 
            AttnBlock(features_dim, 16, 4),
        )

        self.linear = nn.Sequential(nn.Linear(features_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.permute(0,2,1))).mean(dim=1)

policy_kwargs = dict(
    features_extractor_class=Net,
    features_extractor_kwargs=dict(features_dim=512),
)

env = OptimEnv()
# It will check your custom environment and output additional warnings if needed
# check_env(env)
# model = PPO("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

path = '/work3/s212645/DiffusionAirfoil/PPO/stablebaseline_ppo'
try:
    model.load(path)
except Exception as e:
    print(e)
model.learn(total_timesteps=100_000)
model.save(path)
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100_000)

path = '/work3/s212645/DiffusionAirfoil/PPO/sac'
# model = SAC("MlpPolicy", env, verbose=1)
model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save(path)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
path = '/work3/s212645/DiffusionAirfoil/PPO/td3'
# model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model = TD3("MlpPolicy", env, policy_kwargs=policy_kwargs, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save(path)