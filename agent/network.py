import sys
sys.path.append('./')
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import functools
import torch.nn.functional as F
import math

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate) -> None:
        super().__init__()
        layer = [nn.Linear(in_channels, growth_rate),
                nn.ReLU()]
        self.Net = nn.Sequential(*layer)
        
    def forward(self, input):
        if len(input.shape) == 2:
            return torch.concat([input, self.Net(input)], dim=1)
        else:
            return torch.concat([input, self.Net(input)])
    
class DenseBlock(nn.Module):
    def __init__(self, 
                num_layers,
                num_input_features,
                growth_rate) -> None:
        super().__init__()
        model = []
        for i in range(num_layers):
            model += [DenseLayer(num_input_features + growth_rate * i, growth_rate =growth_rate)]
        self.Net = nn.Sequential(*model)
        
    def forward(self, input):
        return self.Net(input)

class DenseNet(nn.Module):
    def __init__(self, inchannels, outchannels, startch = 128, num_blocks = 6, growth_rate = 16, num_layers_per_block = 1):
        super(DenseNet, self).__init__()
        self.blocks = nn.ModuleList()
        
        in_channels = startch  # 初始输入通道数
        self.embed = nn.Sequential(
            nn.Linear(inchannels, in_channels), 
            nn.ReLU())
        for i in range(num_blocks):
            block = DenseBlock(num_layers_per_block, in_channels, growth_rate)
            self.blocks.append(block)
            in_channels += num_layers_per_block * growth_rate  # 更新输入通道数
        self.out = nn.Sequential(
            nn.Linear(in_channels, outchannels), 
            nn.Tanh())

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, hiddensize):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hiddensize),
            nn.ReLU(), 
            nn.Linear(hiddensize, in_channels))
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.net(x) + x)

class ResNet(nn.Module):
    def __init__(self, in_channels, outchannels, n_blocks = 2) -> None:
        super().__init__()
        hiddensize = 256
        blocks = []
        blocks.append(nn.Linear(in_channels, hiddensize))
        blocks.append(nn.ReLU())
        for i in range(n_blocks):
            blocks.append(ResBlock(hiddensize, hiddensize))
        blocks.append(nn.Linear(hiddensize, outchannels))
        blocks.append(nn.Tanh())
        self.net = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    model = DenseNet(19, 1).cuda()
    input = torch.rand([19]).cuda()
    output = model(input)
    print(output.shape)