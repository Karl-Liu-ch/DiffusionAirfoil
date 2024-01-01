import sys
sys.path.append('/')
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, inchannels, hiddensize):
        super().__init__()
        self.linear1 = nn.Linear(inchannels,hiddensize)
        self.linear2 = nn.Linear(hiddensize,inchannels)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.num_heads = n_heads
        self.heads_dim = dim // n_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.scale = nn.Parameter(torch.ones(n_heads, 1, 1))
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
        )
        
    def forward(self, x):
        # x: [b, c, c]
        b, n, c = x.shape
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.view(b, n, c)
        x = self.to_out(x)
        return x
        
class AttnBlock(nn.Module):
    def __init__(self, 
            dim,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        assert dim % heads == 0
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MultiHeadAttention(dim=dim, n_heads=heads),
                FFN(dim, dim), 
                nn.LayerNorm(dim)
                ]))

    def forward(self, x):
        for (attn, ff, ln) in self.blocks:
            x = ln(attn(x) + x)
            x = ln(ff(x) + x)
        out = x
        return out

class Network(nn.Module):
    def __init__(self, inchannels, outchannels, hiddensize = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inchannels, hiddensize), 
            AttnBlock(hiddensize, 16, 6),
            nn.Linear(hiddensize, outchannels), 
            nn.Tanh()
        )
        
    def forward(self, x):
        B, C = x.shape
        x = x.unsqueeze(dim=1).expand(B, C, C)
        x = self.net(x)
        x = x.mean(dim=1)
        return (x + 1.0) / 2.0

BATCH = 2 ** 10
max_prop = 1000
coaxialeff = 450000/(max_prop**2)
coefthrust = 5.7199 / (max_prop**2)
coeftorque = 0.4008 / (max_prop**2)
k = coefthrust
b = coeftorque
Lx = 0.36 / 2
Ly = 0.26 / 2 
Lz = 0.0221
controlmatrix = np.array([[Lx*coefthrust, Lx*coefthrust, -Lx*coefthrust, -Lx*coefthrust, -Lx*coefthrust, -Lx*coefthrust, Lx*coefthrust, Lx*coefthrust],
    [-Ly*coefthrust, -Ly*coefthrust, -Ly*coefthrust, -Ly*coefthrust, Ly*coefthrust, Ly*coefthrust, Ly*coefthrust, Ly*coefthrust],
    [coeftorque, -coeftorque, -coeftorque, coeftorque, coeftorque, -coeftorque, -coeftorque, coeftorque],
    [coefthrust,coefthrust,coefthrust,coefthrust,coefthrust,coefthrust,coefthrust,coefthrust]], dtype=np.float32)

# controlmatrix = torch.from_numpy(controlmatrix).cuda()
# controlmatrix = controlmatrix.unsqueeze(dim=0).expand(BATCH, 4, 8)
# outstd = torch.tensor([max_prop ** 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2]).cuda()
# outmean = torch.tensor([max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2, max_prop ** 2 / 2]).cuda()

def train(params_path='agent/controlalocation/model.pth'):
    model = Network(6, 8).cuda()
    try:
        model.load_state_dict(torch.load(params_path))
        print('model loaded')
    except:
        print('model loading failed')
    learning_rate = 1e-4
    std = torch.tensor([0.5, 0.5, 2, 5]).cuda()
    mean = torch.tensor([0, 0, 0, 20]).cuda()
    d_scale = 0.43/2.0
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1000, eta_min=1e-6)
    for i in range(1000):
        for k in range(100):
            optim.zero_grad()
            lr = optim.param_groups[0]['lr']
            target = torch.randn(size=[BATCH, 4]).cuda()
            alpha = torch.randn(size=[BATCH, 1]).cuda()
            d = torch.randn(size=[BATCH, 1]).cuda()
            target = torch.concat([target, alpha, d], dim=1)
            output = model(target)
            output = output * (max_prop ** 2)
            output = output.unsqueeze(dim=-1)
            controlmatrix = mixmatrix(Lx, Ly, Lz, k, b, d, alpha, BATCH)
            actuator = controlmatrix @ output
            actuator = actuator.squeeze(dim=-1)
            actuator = (actuator - mean) / std
            loss = nn.MSELoss()(target[:,:4], actuator)
            # loss += nn.L1Loss()(output, torch.zeros_like(output)) * 1e-8
            loss.backward()
            optim.step()
        scheduler.step()
        print('epoch: ', i, ' loss: ', loss.item(), ' learning rate: ', lr)
        torch.save(model.state_dict(), params_path)

def mixmatrix(Lx, Ly, Lz, k, b, d, alpha, BATCH):
    d_scale = 0.43/2.0
    dcg = d * d_scale + d_scale
    c = torch.cos(alpha*torch.pi/2.0)
    s = torch.sin(alpha*torch.pi/2.0)
    matrixfront = torch.Tensor([[Lx*k, Lx*k, -Lx*k, -Lx*k], 
                         [-Ly*k, -Ly*k, -Ly*k, -Ly*k], 
                         [b, -b, -b, b], 
                         [k, k, k, k]]).type(torch.float32).cuda()
    matrixfront = matrixfront.unsqueeze(dim=0).expand(BATCH, 4, 4)
    d_bias = torch.Tensor([[0,0,0,0], 
                            [1,1,1,1], 
                            [0,0,0,0], 
                            [0,0,0,0]]).type(torch.float32).cuda()
    d_bias = d_bias.unsqueeze(dim=0).expand(BATCH, 4, 4)
    d_bias = d_bias * dcg.unsqueeze(dim=1).expand(BATCH, 4, 4)
    matrixfront = matrixfront + d_bias * k
    matrixback = torch.Tensor([[-Lx*k, -Lx*k, Lx*k, Lx*k], 
                         [Ly*k, Ly*k, Ly*k, Ly*k], 
                         [b, -b, -b, b], 
                         [k, k, k, k]]).type(torch.float32).cuda()
    matrixback = matrixback.unsqueeze(dim=0).expand(BATCH, 4, 4)
    matrixback = matrixback * c.unsqueeze(dim=1).expand(BATCH, 4, 4)
    matrixback = matrixback + d_bias * k * c.unsqueeze(dim=1).expand(BATCH, 4, 4)
    bias_roll = torch.Tensor([[-1,-1,-1,-1], 
                            [0,0,0,0], 
                            [0,0,0,0], 
                            [0,0,0,0]]).type(torch.float32).cuda()
    bias_roll = bias_roll.unsqueeze(dim=0).expand(BATCH, 4, 4)
    bias_roll = bias_roll * s.unsqueeze(dim=1).expand(BATCH, 4, 4) * b
    matrixback = matrixback + bias_roll
    bias_z = torch.Tensor([[0,0,0,0], 
                        [-1,-1,-1,-1], 
                        [0,0,0,0], 
                        [0,0,0,0]]).type(torch.float32).cuda()
    bias_z = bias_z.unsqueeze(dim=0).expand(BATCH, 4, 4)
    bias_z = bias_z * s.unsqueeze(dim=1).expand(BATCH, 4, 4) * Lz * k
    matrixback = matrixback + bias_z
    controlmatrix = torch.concat([matrixfront, matrixback], dim=2)
    return controlmatrix

def prediction(params_path, data):
    model = Network(6, 8).cuda()
    model.load_state_dict(torch.load(params_path))
    model.eval()
    data = np.array(data)
    data = (data - np.array(([0, 0, 0, 20, 0, 0.43/2.0]))) / np.array([0.5, 0.5, 2, 5, np.pi, 0.43/2.0])
    data = torch.tensor(data, dtype=torch.float32).cuda()
    data = data.unsqueeze(dim=0)

    with torch.no_grad():
        pred = model(data)
    pred = pred.detach().cpu().numpy().squeeze()
    pred = np.ascontiguousarray(pred.T) * (max_prop ** 2)

    return pred

if __name__ == '__main__':
    train()
    params_path = 'agent/controlalocation/model.pth'
    data = [0.0791,-0.0162,-0.1046,22]
    pred = prediction(params_path, data)
    print(pred)
    actuator = controlmatrix.cpu().numpy() @ pred
    print(actuator)