import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

def check_unitFraction(x):
    a = 1/x
    return a==int(a)

class S2Conv(nn.Module):
    # q is parameter that control output channel size.
    def __init__(self, c_in, shift_size, q=1, dilation=1):
        super(S2Conv, self).__init__()
        self.c_in = c_in
        self.shift_size = shift_size
        self.d = dilation
        self.q = q
        if q<1:
            self.conv = nn.Conv1d(int(self.shift_size*c_in*q), int(c_in*q), kernel_size=1, padding=0, stride=1, bias=False, groups=int(c_in*q))
        elif q>=1:
            self.conv = nn.Conv1d(self.shift_size*c_in, int(c_in*q), kernel_size=1, padding=0, stride=1, bias=False, groups=c_in)
        self.bn = nn.BatchNorm1d(int(c_in*q))

    def shift(self, x):
        out = torch.zeros_like(x)
        for k in range(self.c_in):
            sh_n = (k % self.shift_size) - (self.shift_size//2) # if shift_size=7, sh_n -> -3~3
            sh_n = sh_n * self.d
            if sh_n < 0:
                out[:, k, :sh_n] = x[:, k, -sh_n:]  # shift left
            elif sh_n >0:
                out[:, k, sh_n:] = x[:, k, :-sh_n]  # shift right
            else:
                out[:,k,:] = x[:,k,:] # not shift

        return out

    def forward(self, x):
        x = self.shift(x)
        n, c, t = x.size()
        pad = self.shift_size//2
        t_p = Variable(torch.zeros(n, pad, t))
        if self.shift_size % 2 ==0:
            pad = pad -1
        b_p = Variable(torch.zeros(n, pad, t))

        if next(self.parameters()).is_cuda:
            t_p = t_p.cuda()
            b_p = b_p.cuda()

        x = torch.cat((t_p, x, b_p), dim=1)
        if self.q <= 1:
            stride=int(1/self.q)
        else:
            stride = 1
        h = torch.cat([x[:,i*stride:i*stride+self.shift_size,:] for i in range((x.size(1)-self.shift_size+1)//stride)], dim=1)
        out = self.conv(h)
        out = self.bn(out)
        return F.relu_(out)

class BasicConv1d(nn.Module):
    def __init__(self, i_nc, o_nc, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(i_nc, o_nc, **kwargs),
            nn.BatchNorm1d(o_nc)
        )
    def forward(self, x):
        x = self.conv(x)
        return F.relu_(x)

class MTS_module(nn.Module):
    def __init__(self, i_nc, o_nc, n_b):
        super(MTS_module, self).__init__()
        self.o_ncs = self.split_branch(o_nc, n_b+1)
        self.l1 = BasicConv1d(i_nc, self.o_ncs[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.ls = nn.ModuleList()
        for i in range(n_b):
            shift_size = i*2+3
            self.ls.append(S2Conv(self.o_ncs[i+1], shift_size))
    def forward(self, x):
        x1 = self.l1(x)
        tx = x1[:,:self.o_ncs[1],:]
        out = torch.cat([conv(tx) for conv in self.ls], dim=1)
        outputs = torch.cat([x1, out], dim=1)
        return outputs
    def split_branch(self, total_channels, num_groups):
        import numpy as np
        split = [int(np.floor(total_channels / num_groups)) for _ in range(num_groups)]
        split[0] += total_channels - sum(split)
        return split

class MTS_ConvNet(nn.Module):
    def __init__(self, in_nc, class_num, segment_size):
        super(MTS_ConvNet, self).__init__()
        n_fs = 64
        self.T = segment_size
        n_b = 3

        self.stem = BasicConv1d(in_nc, n_fs, kernel_size=1, stride=1, padding=1,bias=False)

        if in_nc == 113: # in case of OPPORTUNITY
            self.conv1 = MTS_module(n_fs, n_fs*2, n_b)
            self.conv2 = MTS_module(n_fs*2, n_fs, n_b)
        else:
            self.conv1 = MTS_module(n_fs, n_fs, n_b)
            self.conv2 = MTS_module(n_fs, n_fs, n_b)
        self.conv3 = MTS_module(n_fs, n_fs, n_b)
        self.conv4 = MTS_module(n_fs, n_fs*2, n_b)
        self.conv5 = MTS_module(n_fs*2, n_fs*4, n_b)

        self.fc = nn.Linear(n_fs*4, class_num)

    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  
        x = F.adaptive_max_pool1d(x,self.T//2)
        x = self.conv4(x)  
        x = F.adaptive_max_pool1d(x,self.T//4)
        x = self.conv5(x)
        x = F.adaptive_avg_pool1d(x,1)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x
