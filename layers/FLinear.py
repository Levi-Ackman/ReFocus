import torch
import torch.nn as nn
import torch.nn.functional as F
import math
        
class FLinear(nn.Module):
    def __init__(self, inp, out):
        super(FLinear, self).__init__()
        self.inp_size = inp // 2 + 1
        self.out_size = out // 2 + 1
        self.proj = nn.Linear(self.inp_size, self.out_size).to(torch.cfloat)
        
    def forward(self, x):
        return torch.fft.irfft(self.proj(torch.fft.rfft(x, dim=-1)), dim=-1)

    def initial(self):
        init_value = 1 / self.inp_size
        real_part = torch.full((self.out_size, self.inp_size), init_value)
        imaginary_part = torch.full((self.out_size, self.inp_size), init_value)
        complex_weights = torch.complex(real_part, imaginary_part)
        self.proj.weight = nn.Parameter(complex_weights)
        
class Filter(nn.Module):
    def __init__(self,channel=1,kernel_size=25):
        super(Filter, self).__init__()
        self.kernel_size=kernel_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, stride=1, 
                              padding=int(kernel_size//2), padding_mode='replicate', bias=True,groups=channel)
        self.conv.weight = nn.Parameter(
                (1 / kernel_size) * torch.ones([channel, 1, kernel_size]))
    def forward(self, inp):
        out = self.conv(inp.transpose(1,2)).transpose(1,2)
        return out

