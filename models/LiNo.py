import torch
import torch.nn as nn
from layers.FLinear import FLinear,Filter
from layers.Encoder import Encoder
from layers.RevIN import RevIN
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.revin_layer = RevIN(configs.enc_in)
        self.pred_len=configs.pred_len
        self.Encoders=nn.ModuleList([Encoder(configs) for _ in range(configs.layers)]) 
        self.embed = FLinear(configs.seq_len, configs.d_model) 
        self.projection = FLinear(configs.d_model, configs.pred_len)
        self.Filter=Filter(configs.enc_in,kernel_size=25)
        self.beta=configs.beta
        if configs.initial:
            self.projection.initial()
    def forward(self, x_enc,emb=None):
        # [B, T, N]
        x_enc=self.revin_layer(x_enc,'norm')
        x_enc=x_enc-self.beta*self.Filter(x_enc)
        x_embed=self.embed(x_enc.transpose(1,2))
        for encoder in self.Encoders:
            x_embed=encoder(x_embed)
        pred=self.projection(x_embed).transpose(1,2)
        pred=self.revin_layer(pred,'denorm')
        # [B, F, N]
        # [B, F, N]
        if emb is None: return pred
        else:  return pred,x_embed