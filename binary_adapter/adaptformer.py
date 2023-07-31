import torch
from torch import nn
from utils import QLinear
import timm


def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.adapter_mlp(x) * self.s
    return x


class Adapter(nn.Module):
    def __init__(self, dim, bit):
        super().__init__()

        if bit == 32:
            self.adapter_down = nn.Linear(768, dim, bias=False)
            self.adapter_up = nn.Linear(dim, 768, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
        else:
            self.adapter_down = QLinear(768, dim, bit)
            self.adapter_up = QLinear(dim, 768, bit)
            nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        return x_up


def set_adapter(model, dim=32, s=1, bit=1):
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Block:
            layer.adapter_mlp = Adapter(dim, bit)
            layer.s = s
            bound_method = forward_block.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_adapter(layer, dim, s, bit)
