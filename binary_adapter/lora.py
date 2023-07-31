import torch
from torch import nn
from utils import QLinear
import timm


def forward_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    delta_q = self.lora_q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    delta_v = self.lora_v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
    q, v = q + delta_q[0], v + delta_v[0]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
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
        self.act = nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        return x_up


def set_adapter(model, dim=32, s=1, bit=1):
    for layer in model.children():
        if type(layer) == timm.models.vision_transformer.Attention:
            layer.lora_q = Adapter(dim, bit)
            layer.lora_v = Adapter(dim, bit)
            layer.s = s
            bound_method = forward_attn.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
        elif len(list(layer.children())) != 0:
            set_adapter(layer, dim, s, bit)
