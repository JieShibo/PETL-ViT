import torch
import struct
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from scipy.stats import norm
import numpy as np
import vtab
import yaml
import os
import random


class QLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bits=1):
        super(QLinear, self).__init__(in_channels, out_channels, bias=False)
        self.q = self.quantize(bits)
        self.fake_quan = True
        self.bits = bits

    def forward(self, inputs):
        if not self.fake_quan:
            output = F.linear(inputs, self.weight, None)
            return output

        w = self.weight
        mean_w = w.mean()
        std_w = w.std()
        w = (w - mean_w) / (std_w + 1e-5)
        wb = self.q(w).data + w - w.data
        weight = wb * std_w + mean_w
        output = F.linear(inputs, weight, None)
        return output

    def dump(self):
        w = self.weight
        mean_w = w.mean()
        std_w = w.std()
        w = (w - mean_w) / (std_w + 1e-5)
        quantized = self.q(w).data.reshape(-1, 1)
        quantized = (quantized - self.code.reshape(1, -1)).abs().argmin(dim=1)
        byte_str = b''
        byte = 0
        for i in range(len(quantized)):
            if i % (8 // self.bits) == 0:
                byte = 0
            byte += quantized[i].item() * (2 ** self.bits) ** (i % (8 // self.bits))
            if i % (8 // self.bits) == (8 // self.bits) - 1:
                byte_str += byte.to_bytes(1, 'big')
        return byte_str, mean_w, std_w

    def load(self, byte_str, mean_w, std_w):
        self.fake_quan = False
        quantized = torch.zeros_like(self.weight.reshape(-1))
        for i in range(len(byte_str)):
            byte = byte_str[i]
            for j in range(8 // self.bits):
                quantized[i * (8 // self.bits) + j].data += self.code[byte % (2 ** self.bits)].data
                byte //= (2 ** self.bits)
        quantized = quantized * std_w + mean_w
        self.weight.data = quantized.reshape(*self.weight.size())

    def quantize(self, bit=1):
        j = 2 ** bit * 2
        ppf = np.array([0 for _ in range(1, j)])
        values = ppf[::2]
        ranges = ppf[1::2]
        for i in range(500):
            ranges = (values[1:] + values[:-1]) / 2
            pv = norm.cdf(ranges)
            pv = np.insert(pv, 0, 0)
            pv = np.insert(pv, len(pv), 1)
            pv = (pv[1:] + pv[:-1]) / 2
            values = norm.ppf(pv)
        value = torch.tensor(values).float()
        self.code = value
        pos = torch.tensor(ranges).float()
        delta = (value[1:] - value[:-1]) / 2

        def func(x):
            pos_ = pos.to(x.device)
            delta_ = delta.to(x.device)
            x = x.unsqueeze(dim=-1)
            x = x - pos_
            x = torch.sign(x)
            x *= delta_
            return x.sum(-1)

        return func


def adapter2byte(model):
    li = []
    for layer in model.children():
        if type(layer) == QLinear:
            li.append(layer.dump())
        elif len(list(layer.children())) != 0:
            li += adapter2byte(layer)
    return li


def byte2adapter(model, byte):
    for layer in model.children():
        if type(layer) == QLinear:
            size = layer.weight.data.numel()
            byte_str = byte[:size // (8 // layer.bits)]
            mean_w = struct.unpack('f', byte[size // (8 // layer.bits):size // (8 // layer.bits) + 4])[0]
            std_w = struct.unpack('f', byte[size // (8 // layer.bits) + 4:size // (8 // layer.bits) + 8])[0]
            layer.load(byte_str, mean_w, std_w)
            byte = byte[size // (8 // layer.bits) + 8:]
        elif len(list(layer.children())) != 0:
            byte = byte2adapter(layer, byte)
    return byte


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path, bihead=False):
    byte_list = adapter2byte(model)
    byte_str = b''
    for li in byte_list:
        byte_str += li[0]
        byte_str += struct.pack('f', li[1].item()) + struct.pack('f', li[2].item())

    if not bihead:
        head_weight = model.head.weight.data.reshape(-1)
        for i in range(len(head_weight)):
            byte_str += struct.pack('f', head_weight[i].item())
        head_bias = model.head.bias.data.reshape(-1)
        for i in range(len(head_bias)):
            byte_str += struct.pack('f', head_bias[i].item())
    with open(path, 'wb') as f:
        f.write(byte_str)


def save(args, model):
    model.eval()
    model = model.cpu()
    if not os.path.exists(os.path.join(args.model_path, 'models_bin/%s' % args.method)):
        os.makedirs(os.path.join(args.model_path, 'models_bin/%s' % args.method))
    if not os.path.exists(os.path.join(args.config_path, 'configs/%s' % args.method)):
        os.makedirs(os.path.join(args.config_path, 'configs/%s' % args.method))

    save_model(model, os.path.join(args.model_path, 'models_bin/%s/%s-bit%d-dim%d.bin' % (
        args.method, args.dataset, args.bit, args.dim)), bihead='bihead' in args.method)
    with open(
            os.path.join(args.config_path, 'configs/%s/%s-bit%d-dim%d.yaml' % (args.method, args.dataset, args.bit, args.dim)),
            'w') as f:
        config = {'dataset': args.dataset, 'num_class': vtab.get_classes_num(args.dataset), 'backbone': args.model,
                  'method': args.method, 'dim': args.dim, 'bit': args.bit, 'scale': args.scale}
        yaml.dump(config, f)


def load_model(model, path, bihead=False):
    with open(path, 'rb') as f:
        byte_str = f.read()
    byte_str = byte2adapter(model, byte_str)

    if not bihead:
        head_weight = torch.zeros_like(model.head.weight.data.reshape(-1))
        for i in range(len(head_weight)):
            head_weight[i].data += struct.unpack('f', byte_str[i * 4: i * 4 + 4])[0]
            model.head.weight.data = head_weight.data.reshape(*model.head.weight.data.size())
        byte_str = byte_str[len(head_weight) * 4:]
        head_bias = torch.zeros_like(model.head.bias.data.reshape(-1))
        for i in range(len(head_bias)):
            head_bias[i].data += struct.unpack('f', byte_str[i * 4: i * 4 + 4])[0]
            model.head.bias.data = head_bias.data.reshape(*model.head.bias.data.size())
        byte_str = byte_str[len(head_bias) * 4:]

    assert len(byte_str) == 0


def load_config(args):
    with open(
            os.path.join(args.config_path, 'configs/%s/%s-bit%d-dim%d.yaml' % (args.method, args.dataset, args.bit, args.dim)),
            'r') as f:
        args.scale = yaml.load(f, Loader=yaml.FullLoader)['scale']


def load(args, model):
    model.eval()
    model = model.cpu()
    load_model(model, os.path.join(args.model_path, 'models_bin/%s/%s-bit%d-dim%d.bin' % (
        args.method, args.dataset, args.bit, args.dim)), bihead='bihead' in args.method)


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, output, label):
        self.sum += (output.argmax(dim=1).view(-1) == label.view(-1)).long().sum()
        self.count += label.size(0)

    def result(self):
        return self.sum / self.count
