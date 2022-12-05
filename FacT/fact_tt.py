import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
import numpy as np
import random
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
import yaml


def train(args, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(vit, test_dl)[1]
            if acc > args.best_acc:
                args.best_acc = acc
                save(args, model, acc, ep)
            pbar.set_description(str(acc) + '|' + str(args.best_acc))

    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    # pbar = tqdm(dl)
    model = model.cuda()
    for batch in dl:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 1)

    return acc.result()


def fact_forward_attn(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x)

    q = vit.FacTv(self.dp(self.q_FacTs(vit.FacTu(x))))
    k = vit.FacTv(self.dp(self.k_FacTs(vit.FacTu(x))))
    v = vit.FacTv(self.dp(self.v_FacTs(vit.FacTu(x))))

    qkv += torch.cat([q, k, v], dim=2) * self.s

    qkv = qkv.reshape(B, N, 3,
                      self.num_heads,
                      C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    proj = self.proj(x)
    proj += vit.FacTv(self.dp(self.proj_FacTs(vit.FacTu(x)))) * self.s
    x = self.proj_drop(proj)
    return x


def fact_forward_mlp(self, x):
    B, N, C = x.shape
    h = self.fc1(x)  # B n 4c
    # print(x.size(), h.size())
    h += vit.FacTv(self.dp(self.fc1_FacTs(vit.FacTu(x))).reshape(
        B, N, 4, self.dim)).reshape(
        B, N, 4 * C) * self.s
    x = self.act(h)
    x = self.drop(x)
    h = self.fc2(x)
    x = x.reshape(B, N, 4, C)
    h += vit.FacTv(self.dp(self.fc2_FacTs(vit.FacTu(x).reshape(
        B, N, 4 * self.dim)))) * self.s
    x = self.drop(h)
    return x


def set_FacT(model, dim=8, s=1):
    if type(model) == timm.models.vision_transformer.VisionTransformer:
        model.FacTu = nn.Linear(768, dim, bias=False)
        model.FacTv = nn.Linear(dim, 768, bias=False)
        nn.init.zeros_(model.FacTv.weight)
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Attention:
            _.q_FacTs = nn.Linear(dim, dim, bias=False)
            _.k_FacTs = nn.Linear(dim, dim, bias=False)
            _.v_FacTs = nn.Linear(dim, dim, bias=False)
            _.proj_FacTs = nn.Linear(dim, dim, bias=False)
            _.dp = nn.Dropout(0.1)
            _.s = s
            _.dim = dim
            bound_method = fact_forward_attn.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif type(_) == timm.models.layers.mlp.Mlp:
            _.fc1_FacTs = nn.Linear(dim, dim * 4, bias=False)
            _.fc2_FacTs = nn.Linear(4 * dim, dim, bias=False)
            _.dim = dim
            _.s = s
            _.dp = nn.Dropout(0.1)
            bound_method = fact_forward_mlp.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
        elif len(list(_.children())) != 0:
            set_FacT(_, dim, s)


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_config(dataset_name):
    with open('./configs/tt/%s.yaml' % (dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


@torch.no_grad()
def save(args, model, acc, ep):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in vit.named_parameters():
        if 'FacT' in n or 'head' in n:
            trainable[n] = p.data
    torch.save(trainable, 'models/tt/' + args.dataset + '.pt')
    with open('models/tt/' + args.dataset + '.log', 'w') as f:
        f.write(str(ep) + ' ' + str(acc))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    args = parser.parse_args()
    print(args)
    seed = args.seed
    set_seed(seed)
    name = args.dataset
    args.best_acc = 0
    vit = create_model(args.model, checkpoint_path='../ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(name)

    set_FacT(vit, dim=args.dim, s=args.scale)

    trainable = []
    vit.reset_classifier(get_classes_num(name))
    total_param = 0
    for n, p in vit.named_parameters():
        if 'FacT' in n or 'head' in n:
            trainable.append(p)
            if 'head' not in n:
                total_param += p.numel()
        else:
            p.requires_grad = False
    print('total_param', total_param)
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=100,
                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
    vit = train(args, vit, train_dl, opt, scheduler, epoch=100)
    print(args.best_acc)
