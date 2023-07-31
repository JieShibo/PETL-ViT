import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import save, load, load_config, set_seed, QLinear, AverageMeter
import adaptformer
import lora


def train(args, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
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
            acc = test(vit, test_dl)
            if acc > args.best_acc:
                args.best_acc = acc
                save(args, model)
            pbar.set_description('best_acc ' + str(args.best_acc))

    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = AverageMeter()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out, y)
    return acc.result().item()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bit', type=int, default=1, choices=[1, 2, 4, 8, 32])
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='adaptformer',
                        choices=['adaptformer', 'adaptformer-bihead', 'lora', 'lora-bihead'])
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='.')
    parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--load_config', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    if args.eval or args.load_config:
        load_config(args)
    set_seed(args.seed)
    args.best_acc = 0
    vit = create_model(args.model, checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(args.dataset, normalize=False)

    if args.method == 'adaptformer':
        adaptformer.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))
    elif args.method == 'adaptformer-bihead':
        adaptformer.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.head = QLinear(768, get_classes_num(args.dataset), 1)
    elif args.method == 'lora':
        lora.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))
    elif args.method == 'lora-bihead':
        lora.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.head = QLinear(768, get_classes_num(args.dataset), 1)

    if not args.eval:
        trainable = []
        for n, p in vit.named_parameters():
            if ('adapter' in n or 'head' in n) and p.requires_grad:
                trainable.append(p)
            else:
                p.requires_grad = False
        opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        scheduler = CosineLRScheduler(opt, t_initial=100,
                                      warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
        vit = train(args, vit, train_dl, opt, scheduler, epoch=100)

    else:
        load(args, vit)
        args.best_acc = test(vit, test_dl)

    print('best_acc:', args.best_acc)
