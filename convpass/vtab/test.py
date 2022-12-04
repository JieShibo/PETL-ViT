import torch
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from argparse import ArgumentParser
from vtab import *
from utils import *
from convpass import set_Convpass


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    pbar = tqdm(dl)
    model = model.cuda()
    for batch in pbar:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 0)

    return acc.result()[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='convpass')
    args = parser.parse_args()
    print(args)
    config = get_config(args.method, args.dataset)
    model = create_model(args.model, checkpoint_path='../ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(args.dataset)

    set_Convpass(model, args.method, dim=8, s=config['scale'], xavier_init=config['xavier_init'])

    trainable = []
    model.reset_classifier(config['class_num'])
    
    model = load(args.method, config['name'], model)
    acc = test(model, test_dl)
    print('Accuracy:', acc)
    
