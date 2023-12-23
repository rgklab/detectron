import os
from glob import glob
from typing import Optional

import torch
import xgboost as xgb
from tqdm import tqdm
from torchvision import models
from models.classifier import TorchvisionClassifier, MLP

WILDS_PATH = '/voyager/projects/tomginsberg/wilds_models'
CKPT_PATH = '/voyager/projects/tomginsberg/detectron_old/checkpoints'


def to_device(device):
    def to_device_fn(model):
        if device is None:
            return model
        return model.to(device)

    return to_device_fn


def camelyon_model_collection(return_names=False, device='cuda:1', wilds=False, eval_=True):
    to_device_fn = to_device(device)
    models = [to_device_fn(camelyon_model(seed=s, wilds=wilds)) for s in tqdm(range(10))]
    if eval_:
        for m in models:
            m.eval()
    if not return_names:
        return models
    return models, [f'camelyon17_{"erm_densenet121" if wilds else "resnet18_pretrained"}_seed{seed}' for seed in
                    range(10)]


def camelyon_model(seed=0, wilds=False):
    """
    Loads the pretrained model from the Camelyon dataset.
    """
    if wilds:

        ckpt = f'{WILDS_PATH}/camelyon17_erm_densenet121_seed{seed}/best_model.pth'
        model = TorchvisionClassifier(
            model='densenet121',
            out_features=2,
        )
        model.load_state_dict(torch.load(ckpt)['algorithm'])
    else:
        ckpt = glob(f'{CKPT_PATH}/camelyon/baselines/camelyon_resnet18_pretrained_seed{seed}/*.ckpt')[0]
        model = TorchvisionClassifier.load_from_checkpoint(ckpt)

    return model


def resnet18_trained_on_cifar10(ckp='cifar/cifar10_resnet18/epoch=197-step=77417.ckpt',
                                prefix: Optional[str] = CKPT_PATH):
    model = TorchvisionClassifier(out_features=10, model='resnet18')

    if isinstance(prefix, str):
        ckp = os.path.join(prefix, ckp)
    model.load_state_dict(torch.load(ckp)['state_dict'])

    return model


def resnet18_collection_trained_on_cifar10(return_names=False, device='cuda:1', eval_=True):
    checkpoints = glob(os.path.join(CKPT_PATH, 'cifar/baselines/*/*.ckpt'))
    to_device_fn = to_device(device)
    models = [to_device_fn(resnet18_trained_on_cifar10(c, prefix=None)) for c in
              tqdm(sorted(checkpoints, key=lambda x: int(x.split('/')[-2][-1])))]
    if eval_:
        for m in models:
            m.eval()
    if not return_names:
        return models
    return models, [f'cifar10_resnet18_pretrained_seed{seed}' for seed in
                    range(10)]


def mlp_trained_on_uci_heart(seed=0):
    if seed == 16:
        path = glob(f'{CKPT_PATH}/uci/baselines2/uci_mlp16_seed0/*.ckpt')[0]
    else:
        path = glob(f'{CKPT_PATH}/uci/baselines2/uci_mlp_seed{seed}/*.ckpt')[0]
    return MLP.load_from_checkpoint(path)


def mlp_collection_trained_on_uci_heart(return_names=False, device='cuda:1', eval_=True):
    to_device_fn = to_device(device)
    models = [to_device_fn(mlp_trained_on_uci_heart(seed)) for seed in range(10)]
    if eval_:
        for i, m in enumerate(models):
            models[i] = m.eval()
    if not return_names:
        return models
    return models, [f'uci_heart_seed{seed}' for seed in
                    range(10)]


def xgb_trained_on_uci_heart(seed=0):
    bst = xgb.Booster({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'nthread': 4,
        'tree_method': 'gpu_hist'
    })
    bst.load_model(f'models/UCI/uci_heart_{seed}.model')
    return bst


def xgb_collection_trained_on_uci_heart(return_names=False):
    models = [xgb_trained_on_uci_heart(seed) for seed in range(10)]

    if not return_names:
        return models
    return models, [f'uci_heart_{seed=}' for seed in
                    range(10)]


def resnet50_pretrained_on_imagenet1k():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    return model
