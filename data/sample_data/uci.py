import os
from os.path import join

import numpy as np
import torch
import xgboost as xgb
from scipy.io import loadmat
from torch.utils.data import TensorDataset

from utils.config import Config

DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'nthread': 4,
    'tree_method': 'gpu_hist'
}


def uci_heart(split='train', dataset_format='torch', normalize=False):
    dirname = os.path.dirname(__file__)
    if split not in {'train', 'val', 'iid_test', 'ood_test'}:
        raise ValueError(f'Invalid split: {split}')

    cfg = Config()
    data = torch.load(join(dirname, 'uci_heart_torch.pt'))
    data = data[split]
    data, labels = list(zip(*data))
    data = torch.stack(data)
    labels = torch.tensor(labels)
    if normalize:
        min_ = data.min(0).values
        max_ = data.max(0).values
        data = (data - min_) / (max_ - min_)

    if dataset_format == 'torch':
        return TensorDataset(data, labels)

    elif dataset_format in {'xgboost', 'xbg'}:
        return xgb.DMatrix(data.numpy(), label=labels.numpy())

    elif dataset_format == 'numpy':
        return data.numpy(), labels.numpy()

    else:
        raise ValueError(f'Unknown dataset format: {dataset_format}')


def uci_heart_xgb(data=None):
    if data is None:
        data = uci_heart_numpy()

    data_dict = {}
    for key in ['train', 'iid_test', 'val', 'ood_test']:
        data_dict[key] = xgb.DMatrix(data[f'{key}_data'], label=data[f'{key}_labels'])
    return data_dict


def uci_heart_numpy():
    dirname = os.path.dirname(__file__)
    return loadmat(join(dirname, 'uci_heart_processed.mat'))


def train_and_test(params=DEFAULT_PARAMS, model_path='/voyager/datasets/UCI', n_seeds=10):
    data = uci_heart_xgb()
    # train <n_seeds> xgb models on the uci dataset and evaluate their in and out of distribution AUC
    iid_auc = []
    ood_auc = []

    for seed in range(n_seeds):
        path = os.path.join(model_path, f'uci_heart_{seed}.model')
        if os.path.exists(path):
            bst = xgb.Booster()
            bst.load_model(path)
            iid_auc.append(float(bst.eval(data['iid_test']).split(':')[1]))
            ood_auc.append(float(bst.eval(data['ood_test']).split(':')[1]))
        else:
            evallist = [(data['val'], 'eval'), (data['train'], 'train')]
            params['seed'] = seed
            num_round = 10
            bst = xgb.train(params, data['train'], num_round, evallist)

            iid_auc.append(float(bst.eval(data['iid_test']).split(':')[1]))
            ood_auc.append(float(bst.eval(data['ood_test']).split(':')[1]))

            bst.save_model(path)

    iid_auc = np.array(iid_auc)
    ood_auc = np.array(ood_auc)
    print(f'IID: {np.mean(iid_auc):.3f} \\pm {np.std(iid_auc):.3f}')
    print(f'OOD: {np.mean(ood_auc):.3f} \\pm {np.std(ood_auc):.3f}')
