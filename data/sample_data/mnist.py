from os.path import join

import torch
import torchvision

from data.core import split_dataset
from utils.config import Config

import pickle

from torch.utils.data import TensorDataset, Dataset


class SpuriousMNIST(Dataset):
    def __int__(self, source_data: Dataset):
        self.source_data = source_data

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, idx):
        x, y = self.source_data[idx]
        x[0, 0] = .5
        return x, y


def mnist(split='train'):
    assert split in {'train', 'val', 'iid_test', 'ood_test'}
    cfg = Config()
    train = (split == 'train') or (split == 'val')
    dataset = torchvision.datasets.MNIST(
        root=cfg.get_dataset_path('mnist'),
        train=train,
        download=False,
        transform=torchvision.transforms.ToTensor()
    )
    if train:
        train, val = split_dataset(dataset, len(dataset) - 10000, random_seed=0)
        if split == 'train':
            return train
        return val
    else:
        iid_test, ood_test = split_dataset(dataset, len(dataset) // 2, random_seed=0)
        if split == 'iid_test':
            return iid_test
        return SpuriousMNIST(ood_test)


def fake_mnist():
    # gdown https://drive.google.com/uc?id=13JpGbp7PEm4PfZ6VeqpFiy0lHfVpy5Z5
    cfg = Config()
    file = join(cfg.get_dataset_path('fake_mnist'), 'Fake_MNIST_data_EP100_N10000.pckl')
    data = (pickle.load(open(file, 'rb'))[0] + 1) / 2
    data = torch.from_numpy(data).float()
    return TensorDataset(data)
