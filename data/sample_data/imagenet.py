import os.path

import torchvision
from torchvision import models, datasets

from data.core import split_dataset
from utils.config import Config

DEFAULT_PROCESSING = models.ResNet50_Weights.DEFAULT.transforms()


def imagenet_v2(transform=DEFAULT_PROCESSING):
    cfg = Config()
    root = cfg.get_dataset_path('imagenetV2')
    return datasets.ImageFolder(root=root, transform=transform)


def imagenet(split='train', transform=DEFAULT_PROCESSING):
    cfg = Config()
    root = cfg.get_dataset_path('imagenet1k')
    test = False
    if split == 'test':
        test = True
        split = 'val'
    dataset = datasets.ImageNet(root=root, transform=transform, split=split)
    if split == 'val':
        val, test = split_dataset(dataset, num_samples=2048, random_seed=0)  # keep 2048 samples for validation
        if test:
            return test
        return val
    return dataset
