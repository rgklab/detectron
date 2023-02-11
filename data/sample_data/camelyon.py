from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
import os.path
from functools import partial
from typing import Union

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from tqdm import tqdm

from data.core import split_dataset
from utils.config import Config


class QuantizedCamelyonDataset(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split
        self.labels = torch.load(os.path.join(root_dir, split, 'labels.pt'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = torch.load(os.path.join(self.root_dir, self.split, f'{index}.pt'))
        return torch.dequantize(data) / 255, self.labels[index]


def download_camelyon(root_dir):
    cfg = Config()
    Camelyon17Dataset(root_dir=root_dir, download=True)
    cfg.write_dataset_path('camelyon17', root_dir)


def camelyon(split='train', num_samples: Union[int, str] = 'all', quantized=True):
    cfg = Config()
    if quantized:
        # if you've run quantize_camelyon()
        ds = QuantizedCamelyonDataset(root_dir=cfg.get_dataset_path('camelyon_quantized'), split=split)
    else:
        # otherwise
        splits = {'train': 'train',
                  'val': 'id_val',
                  'harmful': 'test',
                  'not_harmful': 'val',
                  'test': 'id_val'}
        split_mapped = splits[split]
        try:
            dataset = Camelyon17Dataset(root_dir=cfg.get_dataset_path('camelyon17'), download=False)
        except FileNotFoundError:
            response = input('Camelyon17 dataset not found. Download? [y/n]: ')
            if response == 'y':
                path = input(f'Enter root directory for download or press '
                             f'Enter to use default [{cfg.get_dataset_path()}]: ')
                if path == '':
                    path = cfg.get_dataset_path()
                if not os.path.exists(p := os.path.realpath(path)):
                    raise NotADirectoryError(p)

                download_camelyon(path)
                dataset = Camelyon17Dataset(root_dir=cfg.get_dataset_path('camelyon17'), download=False)
            else:
                print('Download Camelyon17 manually and update `camelyon17` in config.json')
                raise FileNotFoundError('Camelyon17 dataset not found.')
        ds = dataset.get_subset(split_mapped, transform=Compose([Resize((224, 224)), ToTensor()]))

        if split == 'val' or split == 'test':
            val, test = split_dataset(ds, random_seed=0, num_samples=int(len(ds) * 0.5))
            if split == 'val':
                ds = val
            else:
                ds = test

    if num_samples != 'all':
        assert isinstance(num_samples, int)
        ds, _ = split_dataset(ds, num_samples=num_samples, random_seed=0)

    return ds


def write(root_dir, data):
    index, data = data
    torch.save((255 * data[0]).type(torch.uint8), os.path.join(root_dir, f'{index}.pt'))
    return data[1].item()


def quantize_camelyon(root_dir='/voyager/datasets/camelyon17'):
    os.makedirs(root_dir, exist_ok=True)
    for i, split in enumerate(('train', 'val', 'harmful', 'not_harmful', 'test')):
        print(f'Preprocessing ({split}) {i + 1}/5 ...')
        ds = camelyon(split)
        os.makedirs(os.path.join(root_dir, split), exist_ok=True)
        func = partial(write, os.path.join(root_dir, split))
        labels = [func(data) for data in enumerate(tqdm(ds, total=len(ds)))]
        torch.save(torch.tensor(labels), os.path.join(root_dir, split, 'labels.pt'))
    cfg = Config()
    cfg.write_dataset_path('camelyon17', root_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    args = parser.parse_args()
    quantize_camelyon(root_dir=args.root_dir)
