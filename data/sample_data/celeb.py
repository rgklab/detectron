import pandas as pd
from torch.utils.data import Dataset
from utils.config import Config
from os.path import join
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


def train_val_test_split(dataset: pd.DataFrame, seed: int, ratios=(.7, .2, .1)):
    assert abs(sum(ratios) - 1) < 1e-8
    length = len(dataset)
    # deterministic shuffle of rows
    dataset = dataset.sample(frac=1, random_state=seed)
    return {'train': dataset.iloc[:int(length * ratios[0])], 'val': dataset.iloc[int(length * ratios[0]):int(
        length * ratios[0] + length * ratios[1])],
            'iid_test': dataset.iloc[int(length * ratios[0] + length * ratios[1]):]}


class Celeb(Dataset):
    def __init__(self, split_attr='Pale_Skin', target_attr='Male', split='train'):
        assert split in ['train', 'val', 'iid_test', 'ood_test']
        root = Config().get_dataset_path('celeb')
        self.root = root
        df = pd.read_csv(join(root, 'celeba.csv'))
        iid, ood = df.query(f'{split_attr}==1'), df.query(f'{split_attr}==0')
        self.split = split
        if split == 'ood_test':
            df = ood
        else:
            df = train_val_test_split(iid, seed=0)[split]
        self.df = df
        self.target_attr = target_attr
        self.transform = Compose([CenterCrop(178), ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(join(self.root, 'img_align_celeba', row['File_Name']))
        image = self.transform(image)
        return image, row[self.target_attr]

    @staticmethod
    def process_attributes(root):
        with open(f'{root}/list_attr_celeba.txt', 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]

        df = pd.DataFrame(lines[1:], columns=lines[0])
        df = df.replace('-1', 0).replace('1', 1)
        df.to_csv(join(root, 'celeba.csv'), index=False)
