from torch.utils.data import Dataset
from torch import Tensor


class LabeledTensorDataset(Dataset):
    def __init__(self, data: Tensor, labels: Tensor | list[int], transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]
