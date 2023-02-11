from __future__ import annotations

import csv
from typing import Optional, Sized, Iterator, SupportsIndex

import pytorch_lightning as pl
from torch import BoolTensor
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
from torch.utils.data.sampler import T_co


class MaskedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, mask=True, pseudo_labels: Optional[list] = None):
        self.dataset = dataset
        self.mask = mask
        self.pseudo_labels = pseudo_labels
        # noinspection PyTypeChecker
        self.indices = list(range(len(self.dataset)))
        self.original_indices = self.indices

    def __getitem__(self, index):
        index = self.indices[index]
        data = self.dataset[index]
        x = data[0]
        y = data[1]
        if self.pseudo_labels is not None:
            y_hat = self.pseudo_labels[index]
        else:
            y_hat = y
        return x, y_hat, y, self.mask

    def refine(self, mask: BoolTensor):
        """
        Refine the dataset by removing the samples that are masked.
        :param mask:
        """
        self.indices = [self.indices[i] for i, m in enumerate(mask) if m]

    def original(self):
        return MaskedDataset(self.dataset, mask=False, pseudo_labels=self.pseudo_labels)

    def rest_index(self):
        self.indices = self.original_indices

    def save_indices(self, path: str):
        # TODO testme
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.indices)

    def load_indices(self, path: str):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            self.indices = list(map(int, next(reader)))

    def __len__(self):
        return len(self.indices)


class PQSampler(BatchSampler):
    def __init__(self, p: int | Sized, q: int | Sized, batch_size: int, q_samples_per_batch: int = -1):

        self.p_len, self.q_len = [PQSampler._len_if_sized(x) for x in [p, q]]
        assert q_samples_per_batch <= self.q_len, f'q_samples_per_batch ({q_samples_per_batch}) must be <= len(q) ({self.q_len})'
        n_p = list(range(self.p_len))
        n_q = [x + 1 + n_p[-1] for x in range(self.q_len)]
        self.q_sampler = n_q
        self.length = self.p_len // (batch_size - self.q_len)
        super().__init__(sampler=BatchSampler(RandomSampler(n_p), batch_size - self.q_len, drop_last=True),
                         batch_size=batch_size, drop_last=True)

    def __iter__(self) -> Iterator[T_co]:
        p_iter = iter(self.sampler)
        while True:
            try:
                p_batch = next(p_iter)
            except StopIteration:
                return
            yield p_batch + self.q_sampler

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _len_if_sized(obj: int | Sized) -> int:
        if isinstance(obj, int):
            return obj
        else:
            return len(obj)


class DetectronLoader(pl.LightningDataModule):
    def __init__(self,
                 p_train: Dataset | list,
                 p_val: Dataset | list,
                 q: Dataset | list,
                 p_train_pseudo_labels: list,
                 p_val_pseudo_labels: list,
                 q_pseudo_labels: list,
                 batch_size: int = 512,
                 num_workers: int = 16,
                 ):
        super().__init__()

        self.train = MaskedDataset(p_train, mask=True, pseudo_labels=p_train_pseudo_labels)
        self.val = MaskedDataset(p_val, mask=True, pseudo_labels=p_val_pseudo_labels)
        self.q = MaskedDataset(q, mask=False, pseudo_labels=q_pseudo_labels)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pq_sampler = PQSampler(
            p=self.train, q=self.q,
            batch_size=self.batch_size,
        )
        self.pq_dataset = self.train + self.q

    def save_indices(self, path: str):
        self.q.save_indices(path)

    def load_indices(self, path: str):
        self.q.load_indices(path)

    def refine(self, mask: BoolTensor, verbose=True, symbol='Q'):
        count = mask.float().sum().long().item()
        if verbose:
            print(f'|{symbol}| ({len(mask)} → {count})')

        self.q.refine(mask)
        self.pq_sampler = PQSampler(
            p=self.train, q=self.q,
            batch_size=self.batch_size,
        )
        self.pq_dataset = self.train + self.q
        return count

    def train_dataloader(self):
        return DataLoader(
            self.pq_dataset,
            batch_sampler=self.pq_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.q,
            shuffle=False,
            batch_size=len(self.q),
            num_workers=self.num_workers,
        )

    def p_train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def q_dataloader(self):
        original_q = self.q.original()
        return DataLoader(
            original_q,
            shuffle=False,
            batch_size=len(original_q),
            num_workers=self.num_workers,
        )
