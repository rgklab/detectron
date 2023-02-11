from typing import Tuple

from torch.utils.data import Dataset, random_split
from torch import Tensor
import torch


def split_dataset(dataset: Dataset, num_samples: int, random_seed: int = 42) -> Tuple[Dataset, Dataset]:
    # noinspection PyTypeChecker
    return random_split(dataset, [num_samples, len(dataset) - num_samples],
                        generator=torch.Generator().manual_seed(random_seed))
