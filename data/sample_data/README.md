# Instructions for installing datasets used in this project

### CIFAR-10

```python
import torchvision

torchvision.datasets.CIFAR10(root='path/for/install', download=True)
# add the following line to the config.yml file
# datasets:
#     cifar10: path/for/install
```

### CIFAR 10.1

```shell
cd path/for/install
wget https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_data.npy
wget https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_data.npy
# add the following line to the config.yml file
# datasets:
#     cifar10_1: path/for/install
```

### Camelyon17

```python
# make sure wilds is installed (pip install wilds)
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

Camelyon17Dataset(root_dir='path/for/install', download=True)
# add the following line to the config.yml file
# datasets:
#     camelyon17: path/for/install
```

We also create a 8bit quantized version of Camelyon17 for use in the experiments for significantly faster dataloading.
Use the following utility script to create the quantized version of the dataset.

```shell
python -m data.sample_data.camelyon --root_dir path/for/install
# add the following line to the config.yml file
# datasets:
#     camelyon17: path/for/install
```

### UCI Heart Disease

Our preprocessed version of the UCI heart disease dataset is available directly in this repo
at `data/sampledata/uci_heart_preprocessed.mat`.
The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). 