# Instructions for installing datasets used in this project

### CIFAR-10

```python
import torchvision

torchvision.datasets.CIFAR10(root='path/for/install', download=True)
# add the following line to the config.yml file
# datasets:
#     cifar10: 'path/for/install'
```

### CIFAR 10.1

```shell
cd /path/to/desired/cifar10_1/root
wget https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_data.npy
wget https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_data.npy
```

### Camelyon17

### UCI Heart Disease

### CelebA

### Fake MNIST