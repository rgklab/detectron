![](logo.svg)
___
**Official implementation of the ICLR 2023 paper [A Learning Based Hypothesis Test for Harmful Covariate Shift
](https://arxiv.org/abs/2212.02742)**

![](figure.png#gh-dark-mode-only)
![](dark_figure.png#gh-light-mode-only)

## Intro
We introduce the **Detectron**, a learning based hypothesis test for harmful covariate shift. Given a pretrained model $f: X\to Y$ and an unlabeled dataset $Q=\{x\}_i^n$ Detectron aims to automatically decide if $Q$ is similar enough to the $f(x)$'s training domain such that we can trust it to make reliable predictions.  

The algorithm works in two major steps:

First, we estimate the distribution of the test statistic $\phi$ which is computed as the *empirical disagreement rate* of a classifier $g(x)$ trained to explicitly disagree with a pretrained model $f(x)$ on i.i.d samples from the training set.  In practice, we create $g(x)$ by finetuning $f(x)$ using the _diagreement cross entropy_ defined formally in the paper. It is also important to limit the hypothesis space for $g(x)$ by forcing it to agree with $f(x)$ on the original training set while giving it a limited compute budget to prevent overfitting. Conceptually we can interpret $\phi$ as the degree of underspecification $f(x)$ admits on its training domain.

![](gif1.gif)

Next, we train another classifier $g^\star(x)$ in the exact same way as $g(x)$ but we use the unlabeled data $Q$. We detect covariate shift at a significance level $\alpha$ by comparing the empirical disagreement rate of $g^\star(x)$ on $Q$ (denoted $\phi^\star$) to the estimated distribution of $\phi$.

![](gif2.gif)

In our paper, we further show how to boost the power of the test using emsembling and by replacing the disagreement statistic $\phi$ with the related predictive entropy.  

## Benchmarks 
Test power at $5\%$ significance level for Detectron and baselines. We use a very small sample size of $|Q|=10$. Results for other samples sizes can be found in the paper.

| | CIFAR 10.1 [[Recht et al.]](https://arxiv.org/abs/1806.00451) |	Camelyon 17 |	UCI Heart Disease |
|---| :---: | :---: | :---: |
|Black Box Shift Detection [[Lipton et al.]](https://arxiv.org/abs/1802.03916)	|$.07\pm.03$ | $.05 \pm .02$ | $.12 \pm .03$ |
| Rel. Mahalanobis Distance [[Ren et al.]](https://arxiv.org/abs/2106.09022) | $.05 \pm .02$ | $.03 \pm .03$ | $.04 \pm .02$ |
|Deep Ensemble (Disagreement) [Ablation]	| $.05 \pm .02$ | $.03 \pm .03$ | $.04 \pm .02$ |
|Deep Ensemble (Entropy) [Ablation]	| $\mathit{.33 \pm .05}$ | $\mathit{.52 \pm .05}$ | $.68 \pm .05$ |
|Classifier Two Sample Test (CTST) [[Lopez-Paz et al.]](https://arxiv.org/abs/1610.06545)|	 $.03 \pm .02$  |  $.04 \pm .02$  |   $.04 \pm .02$ |
|Deep Kernel MMD [[Liu et al.]](https://arxiv.org/abs/2002.09116)	| $.24 \pm .04$ |  $.10 \pm .03$ |  $.05 \pm .02$ |
|H-Divergence [[Zhao et al.]](https://openreview.net/forum?id=KB5onONJIAU)|	$.02\pm .01$   |  $.05\pm .02$ |  $.04\pm .02$ |
|**Detectron (Disagreement)** [[Ours]](https://arxiv.org/abs/2212.02742) | $\mathbf{.37 \pm .05}$  |  $\underline{.54 \pm .05}$  |   $.83 \pm .04$ |
|**Detectron (Entropy)** [[Ours]](https://arxiv.org/abs/2212.02742) | $\underline{.35 \pm .05}$  |  $\mathbf{.56 \pm .05}$  |   $\mathbf{.92 \pm .03}$|

 The **best** result for each column is bolded, results that are within <ins>2% of the best</ins> are underlined and the _best baseline_ method is italicized.

## Setup

### Environment

`detectron` requires a working build of `pytorch` with the cudatoolkit enabled.
A simple environment setup using `conda` is provided below.

```shell
# create and activate conda environment using a python version >= 3.9
conda create -n detectron python=3.9
conda activate detectron

# install the latest stable release of pytorch (tested for >= 1.9.0)
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

# install additional dependencies with pip
pip install -r requirements.txt
```

### Datasets

We provide a simple config system to store dataset path mappings in the file `detectron/config.yml`

```yaml
datasets:
  default: /datasets
  cifar10_1: /datasets/cifar-10-1
  camelyon17: /datasets/camelyon17
```

for more information on downloading datasets see `detectron/data/sample_data/README.md`.

### Running Detectron

There is work in progress to package Detectron in a robust and easy to deploy system.
For now, all the code needed to reproduce our experiments is in located in the `experiments` directory
and can be run like the following example.

```shell
# run the cifar experiment using the standard config
# use python experiments.detectron_cifar --help for a documented list of options
❯ python -m experiments.detectron_cifar --run_name cifar
```

### Evaluating Detectron

The scratch files will write the output for each seed to a `.pt` file in a directory named `results/<run_name>`.

The script in `experiments/analysis.py` will read these files and produce a summary of the results for each test
described in the paper.

```shell
❯ python -m experiments.analysis --run_name cifar
# Output
→ 600 runs loaded
→ Running Disagreement Test
N = 10, 20, 50
TPR: .37 ± .05 AUC: 0.799 | TPR: .54 ± .05 AUC: 0.902 | TPR: .83 ± .04 AUC: 0.981
→ Running Entropy Test
N = 10, 20, 50
TPR: .35 ± .05 AUC: 0.712 | TPR: .56 ± .05 AUC: 0.866 | TPR: .92 ± .03 AUC: 0.981

```
