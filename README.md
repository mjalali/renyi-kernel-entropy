# RKE score

[Paper: An Information-Theoretic Evaluation of Generative Models in Learning Multi-modal Distributions](https://neurips.cc/virtual/2023/poster/71754)

[Mohammad Jalai](https://mjalali.github.io/) <sup>1*</sup>, [Cheuk Ting Li](https://www.ie.cuhk.edu.hk/faculty/li-cheuk-ting/) <sup>2</sup>,
[Farzan Farnia](https://www.cse.cuhk.edu.hk/people/faculty/farzan-farnia/) <sup>2</sup>

<sup>1</sup> <sub>**Isfahan University of Technology (IUT)**</sub>, <sup>2</sup> <sub>**The Chinese University of Hong Kong (CUHK)**</sub>

<sub>* Work done during an internship at CUHK</sub>

## 1. Background

### Abstract
The evaluation of generative models has received significant attention in the machine learning community.
When applied to a multi-modal distribution which is common among image datasets, an intuitive evaluation criterion is the number of modes captured by the generative model. 
While several scores have been proposed to evaluate the quality and diversity of a model's generated data, the correspondence between existing scores and the number of modes in the distribution is unclear. 
In this work, we propose an information-theoretic diversity evaluation method for multi-modal underlying distributions. We define the **R\'enyi Kernel Entropy (RKE)** as an evaluation score based on quantum information theory to measure the number of modes in generated samples. To interpret the proposed evaluation method, we show that the RKE score can output the number of modes of a mixture of sub-Gaussian components. We also prove estimation error bounds for estimating the RKE score from limited data, suggesting a fast convergence of the empirical RKE score to the score for the underlying data distribution. Utilizing the RKE score, we conduct an extensive evaluation of state-of-the-art generative models over standard image datasets. The numerical results indicate that while the recent algorithms for training generative models manage to improve the mode-based diversity over the earlier architectures, they remain incapable of capturing the full diversity of real data. Our empirical results provide a ranking of widely-used generative models based on the RKE score of their generated samples.

### R'enyi Kernel Entropy Mode Count (RKE) and Relative R'enyi Kernel Entropy (RRKE)

#### Formulation

<a href="https://latex.codecogs.com/svg.image?\mathrm{RKE}_2(\mathbf{X})=-\log\Bigl(\mathbb{E}_{X,X'\stackrel{\mathrm{iid}}{\sim}P_X}\bigl[k^2(\mathbf{X},\mathbf{X}')\bigr]\Bigr)=\,-\log\biggl(\frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}k^2(\mathbf{x}_i,\mathbf{x}_j)\biggr)" target="_blank"><img src="https://latex.codecogs.com/svg.image?\mathrm{RKE}_2(\mathbf{X})=-\log\Bigl(\mathbb{E}_{X,X'\stackrel{\mathrm{iid}}{\sim}P_X}\bigl[k^2(\mathbf{X},\mathbf{X}')\bigr]\Bigr)=\,-\log\biggl(\frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}k^2(\mathbf{x}_i,\mathbf{x}_j)\biggr)" title="RKE_2(X)" /></a>

<a href="https://latex.codecogs.com/svg.image?\widehat{\mathrm{RRKE}}_{\frac{1}{2}}(\mathbf{X},\mathbf{Y})=-\log\Bigl(\bigl\Vert&space;K_{XY}\bigr\Vert^{2}_{\mathrm{nuc}}\Bigr)" target="_blank"><img src="https://latex.codecogs.com/svg.image?\widehat{\mathrm{RRKE}}_{\frac{1}{2}}(\mathbf{X},\mathbf{Y})=-\log\Bigl(\bigl\Vert&space;K_{XY}\bigr\Vert^{2}_{\mathrm{nuc}}\Bigr)" title="RKE_2(X)" /></a>


#### Toy example: Gaussian Distributions
<p align="center">
    <img src=https://github.com/mjalali/renyi-kernel-entropy/tree/main/assets/figures/gaussians-gans.png> 
</p>

#### Evaluation of Generative Models



## 2. Usage

### Installation

Using PIP

```shell
pip install rke
```

Manually
```shell
git clone https://github.com/mjalali/renyi-kernel-entropy-score
python setup.py install
```

### Example

```python
import numpy as np
from rke import RKE


num_real_samples = num_fake_samples = 10000
feature_dim = 1000

real_features = np.random.normal(loc=0.0, scale=1.0,
                                 size=[num_real_samples, feature_dim])

fake_features = np.random.normal(loc=0.0, scale=1.0,
                                 size=[num_fake_samples, feature_dim])

kernel = RKE(kernel_bandwidth=[0.2, 0.3, 0.4])


print(kernel.compute_rke_mc)
print(kernel.compute_rrke)
```

### Guide to evaluate your model
You can evaluate your model using different feature extractors (mostly used: inceptionV3)


## 3. Cite our work
