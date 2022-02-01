# Density estimation on low-dimensional manifolds: an inflation-deflation approach

*Christian Horvat and Jean-Pascal Pfister 2022*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](http://img.shields.io/badge/arXiv-2003.13913-B31B1B.svg)](https://arxiv.org/abs/2105.12152)

![illustration figure](inflation-deflation_logo.png)

We introduce a new method, named [inflation-deflation](https://arxiv.org/abs/2105.12152), to learn a density supported on a low-dimensional manifold. The idea is to exploit the ability of standard Normalizing Flows (NFs) to learn any density p(x) supported on entire embedding space. For that, we 

1. add noise to the data-manifold (inflation step),
2. train a standard NF (denoted as F in the figure above) to learn the inflated distribution,
3. scale the learned density to approximate p(x) (deflation step),
4. find conditions on the type of noise and on the manifold such that the approximation is exact (main Theorem).

Crucially, for step 4. the noise must be added in the manifolds normal space. Thus, the manifold must be known beforehand. However, we show that a standard Gaussian can be well used to approximate a Gaussian in the normal space whenever the manifold dimension d is much smaller than the dimensionality of the embedding space.

### Related Work

The inflation-deflation method ties in with the [Manifold Flow](https://github.com/johannbrehmer/manifold-flow),  introduced by Johann Brehmer and Kyle Cramner, to use NFs to learn densities supported on a low-dimensional manifold. 

### Experiments

We have shown that our method performes well on a wide range of manifolds. In the following table, we refer to the corresponding arguments to reproduce the results from the paper.

Manifold | Data dimension | Manifold dimension | Arguments to `train.py`, and `evaluate.py`
--- | --- | --- | ---
Sphere | 3 | 2 |  `--dataset gan2d`
Torus | 3 | 2|  `--dataset gan64d`
Hyperboloid | 3 | 2 |  `--dataset celeba`
Thin spiral | 2 | 1 |  `--dataset thin_spiral`
Swiss Roll | 3 | 2 |  `--dataset thin_spiral`
Hyperboloid-Sphere | 3 | 2| `--dataset thin_spiral`
Stiefel, SO(2)| 4 | 1| `--dataset thin_spiral`
MNIST digit 1 | 784 | ? `--dataset thin_spiral`

To use the model for your data, you need to create a simulator (see [experiments/datasets](experiments/datasets)), and add it to [experiments/datasets/__init__.py](experiments/datasets/__init__.py). If you have problems with that, please don't hesitate to contact us.


### Training & Evaluation

The configurations for the models and hyperparameter settings used in the paper can be found in [experiments/configs](experiments/configs). 

### Acknowledgements

We thank Johann Brehmer and Kyle Cramner for publishing their implementation of the [Manifold Flow](https://github.com/johannbrehmer/manifold-flow). Our code is partially based on their implementation. As standard NF, we have mainly used the [Block Neural Autoregressive Flow](https://github.com/nicola-decao/BNAF) which was re-implemented [here](https://github.com/kamenbliznashki/normalizing_flows) and served as inspiration for our main script.
We thank the community for fostering the open-source culture.
