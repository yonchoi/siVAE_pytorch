# scalable, interpretable Variational Autoencoder (siVAE)

siVAE is an extension of traditional VAE that learns feature embeddings that guide the interpretation of the sample embeddings, in a manner analogous to factor loadings of factor analysis/PCA. siVAE is as powerful and nearly as fast to train as the standard VAE, but achieves full interpretability of the latent dimensions, as well as all hidden layers of the decoder. In addition, siVAE uses similarity between embeddings and gene regulatory networks to infer aspects of GRN.

This implementation of siVAE is simple pytorch implementation for easier installation

## Requirements
Operation systems: Linux
Programing language: Python 3.8


## Installation

siVAE requires installation of siVAE package as well as modified deepexplain from (https://github.com/marcoancona/DeepExplain), tensorflow-forward-ad (https://github.com/renmengye/tensorflow-forward-ad), and scvi-tools (https://github.com/YosefLab/scvi-tools)

Install siVAE by running the following command on the package file.

```
pip install git+https://github.com/yonchoi/siVAE_pytorch.git --quiet
```

The installation typically takes under an hour.

## Running the model

Example of applying siVAE on subset of fetal liver dataset is shown in colab notebook.

https://colab.research.google.com/drive/1xwWz2ZvGtvbuB2CZdBKXrdU3gA2UlZSR?usp=sharing
