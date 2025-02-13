## Biomed-LDM
This repository contains the implementation of a \textbf{Domain Knowledge Guided Latent Diffusion Model for Biomedical Text Generation}, incorporating self-conditioning and residual-connected guidance for improved contextual text generation. Our approach enhances the generation of biomedical sentences by preserving domain-specific semantics through an iterative noising and denoising process.
To Train the Autoencoder:
Import the data in dataset directory.

### Train Autoencoder:
CUDA_VISIBLE_DEVICES=1 python pretrain_autoencoder.py

### Train BIO-LDM:
Import data in dataset directory.
Change the path of pretrained autoencoder model.pt 
CUDA_VISIBLE_DEVICES=2 python train_bioldm.py

### Acknowledgement
The code is built on top of https://github.com/justinlovelace/latent-diffusion-for-language.git and https://github.com/lucidrains.
