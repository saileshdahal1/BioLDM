# Biomed-LDM
To Train the Autoencoder:
Import the data in dataset directory.

# Train Autoencoder:
CUDA_VISIBLE_DEVICES=1 python pretrain_autoencoder.py

# Train BIO-LDM:
CUDA_VISIBLE_DEVICES=2 python train_bioldm.py
