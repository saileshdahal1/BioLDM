## Domain Knowledge Guided Latent Diffusion Model for Biomedical Text Generation
This repository contains the implementation of a **Domain Knowledge Guided Latent Diffusion Model for Biomedical Text Generation**, incorporating an improved mechanism of guiding the latent diffusion model with domain knowledge to generate complex biomedical texts with syntactical coherence and fluency. 

### Installation
Clone the repo
cd Biomed-LDM

pip install -r requirements.txt

### Train Autoencoder:
cd autoencoder
Import data in dataset directory
python pretrain_autoencoder.py
The trained model will be stored at ./autoencoder_pretrained/dataset_name/model_train_time/model.pt

### Train BIO-LDM:
cd bio-ldm
Import data in dataset directory
edit parser.add_argument("--latent_model_path", type=str, default="../autoencoder_pretrained/dataset_name/model_train_time/model.pt") to change the path of pretrained autoencoder model.pt
python train_bioldm.py
The trained model will be stored at ./trained_bioldm/dataset_name/model_train_time/model.pt

### Sample:
python train_bioldm.py --eval --resume_dir ./trained_bioldm/dataset_name/model_train_time/model.pt

### Acknowledgement
The code is built on top of https://github.com/justinlovelace/latent-diffusion-for-language.git and https://github.com/lucidrains.
