# Generative Model Seminar 2024
Presentation slides are [here](https://en.wikipedia.org/wiki/Diffusion_model).

## Setup
Create vitual environment.
* For Docker User
```
docker-compose build
docker-compose up -d
docker-compose exec kasai-lab bash
```
* For Conda User
```
conda create -n gm-seminar python=3.12
pip install -r requirments.txt
```
## Dataset
* MNIST (VAE, ...)
* 

## Training
```
python train_vae.py --dataset mnist
```

## Generate Samples
```
python main.py --model_type vae --checkpoint save/model_vae_mnist.pth
```