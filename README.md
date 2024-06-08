# Generative Model Seminar 2024
Presentation slides are [here](https://waseda.app.box.com/folder/268416256524).

## Setup
Create vitual environment.
* Docker
```
docker-compose build
docker-compose up -d
docker-compose exec kasai-lab bash
```
* Conda
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

## コードの詳細
```
.
├── README.md
├── configs # 学習や推論に必要なハイパーパラメータ等を記述するためのファイルを保存
│   └── config_vae.json # VAEの学習時に使用
├── data
│   └── dataset.py # train_loader, test_loaderを作成するための関数が記述されている
├── logs # 学習途中の各種パラメータ等を保存
│   ├── checkpoint.pth.tar
├── main.py # モデルの推論に使用
├── models # モデルを記述
│   ├── ae.py
│   └── vae.py
├── outputs # モデルの出力を保存（後の分析に使用するため）
│   └── reconstructed.npy 
├── requirements.txt # インストールするpythonライブラリを列挙
├── save # 学習したモデルを保存
│   └── model_vae_mnist.pth
├── train_vae.py # モデルの学習に使用
└── utils
    ├── plot.py # 描画するための関数を記述
    └── utils.py # よく使う関数を記述
```

## Reference

## Citation