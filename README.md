# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

This repository contains a PyTorch-lightning implementation of [RandLA-Net](http://arxiv.org/abs/1911.11236).

## Preparation

1. Clone this repository

```sh
https://github.com/BigCiLeng/RandLA_pl.git
```

2. Install all Python dependencies

```sh
pip install -r requirements
```

***Common issue***: *the setup file from `torch-points-kernels` package needs PyTorch to be previously installed. You may thus need to install PyTorch first and then torch-points-kernels.*

4. Download a dataset and prepare it. We conducted experiments with [Semantic3D](http://www.semantic3d.net/) and [S3DIS](http://buildingparser.stanford.edu/dataset.html).

  To setup Semantic3D:

```sh
cd RandLA-Net-pytorch/utils
./download_semantic3d.sh
python3 prepare_semantic3d.py # Very slow operation
```

   To setup SDIS, register and then download the `zip` archive containing the files [here](http://buildingparser.stanford.edu/dataset.html#Download). We used the archive which contains only the 3D point clouds with ground truth annotations.

   Assuming that the archive is located in folder `RandLA-Net-pytorch/datasets`, then run:

```sh
cd RandLA-Net-pytorch/utils
python3 prepare_s3dis.py
```

5. Finally, in order to subsample the point clouds using a grid subsampling, run:
```sh
cd RandLA-Net-pytorch/utils/cpp_wrappers
./compile_wrappers.sh   # you might need to chmod +x before
cd ..
python3 subsample_data.py
```


## Usage

- Train a model

```sh
cd scripts/
sh train_xxx.sh
```

### Visualization

One can visualize the evolution of the loss with Wandb.

## Citation

This work implements the work presented in [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](http://arxiv.org/abs/1911.11236).

The original implementation (in TensorFlow 1) can be found [here](https://github.com/QingyongHu/RandLA-Net).

The Pytorch implementation [aRI0U/RandLA-Net-pytorch](https://github.com/aRI0U/RandLA-Net-pytorch).

To cite the original paper:
```
@article{RandLA-Net,
 arxivId = {1911.11236},
 author = {Hu, Qingyong and Yang, Bo and Xie, Linhai and Rosa, Stefano and Guo, Yulan and Wang, Zhihua and Trigoni, Niki and Markham, Andrew},
 eprint = {1911.11236},
 title = {{RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds}},
 url = {http://arxiv.org/abs/1911.11236},
 year = {2019}
}
```
<!--

## Warning

*This repository is still on update, and the segmentation results we reach with our implementation are for now not as good as the ones obtained by the original paper.*
