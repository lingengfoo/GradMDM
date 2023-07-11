# GradMDM
This repository contains the result and the sample code for the work:
GradMDM: Adversarial Attack on Dynamic Networks

# To perturb adversarial samples to SkipNet on the ImageNet validation dataset
## Prerequisite 
1. We support training with Pytorch 1.10.0. To install required packages
```
conda install pytorch=1.10 torchvision cudatoolkit=<the CUDA version you want> numpy
```

2. To prepare ImageNet dataset, please follow this [link](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

3. To prepare SkipNet pretrained model, please follow this [link](https://github.com/ucbdrive/skipnet/tree/master/imagenet).

## Training 
1. To train the adversarial samples with $gamma=100$, run
```
python -u train_gradmdm.py --model-type rl --gamma 100
```
2. To train the adversarial samples without accuracy drop, run
```
python -u train_gradmdm.py --model-type rl --gamma 100 --acc-maintain
```

## Citation
If you find our project useful in your research, please consider citing:
```
@article{pan2023gradmdm,
  title={GradMDM: Adversarial Attack on Dynamic Networks},
  author={Pan, Jianhong and Foo, Lin Geng and Zheng, Qichen and Fan, Zhipeng and Rahmani, Hossein and Ke, Qiuhong and Liu, Jun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
