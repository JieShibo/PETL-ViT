# Convpass

Source code of "Convolutional Bypasses Are Better Vision Transformer Adapters" 

[arxiv](http://arxiv.org/abs/2207.07039) 2022/07

## Requirements
Python == 3.8

torch == 1.10.0

torchvision == 0.11.1

timm == 0.4.12

avalanche-lib == 0.1.0

## Data Preparation

To download the datasets, please refer to https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation (thank [ZhangYuanhan-AI](https://github.com/ZhangYuanhan-AI) for their code). Then move the dataset folders to `./vtab/data/`. 

## Usage
### Pretrained Model
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `./ViT-B_16.npz`

### Train & Save
```sh
cd vtab
bash run.sh
```
Only the trainable parameters are saved (1.4MB/task on average).

### Test
```sh
python test.py --method convpass --dataset <dataset-name>
python test.py --method convpass_attn --dataset <dataset-name>
```
### Performance (seed = 42)
| Natural | cifar | caltech101 | dtd | oxford_flowers102 | oxford_iiit_pet | svhn | sun397 | Average |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|Full FT|68.9| 87.7| 64.3| 97.2| 86.9| 87.4| 38.8| **75.88** |
| [VPT](https://github.com/KMnP/vpt) | 78.8 | 90.8 | 65.8 | 98.0 | 88.3 | 78.1 | 49.6 | **78.48** |
| Convpass-attn | 71.84 | 90.62 | 71.97 | 99.06 | 90.98 | 89.93 | 54.25 | **81.24** |
| Convpass | 72.29 | 91.16 | 72.18 | 99.15 | 90.90 | 91.26 | 54.87 | **81.69** |

| Specialized | patch_camelyon | eurosat | resisc45 | diabetic_retinopathy | Average |
| ---- | ---- | ---- | ---- | ---- | ---- |
|Full FT|79.7| 95.7| 84.2| 73.9|**83.36**|
| [VPT](https://github.com/KMnP/vpt) | 81.8 | 96.1 | 83.4 | 68.4 | **82.43** |
| Convpass-attn | 85.17 | 95.57 | 83.43 | 74.76 | **84.73** |
| Convpass | 84.19 | 96.13 | 85.32 | 75.60 | **85.31** |

| Structured | clevr_count | clevr_dist | dmlab | kitti | dsprites_loc | dsprites_ori | smallnorb_azi | smallnorb_ele | Average |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|Full FT|56.3 |58.6| 41.7| 65.5| 57.5| 46.7| 25.7| 29.1| **47.64**|
|[VPT](https://github.com/KMnP/vpt)|68.5| 60.0| 46.5| 72.8| 73.6| 47.9| 32.9| 37.8| **54.98**|
| Convpass-attn | 79.86 | 66.97 | 50.31 | 79.89 | 84.27 | 53.2 | 34.81 | 42.95 | **61.53** |
| Convpass | 82.26 | 67.86 | 51.31 | 80.03 | 85.94 | 53.13 | 36.43 | 44.44 | **62.68** |


| VTAB-1k Average | (Natural + Specialized + Structured) / 3 |
| ---- | ---- |
|Full FT|**68.96**|
|[VPT](https://github.com/KMnP/vpt)|**71.96**|
| Convpass-attn| **75.83** |
|Convpass| **76.56** |


Checkpoints can be found [here](https://drive.google.com/file/d/19UjWeCuPTJG32RaVyOxV-EXFz-MYW8UT/view?usp=sharing). Extract `models.zip` to `./vtab/`.

## Citation
```
@inproceedings{jie2022convpass,
      title={Convolutional Bypasses Are Better Vision Transformer Adapters}, 
      author={Shibo Jie and Zhi-Hong Deng},
      year={2022},
      archivePrefix={arXiv},
}
```

## Acknowledgments
Part of the code is borrowed from [NOAH](https://github.com/ZhangYuanhan-AI/NOAH) and [timm](https://github.com/rwightman/pytorch-image-models).
