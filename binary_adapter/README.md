# Binary Adapter

Source code of "Revisiting the Parameter Efficiency of Adapters from the Perspective of Precision Redundancy".

[ICCV 2023](https://arxiv.org/) 


## Requirements
Python 3.8

torch 1.10.0

torchvision 0.11.1

timm 0.4.12

## Dataset

To download the datasets, please refer to https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation. Then move the dataset folders to `<YOUR PATH>/binary_adapter/data/`. 

Note: Unlike FacT and NOAH, we here use unnormalized inputs following VPT and SSF.

## Pretrained Model
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `<YOUR PATH>/binary_adapter/ViT-B_16.npz`

## Run
##### Train on VTAB-1K
```sh
bash train.sh
```
##### Evaluation
```sh
bash test.sh
```

| Method | Average Accuracy (%) | Average Size (KB / task) |
| --- | --- | --- |
| Full | 68.9 | 3.4E5 |
| Linear | 57.6 | 140.8 |
| VPT | 72.0 | 2219.5 |
| Bi-AdaptFormer<sub>32</sub> | 77.0 | 212.9 |
| Bi-Adaptformer<sub>32</sub> (Bi-Head) | 76.9 | 76.6 |
| Bi-Adaptformer<sub>1</sub> (Bi-Head) | 73.8 | 6.8 |
| Bi-LoRA<sub>32</sub> | 76.7 | 285.1 |
| Bi-LoRA<sub>32</sub> (Bi-Head) | 76.3 | 148.8 |
| Bi-LoRA<sub>1</sub> (Bi-Head) | 74.6 | 9.2 |

## Citation
```
@inproceedings{jie2023revisiting,
    author    = {Jie, Shibo and Wang, Haoqing and Deng, Zhi-Hong},
    title     = {Revisiting the Parameter Efficiency of Adapters from the Perspective of Precision Redundancy},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
}
```

## Acknowledgments
Part of the code is borrowed from [NOAH](https://github.com/ZhangYuanhan-AI/NOAH) and [timm](https://github.com/rwightman/pytorch-image-models).

