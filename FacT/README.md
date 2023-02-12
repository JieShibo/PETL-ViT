# FacT

Source code of "FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer"

[AAAI 2023 Oral](https://arxiv.org/abs/2212.03145) 


## Requirements
Python == 3.8

torch == 1.10.0

torchvision == 0.11.1

timm == 0.4.12

avalanche-lib == 0.1.0

## Dataset

To download the datasets, please refer to https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation. Then move the dataset folders to `<YOUR PATH>/FacT/data/`. 


## Pretrained Model
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `<YOUR PATH>/FacT/ViT-B_16.npz`

## Run
```sh
cd <YOUR PATH>/FacT
bash run.sh
```
## Citation
```
@inproceedings{jie2023fact,
      title={FacT: Factor-Tuning For Lightweight Adaptation on Vision Transformer}, 
      author={Shibo Jie and Zhi-Hong Deng},
      booktitle={AAAI Conference on Artificial Intelligence},
      booktitle={Proceedings of AAAI Conference on Artificial Intelligence (AAAI)},
      year={2023},
}
```

## Acknowledgments
Part of the code is borrowed from [NOAH](https://github.com/ZhangYuanhan-AI/NOAH) and [timm](https://github.com/rwightman/pytorch-image-models).

