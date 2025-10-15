<div align="center">

# Mosaic3D: Foundation Dataset and Model for Open-vocabulary 3D Segmentation (CVPR 2025)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://nvlabs.github.io/Mosaic3D/)
[![Paper](https://img.shields.io/badge/CVPR-2025-green)](https://arxiv.org/abs/YOUR_ARXIV_ID)

**[Junha Lee¹'²*](https://junha-l.github.io/), [Chunghyun Park¹'²*](https://chrockey.github.io/), [Jaesung Choe¹](https://jaesung-choe.github.io/), [Frank Wang¹](https://vllab.ee.ntu.edu.tw/ycwang.html), [Jan Kautz¹](https://research.nvidia.com/person/jan-kautz), [Minsu Cho²](https://cvlab.postech.ac.kr/~mcho/), [Chris Choy¹](https://chrischoy.github.io/)**

*equal contribution\
¹NVIDIA, ²POSTECH

</div>

## Overview

We present **Mosaic3D**, a comprehensive solution for open-vocabulary 3D scene understanding that addresses three essential aspects: precise 3D region segmentation, comprehensive textual descriptions, and sufficient dataset scale. Our approach combines state-of-the-art open-vocabulary image segmentation models with region-aware vision-language models to create an automatic pipeline for generating high-quality 3D mask-text pairs.

### Key Contributions

- **Mosaic3D-5.6M Dataset**: The largest 3D mask-text paired dataset to date, encompassing over 30K indoor scenes and approximately 1M RGB-D frames, yielding 5.6M region captions with 30M total text tokens
- **Mosaic3D Model**: A 3D visual foundation model (3D-VFM) combining a 3D encoder trained with contrastive learning and a lightweight mask decoder for open-vocabulary 3D semantic and instance segmentation
- **State-of-the-art Performance**: Achieves leading results on open-vocabulary 3D semantic and instance segmentation benchmarks including ScanNet200, Matterport3D, and ScanNet++

### Dataset Advantages

Our Mosaic3D-5.6M dataset offers significant advantages over existing datasets:

- **Scale**: 5.6M mask-text pairs across 30K+ scenes (significantly larger than existing datasets)
- **Precision**: Leverages advanced open-vocabulary segmentation for precise region boundaries
- **Rich Descriptions**: Captures object attributes, spatial relationships, and scene context
- **Quality**: Combines robust region-aware VLMs for comprehensive textual annotations

## Dataset

### Mosaic3D-5.6M Download

The dataset can be found in [Huggingface](https://huggingface.co/datasets/junhalee/Mosaic3D). Follow the instruction there to download and organize the data into required structure.


## Environment Setup

### Docker (Recommended)

```bash
# Build docker image
bash docker/docker_build.sh

# Run docker container with dataset path
bash docker/docker_run.sh /path/to/datasets
```

### Conda Environment

```bash
# Create conda environment
conda env create -f environment.yaml

# Install requirements
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

## Model Architecture

Mosaic3D employs a two-stage training approach:

1. **Per-point Language Alignment**: Trains a 3D encoder using contrastive learning to align 3D point features with textual descriptions
2. **Mask Decoder Training**: Trains a lightweight mask decoder to predict instance segments from the aligned features

This design enables effective open-vocabulary 3D semantic and instance segmentation across diverse indoor scenes.

## Training

### Encoder Training

```bash
# Train Mosaic3D model with default configuration
python src/train.py experiment=train_spunet_multidata_ppt data=sc trainer.ddp trainer.devices=8 logger=wandb
```

### Mask Decoder Training

```bash
# Download Segment3D checkpoint
python src/models/networks/opensegment3d/download_ckpt.py

# Train a lightweight mask decoder with default configuration
python src/train.py experiment=train_opensegment3d_scannet model.net.backbone_ckpt=/path/to/encoder.ckpt trainer.ddp trainer.devices=8 logger=wandb
```

### Configuration Override

You can override any configuration parameter from the command line:

```bash
python src/train.py experiment=train_spunet_multidata_ppt data=sc+ar model=spunet34c trainer.max_epochs=100
```

## Evaluation

The model achieves state-of-the-art results on multiple benchmarks:

- **Annotation-free 3D semantic segmentation**: ScanNet20 & ScanNet200, Matterport3D, ScanNet++
- **Annotation-free 3D instance segmentation**: ScanNet200

### Annotation-free 3D semantic segmentation on ScanNet20 & ScanNet200.

Run the following commands to evaluate the pretrained models on ScanNet20 and ScanNet200 validation.

```bash
python src/eval.py experiment=train_spunet_multidata_ppt data=sc ckpt_path=[path/to/model/checkpoint]
```

## Model Zoo

We provide pretrained models for both **model_scale** and **data_scale** experiments. All models are available on [Hugging Face](https://huggingface.co/junhalee/Mosaic3D/tree/main).

### Available Models

#### Data-Scale Experiments
Models trained on different combinations of datasets:

| Model | Training Data | Size | f-mIoU on ScanNet200 |
|-------|--------------|------|-------------|
| `sc.ckpt` | ScanNet | 2.74 GB | 13.0% |
| `sc+ar.ckpt` | ScanNet + ARKitScenes | 2.74 GB | 14.8% |
| `sc+ar+sc++.ckpt` | ScanNet + ARKitScenes + ScanNet++ | 2.74 GB | 15.5% |
| `sc+ar+sc+++ma.ckpt` | SC + AR + SC++ + Matterport3D | 2.74 GB | 15.4% |
| `sc+ar+sc+++ma+st.ckpt` | SC + AR + SC++ + MA + Structured3D | 2.74 GB | 15.7% |


Evaluate the impact of training data scale by testing models trained on different dataset combinations. 
Set the `data` configuration to match the training data of your checkpoint:

```bash
# Model trained on ScanNet only
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc \
  ckpt_path=ckpt_raw/converted/sc.ckpt

# Model trained on ScanNet + ARKitScenes
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc+ar \
  ckpt_path=ckpt_raw/converted/sc+ar.ckpt

# Model trained on ScanNet + ARKitScenes + ScanNet++
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc+ar+sc++ \
  ckpt_path=ckpt_raw/converted/sc+ar+sc++.ckpt

# Model trained on four datasets (+ Matterport3D)
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc+ar+sc+++ma \
  ckpt_path=ckpt_raw/converted/sc+ar+sc+++ma.ckpt

# Model trained on all five datasets (+ Structured3D)
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc+ar+sc+++ma+st \
  ckpt_path=ckpt_raw/converted/sc+ar+sc+++ma+st.ckpt
```

#### Model-Scale Experiments 
Models with different backbone architectures (trained on ScanNet + ARKitScenes + ScanNet++):

| Model | Architecture | Parameters | Size | f-mIoU on ScanNet200 |
|-------|--------------|------------|------|-------------|
| `spunet14a.ckpt` | SparseUNet-14A | ~14M | 2.61 GB | 13.2% |
| `spunet18a.ckpt` | SparseUNet-18A | ~18M | 2.64 GB | 14.5% |
| `spunet34c.ckpt` | SparseUNet-34C | ~34M | 2.74 GB | 15.5% |
| `spunet50.ckpt` | SparseUNet-50 | ~50M | 2.97 GB | 15.8% |
| `spunet101.ckpt` | SparseUNet-101 | ~101M | 3.59 GB | 16.0% |

Evaluate different model architectures on the combined dataset (ScanNet + ARKitScenes + ScanNet++):

```bash
# SparseUNet-14A (smallest, fastest)
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc+ar+sc++ \
  model=spunet14a+ppt \
  ckpt_path=ckpt_raw/converted/spunet14a.ckpt

# SparseUNet-18A
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc+ar+sc++ \
  model=spunet18a+ppt \
  ckpt_path=ckpt_raw/converted/spunet18a.ckpt

# SparseUNet-34C (recommended for balance)
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc+ar+sc++ \
  model=spunet34c+ppt \
  ckpt_path=ckpt_raw/converted/spunet34c.ckpt

# SparseUNet-50
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc+ar+sc++ \
  model=spunet50+ppt \
  ckpt_path=ckpt_raw/converted/spunet50.ckpt

# SparseUNet-101 (largest, best performance)
python src/eval.py experiment=train_spunet_multidata_ppt \
  data=sc+ar+sc++ \
  model=spunet101+ppt \
  ckpt_path=ckpt_raw/converted/spunet101.ckpt
```


## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{lee2025mosaic3d,
  title={Mosaic3d: Foundation dataset and model for open-vocabulary 3d segmentation},
  author={Lee, Junha and Park, Chunghyun and Choe, Jaesung and Wang, Yu-Chiang Frank and Kautz, Jan and Cho, Minsu and Choy, Chris},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={14089--14101},
  year={2025}
}
```


## Acknowledgments

Our work builds upon several fantastic open-source projects. We'd like to express our gratitude to the authors of:
- [Pointcept](https://github.com/Pointcept/Pointcept)
- [PLA & RegionPLC](https://github.com/CVMI-Lab/PLA)
- [SPConv](https://github.com/traveller59/spconv)
- [Segment3D](https://github.com/LeapLabTHU/Segment3D)
- [OpenIns3D](https://github.com/Pointcept/OpenIns3D)
