# DPLC: Dirichlet Process Guided Long-tail Clustering

This repository contains the core implementation of our NeurIPS 2026 submission. 

[Openreview link](https://openreview.net/forum?id=FmLEIeg5hU#discussion)

## 1. Environment Setup

We use Conda to manage dependencies. Please run:

```bash
conda env create -f requirements.yml

conda activate DPLC
```

## 2. Dataset Preparation

Download the datasets and place them in your preferred data folder. The following datasets are supported:
- **CIFAR-10/100**: [Download Link](https://cave.cs.toronto.edu/kriz/cifar.html)
- **STL-10**: [Download Link](https://cs.stanford.edu/~acoates/stl10/)
- **ImageNet-100**: [HuggingFace Link](https://huggingface.co/datasets/clane9/imagenet-100/tree/main)
- **ImageNet-50**: [Kaggle Link](https://www.kaggle.com/datasets/ambityga/imagenet50)
- **Tiny-ImageNet-LT**: [Download Link](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

## 3. Running the Code

**Step 1: Pre-train BYOL backbone**
```bash
python main_byol.py
```
> **Note**: This runs for 1000 epochs and takes nearly 6 hours. You can run the code directly while using DINOv3 backbone. 

**Step 2: Run DPLC main training & evaluation**
```bash
python DPLC.py
```

> **Hyperparameters & Backbone**: You can modify hyperparameters, paths, and backbone settings in `get_config.py`. They're mostly predifined. We highly recommend using the **DINO** backbone for better performance.

## 4. Code Structure

- `core.py`: Core utilities and main algorithm components.
- `get_config.py`: Configuration, hyperparameter settings, and ablation setups.
- `main_byol.py`: BYOL pre-training script.
- `DPLC.py`: Core mathematical operations and the main entry point for DPLC training and testing.

## 5. Pretrained Weight
To enable quick reproduction, we release our pretrained BYOL backbone on Hugging Face. The CIFAR-10 encoder trained for 1000 epochs (Step 1) is already available — download it, pass the path to the `checkpoint` argument, and skip pre-training to go straight to DPLC training & evaluation. More pretrained BYOL weights for CIFAR-20 and STL-10 will be added if time permits. 

- [HuggingFace Link]<https://huggingface.co/tsnd7931/DPLC/tree/main>

## 6. Citation

If you find this code useful, please cite our paper (to be added upon publication).
