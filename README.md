# Deepfake-Detection

Experiments for distinguishing AI-generated images from real photos using CNN and transformer architectures.

## Overview

This repository contains comprehensive experiments comparing different deep learning approaches for detecting AI-generated images. The work includes both a lightweight custom CNN implementation and transfer learning experiments with state-of-the-art architectures.

## Dataset

**Source:** "AI vs Human Generated Images" (Kaggle)
- **Size:** 15,990 images (subset used)
- **Classes:** Human (49.93%) vs AI Generated (50.07%)
- **Split:** 70% train / 15% validation / 15% test
- **Augmentation:** Random flips, rotations, color jitter

## Experiments

### Part 1: Custom CNN Baseline
Built a lightweight CNN with 8.4M parameters as a baseline:
- **Architecture:** 2 conv layers (32→64 filters) + 2 FC layers
- **Training:** 20 epochs, batch size 128, early stopping
- **Optimizer comparison:** Adam vs SGD vs SGD+Momentum

**Results:**
- Adam: **86.87%** test accuracy (best baseline)
- SGD+Momentum: 74.86% test accuracy  
- SGD: 53.36% test accuracy

### Part 2: Transfer Learning
Fine-tuned pre-trained ImageNet models with three strategies:
1. **Head-only:** Freeze backbone, train classification head
2. **Last-block + head:** Unfreeze last layer + head
3. **Entire network:** Fine-tune full model

**Architectures tested:**
- ResNet variants
- EfficientNet-Lite
- Vision Transformer (ViT)
- SqueezeNet

**Best Results:**
- **EfficientNet-Lite (full fine-tuning): 91.53%** 
- ResNet (full fine-tuning): 90.83%
- Custom CNN baseline: 86.87%

## Key Findings

1. **Transfer learning wins:** Pre-trained models outperformed custom CNN when fully fine-tuned
2. **Fine-tuning strategy matters:** Head-only approaches performed poorly; full fine-tuning essential
3. **Optimizer choice significant:** Adam substantially outperformed SGD variants for custom CNN

## Repository Contents

- **Notebooks:** `ai-vs-human-*.ipynb` - experiments for different architectures
- **Custom CNN:** `custom_cnn/` - assignment implementation and results
- **Results:** `results/` - model checkpoints, plots, metrics
- **Research:** `research_papers/` - reference links

## Results Artifacts

The `results/` folder contains:
- Model checkpoints: `best_model_*.pth`
- Training curves: `training_history_*.png`
- Confusion matrices: `confusion_matrix_*.png`
- Metrics summaries: `results_summary.json`

## Quick Usage

```python
import torch
# Load saved model (adapt to your model class)
model = YourModel(num_classes=2)
checkpoint = torch.load('results/best_model_adam.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    output = model(image_tensor.unsqueeze(0))
    pred = torch.softmax(output, dim=1).argmax(dim=1).item()
```

## Requirements

- Python 3.8+
- PyTorch, torchvision
- numpy, pandas, matplotlib, scikit-learn
- tqdm, seaborn

**Note:** Some experiments were run on Kaggle with GPU acceleration.

## Contact

Open an issue for questions or to reproduce specific experiments.