# Prithvi EO 2.0 Fine-Tuning Module

This module implements the core fine-tuning logic for specializing the Prithvi EO 2.0 foundation model for burn scar detection.

**Author**: Tushar Thokdar

## 📋 Overview

The fine-tuning process utilizes a multi-temporal approach, feeding the model pre-fire and post-fire imagery along with an explicitly calculated "delta" change channel.

## 🎯 Key Features

- **Architectural Specialization**: ViT backbone paired with a **UperNet** decoder.
- **Temporal Stacking**: Specialized 3D input handling `(Time=3, Channels=6)`.
- **Training Strategy**: Two-stage "Freeze-then-Unfreeze" pipeline to protect foundation knowledge while adapting to new classes.
- **Robust Optimization**: Hybrid Loss (Weighted Cross-Entropy + Dice) to handle extreme class imbalances.

## 🚀 Quick Start

### Prerequisites

```python
pip install torch lightning terratorch segmentation-models-pytorch
```

### Core Configuration

```python
# Key Hyperparameters
LR = 1e-4
BACKBONE = "prithvi_eo_v2_tiny_tl"
DECODER = "UperNet"
FREEZE_EPOCHS = 5
LOSS = "CrossEntropy + 0.8 * Dice"
```

## 🧠 Training Strategy details

For a deep dive into the specific algorithms and performance metrics, please refer to the [Technical Deep Dive](../Prithvi_Technical_Deep_Dive.md).

## 📁 Files

- `prithivi_finetune_with_delta.ipynb`: Main interactive notebook for training and validation.

---

**Last Updated**: February 1, 2026
