# Weakly-Supervised Semantic Segmentation of Cracks

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-Fast_Package_Manager-de5d43?style=flat-square&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8_/_12.8-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

CNN-based crack detection that localizes defects using only image-level labels (crack/no crack) during training. This project implements a two-stage pipeline using **Grad-CAM++** for pseudo-label generation and a **U-Net** for refined pixel-wise segmentation.

Our solution was domain-agnostic, and could be applied to other real-world problems.

## 📊 Dataset

This project was developed and evaluated using the **Crack Segmentation Dataset** from Kaggle:

- **Dataset:** [Crack Segmentation Dataset](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset)
- **Content:** RGB images of concrete surfaces with visible cracks
- **Original labels:** Pixel-level crack masks
- Only **image-level labels (crack / no-crack)** were used during training.
- Pixel-level masks were **not used** and served only for final evaluation during test.

The dataset was provided as part of the following Kaggle competition:
  [Deep Learning 2025 – Project 2](https://www.kaggle.com/competitions/dl-2025-project-2-pro/overview)

---

## 🧠 Architecture

**Detailed Analysis and Project Description:** [Project Overview PDF](project-overview.pdf)

The system addresses the challenge of segmenting cracks when only image-level labels (crack/no-crack) are available, eliminating the need for expensive pixel-level manual masks.

### 1. Classification & Attention (Dilated ResNet-34)
Standard ResNet architectures downsample images too aggressively for detailed crack localization. We adapted ResNet-34 to keep more spatial information:
* **Dilated Convolutions:** We swapped stride-2 for stride-1 in Layers 3 and 4, using dilation ($d=2, 4$). This keeps the feature maps at $28 \times 28$ (instead of shrinking to $7 \times 7$).
* **Grad-CAM++ & Fusion:** To catch thin cracks that standard Grad-CAM misses, we use Grad-CAM++ and fuse maps from Layers 2, 3, and 4. This mixes fine edge details with high-level features.



### 2. Pseudo-label Generation
We convert the attention maps into training targets for the next stage:
* **Confidence Check:** If the classifier isn't sure an image has a crack (based on an F1-optimized threshold), we generate an empty mask.
* **Thresholding:** We only take the top 5% of activated pixels as the "crack." This keeps the pseudo-labels clean and reduces noise.



### 3. Segmentation Stage (U-Net)
Attention maps were providing blurry blobs, not precise masks. We use them as noisy (pseudo) labels to train a U-Net.
* **Why it works:** The U-Net sees the original RGB image, so it learns to snap the blurry pseudo-labels to the actual crack edges.
* **Result:** This step cleans up the "blobs" and refines the actual crack edges. It boosted the Dice score by ~0.10.

---

## 🚀 Setup and Reproducibility

This project uses `uv` for extremely fast and reproducible dependency management.

### Prerequisites
* [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system.
* NVIDIA GPU with Drivers (CUDA support).

### Quick Start
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/jakubradziejewski/crack-detection.git](https://github.com/jakubradziejewski/crack-detection.git)
   cd crack-detection

2. Sync dependencies using `uv`:

   ```bash
   uv sync
   ```
3. Run the data setup/preprocessing script:

   ```bash
   uv run src/setup.py
   ```
4. Run the training script:

   ```bash
   uv run src/train.py
   ```
