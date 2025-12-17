# crack-detection
 CNN-based crack detection that localizes defects using only image-level labels (crack/no crack) during training. 

## Setup and Reproducibility

This project uses `uv` for extremely fast and reproducible dependency management. 

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system.
- NVIDIA GPU with Drivers (for CUDA 11.8 support).

### Quick Start
1. Clone the repository.
2. Place the dataset in the `data/` folder.
3. Run the training script:
   ```bash
   uv run main.py