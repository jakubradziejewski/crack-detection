# crack-detection

CNN-based crack detection that localizes defects using only image-level labels (crack/no crack) during training.

## Setup and Reproducibility

This project uses `uv` for extremely fast and reproducible dependency management.

### Prerequisites

* [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system.
* NVIDIA GPU with Drivers (for CUDA 11.8 support).

### Quick Start

1. Clone the repository.
2. Sync dependencies using `uv`:

   ```bash
   uv sync
   ```
3. Run the data setup/preprocessing script:

   ```bash
   uv run setup_data.py
   ```
4. Run the training script:

   ```bash
   uv run main.py
   ```
