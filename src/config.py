from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_stats():
    stats_path = BASE_DIR / "data" / "dataset_stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
            return stats["mean"], stats["std"]
    # If stats don't exist yet, they'll be computed on first run
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


DATASET_MEAN, DATASET_STD = _load_stats()

CONFIG = {
    "root_dir": BASE_DIR / "data",
    # Image Config
    "image_size": 448,
    # Data Config
    "batch_size": 2,
    "batch_size1": 16,
    "val_split": 0.15,
    "seed": 42,
    "num_workers": 0,
    # Augmentations
    "use_augmentation": True,
    "use_rotation_aug": True,
    "use_oversampling": True,
    # Training Hyperparameters
    "classifier_epochs": 1,
    "seg_epochs": 1,
    "lr_classifier": 1e-4,
    "lr_seg": 1e-3,
    # Semi-Supervised Logic
    "confidence_threshold": 0.7,
    "cam_percentile": 98,
    # Model Artifacts
    "cls_model_path": BASE_DIR / "classifier_best.pth",
    "seg_model_path": BASE_DIR / "crack_seg_model.pth",
    "submission_file": BASE_DIR / "submission.csv",
    # Dataset Statistics
    "dataset_mean": DATASET_MEAN,
    "dataset_std": DATASET_STD,
    "stats_cache_path": BASE_DIR / "data" / "dataset_stats.json",
    "stats_sample_size": 1000
}
