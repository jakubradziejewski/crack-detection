from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "root_dir": BASE_DIR / "data",
    # Data Config
    "batch_size": 32,
    "val_split": 0.15,
    "seed": 42,
    "num_workers": 4,

    # Augmentations
    "use_augmentation": True,
    "use_rotation_aug": True,
    "use_oversampling": True,

    # Training Hyperparameters
    "classifier_epochs": 3,
    "seg_epochs": 4,
    "lr_classifier": 1e-4,
    "lr_seg": 1e-3,

    # Semi-Supervised Logic
    "confidence_threshold": 0.82,  # Confidence to accept a pseudo-label
    "cam_percentile": 87,          # Percentile for CAM thresholding

    # Model Artifacts
    "cls_model_path": BASE_DIR / "classifier_best.pth",
    "seg_model_path": BASE_DIR / "crack_seg_model.pth",
    "submission_file": BASE_DIR / "submission.csv",
    
    # Testing
    "test_thresholds": [0.2, 0.25, 0.3, 0.4, 0.5]
}