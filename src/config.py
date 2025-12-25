from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

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
    "cam_percentile": 95,

    # Model Artifacts
    "cls_model_path": BASE_DIR / "classifier_best.pth",
    "seg_model_path": BASE_DIR / "crack_seg_model.pth",
    "submission_file": BASE_DIR / "submission.csv",
}