CONFIG = {
    "root_dir": "./data",
    "batch_size": 32,
    "val_split": 0.15,
    "seed": 42,
    "num_workers": 4,
    
    # Feature Flags
    "use_augmentation": True,      # Color jitter, flips
    "use_rotation_aug": True,      # 90, 180, 270 rotations
    "use_oversampling": True       # Handle class imbalance
}