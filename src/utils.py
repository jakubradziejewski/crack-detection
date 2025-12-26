import torch
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms(config, mode="train"):
    """
    Returns transforms based on config and mode ('train' or 'val').
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_size = config.get("image_size", 224)

    if mode == "val":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

    # Training Transforms
    ops = [transforms.Resize((img_size, img_size))]
    
    # 1. Rotation Augmentation
    if config.get("use_rotation_aug", False):
        ops.append(transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ])
        ], p=0.5))

    # 2. Standard Augmentation
    if config.get("use_augmentation", False):
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(ops)


def get_oversampler(labels):
    """
    Creates a WeightedRandomSampler to balance classes.
    """
    labels_tensor = torch.as_tensor(labels)
    class_counts = torch.bincount(labels_tensor)
    weights = 1. / class_counts.float()
    samples_weights = weights[labels_tensor]
    
    return WeightedRandomSampler(samples_weights, len(samples_weights))


def mask2rle(img):
    """Convert binary mask to RLE encoding for submission"""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# For debugging purposes, running test.py standalone
def load_model(model, path, device):
    """Load model weights from checkpoint."""
    from pathlib import Path
    if Path(path).exists():
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print(f"Warning: Model path {path} not found.")

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)

def calculate_dice(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    sum_area = pred_mask.sum() + true_mask.sum()
    return (2. * intersection) / sum_area if sum_area > 0 else 1.0

def calculate_precision_recall(pred_mask, true_mask):
    tp = np.logical_and(pred_mask, true_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(true_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), true_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall