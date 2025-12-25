import torch
import numpy as np


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