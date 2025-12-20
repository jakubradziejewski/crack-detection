import sys
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import glob
from tqdm import tqdm
from PIL import Image
import argparse

from src.models import UNetLight

# --- Core Metric Calculations ---

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

# --- Evaluation Function (No Threshold Search!) ---

def evaluate_test_set(config, threshold=0.5, verbose=True):
    """
    Evaluate model on test set with a GIVEN threshold.
    This function should only be called ONCE at the end with the optimal threshold
    found during validation.
    
    Args:
        config: Configuration dictionary
        threshold: Pre-determined optimal threshold (from validation)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with mean IoU, Dice, Precision, Recall
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    
    # Load Weights
    if not config["seg_model_path"].exists():
        print(f"Model not found at {config['seg_model_path']}")
        return None

    model.load_state_dict(torch.load(config["seg_model_path"], map_location=device))
    model.eval()
    
    img_size = config.get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_imgs = sorted(glob.glob(str(config["root_dir"] / 'test' / 'images' / '*.jpg')))
    test_masks = sorted(glob.glob(str(config["root_dir"] / 'test' / 'masks' / '*.jpg')))
    
    if len(test_imgs) == 0:
        print(f"Warning: No test images found at {config['root_dir'] / 'test' / 'images'}")
        return None
    
    ious, dices, precisions, recalls = [], [], [], []
    
    if verbose:
        print(f"\nEvaluating {len(test_imgs)} test images with threshold={threshold:.4f}...")
    
    for img_path, mask_path in tqdm(zip(test_imgs, test_masks), total=len(test_imgs), disable=not verbose):
        img = Image.open(img_path).convert('RGB')
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = true_mask.shape
        
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_t).squeeze().cpu().numpy()
        
        pred = cv2.resize(pred, (original_size[1], original_size[0]))
        pred_binary = (pred > threshold).astype(np.uint8)
        true_binary = (true_mask > 127).astype(np.uint8)
        
        ious.append(calculate_iou(pred_binary, true_binary))
        dices.append(calculate_dice(pred_binary, true_binary))
        p, r = calculate_precision_recall(pred_binary, true_binary)
        precisions.append(p)
        recalls.append(r)
    
    results = {
        'mean_iou': np.mean(ious),
        'mean_dice': np.mean(dices),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'threshold': threshold
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TEST SET RESULTS (Threshold: {threshold:.4f})")
        print(f"{'='*60}")
        print(f"  Mean IoU:       {results['mean_iou']:.4f}")
        print(f"  Mean Dice:      {results['mean_dice']:.4f}")
        print(f"  Mean Precision: {results['mean_precision']:.4f}")
        print(f"  Mean Recall:    {results['mean_recall']:.4f}")
        print(f"{'='*60}\n")
        
    return results


# --- Main function for standalone execution ---

def main():
    """
    Run metrics evaluation from command line.
    
    Usage:
        python src/metrics.py --threshold 0.5
        python src/metrics.py --threshold 0.42 --image_size 448
    """
    parser = argparse.ArgumentParser(description='Evaluate segmentation model on test set')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold (default: 0.5)')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Image size for inference (default: from config)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (default: from config)')
    
    args = parser.parse_args()
    
    # Import config
    from config import CONFIG
    
    # Override config if needed
    config = CONFIG.copy()
    if args.image_size:
        config["image_size"] = args.image_size
    if args.model_path:
        from pathlib import Path
        config["seg_model_path"] = Path(args.model_path)
    
    print(f"\n{'#'*60}")
    print(f"# TEST SET EVALUATION")
    print(f"{'#'*60}")
    print(f"Model: {config['seg_model_path']}")
    print(f"Threshold: {args.threshold:.4f}")
    print(f"Image size: {config['image_size']}×{config['image_size']}")
    
    # Run evaluation
    results = evaluate_test_set(config, threshold=args.threshold, verbose=True)
    
    if results:
        print("✓ Evaluation complete!")
        return 0
    else:
        print("✗ Evaluation failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())