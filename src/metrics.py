import sys
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import glob
from tqdm import tqdm
from PIL import Image
import argparse

from src.models import UNetLight, GradCAMPlusPlus

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

# --- Evaluation Function ---

def evaluate_test_set(config, threshold=0.5, verbose=True):
    """
    Evaluate model on test set with classifier + segmentation pipeline.
    
    CRITICAL FIX: Now uses the classifier to detect if image has cracks BEFORE segmentation.
    This matches the training pipeline where only high-confidence images get pseudo-masks.
    
    Args:
        config: Configuration dictionary
        threshold: Pre-determined optimal threshold for segmentation (from validation)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with mean IoU, Dice, Precision, Recall
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Classifier (to detect if image has cracks)
    classifier = GradCAMPlusPlus().to(device)
    if not config["cls_model_path"].exists():
        print(f"Classifier not found at {config['cls_model_path']}")
        return None
    classifier.load_state_dict(torch.load(config["cls_model_path"], map_location=device))
    classifier.eval()
    
    # 2. Segmentation model (to segment cracks if present)
    seg_model = UNetLight().to(device)
    if not config["seg_model_path"].exists():
        print(f"Segmentation model not found at {config['seg_model_path']}")
        return None
    seg_model.load_state_dict(torch.load(config["seg_model_path"], map_location=device))
    seg_model.eval()
    
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
    
    confidence_threshold = config.get("confidence_threshold", 0.8)
    
    ious, dices, precisions, recalls = [], [], [], []
    classified_as_crack = 0
    classified_as_no_crack = 0
    
    if verbose:
        print(f"\nEvaluating {len(test_imgs)} test images")
        print(f"  - Segmentation threshold: {threshold:.4f}")
        print(f"  - Confidence threshold: {confidence_threshold:.4f}")
    
    for img_path, mask_path in tqdm(zip(test_imgs, test_masks), total=len(test_imgs), disable=not verbose):
        img = Image.open(img_path).convert('RGB')
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = true_mask.shape
        
        img_t = transform(img).unsqueeze(0).to(device)
        
        # ===== STEP 1: Check if image has cracks using classifier =====
        with torch.no_grad():
            cls_output = classifier(img_t, return_cam=False)
            probs = F.softmax(cls_output, dim=1)
            crack_confidence = probs[0, 1].item()
        
        # If low confidence (no crack detected), predict empty mask
        if crack_confidence < confidence_threshold:
            pred_binary = np.zeros(original_size, dtype=np.uint8)
            classified_as_no_crack += 1
        else:
            # ===== STEP 2: If crack detected, run segmentation =====
            classified_as_crack += 1
            
            with torch.no_grad():
                pred = seg_model(img_t).squeeze().cpu().numpy()
            
            pred = cv2.resize(pred, (original_size[1], original_size[0]))
            pred_binary = (pred > threshold).astype(np.uint8)
        
        # Calculate metrics
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
        'threshold': threshold,
        'confidence_threshold': confidence_threshold,
        'classified_as_crack': classified_as_crack,
        'classified_as_no_crack': classified_as_no_crack
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"Classification:")
        print(f"  - Images with CRACK:    {classified_as_crack}/{len(test_imgs)} ({100*classified_as_crack/len(test_imgs):.1f}%)")
        print(f"  - Images with NO CRACK: {classified_as_no_crack}/{len(test_imgs)} ({100*classified_as_no_crack/len(test_imgs):.1f}%)")
        print(f"\nSegmentation Metrics (threshold={threshold:.4f}):")
        print(f"  - Mean IoU:       {results['mean_iou']:.4f}")
        print(f"  - Mean Dice:      {results['mean_dice']:.4f}")
        print(f"  - Mean Precision: {results['mean_precision']:.4f}")
        print(f"  - Mean Recall:    {results['mean_recall']:.4f}")
        print(f"{'='*60}\n")
        
    return results


# --- Main function for standalone execution ---

def main():
    """
    Run metrics evaluation from command line.
    Usage:
        python src/metrics.py --threshold 0.5
    """
    parser = argparse.ArgumentParser(description='Evaluate segmentation model on test set')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold (default: 0.5)')

    args = parser.parse_args()
    
    # Import config
    from config import CONFIG
    config = CONFIG.copy()
    print(f"\n{'#'*60}")
    print(f"# TEST SET EVALUATION")
    print(f"{'#'*60}")
    print(f"Classifier: {config['cls_model_path']}")
    print(f"Segmentation: {config['seg_model_path']}")
    print(f"Segmentation threshold: {args.threshold:.4f}")
    print(f"Confidence threshold: {config['confidence_threshold']:.4f}")
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