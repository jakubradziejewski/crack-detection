import argparse
import sys
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CONFIG
from src.inference import Predictor
from src.metrics import calculate_iou, calculate_dice, calculate_precision_recall

def mask2rle(img):
    """Convert binary mask to RLE encoding for submission"""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def run_evaluation(predictor, test_dir, threshold):
    """Runs metrics on the test set (requires masks)."""
    img_paths = sorted(glob.glob(str(test_dir / 'images' / '*.jpg')))
    mask_paths = sorted(glob.glob(str(test_dir / 'masks' / '*.jpg')))
    
    if not img_paths or not mask_paths:
        print("No test data found for evaluation.")
        return

    print(f"\nEvaluating on {len(img_paths)} images (Thresh: {threshold})...")
    
    ious, dices, precisions, recalls = [], [], [], []
    
    for img_path, mask_path in tqdm(zip(img_paths, mask_paths), total=len(img_paths)):
        # Run Prediction
        result = predictor.predict(img_path, seg_threshold=threshold)
        pred_mask = result['mask']
        
        # Load GT
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_binary = (true_mask > 127).astype(np.uint8)
        
        # Metrics
        ious.append(calculate_iou(pred_mask, true_binary))
        dices.append(calculate_dice(pred_mask, true_binary))
        p, r = calculate_precision_recall(pred_mask, true_binary)
        precisions.append(p)
        recalls.append(r)

    print(f"\nResults:")
    print(f"  Mean IoU:       {np.mean(ious):.4f}")
    print(f"  Mean Dice:      {np.mean(dices):.4f}")
    print(f"  Precision:      {np.mean(precisions):.4f}")
    print(f"  Recall:         {np.mean(recalls):.4f}")

def generate_submission(predictor, test_dir, output_file, threshold):
    """Generates submission.csv (no masks required)."""
    img_paths = sorted(glob.glob(str(test_dir / 'images' / '*.jpg')))
    print(f"\nGenerating submission for {len(img_paths)} images to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write('ImageId,EncodedPixels\n')
        
        for path in tqdm(img_paths):
            img_id = os.path.basename(path).replace('.jpg', '')
            
            # Predict
            result = predictor.predict(path, seg_threshold=threshold)
            
            if not result['has_crack'] or result['mask'].sum() == 0:
                f.write(f'{img_id},\n') # Empty prediction
            else:
                rle = mask2rle(result['mask'])
                f.write(f'{img_id},{rle}\n')

if __name__ == "__main__":
    # Hardcode threshold here for debugging/standalone runs
    DEBUG_THRESHOLD = 0.2078 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'submit'], 
                        help='Action: "eval" (calculate metrics) or "submit" (generate CSV)')
    parser.add_argument('--threshold', type=float, default=DEBUG_THRESHOLD, 
                        help='Segmentation threshold')
    args = parser.parse_args()

    # Initialize Logic
    predictor = Predictor(CONFIG)
    test_dir = CONFIG["root_dir"] / 'test'
    
    if args.mode == 'eval':
        run_evaluation(predictor, test_dir, args.threshold)
    elif args.mode == 'submit':
        generate_submission(predictor, test_dir, CONFIG["submission_file"], args.threshold)