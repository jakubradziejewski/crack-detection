import torch
import numpy as np
from tqdm import tqdm
from skimage.filters import threshold_li, threshold_triangle
import time
import sys
import os
import argparse
import cv2
import glob


def find_threshold_otsu_and_li(model, val_loader, device):
    """
    Use BOTH Otsu and Li, then average them for robustness.
    
    This is the RECOMMENDED approach:
    - Otsu: Good for balanced distributions
    - Li: Good for skewed distributions (common in segmentation)
    - Average: More robust than either alone
    
    Speed: ~60s (predictions collected once, both methods applied)
    """
    print("\n" + "="*60)
    print("COMBINED OTSU + LI THRESHOLD SELECTION")
    print("="*60)
    
    model.eval()
    
    # Step 1: Collect predictions (done once)
    print("Collecting predictions from validation set...")
    start_time = time.time()
    
    all_predictions = []
    with torch.no_grad():
        for imgs, _ in tqdm(val_loader, desc="Processing batches"):
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_values = preds.cpu().numpy().flatten()
            all_predictions.append(pred_values)
    
    all_predictions = np.concatenate(all_predictions)
    collection_time = time.time() - start_time
    
    print(f"✓ Collected {len(all_predictions):,} predictions in {collection_time:.1f}s")
    
    # Convert to 8-bit once
    predictions_8bit = (all_predictions * 255).astype(np.uint8)
    
    # Step 2: Apply Otsu
    print("\nApplying Otsu's method...")
    start_time = time.time()
    otsu_thresh_8bit, _ = cv2.threshold(
        predictions_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_threshold = otsu_thresh_8bit / 255.0
    otsu_time = time.time() - start_time
    print(f"✓ Otsu: {otsu_threshold:.4f} (computed in {otsu_time:.3f}s)")
    
    # Step 3: Apply Li
    print("\nApplying Li's method...")
    start_time = time.time()
    try:
        li_thresh_8bit = threshold_li(predictions_8bit)
        li_threshold = li_thresh_8bit / 255.0
        li_time = time.time() - start_time
        print(f"✓ Li: {li_threshold:.4f} (computed in {li_time:.3f}s)")
    except Exception as e:
        print(f"✗ Li's method failed: {e}, using Otsu value")
        li_threshold = otsu_threshold
        li_time = 0
    
    # Step 4: Combine
    final_threshold = (otsu_threshold + li_threshold) / 2.0
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"  Otsu Threshold:     {otsu_threshold:.4f}")
    print(f"  Li Threshold:       {li_threshold:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  ✓ Final (average):  {final_threshold:.4f}")
    print(f"\n  Total Time:         {collection_time + otsu_time + li_time:.1f}s")
    print(f"    - Collection:     {collection_time:.1f}s")
    print(f"    - Otsu:           {otsu_time:.3f}s")
    print(f"    - Li:             {li_time:.3f}s")
    print(f"{'='*60}")
    
    # Distribution statistics
    print(f"\nPrediction Distribution:")
    print(f"  Mean:   {all_predictions.mean():.4f}")
    print(f"  Median: {np.median(all_predictions):.4f}")
    print(f"  Std:    {all_predictions.std():.4f}")
    
    return final_threshold


def find_threshold_from_checkpoint(checkpoint_path, config, method='both'):
    """
    Load model from checkpoint and find optimal threshold.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        config: Configuration dictionary
        method: 'otsu', 'li', or 'both' (default: 'both')
    
    Usage:
        python threshold_selection.py --checkpoint model.pth --method both
    """
    from src.models import UNetLight
    from src.datasets import CrackSegDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from sklearn.model_selection import train_test_split
    import glob
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = UNetLight().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("✓ Model loaded successfully")
    
    # Create validation loader
    print("\nPreparing validation data...")
    img_size = config.get("image_size", 224)
    
    # Get all training images (we'll split them)
    img_pattern = str(config["root_dir"] / 'train' / 'images' / '*.jpg')
    all_img_paths = sorted(glob.glob(img_pattern))
    
    if not all_img_paths:
        raise RuntimeError(f"No images found at {img_pattern}")
    
    # Create dummy masks (not used, just for DataLoader compatibility)
    dummy_masks = [np.zeros((img_size, img_size), dtype=np.uint8) for _ in all_img_paths]
    
    # Split into train/val
    indices = list(range(len(all_img_paths)))
    _, val_idx = train_test_split(
        indices, 
        test_size=config["val_split"],
        random_state=config["seed"]
    )
    
    print(f"Validation set size: {len(val_idx)} images")
    
    # Create validation dataset
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = CrackSegDataset(
        [all_img_paths[i] for i in val_idx],
        [dummy_masks[i] for i in val_idx],
        transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    # Find threshold based on method
    print(f"\nMethod: {method.upper()}")

    threshold = find_threshold_otsu_and_li(model, val_loader, device)

    return threshold


def main():
    """
    Run threshold selection from command line.
    
    Usage:
        # Using default checkpoint from config
        python threshold_selection.py
        
        # Using custom checkpoint
        python threshold_selection.py --checkpoint my_model.pth
        
        # Using specific method
        python threshold_selection.py --method otsu
        python threshold_selection.py --method li
        python threshold_selection.py --method both
    """
    # Add parent directory to path for imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    parser = argparse.ArgumentParser(description='Find optimal threshold from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: from config)')
    parser.add_argument('--method', type=str, default='both',
                        choices=['otsu', 'li', 'both'],
                        help='Threshold method: otsu, li, or both (default: both)')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Image size (default: from config)')
    
    args = parser.parse_args()
    
    # Import config
    from config import CONFIG
    from pathlib import Path
    
    # Use checkpoint from args or config
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = CONFIG["seg_model_path"]
    
    if not checkpoint_path.exists():
        print(f"✗ Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    # Override image size if provided
    config = CONFIG.copy()
    if args.image_size:
        config["image_size"] = args.image_size
    
    # Find threshold
    print(f"\n{'#'*60}")
    print(f"# THRESHOLD SELECTION FROM CHECKPOINT")
    print(f"{'#'*60}")
    
    threshold = find_threshold_from_checkpoint(checkpoint_path, config, args.method)
    
    print(f"\n{'='*60}")
    print(f"✓ THRESHOLD SELECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Method:     {args.method.upper()}")
    print(f"  Threshold:  {threshold:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())