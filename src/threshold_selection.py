import torch
import numpy as np
from tqdm import tqdm
import time
import sys
import os
import argparse
import cv2
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve


def find_confidence_threshold(model, loader, device):
    """
    Find optimal confidence threshold that maximizes F1 score.
    """
    
    model.eval()
    
    # Collect predictions and ground truth for validation set
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating validation set"):
            imgs = imgs.to(device)
            outputs = model(imgs, return_cam=False)
            probs = F.softmax(outputs, dim=1)
            crack_probs = probs[:, 1].cpu().numpy()
            
            all_probs.extend(crack_probs)
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Confidence threshold that maximizes F1 score
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    print(f"Optimal confidence threshold: {optimal_threshold:.4f} based on F1 score {f1_scores[best_idx]:.4f}")
    
    return optimal_threshold



def find_threshold_otsu(model, loader, device, max_samples=1000):
    """
    Use Otsu's method for threshold selection.
    """

    if max_samples is not None:
        print(f"Using {max_samples} samples for threshold selection...")
    else:
        print(f"Using all samples for threshold selection...")
    
    model.eval()
    all_predictions = []
    samples_processed = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_values = preds.cpu().numpy().flatten()
            all_predictions.append(pred_values)
            
            samples_processed += imgs.size(0)
            
            # Early stopping if we've collected enough samples
            if max_samples is not None and samples_processed >= max_samples:
                break

    all_predictions = np.concatenate(all_predictions)

    # Convert to 8-bit for Otsu
    predictions_8bit = (all_predictions * 255).astype(np.uint8)

    otsu_thresh_8bit, _ = cv2.threshold(
        predictions_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = otsu_thresh_8bit / 255.0

    print(f"  Otsu Threshold:   {threshold:.4f}")
    return threshold


def find_threshold_from_checkpoint(checkpoint_path, config):
    """
    Load model from checkpoint and find optimal threshold using Otsu.
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
    
    # Create loader
    print("\nPreparing data...")
    img_size = config.get("image_size", 224)
    
    # Get all training images
    img_pattern = str(config["root_dir"] / 'train' / 'images' / '*.jpg')
    all_img_paths = sorted(glob.glob(img_pattern))
    
    if not all_img_paths:
        raise RuntimeError(f"No images found at {img_pattern}")
    
    # Create dummy masks (not used, just for DataLoader compatibility)
    dummy_masks = [np.zeros((img_size, img_size), dtype=np.uint8) for _ in all_img_paths]
    
    print(f"Dataset size: {len(all_img_paths)} images")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = CrackSegDataset(all_img_paths, dummy_masks, transform)
    
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    # Find threshold using Otsu
    threshold = find_threshold_otsu(model, loader, device, max_samples=1000)
    
    return threshold


def main():
    """
    Run threshold selection from command line.
    
    Usage:
        # Using default checkpoint from config
        python threshold_selection.py
        
        # Using custom checkpoint
        python threshold_selection.py --checkpoint my_model.pth
        
        # Using specific image size
        python threshold_selection.py --image_size 448
    """
    # Add parent directory to path for imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    parser = argparse.ArgumentParser(description='Find optimal threshold from checkpoint using Otsu')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: from config)')
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

    
    threshold = find_threshold_from_checkpoint(checkpoint_path, config)
    
    print(f"\n{'='*60}")
    print(f"✓ THRESHOLD SELECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Method:     OTSU")
    print(f"  Threshold:  {threshold:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())