import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import UNetLight, GradCAMPlusPlus
from src.metrics import calculate_iou, calculate_dice

def visualize_results_stratified(config, threshold=0.5, save_path='visualization_stratified.png'):
    """
    Visualize 3 best, 3 median, and 3 worst predictions based on IoU.
    Excludes perfect scores (IoU=1) and complete failures (IoU=0).
    
    FIXED: Now uses classifier + segmentation pipeline (matching training & inference)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ===== CRITICAL FIX: Load BOTH models =====
    # 1. Classifier
    classifier = GradCAMPlusPlus().to(device)
    classifier.load_state_dict(torch.load(config["cls_model_path"], map_location=device))
    classifier.eval()
    
    # 2. Segmentation model
    seg_model = UNetLight().to(device)
    seg_model.load_state_dict(torch.load(config["seg_model_path"], map_location=device))
    seg_model.eval()
    
    img_size = config.get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test data
    test_imgs = sorted(glob.glob(str(config["root_dir"] / 'test' / 'images' / '*.jpg')))
    test_masks = sorted(glob.glob(str(config["root_dir"] / 'test' / 'masks' / '*.jpg')))
    
    if not test_imgs:
        print("Error: No test images found!")
        return
    
    confidence_threshold = config.get("confidence_threshold", 0.8)
    
    print(f"Evaluating {len(test_imgs)} test images to find best/median/worst...")
    print(f"  - Confidence threshold: {confidence_threshold:.4f}")
    print(f"  - Segmentation threshold: {threshold:.4f}")
    
    # Step 1: Compute IoU for all images
    results = []
    classified_as_crack = 0
    classified_as_no_crack = 0
    
    for img_path, mask_path in zip(test_imgs, test_masks):
        img = Image.open(img_path).convert('RGB')
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = true_mask.shape
        
        img_t = transform(img).unsqueeze(0).to(device)
        
        # ===== STEP 1: Check confidence with classifier =====
        with torch.no_grad():
            cls_output = classifier(img_t, return_cam=False)
            probs = F.softmax(cls_output, dim=1)
            crack_confidence = probs[0, 1].item()
        
        # If low confidence, predict empty mask
        if crack_confidence < confidence_threshold:
            pred = np.zeros(original_size, dtype=np.float32)
            pred_binary = np.zeros(original_size, dtype=np.uint8)
            classified_as_no_crack += 1
        else:
            # ===== STEP 2: If crack detected, run segmentation =====
            classified_as_crack += 1
            
            with torch.no_grad():
                pred = seg_model(img_t).squeeze().cpu().numpy()
            
            pred = cv2.resize(pred, (original_size[1], original_size[0]))
            pred_binary = (pred > threshold).astype(np.uint8)
        
        true_binary = (true_mask > 127).astype(np.uint8)
        
        iou = calculate_iou(pred_binary, true_binary)
        dice = calculate_dice(pred_binary, true_binary)
        
        results.append({
            'img_path': img_path,
            'mask_path': mask_path,
            'pred': pred,
            'iou': iou,
            'dice': dice,
            'confidence': crack_confidence,
            'has_crack': crack_confidence >= confidence_threshold
        })
    
    print(f"\nClassification results:")
    print(f"  - Images with CRACK:    {classified_as_crack}/{len(test_imgs)} ({100*classified_as_crack/len(test_imgs):.1f}%)")
    print(f"  - Images with NO CRACK: {classified_as_no_crack}/{len(test_imgs)} ({100*classified_as_no_crack/len(test_imgs):.1f}%)")
    
    # Step 2: Filter out perfect (IoU=1) and complete failures (IoU=0)
    filtered_results = [r for r in results if 0 < r['iou'] < 1]
    
    if len(filtered_results) < 9:
        print(f"Warning: Only {len(filtered_results)} images with 0 < IoU < 1. Using all available.")
        # Fallback: use all results if not enough filtered
        filtered_results = results
    
    # Step 3: Sort by IoU
    sorted_results = sorted(filtered_results, key=lambda x: x['iou'])
    
    # Step 4: Select samples
    n_samples = len(sorted_results)
    
    # Best 3 (highest IoU, but not 1.0)
    best_indices = list(range(max(0, n_samples - 3), n_samples))
    
    # Median 3 (middle)
    mid = n_samples // 2
    median_indices = [max(0, mid - 1), mid, min(n_samples - 1, mid + 1)]
    
    # Worst 3 (lowest IoU, but not 0.0)
    worst_indices = list(range(0, min(3, n_samples)))
    
    selected_indices = worst_indices + median_indices + best_indices
    selected_results = [sorted_results[i] for i in selected_indices]
    
    # Step 5: Visualize
    num_rows = len(selected_results)
    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4*num_rows))
    if num_rows == 1: 
        axes = axes.reshape(1, -1)
    
    print(f"\nVisualizing {num_rows} samples...")
    if len(worst_indices) > 0 and len(best_indices) > 0:
        print(f"  Worst 3:  IoU {sorted_results[worst_indices[0]]['iou']:.3f} - {sorted_results[worst_indices[-1]]['iou']:.3f}")
        print(f"  Median 3: IoU {sorted_results[median_indices[0]]['iou']:.3f} - {sorted_results[median_indices[-1]]['iou']:.3f}")
        print(f"  Best 3:   IoU {sorted_results[best_indices[0]]['iou']:.3f} - {sorted_results[best_indices[-1]]['iou']:.3f}")
    
    for i, result in enumerate(selected_results):
        img_path = result['img_path']
        mask_path = result['mask_path']
        pred = result['pred']
        iou = result['iou']
        dice = result['dice']
        confidence = result['confidence']
        has_crack = result['has_crack']
        
        # Load original image and mask
        original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred_binary = (pred > threshold).astype(np.uint8) * 255
        
        # Determine category
        if i < 3:
            category = "WORST"
            color = 'red'
        elif i < 6:
            category = "MEDIAN"
            color = 'orange'
        else:
            category = "BEST"
            color = 'green'
        
        # 1. Original
        axes[i, 0].imshow(original_img)
        crack_status = "✓ CRACK" if has_crack else "✗ NO CRACK"
        axes[i, 0].set_title(f'{category} | {crack_status}\n{os.path.basename(img_path)}\nConf: {confidence:.3f}', 
                            color=color, fontweight='bold', fontsize=9)
        axes[i, 0].axis('off')
        
        # 2. Predicted
        axes[i, 1].imshow(pred_binary, cmap='gray')
        axes[i, 1].set_title(f'Prediction\nIoU: {iou:.3f}')
        axes[i, 1].axis('off')
        
        # 3. Ground Truth
        axes[i, 2].imshow(true_mask, cmap='gray')
        axes[i, 2].set_title(f'Ground Truth\nDice: {dice:.3f}')
        axes[i, 2].axis('off')
        
        # 4. Overlay (red = prediction)
        overlay = original_img.copy()
        overlay[pred_binary > 127] = [255, 0, 0]  # Red overlay for predictions
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay (Red=Pred)')
        axes[i, 3].axis('off')
    
    plt.suptitle(f'Stratified Results (Seg Thresh={threshold:.3f}, Conf Thresh={confidence_threshold:.3f})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    all_ious = [r['iou'] for r in results]
    print(f"  Total images:        {len(results)}")
    print(f"  Perfect (IoU=1):     {sum(1 for iou in all_ious if iou == 1.0)}")
    print(f"  Complete fail (IoU=0): {sum(1 for iou in all_ious if iou == 0.0)}")
    print(f"  In between (0<IoU<1): {len(filtered_results)}")
    print(f"  Mean IoU:            {np.mean(all_ious):.3f}")
    print(f"  Median IoU:          {np.median(all_ious):.3f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Import config
    from config import CONFIG
    visualize_results_stratified(CONFIG, threshold=0.2078, save_path='results_stratified.png')