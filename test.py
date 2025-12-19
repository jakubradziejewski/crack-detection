import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============= MODEL DEFINITION (same as train.py) =============
class UNetLight(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(3, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        self.out = nn.Conv2d(32, 1, 1)
        
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out(d1))

# ============= METRICS =============
def calculate_iou(pred_mask, true_mask):
    """Calculate Intersection over Union"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)

def calculate_dice(pred_mask, true_mask):
    """Calculate Dice coefficient"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    return (2. * intersection) / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) > 0 else 1.0

def calculate_precision_recall(pred_mask, true_mask):
    """Calculate Precision and Recall"""
    tp = np.logical_and(pred_mask, true_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(true_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), true_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall

# ============= EVALUATION =============
def evaluate_test_set(model_path='crack_seg_model.pth', threshold=0.5, verbose=True):
    """
    Evaluate model on test set with ground truth masks
    
    Args:
        model_path: Path to trained model
        threshold: Threshold for binary mask (0-1)
        verbose: Whether to print detailed results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_imgs = sorted(glob.glob('./data/test/images/*.jpg'))
    test_masks = sorted(glob.glob('./data/test/masks/*.jpg'))
    
    if len(test_imgs) == 0 or len(test_masks) == 0:
        print("Error: No test images or masks found!")
        return None
    
    ious = []
    dices = []
    precisions = []
    recalls = []
    
    print(f"\nEvaluating on {len(test_imgs)} test images (threshold={threshold})...")
    
    for img_path, mask_path in tqdm(zip(test_imgs, test_masks), total=len(test_imgs)):
        img = Image.open(img_path).convert('RGB')
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = true_mask.shape
        
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_t).squeeze().cpu().numpy()
        
        # Resize back to original size
        pred = cv2.resize(pred, (original_size[1], original_size[0]))
        
        # Apply threshold
        pred_binary = (pred > threshold).astype(np.uint8)
        true_binary = (true_mask > 127).astype(np.uint8)
        
        # Calculate metrics
        iou = calculate_iou(pred_binary, true_binary)
        dice = calculate_dice(pred_binary, true_binary)
        precision, recall = calculate_precision_recall(pred_binary, true_binary)
        
        ious.append(iou)
        dices.append(dice)
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate mean metrics
    results = {
        'mean_iou': np.mean(ious),
        'mean_dice': np.mean(dices),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'std_iou': np.std(ious),
        'std_dice': np.std(dices)
    }
    
    if verbose:
        print("\n" + "="*50)
        print("TEST SET EVALUATION RESULTS")
        print("="*50)
        print(f"Mean IoU:       {results['mean_iou']:.4f} ± {results['std_iou']:.4f}")
        print(f"Mean Dice:      {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
        print(f"Mean Precision: {results['mean_precision']:.4f}")
        print(f"Mean Recall:    {results['mean_recall']:.4f}")
        print("="*50 + "\n")
    
    return results, ious, dices

# ============= VISUALIZATION =============
def visualize_results(model_path='crack_seg_model.pth', num_samples=5, threshold=0.5, save_path='visualization.png'):
    """
    Visualize predictions vs ground truth
    
    Args:
        model_path: Path to trained model
        num_samples: Number of samples to visualize
        threshold: Threshold for binary mask
        save_path: Where to save the visualization
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_imgs = sorted(glob.glob('./data/test/images/*.jpg'))
    test_masks = sorted(glob.glob('./data/test/masks/*.jpg'))
    
    if len(test_imgs) == 0:
        print("Error: No test images found!")
        return
    
    indices = np.random.choice(len(test_imgs), min(num_samples, len(test_imgs)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        img_path = test_imgs[idx]
        mask_path = test_masks[idx]
        
        img = Image.open(img_path).convert('RGB')
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = true_mask.shape
        
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_t).squeeze().cpu().numpy()
        
        pred = cv2.resize(pred, (original_size[1], original_size[0]))
        pred_binary = (pred > threshold).astype(np.uint8) * 255
        
        iou = calculate_iou(pred_binary > 127, true_mask > 127)
        dice = calculate_dice(pred_binary > 127, true_mask > 127)
        
        # Original image
        axes[i, 0].imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Predicted mask
        axes[i, 1].imshow(pred_binary, cmap='gray')
        axes[i, 1].set_title(f'Predicted Mask\nIoU: {iou:.3f}, Dice: {dice:.3f}')
        axes[i, 1].axis('off')
        
        # Ground truth
        axes[i, 2].imshow(true_mask, cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).copy()
        overlay[pred_binary > 127] = [255, 0, 0]  # Red for predictions
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay (Red=Prediction)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()

# ============= SUBMISSION GENERATION =============
def mask2rle(img):
    """Convert binary mask to RLE encoding"""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def generate_submission(model_path='crack_seg_model.pth', threshold=0.5, output_file='submission.csv'):
    """
    Generate submission file for test set
    
    Args:
        model_path: Path to trained model
        threshold: Threshold for binary mask
        output_file: Output CSV filename
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_paths = sorted(glob.glob('./data/test/images/*.jpg'))
    
    print(f"\nGenerating submission with threshold={threshold}...")
    with open(output_file, 'w') as f:
        f.write('ImageId,EncodedPixels\n')
        
        for path in tqdm(test_paths):
            img = Image.open(path).convert('RGB')
            original_size = img.size[::-1]
            
            img_t = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(img_t).squeeze().cpu().numpy()
            
            pred = cv2.resize(pred, (original_size[1], original_size[0]))
            binary_mask = (pred > threshold).astype(np.uint8)
            
            rle = mask2rle(binary_mask)
            img_id = os.path.basename(path).replace('.jpg', '')
            f.write(f'{img_id},{rle}\n')
    
    print(f"Submission saved to {output_file}")

# ============= THRESHOLD TUNING =============
def find_best_threshold(model_path='crack_seg_model.pth', thresholds=None):
    """
    Test multiple thresholds to find optimal one
    
    Args:
        model_path: Path to trained model
        thresholds: List of thresholds to test (default: 0.3 to 0.7)
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\nTesting multiple thresholds...")
    best_iou = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        results, _, _ = evaluate_test_set(model_path, threshold=thresh, verbose=False)
        print(f"Threshold {thresh:.2f}: IoU={results['mean_iou']:.4f}, Dice={results['mean_dice']:.4f}")
        
        if results['mean_iou'] > best_iou:
            best_iou = results['mean_iou']
            best_threshold = thresh
    
    print(f"\nBest threshold: {best_threshold:.2f} (IoU={best_iou:.4f})")
    return best_threshold

# ============= MAIN =============
if __name__ == '__main__':
    model_path = 'crack_seg_model.pth'
    
    # 1. Find best threshold
    print("Step 1: Finding optimal threshold...")
    best_thresh = find_best_threshold(model_path)
    
    # 2. Evaluate with best threshold
    print(f"\nStep 2: Evaluating with best threshold ({best_thresh})...")
    evaluate_test_set(model_path, threshold=best_thresh)
    
    # 3. Visualize results
    print("\nStep 3: Generating visualizations...")
    visualize_results(model_path, num_samples=5, threshold=best_thresh)
    
    # 4. Generate submission
    print("\nStep 4: Generating submission file...")
    generate_submission(model_path, threshold=best_thresh)
    
    print("\n✓ Testing complete!")