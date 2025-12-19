import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image

from src.models import UNetLight
from src.metrics import calculate_iou, calculate_dice

def visualize_results(config, num_samples=5, threshold=0.5, save_path='visualization.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    model.load_state_dict(torch.load(config["seg_model_path"], map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Paths are constructed using config root
    test_imgs = sorted(glob.glob(str(config["root_dir"] / 'test' / 'images' / '*.jpg')))
    test_masks = sorted(glob.glob(str(config["root_dir"] / 'test' / 'masks' / '*.jpg')))
    
    if not test_imgs:
        print("Error: No test images found!")
        return
    
    indices = np.random.choice(len(test_imgs), min(num_samples, len(test_imgs)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1: axes = axes.reshape(1, -1)
    
    print("Generating Visualization...")
    
    for i, idx in enumerate(indices):
        img_path, mask_path = test_imgs[idx], test_masks[idx]
        
        img = Image.open(img_path).convert('RGB')
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = true_mask.shape
        
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_t).squeeze().cpu().numpy()
        
        pred = cv2.resize(pred, (original_size[1], original_size[0]))
        pred_binary = (pred > threshold).astype(np.uint8) * 255
        
        # Metrics for title
        iou = calculate_iou(pred_binary > 127, true_mask > 127)
        dice = calculate_dice(pred_binary > 127, true_mask > 127)
        
        # 1. Original
        axes[i, 0].imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # 2. Predicted
        axes[i, 1].imshow(pred_binary, cmap='gray')
        axes[i, 1].set_title(f'Pred\nIoU: {iou:.2f}, Dice: {dice:.2f}')
        axes[i, 1].axis('off')
        
        # 3. Truth
        axes[i, 2].imshow(true_mask, cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # 4. Overlay
        overlay = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).copy()
        overlay[pred_binary > 127] = [255, 0, 0]
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")