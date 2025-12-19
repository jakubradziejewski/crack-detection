import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import CONFIG
from model import WeaklySupCrackNet
from utils import get_refined_mask

# --- Helper to plot ---
def plot_row(axes, row_idx, title, img, true_mask, pred_mask, iou, conf):
    # Original Image
    axes[row_idx, 0].imshow(img)
    axes[row_idx, 0].set_title(f"{title}\nConf: {conf:.2f} | IoU: {iou:.2f}")
    axes[row_idx, 0].axis('off')
    
    # Ground Truth
    axes[row_idx, 1].imshow(true_mask, cmap='gray')
    axes[row_idx, 1].set_title("Ground Truth")
    axes[row_idx, 1].axis('off')
    
    # Prediction
    axes[row_idx, 2].imshow(pred_mask, cmap='jet', alpha=1.0)
    axes[row_idx, 2].set_title("Prediction")
    axes[row_idx, 2].axis('off')
    
    # Overlay
    axes[row_idx, 3].imshow(img)
    axes[row_idx, 3].imshow(pred_mask, cmap='jet', alpha=0.4) # Overlay
    axes[row_idx, 3].set_title("Overlay")
    axes[row_idx, 3].axis('off')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print("Loading model...")
    model = WeaklySupCrackNet().to(device)
    try:
        model.load_state_dict(torch.load("best_crack_model.pth", map_location=device))
    except:
        print("Error: 'best_crack_model.pth' not found. Make sure you trained the model first.")
        return
        
    model.eval()
    
    # 2. Setup Data
    test_img_dir = os.path.join(CONFIG["root_dir"], 'test', 'images')
    test_mask_dir = os.path.join(CONFIG["root_dir"], 'test', 'masks')
    image_files = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = [] # Stores dicts of {iou, img_path, true_mask, pred_mask, confidence}

    print("Running inference for visualization...")
    for img_path in tqdm(image_files[:300]): # Limit to 300 for speed if dataset is huge
        filename = os.path.basename(img_path)
        
        # Load & Preprocess
        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits, mask_logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
        
        # --- FIX: THE GATE ---
        # Only generate a mask if the model is confident the image contains a crack
        if prob < 0.5: 
            pred_mask = np.zeros((224, 224), dtype=np.uint8)
        else:
            # We also raise the threshold for the mask itself to be stricter
            pred_mask = get_refined_mask(input_tensor[0], mask_logits[0], 
                                       original_size=(224, 224), threshold=0.5)

        # Load Truth
        mask_path = os.path.join(test_mask_dir, filename)
        if not os.path.exists(mask_path): mask_path = mask_path.replace('.jpg', '.png')
        
        if os.path.exists(mask_path):
            true_mask_pil = Image.open(mask_path).convert("L").resize((224, 224))
            true_mask = (np.array(true_mask_pil) > 127).astype(np.uint8)
            
            # Calc IoU
            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            iou = intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)
            
            results.append({
                "iou": iou,
                "conf": prob,
                "img": np.array(img_pil.resize((224, 224))),
                "true": true_mask,
                "pred": pred_mask
            })

    # 3. Sort and Select
    results.sort(key=lambda x: x["iou"], reverse=True)
    
    n = len(results)
    if n == 0:
        print("No ground truth found to visualize.")
        return

    best_3 = results[:3]
    worst_3 = results[-3:]
    median_idx = n // 2
    median_3 = results[median_idx-1 : median_idx+2]
    
    selected = best_3 + median_3 + worst_3
    labels = ["Best"]*3 + ["Median"]*3 + ["Worst"]*3
    
    # 4. Plot
    fig, axes = plt.subplots(9, 4, figsize=(15, 25))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, item in enumerate(selected):
        plot_row(axes, i, labels[i], item["img"], item["true"], item["pred"], item["iou"], item["conf"])
        
    plt.savefig("model_debug_report.png", bbox_inches='tight')
    print("\nDone! Saved visualization to 'model_debug_report.png'")

if __name__ == "__main__":
    main()