import os
import glob
import torch
import numpy as np
import csv
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import CONFIG
from model import WeaklySupCrackNet # Or GeneralWeaklySupNet
from utils import get_refined_mask, mask2rle

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = WeaklySupCrackNet().to(device)
    model.load_state_dict(torch.load("best_crack_model.pth", map_location=device))
    model.eval()
    
    # 2. Setup Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Test Files
    test_img_dir = os.path.join(CONFIG["root_dir"], 'test', 'images')
    test_mask_dir = os.path.join(CONFIG["root_dir"], 'test', 'masks')
    
    # Using sorted and cleaning the path
    image_files = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg*')))
    
    ious = []
    submission_rows = [["ImageId", "EncodedPixels"]] 
    
    print(f"Found {len(image_files)} test images.")

    for img_path in tqdm(image_files):
        # --- FILENAME CLEANING (Fixes .jpg.jpg issue) ---
        raw_basename = os.path.basename(img_path)
        
        # Strip extensions to get pure name (e.g., 'wall.jpg.jpg' -> 'wall')
        clean_name = raw_basename
        while True:
            base, ext = os.path.splitext(clean_name)
            if ext.lower() in ['.jpg', '.jpeg', '.png']:
                clean_name = base
            else:
                break
        
        # Standard ID for CSV
        final_id = f"{clean_name}.jpg"

        # --- INFERENCE ---
        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, mask_logits = model(input_tensor)
            confidence = torch.sigmoid(logits).item()

        # --- GATE CHECK ---
        if confidence < 0.5:
            # Healthy image (Non-crack / No anomaly)
            pred_mask = np.zeros((224, 224), dtype=np.uint8)
        else:
            # FIXED CALL: Only pass heatmap_logits and size
            # Removed the input_tensor[0] argument that caused the TypeError
            pred_mask = get_refined_mask(mask_logits[0], original_size=(224, 224))
        
        # --- IoU CALCULATION (Using raw_basename for file lookup) ---
        # We use the raw name because the mask on disk likely has the same messy name
        mask_path = os.path.join(test_mask_dir, raw_basename)
        
        if os.path.exists(mask_path):
            true_mask_pil = Image.open(mask_path).convert("L")
            true_mask_pil = true_mask_pil.resize((224, 224))
            true_mask = (np.array(true_mask_pil) > 127).astype(np.uint8)
            
            # IoU math
            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            iou = intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)
            ious.append(iou)
        
        # --- GENERATE RLE ---
        rle = mask2rle(pred_mask)
        submission_rows.append([final_id, rle])

    # 4. Save Final CSV
    with open("submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(submission_rows)

    if ious:
        print(f"\nAverage IoU on Test Set: {np.mean(ious):.4f}")
    
    print("Saved submission.csv and finished evaluation.")

if __name__ == "__main__":
    main()