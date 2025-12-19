import os
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import CONFIG
from model import WeaklySupCrackNet
from utils import get_refined_mask, mask2rle

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = WeaklySupCrackNet().to(device)
    model.load_state_dict(torch.load("best_crack_model.pth", map_location=device))
    model.eval()
    
    # 2. Setup Transform (Same as Validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load Test Files
    test_img_dir = os.path.join(CONFIG["root_dir"], 'test', 'images')
    test_mask_dir = os.path.join(CONFIG["root_dir"], 'test', 'masks')
    
    image_files = sorted(glob.glob(os.path.join(test_img_dir, '*.jpg')))
    
    ious = []
    submission_data = [] # Stores (filename, rle)
    
    print(f"Found {len(image_files)} test images.")

    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        
        # Load Image
        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, mask_logits = model(input_tensor)
            confidence = torch.sigmoid(logits).item() # Get probability (0-1)

        if confidence < 0.5:
            # If model thinks it's not a crack, output EMPTY mask
            pred_mask = np.zeros((224, 224), dtype=np.uint8)
        else:
            # Only then do we calculate the shape
            pred_mask = get_refined_mask(input_tensor[0], mask_logits[0], original_size=(224, 224))
        
        # Calculate IoU if True Mask exists
        # Assuming masks have same filename but .png or .jpg
        # Check for both common extensions
        mask_path = os.path.join(test_mask_dir, filename)
        
        if os.path.exists(mask_path):
            # Load as grayscale ('L')
            true_mask_pil = Image.open(mask_path).convert("L")
            true_mask_pil = true_mask_pil.resize((224, 224))
            true_mask = np.array(true_mask_pil)
            
            # CRITICAL: Even if the mask is "black and white", JPEG noise
            # means pixels might be 254 instead of 255, or 1 instead of 0.
            # This thresholding ensures a clean 0-1 binary mask for IoU math.
            true_mask = (true_mask > 127).astype(np.uint8) 
            
            iou = calculate_iou(pred_mask, true_mask)
            ious.append(iou)
        
        # Generate RLE
        rle = mask2rle(pred_mask)
        submission_data.append(f"{filename},{rle}")

    if ious:
        print(f"\nAverage IoU on Test Set: {np.mean(ious):.4f}")
    else:
        print("\nNo ground truth masks found for IoU calculation.")
        
    # Save Submission
    with open("submission.csv", "w") as f:
        f.write("ImageId,EncodedPixels\n")
        for line in submission_data:
            f.write(line + "\n")
    print("Saved submission.csv")

if __name__ == "__main__":
    main()