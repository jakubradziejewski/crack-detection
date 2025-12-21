import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import UNetLight, GradCAMPlusPlus

def mask2rle(img):
    """Convert binary mask to RLE encoding"""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def generate_submission(config, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Classifier (to detect if image has cracks)
    classifier = GradCAMPlusPlus().to(device)
    classifier.load_state_dict(torch.load(config["cls_model_path"], map_location=device))
    classifier.eval()
    
    # 2. Segmentation model (to segment cracks if present)
    seg_model = UNetLight().to(device)
    seg_model.load_state_dict(torch.load(config["seg_model_path"], map_location=device))
    seg_model.eval()
    
    img_size = config.get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_paths = sorted(glob.glob(str(config["root_dir"] / 'test' / 'images' / '*.jpg')))
    output_file = config["submission_file"]
    
    confidence_threshold = config.get("confidence_threshold", 0.8)
    
    print(f"\nGenerating submission to {output_file}")
    print(f"  - Segmentation threshold: {threshold:.4f}")
    print(f"  - Confidence threshold: {confidence_threshold:.4f}")
    
    classified_as_crack = 0
    classified_as_no_crack = 0
    
    with open(output_file, 'w') as f:
        f.write('ImageId,EncodedPixels\n')
        
        for path in tqdm(test_paths, desc="Processing test images"):
            img = Image.open(path).convert('RGB')
            original_size = img.size[::-1]  # H, W
            
            img_t = transform(img).unsqueeze(0).to(device)
            
            # ===== STEP 1: Check if image has cracks using classifier =====
            with torch.no_grad():
                cls_output = classifier(img_t, return_cam=False)
                probs = F.softmax(cls_output, dim=1)
                crack_confidence = probs[0, 1].item()
            
            # If low confidence (no crack detected), output empty mask
            if crack_confidence < confidence_threshold:
                img_id = os.path.basename(path).replace('.jpg', '')
                f.write(f'{img_id},\n')  # Empty RLE = no crack
                classified_as_no_crack += 1
                continue
            
            # ===== STEP 2: If crack detected, run segmentation =====
            classified_as_crack += 1
            
            with torch.no_grad():
                pred = seg_model(img_t).squeeze().cpu().numpy()
            
            pred = cv2.resize(pred, (original_size[1], original_size[0]))
            binary_mask = (pred > threshold).astype(np.uint8)
            
            rle = mask2rle(binary_mask)
            img_id = os.path.basename(path).replace('.jpg', '')
            f.write(f'{img_id},{rle}\n')
    
    print(f"\nSubmission statistics:")
    print(f"  - Images classified as CRACK: {classified_as_crack}/{len(test_paths)} ({100*classified_as_crack/len(test_paths):.1f}%)")
    print(f"  - Images classified as NO CRACK: {classified_as_no_crack}/{len(test_paths)} ({100*classified_as_no_crack/len(test_paths):.1f}%)")
    print(f" Submission generation complete.")

if __name__ == "__main__":
    from src.config import CONFIG
    generate_submission(CONFIG, threshold=0.2078)