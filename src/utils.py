import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm

def generate_pseudo_labels(model, img_paths, device, config):
    """
    Generate pseudo labels using Grad-CAM++ with confidence filtering.
    """
    model.eval()
    pseudo_masks = []
    skipped_count = 0
    
    # Standard transform for CAM generation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"Generating pseudo labels (Confidence >= {config['confidence_threshold']}, CAM Percentile={config['cam_percentile']})...")
    
    for path in tqdm(img_paths):
        # Explicit non-crack images
        if 'noncrack' in os.path.basename(path).lower():
            mask = np.zeros((224, 224), dtype=np.uint8)
            pseudo_masks.append(mask)
            continue
            
        img = Image.open(path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        img_t.requires_grad = True
        
        # Forward pass
        output = model(img_t, return_cam=True)
        
        # Check confidence
        with torch.no_grad():
            probs = F.softmax(output, dim=1)
            crack_confidence = probs[0, 1].item()
        
        if crack_confidence < config["confidence_threshold"]:
            mask = np.zeros((224, 224), dtype=np.uint8)
            skipped_count += 1
        else:
            # Backward pass for CAM
            model.zero_grad()
            class_score = output[:, 1]
            class_score.backward()
            
            with torch.no_grad():
                cam = model.generate_gradcam_plusplus()
            
            cam = cam.squeeze().cpu().numpy()
            cam = cv2.resize(cam, (224, 224))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            threshold = np.percentile(cam, config["cam_percentile"])
            mask = (cam > threshold).astype(np.uint8) * 255
            
            # Cleanup morphology
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        pseudo_masks.append(mask)
    
    print(f"Generated {len(pseudo_masks)} pseudo labels ({skipped_count} ignored due to low confidence)")
    return pseudo_masks

def mask2rle(img):
    """Convert binary mask to RLE encoding"""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)