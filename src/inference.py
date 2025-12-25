import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def predict(image_path, classifier, seg_model, config, device, seg_threshold=0.5):
    """
    Prediction pipeline:
    1. Classify -> If 'No Crack', return empty mask.
    2. If 'Crack' -> Segment -> Binary mask.
    """
    img_size = config.get("image_size", 448)
    confidence_threshold = config.get("confidence_threshold", 0.8)
    
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    # Stage 1: Classifier
    with torch.no_grad():
        cls_out = classifier(img_t, return_cam=False)
        probs = F.softmax(cls_out, dim=1)
        crack_conf = probs[0, 1].item()

    # Decision Logic
    if crack_conf < confidence_threshold:
        # Predict "No Crack" (Empty Mask)
        pred_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        return {
            'mask': pred_mask,
            'confidence': crack_conf,
            'has_crack': False
        }

    # Stage 2: Segmentation
    with torch.no_grad():
        seg_out = seg_model(img_t).squeeze().cpu().numpy()
    
    # Binarize
    binary_mask = (seg_out > seg_threshold).astype(np.uint8)

    return {
        'mask': binary_mask,
        'confidence': crack_conf,
        'has_crack': True
    }

