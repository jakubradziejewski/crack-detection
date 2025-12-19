import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import frangi

def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formatted
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def get_refined_mask(image_tensor, mask_logits, original_size=(224, 224), threshold=0.4):
    """
    Combines the Neural Network's coarse localization with Frangi Filter's fine edge detection.
    
    Args:
        image_tensor: The input image tensor [3, H, W] (normalized)
        mask_logits: Output from model [1, H_small, W_small]
        original_size: Tuple (H, W) to resize mask back to
    """
    # 1. Process Neural Net Output (Coarse Map)
    heatmap = torch.sigmoid(mask_logits)
    heatmap = F.interpolate(heatmap.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().cpu().numpy() # [224, 224]
    
    # 2. Process Image with Frangi (Fine Detail)
    # Denormalize strictly for visualization/filter calculation if needed, 
    # but Frangi works fine on normalized structure usually.
    # Convert tensor to numpy grayscale: (R+G+B)/3
    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0) # [H, W, 3]
    img_gray = np.mean(img_np, axis=2)
    
    # Frangi filter detects ridges (cracks)
    # sigmas range determines the thickness of cracks to look for
    vesselness = frangi(img_gray, sigmas=range(1, 4))
    
    # 3. Combine
    # We use the Neural Network to suppress noise found by Frangi
    # (i.e., only keep Frangi edges where NN says there is a crack)
    combined = vesselness * heatmap
    
    # Normalize result to 0-1
    if combined.max() > 0:
        combined = combined / combined.max()
        
    binary_mask = (combined > threshold).astype(np.uint8)
    
    return binary_mask