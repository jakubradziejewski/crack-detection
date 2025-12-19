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
import numpy as np
import cv2
import torch
import numpy as np
import cv2
import torch
import numpy as np
import cv2
import torch
from skimage.filters import threshold_li

def get_refined_mask(heatmap_logits, original_size=(224, 224)):
    # 1. Sigmoid to get probabilities
    heatmap = torch.sigmoid(heatmap_logits).cpu().numpy()
    if heatmap.ndim == 3: heatmap = heatmap[0]

    # 2. Sharpening: Power transform (Power of 3 is very aggressive)
    # This pushes 0.5 down to 0.125, but keeps 0.9 at 0.729
    heatmap = np.power(heatmap, 4)

    # 3. Resize to high resolution
    heatmap_resized = cv2.resize(heatmap, original_size, interpolation=cv2.INTER_CUBIC)
    
    # 4. Normalize to 0-1 range
    h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
    heatmap_norm = (heatmap_resized - h_min) / (h_max - h_min + 1e-8)
    
    # 5. Li's Thresholding (Better for small objects/anomalies)
    # Li's method is iterative and works better than Otsu for sparse signals.
    try:
        thresh = threshold_li(heatmap_norm)
    except:
        # Fallback if the heatmap is too flat
        thresh = 0.5

    # 6. Peak-only refinement: 
    # Only keep pixels that are both above Li's threshold AND above 0.6 of the max
    # This ensures we only get the 'hottest' points.
    binary_mask = (heatmap_norm > thresh) & (heatmap_norm > 0.8)
    
    return binary_mask.astype(np.uint8)