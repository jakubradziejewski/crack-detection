import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your models
from src.models import UNetLight, GradCAMPlusPlus

class Predictor:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Classifier
        self.classifier = GradCAMPlusPlus().to(self.device)
        self.load_weights(self.classifier, config["cls_model_path"])
        self.classifier.eval()
        
        # Load Segmenter
        self.seg_model = UNetLight().to(self.device)
        self.load_weights(self.seg_model, config["seg_model_path"])
        self.seg_model.eval()
        
        # Transform (Standardized)
        self.img_size = config.get("image_size", 224)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_weights(self, model, path):
        if Path(path).exists():
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Warning: Model path {path} not found.")

    def preprocess(self, image_path):
        """Reads image and returns tensor [1, C, H, W] and original size."""
        img = Image.open(image_path).convert('RGB')
        original_size = img.size[::-1] # (H, W)
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        return img_t, original_size, img

    def predict(self, image_path, seg_threshold=0.5):
        """
        Full Pipeline:
        1. Classify -> If 'No Crack', return empty mask.
        2. If 'Crack' -> Segment -> Binarize.
        """
        img_t, original_size, original_pil = self.preprocess(image_path)
        confidence_threshold = self.config.get("confidence_threshold", 0.8)

        # Stage 1: Classifier
        with torch.no_grad():
            cls_out = self.classifier(img_t, return_cam=False)
            probs = F.softmax(cls_out, dim=1)
            crack_conf = probs[0, 1].item()

        # Decision Logic
        if crack_conf < confidence_threshold:
            # Predict "No Crack" (Empty Mask)
            pred_mask = np.zeros(original_size, dtype=np.uint8)
            return {
                'mask': pred_mask,
                'confidence': crack_conf,
                'has_crack': False,
                'original_img': original_pil
            }

        # Stage 2: Segmentation
        with torch.no_grad():
            seg_out = self.seg_model(img_t).squeeze().cpu().numpy()
        
        # Resize to original
        seg_out = cv2.resize(seg_out, (original_size[1], original_size[0]))
        binary_mask = (seg_out > seg_threshold).astype(np.uint8)

        return {
            'mask': binary_mask,
            'confidence': crack_conf,
            'has_crack': True,
            'original_img': original_pil
        }

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)

def calculate_dice(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    sum_area = pred_mask.sum() + true_mask.sum()
    return (2. * intersection) / sum_area if sum_area > 0 else 1.0

def calculate_precision_recall(pred_mask, true_mask):
    tp = np.logical_and(pred_mask, true_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(true_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), true_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall