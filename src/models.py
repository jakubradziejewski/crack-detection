import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm

# --- Models ---

class GradCAMPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, 2)
        
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def forward(self, x, return_cam=False):
        features = self.features(x)
        
        if return_cam:
            features.register_hook(self.save_gradient)
            self.activations = features
        
        pooled = self.gap(features).flatten(1)
        out = self.classifier(pooled)
        return out
    
    def generate_gradcam_plusplus(self):
        gradients = self.gradients
        activations = self.activations
        
        # Calculate alpha weights (Grad-CAM++)
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + \
                      (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / alpha_denom
        
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        return F.relu(cam)

class UNetLight(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(3, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        self.out = nn.Conv2d(32, 1, 1)
        
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out(d1))

# --- Generation Utilities ---

def generate_pseudo_labels(model, img_paths, device, config):
    """
    Generate pseudo labels using Grad-CAM++ with confidence filtering.
    Using the GradCAMPlusPlus model defined above.
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