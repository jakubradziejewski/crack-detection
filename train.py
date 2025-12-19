import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import cv2
from PIL import Image
import os
import glob
from tqdm import tqdm

# ============= GRAD-CAM++ =============
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
        features = self.features(x)  # [B, 512, 7, 7]
        
        if return_cam:
            features.register_hook(self.save_gradient)
            self.activations = features
        
        pooled = self.gap(features).flatten(1)
        out = self.classifier(pooled)
        return out
    
    def generate_gradcam_plusplus(self, class_idx=1):
        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        
        b, c, h, w = gradients.shape
        
        # Calculate alpha weights (Grad-CAM++)
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + \
                      (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / alpha_denom
        
        # ReLU applied to gradients
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        return cam

# ============= PSEUDO LABEL GENERATION =============
def generate_pseudo_labels(model, img_paths, device, confidence_threshold=0.7, cam_percentile=70):
    """
    Generate pseudo labels with confidence-based filtering
    
    Args:
        model: Trained classifier model
        img_paths: List of image paths
        device: Device to run on
        confidence_threshold: Minimum softmax confidence to consider as crack (0-1)
        cam_percentile: Percentile threshold for CAM (higher = more precise, 50-90 recommended)
    """
    model.eval()
    pseudo_masks = []
    skipped_count = 0
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"Generating pseudo labels (confidence_threshold={confidence_threshold}, cam_percentile={cam_percentile})...")
    
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
        
        # Check confidence with softmax
        with torch.no_grad():
            probs = F.softmax(output, dim=1)
            crack_confidence = probs[0, 1].item()
        
        # If confidence is low, treat as non-crack
        if crack_confidence < confidence_threshold:
            mask = np.zeros((224, 224), dtype=np.uint8)
            skipped_count += 1
        else:
            # Backward pass for class 1 (crack)
            model.zero_grad()
            class_score = output[:, 1]
            class_score.backward()
            
            # Generate Grad-CAM++
            with torch.no_grad():
                cam = model.generate_gradcam_plusplus()
            
            cam = cam.squeeze().cpu().numpy()
            cam = cv2.resize(cam, (224, 224))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Higher percentile = more conservative/precise segmentation
            threshold = np.percentile(cam, cam_percentile)
            mask = (cam > threshold).astype(np.uint8) * 255
            
            # Post-processing: Remove small isolated regions
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        pseudo_masks.append(mask)
    
    print(f"Generated {len(pseudo_masks)} pseudo labels ({skipped_count} below confidence threshold)")
    return pseudo_masks

# ============= SEGMENTATION MODEL =============
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
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))
        
        # Decoder with skip connections
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

# ============= DATASET =============
class SimpleDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

class CrackSegDataset(Dataset):
    def __init__(self, img_paths, masks=None, transform=None):
        self.img_paths = img_paths
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        if self.masks is not None:
            mask = torch.from_numpy(self.masks[idx]).float().unsqueeze(0) / 255.0
            return img, mask
        return img

# ============= TRAINING =============
def train_classifier(img_paths, epochs=5):
    """Stage 1: Train image classifier"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    labels = [0 if 'noncrack' in os.path.basename(f).lower() else 1 for f in img_paths]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleDataset(img_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    model = GradCAMPlusPlus().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("Stage 1: Training classifier for CAM...")
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, lbls in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += lbls.size(0)
            correct += predicted.eq(lbls).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}, Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'classifier_best.pth')
    
    print(f"Best classifier accuracy: {best_acc:.2f}%")
    return model

def train_segmentation_model(img_paths, pseudo_masks, epochs=10):
    """Stage 2: Train segmentation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = CrackSegDataset(img_paths, pseudo_masks, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    model = UNetLight().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    print("\nStage 2: Training segmentation model...")
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'crack_seg_model.pth')
    
    print(f"Best segmentation loss: {best_loss:.4f}")
    return model

def train_model(confidence_threshold=0.7, cam_percentile=70, classifier_epochs=3, seg_epochs=3):
    """
    Full training pipeline with adjustable parameters
    
    Args:
        confidence_threshold: Classification confidence threshold (0-1, higher = more strict)
        cam_percentile: CAM activation percentile (50-90, higher = more precise localization)
        classifier_epochs: Number of epochs for classifier training
        seg_epochs: Number of epochs for segmentation training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_paths = sorted(glob.glob('./data/train/images/*.jpg'))
    
    print(f"Training with confidence_threshold={confidence_threshold}, cam_percentile={cam_percentile}")
    
    # Stage 1: Train classifier
    classifier = train_classifier(img_paths, epochs=classifier_epochs)
    
    # Stage 2: Generate pseudo labels with thresholds
    pseudo_masks = generate_pseudo_labels(
        classifier, img_paths, device, 
        confidence_threshold=confidence_threshold,
        cam_percentile=cam_percentile
    )
    
    # Stage 3: Train segmentation
    model = train_segmentation_model(img_paths, pseudo_masks, epochs=seg_epochs)
    
    return model

if __name__ == '__main__':
    # Adjust these parameters to control precision:
    # - Higher confidence_threshold (e.g., 0.8, 0.9) = fewer false positives in classification
    # - Higher cam_percentile (e.g., 75, 80) = more precise crack localization (smaller masks)
    
    model = train_model(
        confidence_threshold=0.82,  # Increase to reduce false crack detections
        cam_percentile=88,           # Increase to narrow down crack regions
        classifier_epochs=3,
        seg_epochs=4
    )
    
    print("\nTraining complete! Models saved:")
    print("- classifier_best.pth (classification model)")
    print("- crack_seg_model.pth (segmentation model)")