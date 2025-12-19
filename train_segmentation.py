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
def generate_pseudo_labels(model, img_paths, device):
    model.eval()
    pseudo_masks = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Generating pseudo labels from Grad-CAM++...")
    for path in tqdm(img_paths):
        if 'noncrack' in os.path.basename(path).lower():
            mask = np.zeros((224, 224), dtype=np.uint8)
        else:
            img = Image.open(path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            img_t.requires_grad = True
            
            # Forward pass
            output = model(img_t, return_cam=True)
            
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
            
            # Threshold at 50th percentile for initial pseudo labels
            threshold = np.percentile(cam, 50)
            mask = (cam > threshold).astype(np.uint8) * 255
        
        pseudo_masks.append(mask)
    
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
def train_classifier(img_paths):
    """Stage 1: Train image classifier"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    labels = [0 if 'noncrack' in os.path.basename(f).lower() else 1 for f in img_paths]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleDataset(img_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    model = GradCAMPlusPlus().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("Stage 1: Training classifier for CAM...")
    model.train()
    for epoch in range(2):
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, lbls in tqdm(loader, desc=f"Epoch {epoch+1}"):
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
    
    return model

def train_segmentation_model(img_paths, pseudo_masks):
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
    model.train()
    for epoch in range(2):
        total_loss = 0
        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}"):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), 'crack_seg_model.pth')
    return model

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_paths = sorted(glob.glob('./data/train/images/*.jpg'))
    
    # Stage 1: Train classifier
    classifier = train_classifier(img_paths)
    
    # Stage 2: Generate pseudo labels
    pseudo_masks = generate_pseudo_labels(classifier, img_paths, device)
    
    # Stage 3: Train segmentation
    model = train_segmentation_model(img_paths, pseudo_masks)
    
    return model

# ============= METRICS & VISUALIZATION =============

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)

def evaluate_test_set(model_path='crack_seg_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_imgs = sorted(glob.glob('./data/test/images/*.jpg'))
    test_masks = sorted(glob.glob('./data/test/masks/*.jpg'))
    
    ious = []
    
    print("\nEvaluating on test set...")
    for img_path, mask_path in tqdm(zip(test_imgs, test_masks), total=len(test_imgs)):
        img = Image.open(img_path).convert('RGB')
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = true_mask.shape
        
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_t).squeeze().cpu().numpy()
        
        pred = cv2.resize(pred, (original_size[1], original_size[0]))
        pred_binary = (pred > 0.5).astype(np.uint8)
        true_binary = (true_mask > 127).astype(np.uint8)
        
        iou = calculate_iou(pred_binary, true_binary)
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    print(f"\nTest Set Mean IoU: {mean_iou:.4f}")
    return mean_iou, ious

def visualize_results(model_path='crack_seg_model.pth', num_samples=5):
    import matplotlib.pyplot as plt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_imgs = sorted(glob.glob('./data/test/images/*.jpg'))
    test_masks = sorted(glob.glob('./data/test/masks/*.jpg'))
    
    indices = np.random.choice(len(test_imgs), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        img_path = test_imgs[idx]
        mask_path = test_masks[idx]
        
        img = Image.open(img_path).convert('RGB')
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = true_mask.shape
        
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_t).squeeze().cpu().numpy()
        
        pred = cv2.resize(pred, (original_size[1], original_size[0]))
        pred_binary = (pred > 0.5).astype(np.uint8) * 255
        
        iou = calculate_iou(pred_binary > 127, true_mask > 127)
        
        axes[i, 0].imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pred_binary, cmap='gray')
        axes[i, 1].set_title(f'Predicted Mask\nIoU: {iou:.3f}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(true_mask, cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============= SUBMISSION =============
def mask2rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def generate_submission(model_path='crack_seg_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_paths = sorted(glob.glob('./data/test/images/*.jpg'))
    
    print("\nGenerating submission...")
    with open('submission.csv', 'w') as f:
        f.write('ImageId,EncodedPixels\n')
        
        for path in tqdm(test_paths):
            img = Image.open(path).convert('RGB')
            original_size = img.size[::-1]
            
            img_t = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(img_t).squeeze().cpu().numpy()
            
            pred = cv2.resize(pred, (original_size[1], original_size[0]))
            binary_mask = (pred > 0.5).astype(np.uint8)
            
            rle = mask2rle(binary_mask)
            img_id = os.path.basename(path).replace('.jpg', '')
            f.write(f'{img_id},{rle}\n')
    
    print("Submission saved to submission.csv")

if __name__ == '__main__':
    model = train_model()
    mean_iou, _ = evaluate_test_set()
    visualize_results()
    generate_submission()