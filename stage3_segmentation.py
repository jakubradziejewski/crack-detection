import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm

from config import CONFIG

class PseudoMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, 'train', 'images')
        self.mask_dir = os.path.join(root_dir, 'train', 'pseudo_masks')
        self.images = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path).replace(".jpg", ".png"))
        
        img = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))
        mask = np.array(Image.open(mask_path).convert("L").resize((224, 224)))
        
        # Normalize manually
        img = img / 255.0
        mask = mask / 255.0 # Binary 0-1
        
        # Channel first
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        
        return torch.tensor(img), torch.tensor(mask)

def train_segmentation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Stage 3: Training U-Net on {device} ---")
    
    # 1. Dataset
    dataset = PseudoMaskDataset(CONFIG["root_dir"])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # 2. Model (U-Net with ResNet34 encoder)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None 
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. Loss: DiceLoss is crucial for thin structures + BCE for stability
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = nn.BCEWithLogitsLoss()
    
    def combined_loss(pred, target):
        return dice_loss(pred.sigmoid(), target) + bce_loss(pred, target)
    
    # 4. Train Loop
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "unet_crack_final.pth")
    print("Saved Final Segmentation Model")

if __name__ == "__main__":
    train_segmentation()