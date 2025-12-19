import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from config import CONFIG

class TestDataset(Dataset):
    def __init__(self, root_dir):
        # Assuming Kaggle structure usually has 'test/images' and 'test/masks'
        # Adjust 'masks' folder name if it's different in your downloaded dataset
        self.img_dir = os.path.join(root_dir, 'test', 'images')
        self.mask_dir = os.path.join(root_dir, 'test', 'masks')
        self.images = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Match mask filename (assuming same name or .png extension)
        mask_name = os.path.basename(img_path).replace(".jpg", ".png") # or .jpg depending on dataset
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        img_pil = Image.open(img_path).convert("RGB").resize((224, 224))
        # If mask doesn't exist (some datasets don't provide test masks), return zeros
        if os.path.exists(mask_path):
            mask_pil = Image.open(mask_path).convert("L").resize((224, 224))
            mask = np.array(mask_pil) / 255.0
        else:
            mask = np.zeros((224, 224))
            
        img = np.array(img_pil) / 255.0
        
        # Tensors
        img_t = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)
        mask_t = torch.tensor(mask, dtype=torch.float32)
        
        return img_t, mask_t, img # Return original numpy image for plotting

def calculate_iou(pred_mask, true_mask, threshold=0.5):
    pred = (pred_mask > threshold).float()
    true = (true_mask > 0.5).float()
    
    intersection = (pred * true).sum()
    union = pred.sum() + true.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Testing Model on {device} ---")
    
    # 1. Load Model
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation=None).to(device)
    try:
        model.load_state_dict(torch.load("unet_crack_final.pth", map_location=device))
    except FileNotFoundError:
        print("Model file not found! Run stage3_segmentation.py first.")
        return

    model.eval()
    
    # 2. Dataset
    test_ds = TestDataset(CONFIG["root_dir"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    ious = []
    visualization_samples = []
    
    with torch.no_grad():
        for i, (img_t, mask_t, img_np) in enumerate(test_loader):
            img_t = img_t.to(device)
            
            # Predict
            logits = model(img_t)
            pred_prob = torch.sigmoid(logits).cpu().squeeze(0).squeeze(0)
            
            iou = calculate_iou(pred_prob, mask_t.squeeze(0))
            ious.append(iou)
            
            # Save first 3 for viz
            if i < 3:
                visualization_samples.append((img_np.squeeze(0).numpy(), mask_t.squeeze(0).numpy(), pred_prob.numpy(), iou))

    print(f"\nFinal Mean IoU: {np.mean(ious):.4f}")
    
    # 3. Visualization
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.suptitle("Model Predictions (Image | Ground Truth | Prediction)")
    
    for idx, (img, gt, pred, iou) in enumerate(visualization_samples):
        # Image
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title("Input Image")
        axes[idx, 0].axis('off')
        
        # Ground Truth
        axes[idx, 1].imshow(gt, cmap='gray')
        axes[idx, 1].set_title("Ground Truth Mask")
        axes[idx, 1].axis('off')
        
        # Prediction
        axes[idx, 2].imshow(pred > 0.5, cmap='jet')
        axes[idx, 2].set_title(f"Prediction (IoU: {iou:.2f})")
        axes[idx, 2].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()