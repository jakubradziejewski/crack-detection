import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob

# checker function 
def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# --- 2. Custom Dataset Class ---
class CrackClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the data (e.g., './data')
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train' or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
        # Define paths
        if self.mode == 'train':
            # Path: data/train/images/
            self.img_dir = os.path.join(root_dir, 'train', 'images')
        elif self.mode == 'test':
            # Path: data/test/images/
            self.img_dir = os.path.join(root_dir, 'test', 'images')
            
        # Load file names
        # We search for common image extensions
        search_path = os.path.join(self.img_dir, '*')
        all_files = glob.glob(search_path)
        
        # Filter for image files just in case
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_paths = [f for f in all_files if f.lower().endswith(valid_exts)]
        
        # --- LABEL LOGIC (Weakly Supervised) ---
        if self.mode == 'train':
            for img_path in self.image_paths:
                filename = os.path.basename(img_path)
                
                # Logic: If filename starts with 'noncrack', it's healthy (0). Else crack (1).
                if filename.lower().startswith('noncrack'):
                    self.labels.append(0) # Negative
                else:
                    self.labels.append(1) # Positive
        else:
            # For testing, we don't have labels usually, or we produce the submission
            self.labels = [-1] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'train':
            label = self.labels[idx]
            return image, label
        else:
            # In test mode, we might want the filename to create the submission file
            filename = os.path.basename(img_path)
            return image, filename

# --- 3. Execution & Statistics ---

# Basic transformations for loading (resize to standard CNN input, convert to tensor)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Standard ResNet size
    transforms.ToTensor(),
])

# Initialize Dataset
# Ensure this path points to your actual 'data' folder
dataset = CrackClassificationDataset(root_dir='./data', transform=data_transforms, mode='train')

# Calculate Stats
total_images = len(dataset)
crack_count = sum(dataset.labels)
non_crack_count = total_images - crack_count

print("--- Data Loading Statistics ---")
print(f"Total Training Images: {total_images}")
print(f"Crack Images (Positive): {crack_count}")
print(f"Non-Crack Images (Negative): {non_crack_count}")

# Check for Class Imbalance
if total_images > 0:
    positive_ratio = crack_count / total_images
    print(f"Class Balance: {positive_ratio:.2%} Crack vs {1-positive_ratio:.2%} Non-Crack")
else:
    print("Error: No images found. Check your directory structure.")

# Example: Loading a batch
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
images, labels = next(iter(dataloader))
print(f"\nSample Batch Shape: {images.shape}") # Should be [4, 3, 224, 224]
print(f"Sample Batch Labels: {labels}")