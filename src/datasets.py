import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import os
import numpy as np

# --- 1. Augmentations (Merged from augmentations.py) ---

def get_transforms(config, mode="train"):
    """
    Returns transforms based on config and mode ('train' or 'val').
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if mode == "val":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    # Training Transforms
    ops = [transforms.Resize((224, 224))]
    
    # 1. Rotation Augmentation
    if config.get("use_rotation_aug", False):
        ops.append(transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270))
        ]))

    # 2. Standard Augmentation
    if config.get("use_augmentation", False):
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(ops)


# --- 2. Sampling Strategy (Merged from sampler.py) ---

def get_oversampler(labels):
    """
    Creates a WeightedRandomSampler to balance classes.
    """
    labels_tensor = torch.as_tensor(labels)
    class_counts = torch.bincount(labels_tensor)
    weights = 1. / class_counts.float()
    samples_weights = weights[labels_tensor]
    
    return WeightedRandomSampler(samples_weights, len(samples_weights))


# --- 3. Dataset Classes (Merged from old datasets.py) ---

class CrackClsDataset(Dataset):
    """
    Dataset for Stage 1: Classification (Image -> Label)
    """
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class CrackSegDataset(Dataset):
    """
    Dataset for Stage 3: Segmentation (Image -> Mask)
    Handles loading images and their corresponding Pseudo-Masks (numpy arrays).
    """
    def __init__(self, img_paths, masks, transform=None):
        self.img_paths = img_paths
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # Masks are already passed in as a list of numpy arrays (from pseudo-label generation)
        # Convert to tensor [1, H, W] and normalize to 0-1
        mask = torch.from_numpy(self.masks[idx]).float().unsqueeze(0) / 255.0
        
        return img, mask


# --- 4. DataLoader Builders (Merged from data_loader.py) ---

def get_cls_dataloaders(config):
    """
    Builders for Stage 1 (Classification).
    Handles: Globbing files -> Splitting -> Augmentation -> Sampling -> DataLoader
    """
    # 1. Gather Files
    img_pattern = str(config["root_dir"] / 'train' / 'images' / '*.jpg')
    all_files = sorted(glob.glob(img_pattern))
    
    if not all_files:
        raise RuntimeError(f"No images found at {img_pattern}")

    # 2. Generate Labels (0 for 'noncrack', 1 for crack)
    labels = [0 if 'noncrack' in os.path.basename(f).lower() else 1 for f in all_files]

    # 3. Stratified Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_files, labels, 
        test_size=config["val_split"], 
        stratify=labels, 
        random_state=config["seed"]
    )
    
    print(f"Data Split: {len(train_paths)} Train, {len(val_paths)} Val")

    # 4. Create Datasets with Transforms
    train_ds = CrackClsDataset(train_paths, train_labels, transform=get_transforms(config, "train"))
    val_ds = CrackClsDataset(val_paths, val_labels, transform=get_transforms(config, "val"))

    # 5. Setup Sampler (Optional)
    sampler = None
    if config.get("use_oversampling", False):
        print("Using WeightedRandomSampler for class balancing.")
        sampler = get_oversampler(train_labels)

    # 6. Create Loaders
    # Note: shuffle must be False if sampler is used
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["batch_size"], 
        sampler=sampler, 
        shuffle=(sampler is None), 
        num_workers=config["num_workers"]
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"]
    )

    return train_loader, val_loader