import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import os

from src.utils import get_transforms, get_oversampler

class ImageDataset(Dataset):
    """
    Dataset that loads images on-demand.
    For classification: labels are integers
    For segmentation: labels are numpy arrays (pseudo-masks)
    """
    def __init__(self, paths, labels, transform=None, is_segmentation=False):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.is_segmentation = is_segmentation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        
        # For segmentation, convert mask numpy array to tensor
        if self.is_segmentation:
            label = torch.from_numpy(label).float().unsqueeze(0) / 255.0
        
        return img, label


def classifier_dataloader(config):
    """
    Creates train and validation dataloaders for classification.
    Handles: Globbing files -> Splitting -> Augmentation -> Sampling -> DataLoader
    """
    # 1. Gather Files
    img_path = str(config["root_dir"] / 'train' / 'images' / '*.jpg')
    all_files = sorted(glob.glob(img_path))
    
    if not all_files:
        raise RuntimeError(f"No images found at {img_path}")

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

    # 4. Create Datasets
    train_ds = ImageDataset(train_paths, train_labels, get_transforms(config, "train"))
    val_ds = ImageDataset(val_paths, val_labels, get_transforms(config, "val"))

    # 5. Sampler for oversampling non-cracks
    sampler = None
    if config.get("use_oversampling", False):
        print("Using WeightedRandomSampler for class balancing.")
        sampler = get_oversampler(train_labels)

    # 6. Create Loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["batch_size1"], 
        sampler=sampler, 
        shuffle=(sampler is None), 
        num_workers=config["num_workers"]
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config["batch_size1"], 
        shuffle=False, 
        num_workers=config["num_workers"]
    )

    return train_loader, val_loader