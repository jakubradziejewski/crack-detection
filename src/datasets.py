import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


def calculate_and_save_statistics(image_paths, resize_size, sample_size, save_path):
    if sample_size and sample_size < len(image_paths):
        import random
        image_paths = random.sample(image_paths, sample_size)
    
    count = 0
    mean = np.zeros(3)
    M2 = np.zeros(3)
    
    for img_path in tqdm(image_paths, desc="Computing dataset stats"):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((resize_size, resize_size))
            img_array = np.array(img).astype(np.float32) / 255.0
            pixels = img_array.reshape(-1, 3)
            
            for pixel in pixels:
                count += 1
                delta = pixel - mean
                mean += delta / count
                delta2 = pixel - mean
                M2 += delta * delta2
        except:
            continue
    
    variance = M2 / count
    std = np.sqrt(variance)
    
    stats = {'mean': mean.tolist(), 'std': std.tolist()}
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(stats, f)
    
    print(f"Stats saved: mean={mean}, std={std}")
    return stats


def get_transforms(config, mode="train", img_paths=None):
    # Check if stats need to be computed
    if not Path(config["stats_cache_path"]).exists() and img_paths is not None:
        calculate_and_save_statistics(
            img_paths,
            resize_size=config["image_size"],
            sample_size=config["stats_sample_size"],
            save_path=config["stats_cache_path"]
        )
        # Reload config to get new stats
        import importlib
        import config as cfg_module
        importlib.reload(cfg_module)
        config = cfg_module.CONFIG
    
    normalize = transforms.Normalize(mean=config["dataset_mean"], std=config["dataset_std"])
    
    if mode == "val":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    img_size = config["image_size"]
    ops = [transforms.Resize((img_size, img_size))]
    
    if config["use_rotation_aug"]:
        ops.append(transforms.RandomApply([
            transforms.RandomChoice([
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ])
        ], p=0.5))

    if config["use_augmentation"]:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(ops)


def get_oversampler(labels):
    labels_tensor = torch.as_tensor(labels)
    class_counts = torch.bincount(labels_tensor)
    weights = 1. / class_counts.float()
    samples_weights = weights[labels_tensor]
    return WeightedRandomSampler(samples_weights, len(samples_weights))


class CrackClsDataset(Dataset):
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
            
        mask = torch.from_numpy(self.masks[idx]).float().unsqueeze(0) / 255.0
        return img, mask


def get_cls_dataloaders(config):
    img_pattern = str(config["root_dir"] / 'train' / 'images' / '*.jpg')
    all_files = sorted(glob.glob(img_pattern))
    
    if not all_files:
        raise RuntimeError(f"No images found at {img_pattern}")

    labels = [0 if 'noncrack' in os.path.basename(f).lower() else 1 for f in all_files]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_files, labels, 
        test_size=config["val_split"], 
        stratify=labels, 
        random_state=config["seed"]
    )
    
    print(f"Data Split: {len(train_paths)} Train, {len(val_paths)} Val")

    train_ds = CrackClsDataset(
        train_paths, 
        train_labels, 
        transform=get_transforms(config, "train", img_paths=train_paths)
    )
    val_ds = CrackClsDataset(
        val_paths, 
        val_labels, 
        transform=get_transforms(config, "val")
    )

    sampler = None
    if config["use_oversampling"]:
        sampler = get_oversampler(train_labels)

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