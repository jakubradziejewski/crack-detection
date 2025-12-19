import torch
from torch.utils.data import Dataset
from PIL import Image

class SimpleDataset(Dataset):
    """Used for the Classifier Stage"""
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
    """Used for the Segmentation Stage (with Pseudo-masks)"""
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
            # Pseudo-masks are numpy arrays, convert to tensor [1, H, W]
            mask = torch.from_numpy(self.masks[idx]).float().unsqueeze(0) / 255.0
            return img, mask
        return img