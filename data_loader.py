import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import CONFIG
from augmentations import get_transforms
from sampler import get_oversampler

class CrackDataset(Dataset):
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

def get_dataloaders():
    # 1. Load Paths
    img_dir = os.path.join(CONFIG["root_dir"], 'train', 'images', '*.jpg')
    all_files = sorted(glob.glob(img_dir))
    labels = [0 if os.path.basename(f).lower().startswith('noncrack') else 1 for f in all_files]

    # 2. Split (Stratified ensures same crack/non-crack ratio in both sets)
    train_p, val_p, train_l, val_l = train_test_split(
        all_files, labels, test_size=CONFIG["val_split"], stratify=labels, random_state=CONFIG["seed"]
    )

    # 3. Get Transforms
    train_trans, val_trans = get_transforms(CONFIG)

    # 4. Create Datasets
    train_ds = CrackDataset(train_p, train_l, train_trans)
    val_ds = CrackDataset(val_p, val_l, val_trans)

    # 5. Handle Oversampling logic
    sampler = get_oversampler(train_l) if CONFIG["use_oversampling"] else None
    
    # Note: Shuffle must be False if using a Sampler
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], 
        sampler=sampler, shuffle=(sampler is None), num_workers=CONFIG["num_workers"]
    )
    
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, _ = get_dataloaders()
    imgs, lbls = next(iter(train_loader))
    print(f"Batch loaded: {imgs.shape}, Labels: {lbls.tolist()}")