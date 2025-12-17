import torch
from torchvision import transforms

def get_transforms(config):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_ops = [transforms.Resize((224, 224))]
    
    if config["use_rotation_aug"]:
        # Randomly picks 0, 90, 180, or 270 degrees
        train_ops.append(transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270))
        ]))

    if config["use_augmentation"]:
        train_ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

    train_ops.extend([transforms.ToTensor(), normalize])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    return transforms.Compose(train_ops), val_transform