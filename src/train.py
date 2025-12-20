import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import glob
from tqdm import tqdm

from config import CONFIG
from src.models import GradCAMPlusPlus, UNetLight, generate_pseudo_labels
from src.datasets import SimpleDataset, CrackSegDataset
from src.metrics import find_best_threshold
from src.visualize import visualize_results
from src.submit import generate_submission

# --- Generic Training Loop ---

def train_model(model, loader, optimizer, criterion, device, epochs, description="Training"):
    """
    Generic training loop that can be used for both classifier and segmenter.
    """
    model.to(device)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(loader, desc=f"{description} Ep {epoch+1}"):
            # Handle variable unpacking based on dataset type
            if len(batch) == 2:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs = batch.to(device)
                targets = None # Unsupervised case
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss calculation
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Optional: Calculate Accuracy for classification tasks
            if isinstance(criterion, nn.CrossEntropyLoss):
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(loader)
        
        # Logging
        log_msg = f"Epoch {epoch+1} Loss: {avg_loss:.4f}"
        if total > 0:
            acc = 100. * correct / total
            log_msg += f" | Acc: {acc:.2f}%"
        print(log_msg)
        
    return model

# --- Stage Specific Setups ---

def run_classifier_stage(img_paths, device, config):
    print(f"\n[Stage 1] Training Classifier...")
    
    labels = [0 if 'noncrack' in os.path.basename(f).lower() else 1 for f in img_paths]
    
    # Check config for augmentation
    if config.get("use_augmentation", False):
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    else:
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        
    transform = transforms.Compose(transform_list)
    
    dataset = SimpleDataset(img_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    
    model = GradCAMPlusPlus()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_classifier"])
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model = train_model(model, loader, optimizer, criterion, device, 
                        epochs=config["classifier_epochs"], description="Classifier")
    
    # Save
    torch.save(model.state_dict(), config["cls_model_path"])
    return model

def run_segmentation_stage(img_paths, pseudo_masks, device, config):
    print(f"\n[Stage 3] Training Segmentation Model...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = CrackSegDataset(img_paths, pseudo_masks, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=config["num_workers"])
    
    model = UNetLight()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_seg"])
    criterion = nn.BCELoss()
    
    # Train
    model = train_model(model, loader, optimizer, criterion, device, 
                        epochs=config["seg_epochs"], description="Segmentation")
    
    # Save
    torch.save(model.state_dict(), config["seg_model_path"])
    return model

# --- Main Runner ---

def main_runner(config, ClassifierClass=GradCAMPlusPlus, SegmenterClass=UNetLight):
    """
    Main function acts as a dependency injection point.
    You can easily pass different model classes here in the future.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Get Data
    img_dir = config["root_dir"] / 'train' / 'images' / '*.jpg'
    img_paths = sorted(glob.glob(str(img_dir)))
    
    # 2. Train Classifier
    classifier = run_classifier_stage(img_paths, device, config)
    
    # 3. Generate Pseudo Labels
    # Reload best/current classifier weights
    classifier.load_state_dict(torch.load(config["cls_model_path"], map_location=device))
    pseudo_masks = generate_pseudo_labels(classifier, img_paths, device, config)
    
    # 4. Train Segmentation
    run_segmentation_stage(img_paths, pseudo_masks, device, config)
    
    # 5. Evaluation & Testing
    print("\n--- Training Complete. Starting Evaluation ---")
    best_thresh = find_best_threshold(config)
    visualize_results(config, threshold=best_thresh)
    generate_submission(config, threshold=best_thresh)

if __name__ == '__main__':
    main_runner(CONFIG)