import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms # Still needed for seg transform
import glob
from tqdm import tqdm

from config import CONFIG
from src.models import GradCAMPlusPlus, UNetLight, generate_pseudo_labels
from src.datasets import get_cls_dataloaders, CrackSegDataset
from src.metrics import find_best_threshold
from src.visualize import visualize_results
from src.submit import generate_submission

# --- Generic Training Loop ---
def train_model(model, loader, optimizer, criterion, device, epochs, description="Training"):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(loader, desc=f"{description} Ep {epoch+1}"):
            if len(batch) == 2:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs = batch.to(device)
                targets = None
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(loader)
        log_msg = f"Epoch {epoch+1} Loss: {avg_loss:.4f}"
        if total > 0:
            acc = 100. * correct / total
            log_msg += f" | Acc: {acc:.2f}%"
        print(log_msg)
        
    return model

# --- Stage 1: Classifier ---
def run_classifier_stage(device, config):
    print(f"\n[Stage 1] Training Classifier...")
    
    # NEW: Uses the merged dataset logic (Splitting + Sampling + Augmentation)
    train_loader, val_loader = get_cls_dataloaders(config)
    
    model = GradCAMPlusPlus()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_classifier"])
    criterion = nn.CrossEntropyLoss()
    
    # Train
    model = train_model(model, train_loader, optimizer, criterion, device, 
                        epochs=config["classifier_epochs"], description="Classifier")
    
    # Save
    torch.save(model.state_dict(), config["cls_model_path"])
    return model

# --- Stage 3: Segmentation ---
def run_segmentation_stage(img_paths, pseudo_masks, device, config):
    print(f"\n[Stage 3] Training Segmentation Model...")
    
    # Standard transform for UNet (Augmentation usually less aggressive here to keep mask alignment simple)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # We still use CrackSegDataset directly here because Pseudo-Masks are generated in memory
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
def main_runner(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Train Classifier (Now includes weighted sampling automatically)
    classifier = run_classifier_stage(device, config)
    
    # 2. Generate Pseudo Labels
    # We need just the paths for this step
    img_dir = config["root_dir"] / 'train' / 'images' / '*.jpg'
    all_img_paths = sorted(glob.glob(str(img_dir)))
    
    classifier.load_state_dict(torch.load(config["cls_model_path"], map_location=device))
    pseudo_masks = generate_pseudo_labels(classifier, all_img_paths, device, config)
    
    # 3. Train Segmentation
    run_segmentation_stage(all_img_paths, pseudo_masks, device, config)
    
    # 4. Evaluation
    print("\n--- Training Complete. Starting Evaluation ---")
    best_thresh = find_best_threshold(config)
    visualize_results(config, threshold=best_thresh)
    generate_submission(config, threshold=best_thresh)

if __name__ == '__main__':
    main_runner(CONFIG)