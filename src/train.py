import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import glob
import os
from tqdm import tqdm

from config import CONFIG
from src.models import GradCAMPlusPlus, UNetLight
from src.datasets import SimpleDataset, CrackSegDataset
from src.utils import generate_pseudo_labels
from src.evaluate import find_best_threshold, evaluate_test_set
from src.visualize import visualize_results
from src.submit import generate_submission

def train_classifier(img_paths, device):
    """Stage 1: Train Classifier"""
    print(f"\n[Stage 1] Training Classifier for {CONFIG['classifier_epochs']} epochs...")
    
    labels = [0 if 'noncrack' in os.path.basename(f).lower() else 1 for f in img_paths]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleDataset(img_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    
    model = GradCAMPlusPlus().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr_classifier"])
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(CONFIG["classifier_epochs"]):
        model.train()
        correct = 0
        total = 0
        
        for imgs, lbls in tqdm(loader, desc=f"Ep {epoch+1}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += lbls.size(0)
            correct += predicted.eq(lbls).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1} Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CONFIG["cls_model_path"])
            
    print(f"Best Classifier Accuracy: {best_acc:.2f}%")
    return model

def train_segmentation(img_paths, pseudo_masks, device):
    """Stage 3: Train Segmentation Model"""
    print(f"\n[Stage 3] Training Segmentation Model for {CONFIG['seg_epochs']} epochs...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = CrackSegDataset(img_paths, pseudo_masks, transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=CONFIG["num_workers"])
    
    model = UNetLight().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr_seg"])
    criterion = nn.BCELoss()
    
    best_loss = float('inf')
    for epoch in range(CONFIG["seg_epochs"]):
        model.train()
        total_loss = 0
        
        for imgs, masks in tqdm(loader, desc=f"Ep {epoch+1}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CONFIG["seg_model_path"])
            
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Get Data
    img_dir = CONFIG["root_dir"] / 'train' / 'images' / '*.jpg'
    img_paths = sorted(glob.glob(str(img_dir)))
    
    # 2. Train Classifier
    classifier = train_classifier(img_paths, device)
    
    # 3. Generate Pseudo Labels
    # Reload best classifier weights for generation
    classifier.load_state_dict(torch.load(CONFIG["cls_model_path"], map_location=device))
    pseudo_masks = generate_pseudo_labels(classifier, img_paths, device, CONFIG)
    
    # 4. Train Segmentation
    train_segmentation(img_paths, pseudo_masks, device)
    
    # 5. Evaluation & Testing
    print("\n--- Training Complete. Starting Evaluation ---")
    
    # Find best threshold on test set
    best_thresh = find_best_threshold(CONFIG)
    
    # Generate visualization
    visualize_results(CONFIG, threshold=best_thresh)
    
    # Generate submission CSV
    generate_submission(CONFIG, threshold=best_thresh)

if __name__ == '__main__':
    main()