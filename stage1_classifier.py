import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os

from config import CONFIG
from data_loader import get_dataloaders

class CrackClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18 is sufficient for binary classification
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        
    def forward(self, x):
        return self.backbone(x)

def train_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Stage 1: Training Classifier on {device} ---")
    
    train_loader, val_loader = get_dataloaders()
    model = CrackClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    
    for epoch in range(5): # 5 epochs is usually enough for convergence on simple binary tasks
        model.train()
        train_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        print(f"Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f} - Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "classifier_best.pth")
            print("Saved Best Classifier")

if __name__ == "__main__":
    train_classifier()