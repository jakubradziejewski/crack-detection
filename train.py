import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import CONFIG
from data_loader import get_dataloaders
from model import WeaklySupCrackNet

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data
    train_loader, val_loader = get_dataloaders()
    
    # 2. Model
    model = WeaklySupCrackNet().to(device)
    
    # 3. Setup
    criterion = nn.BCEWithLogitsLoss() # Good for binary classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_acc = 0.0
    
    # 4. Loop
    epochs = 10 # Adjust as needed
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            
            # Forward
            # We get both the classification score (logits) and the map
            logits, _ = model(imgs)
            
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)
            train_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())
            
        # Validation
        model.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).float()
                logits, _ = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        acc = val_correct / total_val
        print(f"Epoch {epoch+1} Results: Train Acc: {train_correct/total_train:.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_crack_model.pth")
            print(">>> Model Saved!")

if __name__ == "__main__":
    train_model()