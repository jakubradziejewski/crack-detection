import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
from tqdm import tqdm

from config import CONFIG
from src.models import GradCAMPlusPlus, UNetLight, generate_pseudo_labels
from src.datasets import get_cls_dataloaders, CrackSegDataset
from src.metrics import find_best_threshold
from src.visualize import visualize_results
from src.submit import generate_submission

def train_model_with_val(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path, description="Training"):
    """
    Enhanced training loop with validation monitoring and early stopping.
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    print(f"\n{'='*60}")
    print(f"Starting {description}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Training")
        for batch in pbar:
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
            
            train_loss += loss.item()
            
            # Track accuracy for classification
            if isinstance(criterion, nn.CrossEntropyLoss):
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Validation"):
                if len(batch) == 2:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                else:
                    inputs = batch.to(device)
                    targets = None
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Track accuracy for classification
                if isinstance(criterion, nn.CrossEntropyLoss):
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        # ============ LOGGING ============
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"{'─'*60}\n")
    
    # Load best model
    print(f"\n✓ Training complete. Loading best model from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# --- Stage 1: Classifier with Validation ---
def run_classifier_stage(device, config):
    print(f"\n{'#'*60}")
    print(f"# STAGE 1: CLASSIFIER TRAINING")
    print(f"{'#'*60}")
    
    # Get dataloaders (already includes split, augmentation, sampling)
    train_loader, val_loader = get_cls_dataloaders(config)
    
    model = GradCAMPlusPlus()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_classifier"])
    criterion = nn.CrossEntropyLoss()
    
    # Train with validation monitoring
    model = train_model_with_val(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=config["classifier_epochs"],
        save_path=config["cls_model_path"],
        description="Classifier"
    )
    
    return model


# --- Stage 3: Segmentation (keeping simple for now) ---
def run_segmentation_stage(img_paths, pseudo_masks, device, config):
    print(f"\n{'#'*60}")
    print(f"# STAGE 3: SEGMENTATION TRAINING")
    print(f"{'#'*60}")
    
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
    
    # Simple training (can add validation here too later)
    model.to(device)
    for epoch in range(config["seg_epochs"]):
        model.train()
        total_loss = 0
        
        for inputs, masks in tqdm(loader, desc=f"Segmentation Epoch {epoch+1}"):
            inputs, masks = inputs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{config['seg_epochs']} | Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), config["seg_model_path"])
    return model


# --- Main Runner ---
def main_runner(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 1. Train Classifier with Validation
    classifier = run_classifier_stage(device, config)
    
    # 2. Generate Pseudo Labels
    print(f"\n{'#'*60}")
    print(f"# STAGE 2: PSEUDO-LABEL GENERATION")
    print(f"{'#'*60}")
    
    img_dir = config["root_dir"] / 'train' / 'images' / '*.jpg'
    all_img_paths = sorted(glob.glob(str(img_dir)))
    
    classifier.load_state_dict(torch.load(config["cls_model_path"], map_location=device))
    pseudo_masks = generate_pseudo_labels(classifier, all_img_paths, device, config)
    
    # 3. Train Segmentation
    run_segmentation_stage(all_img_paths, pseudo_masks, device, config)
    
    # 4. Evaluation
    print(f"\n{'#'*60}")
    print(f"# EVALUATION & SUBMISSION")
    print(f"{'#'*60}")
    best_thresh = find_best_threshold(config)
    visualize_results(config, threshold=best_thresh)
    generate_submission(config, threshold=best_thresh)
    
    print(f"\n{'='*60}")
    print(f"✓ PIPELINE COMPLETE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main_runner(CONFIG)