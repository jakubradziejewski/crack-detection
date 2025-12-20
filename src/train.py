import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
    
from config import CONFIG
from src.models import GradCAMPlusPlus, UNetLight, generate_pseudo_labels
from src.datasets import get_cls_dataloaders, CrackSegDataset
from src.visualize import visualize_results
from src.submit import generate_submission

# ==== NEW: IoU Calculation Helper ====
def calculate_iou(pred_mask, true_mask):
    """Calculate IoU between two binary masks"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)


def train_model_with_val(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path, description="Training"):
    """
    Enhanced training loop with validation monitoring and early stopping.
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    is_classification = isinstance(criterion, nn.CrossEntropyLoss)
    is_segmentation = isinstance(criterion, nn.BCELoss)
    
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
            if is_classification:
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            else:  # Segmentation metrics
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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
                if is_classification:
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        # ============ LOGGING ============
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        if is_classification:
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        else:  # Segmentation
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f}")
            
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


# --- Stage 3: Segmentation with Validation ---
def run_segmentation_stage(img_paths, pseudo_masks, device, config):
    print(f"\n{'#'*60}")
    print(f"# STAGE 3: SEGMENTATION TRAINING")
    print(f"{'#'*60}")

    # CHANGED: Use config image_size
    img_size = config.get("image_size", 224)
    
    indices = list(range(len(img_paths)))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=config["val_split"],
        random_state=config["seed"]
    )
    
    print(f"Segmentation split: {len(train_idx)} train, {len(val_idx)} val")
    
    # Create separate datasets
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = CrackSegDataset(
        [img_paths[i] for i in train_idx],
        [pseudo_masks[i] for i in train_idx],
        transform
    )
    
    val_dataset = CrackSegDataset(
        [img_paths[i] for i in val_idx],
        [pseudo_masks[i] for i in val_idx],
        transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"]
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    model = UNetLight()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_seg"])
    criterion = nn.BCELoss()
    
    model = train_model_with_val(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config["seg_epochs"],
        save_path=config["seg_model_path"],
        description="Segmentation"
    )
    
    # CHANGED: Return both model and val_loader for threshold search
    return model, val_loader


# --- Main Runner ---
def main_runner(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Image size: {config['image_size']}×{config['image_size']}")
    
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
    
    # 3. Train Segmentation (now returns val_loader too!)
    seg_model, val_loader = run_segmentation_stage(all_img_paths, pseudo_masks, device, config)
    
    # 4. Find Optimal Threshold on Validation Set
    print(f"\n{'#'*60}")
    print(f"# STAGE 4: THRESHOLD OPTIMIZATION")
    print(f"{'#'*60}")

    from src.threshold_selection import find_threshold_otsu_and_li

    # Use combined Otsu + Li for robustness
    best_thresh = find_threshold_otsu_and_li(seg_model, val_loader, device)
    # 5. Final Evaluation & Submission
    print(f"\n{'#'*60}")
    print(f"# FINAL EVALUATION & SUBMISSION")
    print(f"{'#'*60}")
    
    # Optional: Visualize results with optimal threshold
    visualize_results(config, threshold=best_thresh)
    
    # Generate submission
    generate_submission(config, threshold=best_thresh)
    
    print(f"\n{'='*60}")
    print(f"✓ PIPELINE COMPLETE!")
    print(f"  Optimal Threshold: {best_thresh:.4f}")
    print(f"  Image Resolution: {config['image_size']}×{config['image_size']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main_runner(CONFIG)