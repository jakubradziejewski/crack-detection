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
from src.visualize import visualize_results_stratified
from src.submit import generate_submission


def train_classifier_with_val(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path):
    """
    Training loop for CLASSIFIER ONLY (Stage 1).
    This is the only stage that benefits from validation.
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    print(f"\n{'='*60}")
    print(f"Starting Classifier Training")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # === TRAINING ===
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Training")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # === VALIDATION ===
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # === LOGGING ===
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


def train_segmentation_simple(model, train_loader, optimizer, criterion, device, epochs, save_path):
    """
    Simple training loop for SEGMENTATION (Stage 3).
    No validation needed - we're just learning the pseudo-masks.
    """
    model.to(device)
    
    print(f"\n{'='*60}")
    print(f"Starting Segmentation Training (No Validation)")
    print(f"Train batches: {len(train_loader)}")
    print(f"Training on ALL pseudo-masks")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Training")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"{'─'*60}\n")
    
    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ Training complete. Model saved to {save_path}")
    
    return model


def run_classifier_stage(device, config):
    """Stage 1: Classifier Training (with validation)"""
    print(f"\n{'#'*60}")
    print(f"# STAGE 1: CLASSIFIER TRAINING")
    print(f"{'#'*60}")
    
    # Get dataloaders (includes split, augmentation, sampling)
    train_loader, val_loader = get_cls_dataloaders(config)
    
    model = GradCAMPlusPlus()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_classifier"])
    criterion = nn.CrossEntropyLoss()
    
    # Train with validation monitoring
    model = train_classifier_with_val(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=config["classifier_epochs"],
        save_path=config["cls_model_path"]
    )
    
    return model


def run_segmentation_stage(img_paths, pseudo_masks, device, config):
    """
    Stage 3: Segmentation Training (NO validation split).
    Train on ALL pseudo-masks since they're our only supervision.
    """
    print(f"\n{'#'*60}")
    print(f"# STAGE 3: SEGMENTATION TRAINING")
    print(f"{'#'*60}")

    img_size = config.get("image_size", 224)
    
    print(f"Training on all {len(img_paths)} images with pseudo-masks")
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset with ALL data
    train_dataset = CrackSegDataset(img_paths, pseudo_masks, transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"]
    )
    
    # Initialize model
    model = UNetLight()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_seg"])
    criterion = nn.BCELoss()
    
    # Train (simple loop, no validation)
    model = train_segmentation_simple(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config["seg_epochs"],
        save_path=config["seg_model_path"]
    )
    
    # Return model and loader for threshold selection
    return model, train_loader


def main_runner(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Image size: {config['image_size']}×{config['image_size']}")
    
    # ==========================================
    # STAGE 1: Train Classifier (with validation)
    # ==========================================
    classifier = run_classifier_stage(device, config)
    
    # ==========================================
    # STAGE 2: Generate Pseudo Labels
    # ==========================================
    print(f"\n{'#'*60}")
    print(f"# STAGE 2: PSEUDO-LABEL GENERATION")
    print(f"{'#'*60}")
    
    img_dir = config["root_dir"] / 'train' / 'images' / '*.jpg'
    all_img_paths = sorted(glob.glob(str(img_dir)))
    
    classifier.load_state_dict(torch.load(config["cls_model_path"], map_location=device))
    pseudo_masks = generate_pseudo_labels(classifier, all_img_paths, device, config)
    
    # ==========================================
    # STAGE 3: Train Segmentation (NO validation)
    # ==========================================
    seg_model, train_loader = run_segmentation_stage(all_img_paths, pseudo_masks, device, config)
    
    # ==========================================
    # STAGE 4: Threshold Optimization
    # ==========================================
    print(f"\n{'#'*60}")
    print(f"# STAGE 4: THRESHOLD OPTIMIZATION")
    print(f"{'#'*60}")

    from src.threshold_selection import find_threshold_otsu

    # Use train_loader since we have no val split
    best_thresh = find_threshold_otsu(seg_model, train_loader, device)
    
    # ==========================================
    # STAGE 5: Test Set Evaluation
    # ==========================================
    print(f"\n{'#'*60}")
    print(f"# STAGE 5: TEST SET EVALUATION")
    print(f"{'#'*60}")
    
    from src.metrics import evaluate_test_set
    
    test_results = evaluate_test_set(config, threshold=best_thresh, verbose=True)
    
    # ==========================================
    # STAGE 6: Visualization & Submission
    # ==========================================
    print(f"\n{'#'*60}")
    print(f"# STAGE 6: VISUALIZATION & SUBMISSION")
    print(f"{'#'*60}")
    
    # Visualize results with optimal threshold
    print("\nGenerating visualizations...")
    visualize_results_stratified(config, threshold=best_thresh, save_path='results_stratified.png')
    
    # Generate submission
    print("\nGenerating submission file...")
    generate_submission(config, threshold=best_thresh)
    
    print(f"\n{'='*60}")
    print(f"✓ PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"  Optimal Threshold: {best_thresh:.4f}")
    print(f"  Image Resolution: {config['image_size']}×{config['image_size']}")
    print(f"  Classifier: Trained with validation split")
    print(f"  Segmentation: Trained on ALL pseudo-masks (no val split)")
    if test_results:
        print(f"\n  TEST SET RESULTS:")
        print(f"    Mean IoU:       {test_results['mean_iou']:.4f}")
        print(f"    Mean Dice:      {test_results['mean_dice']:.4f}")
        print(f"    Mean Precision: {test_results['mean_precision']:.4f}")
        print(f"    Mean Recall:    {test_results['mean_recall']:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main_runner(CONFIG)