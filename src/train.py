import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from src.adaptive_thresholds import (
    find_confidence_threshold,
    find_cam_percentile,
    visualize_threshold_analysis
)


def train_classifier_with_val(
    model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path
):
    model.to(device)
    best_val_loss = float("inf")
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

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.*train_correct/train_total:.2f}%",
                }
            )

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # === VALIDATION ===
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Validation"
            ):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

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

    # Load best model
    print(f"\nTraining complete. Loading best model from {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def train_segmentation_simple(
    model, train_loader, optimizer, criterion, device, epochs, save_path
):
    model.to(device)

    print(f"Starting Segmentation Training, no of batches: {len(train_loader)}")

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
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} Train Loss: {avg_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ Training complete. Model saved to {save_path}")

    return model


def run_classifier_stage(device, config):
    # Get dataloaders (includes split, augmentation, sampling)
    train_loader, val_loader = get_cls_dataloaders(config)

    model = GradCAMPlusPlus()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_classifier"])
    criterion = nn.CrossEntropyLoss()

    # Train with validation monitoring
    model = train_classifier_with_val(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=config["classifier_epochs"],
        save_path=config["cls_model_path"],
    )

    return model, val_loader


def run_segmentation_stage(img_paths, pseudo_masks, device, config):
    img_size = config.get("image_size", 224)

    print(f"Training on all {len(img_paths)} images with pseudo-masks")

    # Create transform
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset with ALL data
    train_dataset = CrackSegDataset(img_paths, pseudo_masks, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
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
        save_path=config["seg_model_path"],
    )

    # Return model and loader for threshold selection
    return model, train_loader


def main_runner(config, use_adaptive_thresholds=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Image size: {config['image_size']}×{config['image_size']}")

    # STAGE 1: CLASSIFIER TRAINING
    print(f"\n{'#'*60}")
    print(f"# STAGE 1: CLASSIFIER TRAINING")
    print(f"{'#'*60}")
    classifier, val_loader = run_classifier_stage(device, config)

    # STAGE 1.5: ADAPTIVE THRESHOLD SELECTION (NEW!)
    if use_adaptive_thresholds:
        print(f"\n{'#'*60}")
        print(f"# STAGE 1.5: ADAPTIVE THRESHOLD SELECTION")
        print(f"{'#'*60}")
        
        # Find optimal confidence threshold
        print("\n[1/3] Finding optimal confidence threshold...")
        optimal_confidence, conf_metrics = find_confidence_threshold(
            model=classifier,
            loader=val_loader,
            device=device,
            method='f1_max'
        )
        
        # Update config with optimal threshold
        config["confidence_threshold"] = optimal_confidence
        
        # Find optimal CAM percentile
        print("\n[3/3] Finding optimal CAM percentile...")
        img_dir = config["root_dir"] / "train" / "images" / "*.jpg"
        all_img_paths = sorted(glob.glob(str(img_dir)))
        
        optimal_percentile, perc_analysis = find_cam_percentile(
            model=classifier,
            img_paths=all_img_paths,
            device=device,
            config=config,
            test_percentiles=[90, 92, 94, 95, 96, 97, 98],
            confidence_threshold=optimal_confidence
        )
        
        # Update config with optimal percentile
        config["cam_percentile"] = optimal_percentile
        
        print(f"\n{'='*60}")
        print(f"ADAPTIVE THRESHOLDS DETERMINED:")
        print(f"{'='*60}")
        print(f"  Confidence Threshold: {optimal_confidence:.4f}")
        print(f"  CAM Percentile:       {optimal_percentile}")
        print(f"  F1 Score:             {conf_metrics['f1']:.4f}")
        print(f"  Precision:            {conf_metrics['precision']:.4f}")
        print(f"  Recall:               {conf_metrics['recall']:.4f}")
        print(f"{'='*60}\n")
    else:
        print(f"\nUsing hardcoded thresholds:")
        print(f"  Confidence: {config['confidence_threshold']}")
        print(f"  CAM Percentile: {config['cam_percentile']}")

    # STAGE 2: PSEUDO-LABEL GENERATION
    print(f"\n{'#'*60}")
    print(f"# STAGE 2: PSEUDO-LABEL GENERATION")
    print(f"{'#'*60}")

    img_dir = config["root_dir"] / "train" / "images" / "*.jpg"
    all_img_paths = sorted(glob.glob(str(img_dir)))

    classifier.load_state_dict(
        torch.load(config["cls_model_path"], map_location=device)
    )
    pseudo_masks = generate_pseudo_labels(
        classifier, all_img_paths, device, config, use_multiscale=True
    )

    # STAGE 3: SEGMENTATION TRAINING
    print(f"\n{'#'*60}")
    print(f"# STAGE 3: SEGMENTATION TRAINING")
    print(f"{'#'*60}")
    seg_model, train_loader = run_segmentation_stage(
        all_img_paths, pseudo_masks, device, config
    )

    # STAGE 4: THRESHOLD OPTIMIZATION
    print(f"\n{'#'*60}")
    print(f"# STAGE 4: SEGMENTATION THRESHOLD OPTIMIZATION")
    print(f"{'#'*60}")

    from src.threshold_selection import find_threshold_otsu

    best_thresh = find_threshold_otsu(seg_model, train_loader, device)

    # STAGE 5: TEST SET EVALUATION
    print(f"\n{'#'*60}")
    print(f"# STAGE 5: TEST SET EVALUATION")
    print(f"{'#'*60}")
    from src.test import run_evaluation, generate_submission
    from src.inference import Predictor

    predictor = Predictor(config)
    test_dir = config["root_dir"] / "test"

    run_evaluation(predictor, test_dir, best_thresh)

    # STAGE 6: VISUALIZATION & SUBMISSION
    print(f"\n{'#'*60}")
    print(f"# STAGE 6: VISUALIZATION & SUBMISSION")
    print(f"{'#'*60}")

    generate_submission(predictor, test_dir, config["submission_file"], best_thresh)
    visualize_results_stratified(config, threshold=best_thresh)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Final Configuration:")
    print(f"  Confidence Threshold: {config['confidence_threshold']:.4f}")
    print(f"  CAM Percentile:       {config['cam_percentile']}")
    print(f"  Seg Threshold:        {best_thresh:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train crack detection pipeline')
    parser.add_argument('--no-adaptive', action='store_true',
                        help='Disable adaptive threshold selection (use hardcoded values)')
    args = parser.parse_args()
    
    main_runner(CONFIG, use_adaptive_thresholds=not args.no_adaptive)