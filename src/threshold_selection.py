import torch
import numpy as np
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve


def find_confidence_thresh(model, loader, device):
    """
    Find optimal confidence threshold that maximizes F1 score.
    """
    
    model.eval()
    
    # Collect predictions and ground truth for validation set
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating validation set"):
            imgs = imgs.to(device)
            outputs = model(imgs, return_cam=False)
            probs = F.softmax(outputs, dim=1)
            crack_probs = probs[:, 1].cpu().numpy()
            
            all_probs.extend(crack_probs)
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Confidence threshold that maximizes F1 score
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    print(f"Optimal confidence threshold: {optimal_threshold:.4f} based on F1 score {f1_scores[best_idx]:.4f}")
    
    return optimal_threshold



def find_otsu_thresh(model, loader, device, max_samples=1000):
    """
    Use Otsu's method for threshold selection.
    """

    if max_samples is not None:
        print(f"Using {max_samples} samples for threshold selection...")
    else:
        print(f"Using all samples for threshold selection...")
    
    model.eval()
    all_predictions = []
    samples_processed = 0
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            pred_values = preds.cpu().numpy().flatten()
            all_predictions.append(pred_values)
            
            samples_processed += imgs.size(0)
            
            # Early stopping if we've collected enough samples
            if max_samples is not None and samples_processed >= max_samples:
                break

    all_predictions = np.concatenate(all_predictions)

    # Convert to 8-bit for Otsu
    predictions_8bit = (all_predictions * 255).astype(np.uint8)

    otsu_thresh_8bit, _ = cv2.threshold(
        predictions_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = otsu_thresh_8bit / 255.0

    print(f"  Otsu Threshold:   {threshold:.4f}")
    return threshold

