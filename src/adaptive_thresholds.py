import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve


def find_confidence_threshold(model, loader, device):
    """
    Find optimal confidence threshold that maximizes F1 score.
    """
    print(f"\n{'='*60}")
    print(f"FINDING OPTIMAL CONFIDENCE THRESHOLD")
    print(f"{'='*60}")
    
    model.eval()
    
    # Collect predictions and ground truth
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Collecting predictions"):
            imgs = imgs.to(device)
            outputs = model(imgs, return_cam=False)
            probs = F.softmax(outputs, dim=1)
            crack_probs = probs[:, 1].cpu().numpy()
            
            all_probs.extend(crack_probs)
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"\nDataset composition:")
    print(f"  No Crack (0): {np.sum(all_labels == 0)} samples")
    print(f"  Crack (1):    {np.sum(all_labels == 1)} samples")
    
    # Find threshold that maximizes F1 score
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Calculate metrics at optimal threshold for reporting
    predictions = (all_probs >= optimal_threshold).astype(int)
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    tn = np.sum((predictions == 0) & (all_labels == 0))
    
    accuracy = (tp + tn) / len(all_labels)
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL THRESHOLD: {optimal_threshold:.4f}")
    print(f"{'='*60}")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  Precision:    {precision_val:.4f}")
    print(f"  Recall:       {recall_val:.4f}")
    print(f"  F1 Score:     {f1_val:.4f}")
    print(f"{'='*60}\n")
    
    return optimal_threshold