import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


def find_confidence_threshold(model, loader, device, method='f1_max'):
    """
    Find optimal confidence threshold for crack classification.
    
    Methods:
        - 'f1_max': Maximize F1 score (balanced precision/recall)
        - 'youden': Maximize Youden's J statistic (TPR - FPR)
        - 'precision_90': Get 90% precision (conservative)
        - 'recall_90': Get 90% recall (aggressive)
    
    Returns:
        optimal_threshold (float): Best confidence threshold
        metrics (dict): Performance metrics at that threshold
    """
    print(f"\n{'='*60}")
    print(f"FINDING OPTIMAL CONFIDENCE THRESHOLD")
    print(f"Method: {method}")
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
    
    # Calculate metrics at different thresholds
    if method == 'f1_max':
        # Find threshold that maximizes F1 score
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    
    # Calculate final metrics at optimal threshold
    predictions = (all_probs >= optimal_threshold).astype(int)
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    tn = np.sum((predictions == 0) & (all_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(all_labels)
    
    metrics = {
        'threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL THRESHOLD: {optimal_threshold:.4f}")
    print(f"{'='*60}")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"{'='*60}\n")
    
    return optimal_threshold, metrics


def find_cam_percentile(model, img_paths, device, config, 
                        test_percentiles=[90, 92, 94, 95, 96, 97, 98, 99],
                        confidence_threshold=None):
    import cv2
    from PIL import Image
    from torchvision import transforms
    
    print(f"\n{'='*60}")
    print(f"FINDING OPTIMAL CAM PERCENTILE")
    print(f"Testing percentiles: {test_percentiles}")
    print(f"{'='*60}")
    
    model.eval()
    
    if confidence_threshold is None:
        confidence_threshold = config["confidence_threshold"]
    
    img_size = config.get("image_size", 224)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Sample subset for efficiency (analyze 50-100 crack images)
    crack_paths = [p for p in img_paths if 'noncrack' not in p.lower()]
    sample_size = min(100, len(crack_paths))
    np.random.seed(config.get("seed", 42))
    sample_paths = np.random.choice(crack_paths, sample_size, replace=False)
    
    print(f"\nAnalyzing {sample_size} crack images...")
    
    # Test each percentile
    results = {}
    
    for percentile in test_percentiles:
        print(f"\nTesting percentile {percentile}...")
        
        metrics = {
            'fragmentation': [],  # Number of connected components
            'coverage': [],        # % of image that is crack
            'compactness': [],     # Perimeter² / (4π × Area)
            'aspect_ratio': []     # Width / Height of bounding box
        }
        
        for path in tqdm(sample_paths, desc=f"Percentile {percentile}"):
            img = Image.open(path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device).requires_grad_(True)
            
            # Check confidence
            with torch.no_grad():
                output = model(img_t, return_cam=False)
                probs = F.softmax(output, dim=1)
                crack_conf = probs[0, 1].item()
            
            if crack_conf < confidence_threshold:
                continue
            
            # Generate CAM (using layer4 for efficiency)
            output = model(img_t, return_cam=True, cam_layers=['layer4'])
            model.zero_grad()
            class_score = output[:, 1]
            class_score.backward()
            
            with torch.no_grad():
                cam = model.generate_gradcam_plusplus('layer4')
            
            cam = cam.squeeze().cpu().numpy()
            cam = cv2.resize(cam, (img_size, img_size))
            cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Threshold at current percentile
            threshold = np.percentile(cam_norm, percentile)
            mask = (cam_norm > threshold).astype(np.uint8)
            
            if mask.sum() == 0:  # Empty mask
                continue
            
            # 1. Fragmentation: count connected components
            num_labels, _ = cv2.connectedComponents(mask)
            metrics['fragmentation'].append(num_labels - 1)  # Subtract background
            
            # 2. Coverage: % of pixels that are crack
            coverage = mask.sum() / mask.size
            metrics['coverage'].append(coverage)
            
            # 3. Compactness: perimeter² / (4π × area)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = sum(cv2.arcLength(c, True) for c in contours)
                area = mask.sum()
                compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-8)
                metrics['compactness'].append(compactness)
                
                # 4. Aspect ratio
                x, y, w, h = cv2.boundingRect(contours[0])
                aspect = w / (h + 1e-8)
                metrics['aspect_ratio'].append(min(aspect, 1/aspect))  # Normalize to [0,1]
        
        # Aggregate metrics
        if len(metrics['fragmentation']) > 0:
            avg_metrics = {
                'fragmentation': np.mean(metrics['fragmentation']),
                'coverage': np.mean(metrics['coverage']),
                'compactness': np.mean(metrics['compactness']),
                'aspect_ratio': np.mean(metrics['aspect_ratio']),
                'n_samples': len(metrics['fragmentation'])
            }
            results[percentile] = avg_metrics
            
            print(f"  Fragments:    {avg_metrics['fragmentation']:.2f}")
            print(f"  Coverage:     {avg_metrics['coverage']:.3f}")
            print(f"  Compactness:  {avg_metrics['compactness']:.2f}")
            print(f"  Samples:      {avg_metrics['n_samples']}")
        else:
            print(f"  No valid samples!")
            results[percentile] = None
    
    # Score each percentile (lower is better)
    # Goal: Low fragmentation, reasonable coverage (5-15%), good compactness
    scores = {}
    for percentile, metrics in results.items():
        if metrics is None:
            scores[percentile] = float('inf')
            continue
        
        # Normalize and weight components
        frag_score = metrics['fragmentation']  # Fewer is better
        coverage_score = abs(metrics['coverage'] - 0.10) * 100  # Target ~10%
        compact_score = metrics['compactness'] / 10  # Lower is better
        
        # Combined score (you can adjust weights)
        total_score = (
            0.4 * frag_score +      # Fragmentation is important
            0.3 * coverage_score +  # Coverage should be reasonable
            0.3 * compact_score     # Compactness matters
        )
        
        scores[percentile] = total_score
    
    # Find optimal
    optimal_percentile = min(scores.keys(), key=lambda k: scores[k])
    
    print(f"\n{'='*60}")
    print(f"PERCENTILE ANALYSIS:")
    print(f"{'='*60}")
    for p in sorted(scores.keys()):
        if results[p] is not None:
            marker = " ← OPTIMAL" if p == optimal_percentile else ""
            print(f"  {p}%: Score={scores[p]:.2f} | "
                  f"Frag={results[p]['fragmentation']:.1f} | "
                  f"Cov={results[p]['coverage']:.3f}{marker}")
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL CAM PERCENTILE: {optimal_percentile}")
    print(f"{'='*60}\n")
    
    return optimal_percentile, results