import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import glob
from tqdm import tqdm
from PIL import Image
from src.models import UNetLight

# --- Core Metric Calculations ---

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else (1.0 if intersection == 0 else 0.0)

def calculate_dice(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    sum_area = pred_mask.sum() + true_mask.sum()
    return (2. * intersection) / sum_area if sum_area > 0 else 1.0

def calculate_precision_recall(pred_mask, true_mask):
    tp = np.logical_and(pred_mask, true_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(true_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), true_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall

# --- Evaluation Loops ---

def evaluate_test_set(config, threshold=0.5, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    
    # Load Weights
    if not config["seg_model_path"].exists():
        print(f"Model not found at {config['seg_model_path']}")
        return None

    model.load_state_dict(torch.load(config["seg_model_path"], map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_imgs = sorted(glob.glob(str(config["root_dir"] / 'test' / 'images' / '*.jpg')))
    test_masks = sorted(glob.glob(str(config["root_dir"] / 'test' / 'masks' / '*.jpg')))
    
    ious, dices, precisions, recalls = [], [], [], []
    
    if verbose:
        print(f"Evaluating {len(test_imgs)} images (thresh={threshold})...")
    
    for img_path, mask_path in tqdm(zip(test_imgs, test_masks), total=len(test_imgs), disable=not verbose):
        img = Image.open(img_path).convert('RGB')
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        original_size = true_mask.shape
        
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(img_t).squeeze().cpu().numpy()
        
        pred = cv2.resize(pred, (original_size[1], original_size[0]))
        pred_binary = (pred > threshold).astype(np.uint8)
        true_binary = (true_mask > 127).astype(np.uint8)
        
        ious.append(calculate_iou(pred_binary, true_binary))
        dices.append(calculate_dice(pred_binary, true_binary))
        p, r = calculate_precision_recall(pred_binary, true_binary)
        precisions.append(p)
        recalls.append(r)
    
    results = {
        'mean_iou': np.mean(ious),
        'mean_dice': np.mean(dices),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls)
    }
    
    if verbose:
        print(f"Mean IoU: {results['mean_iou']:.4f} | Mean Dice: {results['mean_dice']:.4f}")
        
    return results

def find_best_threshold(config):
    print("\nFinding optimal threshold...")
    best_iou = 0
    best_thresh = 0.5
    
    for thresh in config["test_thresholds"]:
        results = evaluate_test_set(config, threshold=thresh, verbose=False)
        if results and results['mean_iou'] > best_iou:
            best_iou = results['mean_iou']
            best_thresh = thresh
            print(f"New best: {thresh} (IoU: {best_iou:.4f})")
            
    return best_thresh