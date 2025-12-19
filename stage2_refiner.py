import torch
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# Correct import for the GitHub version of MobileSAM
from mobile_sam import sam_model_registry, SamPredictor 

from config import CONFIG
from stage1_classifier import CrackClassifier

def generate_pseudo_labels():
    # Force CPU to avoid CUDA memory issues on typical laboratory machines
    device = torch.device("cpu")
    print(f"--- Stage 2: Fast Pseudo-Label Generation (MobileSAM on CPU) ---")
    
    # 1. Load Classifier
    model = CrackClassifier().to(device)
    model.load_state_dict(torch.load("classifier_best.pth", map_location=device))
    model.eval()
    
    # 2. Setup Grad-CAM++
    target_layer = [model.backbone.layer4[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layer)
    
    # 3. Setup MobileSAM (40MB brain)
    sam_checkpoint = "mobile_sam.pt" 
    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError("Missing mobile_sam.pt! Download it from the MobileSAM repo.")
        
    sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # 4. Directories
    pseudo_dir = os.path.join(CONFIG["root_dir"], "train", "pseudo_masks")
    os.makedirs(pseudo_dir, exist_ok=True)
    
    img_dir = os.path.join(CONFIG["root_dir"], 'train', 'images')
    image_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. Optimized Loop
    for img_path in tqdm(image_paths):
        fname = os.path.basename(img_path).replace(".jpg", ".png")
        save_path = os.path.join(pseudo_dir, fname)
        
        # SKIP if already done (allows resuming)
        if os.path.exists(save_path): continue

        # INSTANT SKIP for non-cracks (Huge speed boost)
        if "noncrack" in fname.lower():
            Image.fromarray(np.zeros((224, 224), dtype=np.uint8)).save(save_path)
            continue

        # A. Process Image
        orig_img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(orig_img)
        input_tensor = transform(orig_img).unsqueeze(0).to(device)
        
        # B. Grad-CAM++
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # C. Refine with MobileSAM
        # Use top 5% intensity pixels as prompts for the 'curved' structure
        y, x = np.where(grayscale_cam > np.percentile(grayscale_cam, 95))
        
        if len(x) == 0:
            mask = np.zeros((224, 224), dtype=np.uint8)
        else:
            # CPU Optimization: Only use 10 points (plenty for thin cracks)
            step = max(1, len(x) // 10)
            input_points = np.column_stack((x[::step], y[::step]))
            input_labels = np.ones(len(input_points))
            
            predictor.set_image(img_np)
            masks, _, _ = predictor.predict(input_points, input_labels, multimask_output=False)
            mask = (masks[0] * 255).astype(np.uint8)
        
        Image.fromarray(mask).save(save_path)

if __name__ == "__main__":
    generate_pseudo_labels()