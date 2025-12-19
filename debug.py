import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from config import CONFIG
from model import WeaklySupCrackNet
from utils import get_refined_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeaklySupCrackNet().to(device)
    model.load_state_dict(torch.load("best_crack_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Path Handling
    test_img_dir = Path(CONFIG["root_dir"]) / 'test' / 'images'
    debug_out_dir = Path("debug_results")
    debug_out_dir.mkdir(exist_ok=True)

    image_files = sorted(list(test_img_dir.glob("*.jpg*")))[:20] # Debug first 20

    for img_path in tqdm(image_files):
        # Clean filename for display
        clean_name = img_path.name
        while True:
            base, ext = os.path.splitext(clean_name)
            if ext.lower() in ['.jpg', '.png']: clean_name = base
            else: break

        # Load and Inference
        img_pil = Image.open(img_path).convert("RGB")
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, mask_logits = model(input_tensor)
            confidence = torch.sigmoid(logits).item()

        # FIXED CALL: Removed input_tensor[0]
        if confidence > 0.5:
            pred_mask = get_refined_mask(mask_logits[0], original_size=(224, 224))
        else:
            pred_mask = np.zeros((224, 224), dtype=np.uint8)

        # VISUALIZATION
        plt.figure(figsize=(12, 4))
        
        # Subplot 1: Original
        plt.subplot(1, 3, 1)
        plt.imshow(img_pil.resize((224, 224)))
        plt.title(f"Conf: {confidence:.2f}")
        plt.axis('off')

        # Subplot 2: Heatmap (The 'Raw' model output)
        plt.subplot(1, 3, 2)
        heatmap = torch.sigmoid(mask_logits[0]).cpu().numpy()[0]
        plt.imshow(heatmap, cmap='jet')
        plt.title("Neural Activation")
        plt.axis('off')

        # Subplot 3: Final Mask (Otsu Refined)
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("General Refined Mask")
        plt.axis('off')

        plt.savefig(debug_out_dir / f"{clean_name}_debug.png")
        plt.close()

if __name__ == "__main__":
    main()