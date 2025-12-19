import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import glob
import os
from tqdm import tqdm
from PIL import Image

from src.models import UNetLight
from src.utils import mask2rle

def generate_submission(config, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetLight().to(device)
    model.load_state_dict(torch.load(config["seg_model_path"], map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_paths = sorted(glob.glob(str(config["root_dir"] / 'test' / 'images' / '*.jpg')))
    output_file = config["submission_file"]
    
    print(f"\nGenerating submission to {output_file} (thresh={threshold})...")
    
    with open(output_file, 'w') as f:
        f.write('ImageId,EncodedPixels\n')
        
        for path in tqdm(test_paths):
            img = Image.open(path).convert('RGB')
            original_size = img.size[::-1] # H, W
            
            img_t = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred = model(img_t).squeeze().cpu().numpy()
            
            pred = cv2.resize(pred, (original_size[1], original_size[0]))
            binary_mask = (pred > threshold).astype(np.uint8)
            
            rle = mask2rle(binary_mask)
            img_id = os.path.basename(path).replace('.jpg', '')
            f.write(f'{img_id},{rle}\n')
            
    print("Submission generation complete.")