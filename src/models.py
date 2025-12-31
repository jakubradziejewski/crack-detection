import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm

class GradCAMPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        
        # === SWITCHED TO RESNET-34 ===
        # ResNet-34 is deeper than ResNet-18 but uses the same BasicBlock structure.
        # This makes it compatible with our manual dilation patch.
        weights = ResNet34_Weights.IMAGENET1K_V1
        backbone = resnet34(weights=weights)
        
        # Split backbone
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # === MANUAL DILATION PATCH ===
        # ResNet-34 uses 'BasicBlock', so we must manually patch it 
        # just like we did for ResNet-18.
        
        # Layer 3: Output 28x28 (Stride 1, Dilation 2)
        self._modify_layer(self.layer3, dilation=2)
        
        # Layer 4: Output 28x28 (Stride 1, Dilation 4)
        self._modify_layer(self.layer4, dilation=4)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        # ResNet-34 outputs 512 channels (same as ResNet-18)
        self.classifier = nn.Linear(512, 2)
        
        self.gradients = {}
        self.activations = {}

    def _modify_layer(self, layer, dilation):
        """
        Manually change stride to 1 and increase dilation for all Conv2d layers.
        Works for both ResNet-18 and ResNet-34 (BasicBlock architectures).
        """
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                # Change stride (2,2) -> (1,1) to stop downsampling
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                
                # Apply dilation
                module.dilation = (dilation, dilation)
                
                # Adjust padding to match dilation so image size stays constant
                if module.kernel_size == (3, 3):
                    module.padding = (dilation, dilation)

    def save_gradient(self, name):
        def hook(grad):
            self.gradients[name] = grad
        return hook
        
    def forward(self, x, return_cam=False, cam_layers=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)  # 56x56
        
        x = self.layer2(x)  # 28x28 (Standard Stride 2 happened here)
        if return_cam and cam_layers and 'layer2' in cam_layers:
            x.register_hook(self.save_gradient('layer2'))
            self.activations['layer2'] = x
        
        x = self.layer3(x)  # 28x28 (Patched: Stride 1, Dilation 2)
        if return_cam and cam_layers and 'layer3' in cam_layers:
            x.register_hook(self.save_gradient('layer3'))
            self.activations['layer3'] = x
        
        x = self.layer4(x)  # 28x28 (Patched: Stride 1, Dilation 4)
        if return_cam and cam_layers and 'layer4' in cam_layers:
            x.register_hook(self.save_gradient('layer4'))
            self.activations['layer4'] = x
        
        pooled = self.gap(x).flatten(1)
        out = self.classifier(pooled)
        return out
    
    def generate_gradcam_plusplus(self, layer_name='layer4'):
        if layer_name not in self.gradients:
            raise KeyError(f"Layer {layer_name} not found.")
        
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]
        
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + \
                      (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / alpha_denom
        
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        
        return F.relu(cam)


class UNetLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self._conv_block(3, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.bottleneck = self._conv_block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        self.out = nn.Conv2d(32, 1, 1)
        
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.out(d1))


def generate_pseudo_labels(model, img_paths, device, config, use_multiscale=True):
    model.eval()
    pseudo_masks = []
    low_confidence_count = 0
    img_size = config.get("image_size", 224)
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Confidence threshold: {config['confidence_threshold']}, CAM percentile: {config['cam_percentile']}")
    for path in tqdm(img_paths, desc="Generating pseudo-labels"):
        if 'noncrack' in os.path.basename(path).lower():
            mask = np.zeros((img_size, img_size), dtype=np.uint8)
            pseudo_masks.append(mask)
            continue
            
        img = Image.open(path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_t, return_cam=False) 
            probs = F.softmax(output, dim=1)
            crack_confidence = probs[0, 1].item()
        
        if crack_confidence < config["confidence_threshold"]:
            mask = np.zeros((img_size, img_size), dtype=np.uint8)
            low_confidence_count += 1
            pseudo_masks.append(mask)
            continue
        
        img_t = img_t.requires_grad_(True)
        
        if use_multiscale:
            cam_layers = ['layer2', 'layer3', 'layer4']
            cams = {}
            for layer_idx, layer_name in enumerate(cam_layers):
                output = model(img_t, return_cam=True, cam_layers=[layer_name])
                model.zero_grad()
                class_score = output[:, 1]
                retain = (layer_idx < len(cam_layers) - 1)
                class_score.backward(retain_graph=retain)
                
                with torch.no_grad():
                    cam = model.generate_gradcam_plusplus(layer_name)
                    cam = cam.squeeze().cpu().numpy()
                    cam_resized = cv2.resize(cam, (img_size, img_size))
                    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
                    cams[layer_name] = cam_norm
            
            # Weighted average for ResNet-34
            # We keep high weights on layer 3/4 as they have better semantic understanding of cracks
            fused_cam = (0.2 * cams['layer2'] + 0.4 * cams['layer3'] + 0.4 * cams['layer4'])
            cam_final = fused_cam
        else:
            output = model(img_t, return_cam=True, cam_layers=['layer4'])
            model.zero_grad()
            class_score = output[:, 1]
            class_score.backward()
            with torch.no_grad():
                cam = model.generate_gradcam_plusplus('layer4')
            cam = cam.squeeze().cpu().numpy()
            cam = cv2.resize(cam, (img_size, img_size))
            cam_final = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        threshold = np.percentile(cam_final, config["cam_percentile"])
        mask = (cam_final > threshold).astype(np.uint8) * 255
        pseudo_masks.append(mask)
    
    print(f"Generated {len(pseudo_masks)} pseudo labels")
    print(f"Skipped to low confidence: {low_confidence_count}")
    return pseudo_masks