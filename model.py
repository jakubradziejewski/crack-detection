import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class WeaklySupCrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load ResNet18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # We keep spatial resolution by skipping the final pooling/fc layers.
        # Output of layer4 is 512 channels x 7x7 (for 224 input).
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        
        # 1x1 Conv: Collapses 512 features -> 1 "Crack Score" map
        self.classifier_conv = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        feat = self.features(x)  # [Batch, 512, 7, 7]
        
        # Generate the raw heatmap (before pooling)
        # This map shows WHERE the network thinks the crack is.
        mask_logits = self.classifier_conv(feat) # [Batch, 1, 7, 7]
        
        # Global Max Pooling (GMP)
        # We take the maximum value from the heatmap. 
        # Logic: If *any* part of the image has a high crack score, the image is a crack.
        # adaptive_max_pool2d(input, output_size)
        logits = F.adaptive_max_pool2d(mask_logits, (1, 1)).view(-1)
        
        return logits, mask_logits