import torch
import torch.nn as nn
from torchvision import models

class DiscDetectorCNN(nn.Module):
    """
    ResNet18-based model for optic disc coordinate regression.
    Input: (B, 3, 224, 224) RGB images
    Output: (B, 2) Normalized (x, y) coordinates in [0, 1]
    """
    def __init__(self, weights=None):
        super(DiscDetectorCNN, self).__init__()
        
        # Load ResNet18 backbone
        # We use 'weights' parameter if provided, otherwise standard
        if weights == 'IMAGENET1K_V1':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.resnet18(weights=None)
            
        # Replace final fully connected layer
        # ResNet18 has 512 features before FC
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Output: (x, y, h)
            nn.Sigmoid()       # Normalize to [0, 1]
        )
    
    def forward(self, x):
        return self.backbone(x)
