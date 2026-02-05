import torch
import torch.nn as nn
import numpy as np
import warnings
from scipy.ndimage import center_of_mass

try:
    import timm
except ImportError:
    warnings.warn("timm is not installed. Please install it using 'pip install timm'")

class RETFound_UNet(nn.Module):
    """
    RETFound U-Net for Optic Disc Detection via Heatmap Regression.
    
    Architecture:
    - Encoder: RETFound (vit_large_patch16) initialized with CFP weights.
    - Decoder: Standard U-Net upsampling blocks to restore 512x512 resolution.
    - Output: 1-channel Gaussian Heatmap (Sigmoid activation).
    
    References:
    - RETFound: https://github.com/rmaphoh/RETFound_MAE
    """
    def __init__(self, img_size=224, num_classes=1, weights_path=None, freeze_encoder=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = 16
        self.embed_dim = 1024  # ViT-Large dimension
        
        # --------------------------------------------------------
        # 1. Encoder: RETFound (ViT-Large)
        # --------------------------------------------------------
        # Load the backbone using timm
        # We use 'vit_large_patch16_224' as the base architecture.
        self.encoder = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            img_size=img_size,
            num_classes=0,  # No classification head
            in_chans=3,
            global_pool=''  # Keep spatial tokens
        )
        
        # Load RETFound-CFP weights if provided
        if weights_path:
            self._load_retfound_weights(weights_path)
            
        # Freeze encoder for the first stage of training
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("RETFound encoder frozen.")

        # --------------------------------------------------------
        # 2. Decoder: Standard U-Net Upsampling
        # --------------------------------------------------------
        # Input: (B, 1024, 14, 14) -> Output: (B, 1, 224, 224)
        # Note: If input is 512x512, bottleneck is 32x32.
        # This implementation assumes 224x224 input -> 14x14 bottleneck.
        # We need 4 upsampling blocks (16x scale factor) to get back to 224.
        
        self.decoder_blocks = nn.ModuleList([
            UpBlock(1024, 512),  # 14 -> 28
            UpBlock(512, 256),   # 28 -> 56
            UpBlock(256, 128),   # 56 -> 112
            UpBlock(128, 64),    # 112 -> 224
        ])
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _load_retfound_weights(self, weights_path):
        """Loads RETFound weights from a checkpoint file."""
        try:
            print(f"Loading RETFound weights from {weights_path}...")
            # Set weights_only=False for PyTorch 2.6+ to allow loading Namespace objects
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint keys
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Filter and load
            # Relaxed loading to ignore head mismatches
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded: {msg}")
            
        except Exception as e:
            print(f"Error loading weights: {e}")

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image.
            
        Returns:
            heatmap: (B, 1, H, W) predicted heatmap in [0, 1].
        """
        # 1. Encoder
        # timm forward_features returns (B, N, C)
        x_enc = self.encoder.forward_features(x)
        
        # Remove CLS token if present
        if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            x_enc = x_enc[:, 1:, :]
            
        # Reshape to spatial grid (B, C, H/P, W/P)
        B, N, C = x_enc.shape
        H_grid = int(np.sqrt(N))
        W_grid = int(np.sqrt(N))
        # Ensure contiguous before reshape to avoid MPS view errors
        x_enc = x_enc.transpose(1, 2).contiguous().reshape(B, C, H_grid, W_grid)
        
        # 2. Decoder
        d = x_enc
        for up_block in self.decoder_blocks:
            d = up_block(d)
            
        # 3. Output Head
        out = self.final_conv(d)
        
        # Sigmoid activation for heatmap regression (0-1 range)
        return torch.sigmoid(out)

    def unfreeze_encoder(self):
        """Unfreezes encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen.")


class UpBlock(nn.Module):
    """
    Standard U-Net Upsampling Block
    Upsample -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv for refinement
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


def HeatmapGenerator(center_x, center_y, height, width, sigma=20):
    """
    Generates a 2D Gaussian Heatmap target for Optic Disc Detection.
    
    Args:
        center_x, center_y: Coordinates of the Optic Disc center.
        height, width: Dimensions of the heatmap (e.g., 224, 224).
        sigma: Spread of the Gaussian blob (default ~20px).
        
    Returns:
        heatmap: 2D numpy array (height, width) with values in [0, 1].
    """
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    
    dist_sq = (x - center_x)**2 + (y - center_y)**2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    
    return heatmap


def get_coordinates_from_heatmap(heatmap):
    """
    Extracts (x, y) from a predicted heatmap using Center of Mass.
    
    Args:
        heatmap: 2D numpy array (H, W).
        
    Returns:
        (x, y): Predicted sub-pixel coordinates.
    """
    # Use scipy's center_of_mass for sub-pixel accuracy
    # We can threshold slightly to reduce noise influence if needed,
    # but for a clean Gaussian prediction, global CoM or windowed CoM is good.
    
    # Simple thresholding to focus on the peak
    threshold = heatmap.max() * 0.5
    masked_heatmap = np.where(heatmap > threshold, heatmap, 0)
    
    if masked_heatmap.sum() == 0:
        # Fallback to argmax if heatmap is empty/flat
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return float(x), float(y)
        
    y_com, x_com = center_of_mass(masked_heatmap)
    return float(x_com), float(y_com)

# Example Usage
if __name__ == "__main__":
    # Test instantiation
    try:
        model = RETFound_UNet(img_size=224)
        print("Model created successfully.")
        
        # Test Heatmap Generation
        hm = HeatmapGenerator(112, 112, 224, 224, sigma=20)
        print(f"Heatmap generated. Max value: {hm.max():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")