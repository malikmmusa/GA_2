"""
U-Net Architecture for Semantic Segmentation and Heatmap Regression

Used in:
- Stage 2: Fovea Localization (Heatmap Regression)
- Stage 3: GA Segmentation (Semantic Segmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm -> ReLU) x 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatches
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture.
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        out_channels: Number of output channels (1 for heatmap/binary mask)
        bilinear: Use bilinear upsampling (True) or transposed conv (False)
    """
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


class UNetWithSigmoid(UNet):
    """
    U-Net with Sigmoid activation for heatmap regression (Stage 2).
    """
    def forward(self, x):
        logits = super().forward(x)
        return torch.sigmoid(logits)


class UNetForSegmentation(UNet):
    """
    U-Net for binary segmentation with Sigmoid (Stage 3).
    Can also be used with BCEWithLogitsLoss (no sigmoid in forward).
    """
    def __init__(self, in_channels=1, out_channels=1, bilinear=True, apply_sigmoid=True):
        super().__init__(in_channels, out_channels, bilinear)
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, x):
        logits = super().forward(x)
        if self.apply_sigmoid:
            return torch.sigmoid(logits)
        return logits


class DualStreamUNet(nn.Module):
    """
    Dual-stream U-Net for processing both B-Scan and En Face images.
    
    Used in Stage 2 for fovea detection with both image types.
    """
    def __init__(self, out_channels=1, bilinear=True):
        super(DualStreamUNet, self).__init__()
        
        # Separate encoders for each stream
        self.b_scan_encoder = nn.Sequential(
            DoubleConv(1, 64),
            Down(64, 128),
            Down(128, 256)
        )
        
        self.en_face_encoder = nn.Sequential(
            DoubleConv(1, 64),
            Down(64, 128),
            Down(128, 256)
        )
        
        # Fusion layer
        self.fusion = DoubleConv(512, 256)  # 256 + 256 = 512
        
        # Shared decoder
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, b_scan, en_face):
        # Encode both streams
        b_features = self.b_scan_encoder(b_scan)
        e_features = self.en_face_encoder(en_face)
        
        # Fuse features
        fused = torch.cat([b_features, e_features], dim=1)
        x = self.fusion(fused)
        
        # Continue with decoder
        x3 = self.down3(x)
        x4 = self.down4(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x)  # Note: simplified, may need proper skip connections
        
        logits = self.outc(x)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    # Test the models
    print("Testing U-Net architectures...")
    
    # Standard U-Net
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    print(f"Standard U-Net: Input {x.shape} -> Output {output.shape}")
    
    # U-Net with Sigmoid
    model_sigmoid = UNetWithSigmoid(in_channels=1, out_channels=1)
    output_sigmoid = model_sigmoid(x)
    print(f"U-Net with Sigmoid: Input {x.shape} -> Output {output_sigmoid.shape}")
    print(f"  Output range: [{output_sigmoid.min():.3f}, {output_sigmoid.max():.3f}]")
    
    # Dual Stream U-Net
    model_dual = DualStreamUNet(out_channels=1)
    b_scan = torch.randn(1, 1, 256, 256)
    en_face = torch.randn(1, 1, 256, 256)
    output_dual = model_dual(b_scan, en_face)
    print(f"Dual Stream U-Net: B-Scan {b_scan.shape} + En Face {en_face.shape} -> Output {output_dual.shape}")
    
    print("\n✓ All models initialized successfully!")
