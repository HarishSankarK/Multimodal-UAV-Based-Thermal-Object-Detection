import torch
import torch.nn as nn
import sys
import os

# Ensure models directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.backbones import DualBackbone
from models.necks import PANet
from models.heads import YOLOv11Head

class FusionYOLOv11(nn.Module):
    """Multimodal YOLOv11 with mid-fusion for RGB and IR inputs."""
    def __init__(self, num_classes=4, img_size=640):
        super(FusionYOLOv11, self).__init__()
        self.backbone = DualBackbone()
        self.neck = PANet(in_channels=[256, 512, 1024])
        self.head = YOLOv11Head(num_classes=num_classes, in_channels=[256, 512, 1024], img_size=img_size)
        
        # Fusion layers (concatenate RGB and IR features)
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(in_channels * 2, in_channels, 1) for in_channels in [256, 512, 1024]
        ])

    def forward(self, rgb, ir=None):
        # Extract features
        rgb_features, ir_features = self.backbone(rgb, ir)
        
        # Fuse features (concatenate and reduce channels)
        fused_features = []
        for i, (rgb_feat, ir_feat) in enumerate(zip(rgb_features, ir_features if ir is not None else [None] * 3)):
            if ir_feat is not None:
                fused = torch.cat([rgb_feat, ir_feat], dim=1)
                fused = self.fusion_layers[i](fused)
            else:
                fused = rgb_feat  # Use RGB only if IR is absent
            fused_features.append(fused)
        
        # Pass through neck
        neck_features = self.neck(fused_features)
        
        # Detection head
        predictions = self.head(neck_features)
        return predictions

if __name__ == "__main__":
    # Example usage
    model = FusionYOLOv11(num_classes=4, img_size=640)
    rgb = torch.randn(2, 3, 640, 640)
    ir = torch.randn(2, 3, 640, 640)
    predictions = model(rgb, ir)
    for i, pred in enumerate(predictions):
        print(f"Scale {i} prediction shape: {pred.shape}")