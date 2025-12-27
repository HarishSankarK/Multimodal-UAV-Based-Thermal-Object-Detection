import torch
import torch.nn as nn
import sys
import os

# Ensure models directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.backbones import DualBackbone
from models.necks import PANet
from models.heads import YOLOv11Head

class AttentionGating(nn.Module):
    """Lightweight attention gating module G(·) for modality-specific weighting.
    Per paper: F'fused = G(FRGB, FIR) ⊗ Ffused
    """
    def __init__(self, channels):
        super(AttentionGating, self).__init__()
        # Compute attention weights from both modalities
        # Input: concatenated RGB+IR features (channels * 2)
        # Output: attention weights for fused features (channels)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_feat, ir_feat):
        # Concatenate RGB and IR for attention computation
        concat_feat = torch.cat([rgb_feat, ir_feat], dim=1)  # [B, C*2, H, W]
        # Generate attention weights
        attention_weights = self.attention(concat_feat)  # [B, C, 1, 1]
        # Return attention weights (will be applied to fused features in forward)
        return attention_weights

class FusionYOLOv11(nn.Module):
    """Multimodal YOLOv11 with mid-fusion for RGB and IR inputs.
    Implements the architecture described in the paper:
    - Dual-stream SGGF-Net backbone
    - Mid-level fusion with attention gating
    - PANet for multi-scale feature aggregation
    - YOLOv11 detection head
    """
    def __init__(self, num_classes=4, img_size=640, fast_mode=False):
        super(FusionYOLOv11, self).__init__()
        self.fast_mode = fast_mode
        self.backbone = DualBackbone(fast_mode=fast_mode)
        self.neck = PANet(in_channels=[256, 512, 1024])
        self.head = YOLOv11Head(num_classes=num_classes, in_channels=[256, 512, 1024], img_size=img_size)
        
        # Fusion layers (concatenate RGB and IR features)
        # Note: Backbone returns features in order [1024, 512, 256] (s4, s3, s2)
        # So fusion layers must match: [1024*2->1024, 512*2->512, 256*2->256]
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(ch * 2, ch, 1) for ch in [1024, 512, 256]
        ])
        
        # Attention gating modules per paper: G(FRGB, FIR) ⊗ Ffused
        self.attention_gates = nn.ModuleList([
            AttentionGating(ch) for ch in [1024, 512, 256]
        ])

    def forward(self, rgb, ir=None):
        # Extract features using dual-stream SGGF-Net
        rgb_features, ir_features = self.backbone(rgb, ir)
        
        # Mid-level fusion with attention gating (per paper)
        fused_features = []
        for i, (rgb_feat, ir_feat) in enumerate(zip(rgb_features, ir_features if ir is not None else [None] * 3)):
            if ir_feat is not None:
                # Step 1: Concatenate features
                fused = torch.cat([rgb_feat, ir_feat], dim=1)  # [B, C*2, H, W]
                # Step 2: Reduce channels
                fused = self.fusion_layers[i](fused)  # [B, C, H, W]
                # Step 3: Apply attention gating (per paper): F'fused = G(FRGB, FIR) ⊗ Ffused
                attention_weights = self.attention_gates[i](rgb_feat, ir_feat)  # [B, C, 1, 1]
                fused = attention_weights * fused  # Element-wise multiplication
            else:
                fused = rgb_feat  # Use RGB only if IR is absent
            fused_features.append(fused)
        
        # Pass through PANet for multi-scale feature aggregation
        neck_features = self.neck(fused_features)
        
        # YOLOv11 detection head
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