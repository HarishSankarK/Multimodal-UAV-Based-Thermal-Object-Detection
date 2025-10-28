import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Basic convolutional block with Conv2d, BatchNorm, and SiLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class SelfAttention(nn.Module):
    """Self-Attention module for GFEM as channel attention."""
    def __init__(self, in_channels, reduction=16):
        super(SelfAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels * 2, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = self.avg_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        max_pooled = self.max_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)  # [B, 2*C]
        y = self.fc1(pooled)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y  # Channel-wise multiplication

class Bottleneck(nn.Module):
    """CSPDarknet bottleneck with residual connection."""
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels // 2, 1)
        self.conv2 = ConvBlock(out_channels // 2, out_channels, 3, padding=1)
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut:
            out += residual
        return out

class CSPBlock(nn.Module):
    """CSPDarknet block with multiple bottlenecks."""
    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcut=True):
        super(CSPBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 1)
        self.conv2 = ConvBlock(in_channels, out_channels, 1)
        self.bottlenecks = nn.Sequential(*[Bottleneck(out_channels, out_channels, shortcut) for _ in range(num_bottlenecks)])
        self.conv3 = ConvBlock(out_channels * 2, out_channels, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.bottlenecks(y2)
        return self.conv3(torch.cat([y1, y2], dim=1))

class GFEM(nn.Module):
    """Global Feature Extraction Module with Self-Attention."""
    def __init__(self, in_channels):
        super(GFEM, self).__init__()
        self.self_attention = SelfAttention(in_channels)

    def forward(self, x):
        return self.self_attention(x)

class SGGFNet(nn.Module):
    """SGGF-Net backbone integrating GFEM."""
    def __init__(self, in_channels=3):
        super(SGGFNet, self).__init__()
        self.stem = ConvBlock(in_channels, 64, 6, stride=2, padding=2)
        self.stage1 = CSPBlock(64, 128, 3)
        self.stage2 = nn.Sequential(
            ConvBlock(128, 256, 3, stride=2, padding=1),
            CSPBlock(256, 256, 9)
        )
        self.stage3 = nn.Sequential(
            ConvBlock(256, 512, 3, stride=2, padding=1),
            CSPBlock(512, 512, 9)
        )
        self.stage4 = nn.Sequential(
            ConvBlock(512, 1024, 3, stride=2, padding=1),
            CSPBlock(1024, 1024, 3)
        )
        self.gfem = GFEM(1024)  # Apply GFEM to the deepest feature map for global fusion

    def forward(self, x):
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s4 = self.gfem(s4)  # Apply GFEM to the deepest feature map
        return s4, s3, s2  # Feature maps at three scales

class DualBackbone(nn.Module):
    """Dual-stream backbone for RGB and IR inputs."""
    def __init__(self):
        super(DualBackbone, self).__init__()
        self.rgb_backbone = SGGFNet(in_channels=3)
        self.ir_backbone = SGGFNet(in_channels=3)  # IR converted to 3 channels in dataset

    def forward(self, rgb, ir=None):
        rgb_features = self.rgb_backbone(rgb)  # (s2, s3, s4)
        ir_features = self.ir_backbone(ir) if ir is not None else None
        return rgb_features, ir_features

if __name__ == "__main__":
    # Example usage
    backbone = DualBackbone()
    rgb = torch.randn(2, 3, 640, 640)
    ir = torch.randn(2, 3, 640, 640)
    rgb_feats, ir_feats = backbone(rgb, ir)
    for i, (rf, irf) in enumerate(zip(rgb_feats, ir_feats)):
        print(f"Stage {i+2} RGB shape: {rf.shape}, IR shape: {irf.shape}")