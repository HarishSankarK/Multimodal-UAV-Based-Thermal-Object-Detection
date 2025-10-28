import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Basic convolutional block for neck."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class PANet(nn.Module):
    """Path Aggregation Network for feature aggregation."""
    def __init__(self, in_channels=[256, 512, 1024]):
        super(PANet, self).__init__()
        self.in_channels = in_channels
        
        # Top-down path (FPN)
        self.lateral_conv1 = ConvBlock(in_channels[2], in_channels[1], 1)
        self.lateral_conv2 = ConvBlock(in_channels[1], in_channels[0], 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.top_down1 = ConvBlock(in_channels[1] + in_channels[1], in_channels[1], 3, padding=1)
        self.top_down2 = ConvBlock(in_channels[0] + in_channels[0], in_channels[0], 3, padding=1)
        
        # Bottom-up path
        self.downsample1 = ConvBlock(in_channels[0], in_channels[0], 3, stride=2, padding=1)
        self.downsample2 = ConvBlock(in_channels[1], in_channels[1], 3, stride=2, padding=1)
        
        self.bottom_up1 = ConvBlock(in_channels[0] + in_channels[1], in_channels[1], 3, padding=1)
        self.bottom_up2 = ConvBlock(in_channels[1] + in_channels[2], in_channels[2], 3, padding=1)

    def forward(self, features):
        p5, p4, p3 = features  # From backbone: 1024@20x20, 512@40x40, 256@80x80
        
        # Top-down path
        p4_lateral = self.lateral_conv1(p5)
        p4_lateral = self.upsample(p4_lateral)
        p4 = self.top_down1(torch.cat([p4_lateral, p4], dim=1))
        
        p3_lateral = self.lateral_conv2(p4)
        p3_lateral = self.upsample(p3_lateral)
        p3 = self.top_down2(torch.cat([p3_lateral, p3], dim=1))
        
        # Bottom-up path
        p4_down = self.downsample1(p3)
        p4 = self.bottom_up1(torch.cat([p4_down, p4], dim=1))
        
        p5_down = self.downsample2(p4)
        p5 = self.bottom_up2(torch.cat([p5_down, p5], dim=1))
        
        return [p3, p4, p5]  # Return aggregated features

if __name__ == "__main__":
    # Example usage
    neck = PANet(in_channels=[256, 512, 1024])
    features = [
        torch.randn(2, 256, 80, 80),
        torch.randn(2, 512, 40, 40),
        torch.randn(2, 1024, 20, 20)
    ]
    output = neck(features)
    for i, feat in enumerate(output):
        print(f"Neck output {i} shape: {feat.shape}")