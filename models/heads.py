import torch
import torch.nn as nn

class YOLOv11Head(nn.Module):
    """YOLOv11 detection head with decoupled box and class predictions."""
    def __init__(self, num_classes=4, in_channels=[256, 512, 1024], img_size=640, num_anchors=3):
        super(YOLOv11Head, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.strides = [8, 16, 32]  # Corresponding to 80x80, 40x40, 20x20 outputs
        
        self.box_heads = nn.ModuleList([
            nn.Conv2d(ch, num_anchors * 4, 1) for ch in in_channels
        ])
        self.obj_heads = nn.ModuleList([
            nn.Conv2d(ch, num_anchors * 1, 1) for ch in in_channels
        ])
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(ch, num_anchors * num_classes, 1) for ch in in_channels
        ])

    def forward(self, features):
        outputs = []
        for i, (feature, box_head, obj_head, cls_head) in enumerate(zip(features, self.box_heads, self.obj_heads, self.cls_heads)):
            # Box predictions: [batch_size, num_anchors * 4, H, W]
            box = box_head(feature)
            # Objectness: [batch_size, num_anchors * 1, H, W]
            obj = obj_head(feature)
            # Class probabilities: [batch_size, num_anchors * num_classes, H, W]
            cls = cls_head(feature)
            
            # Combine predictions
            batch_size, _, h, w = box.shape
            out = torch.cat([
                box.view(batch_size, self.num_anchors, 4, h, w),
                obj.view(batch_size, self.num_anchors, 1, h, w),
                cls.view(batch_size, self.num_anchors, self.num_classes, h, w)
            ], dim=2)  # [batch_size, num_anchors, 5 + num_classes, H, W]
            out = out.permute(0, 1, 3, 4, 2).contiguous()  # [batch_size, num_anchors, H, W, 5 + num_classes]
            outputs.append(out)
        return outputs

if __name__ == "__main__":
    # Example usage
    head = YOLOv11Head(num_classes=4, in_channels=[256, 512, 1024])
    features = [
        torch.randn(2, 256, 80, 80),
        torch.randn(2, 512, 40, 40),
        torch.randn(2, 1024, 20, 20)
    ]
    predictions = head(features)
    for i, pred in enumerate(predictions):
        print(f"Head output {i} shape: {pred.shape}")