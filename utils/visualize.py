import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torchvision

def decode_predictions(predictions, anchors, strides=[8, 16, 32], img_size=640, conf_thres=0.5, iou_thres=0.5):
    """
    Decode YOLOv11 predictions to bounding boxes.
    Args:
        predictions (list): List of tensors [batch_size, num_anchors, H, W, 5 + num_classes].
        anchors (list): List of anchor boxes per scale.
        strides (list): Strides for each scale (default: [8, 16, 32]).
        img_size (int): Image size (default: 640).
        conf_thres (float): Confidence threshold for objectness.
        iou_thres (float): IoU threshold for NMS.
    Returns:
        list: List of [x_min, y_min, x_max, y_max, conf, class_id] per image.
    """
    batch_size = predictions[0].shape[0]
    detections = []
    
    for batch_idx in range(batch_size):
        batch_dets = []
        for i, pred in enumerate(predictions):
            num_anchors, h, w = pred.shape[1:4]
            pred = pred[batch_idx].view(num_anchors * h * w, -1)  # [num_anchors * H * W, 5 + num_classes]
            
            # Generate grid
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_x = grid_x.reshape(-1).to(pred.device)
            grid_y = grid_y.reshape(-1).to(pred.device)
            
            # Decode box coordinates
            xy = (torch.sigmoid(pred[:, :2]) + torch.stack([grid_x, grid_y], dim=-1)) * strides[i]
            # Convert anchors to tensor if needed
            anchor_tensor = torch.tensor(anchors[i], device=pred.device, dtype=torch.float32)
            wh = torch.exp(pred[:, 2:4]) * anchor_tensor
            boxes = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)  # [x_min, y_min, x_max, y_max]
            
            # Decode objectness and class scores
            obj = torch.sigmoid(pred[:, 4])
            cls = torch.sigmoid(pred[:, 5:])
            conf = obj * cls.max(dim=1)[0]
            
            # Filter by confidence
            mask = conf > conf_thres
            boxes = boxes[mask]
            conf = conf[mask]
            cls = cls[mask].argmax(dim=1)
            
            # NMS
            if boxes.shape[0] > 0:
                indices = torchvision.ops.nms(boxes, conf, iou_thres)
                boxes = boxes[indices]
                conf = conf[indices]
                cls = cls[indices]
                batch_dets.append(torch.cat([boxes, conf[:, None], cls[:, None].float()], dim=1))
        
        if batch_dets:
            batch_dets = torch.cat(batch_dets, dim=0)
        else:
            batch_dets = torch.zeros((0, 6), device=pred.device)
        detections.append(batch_dets)
    
    return detections

def draw_boxes(image, boxes, labels=None, class_names=['person', 'rider', 'bicycle', 'car'], colors=None):
    """
    Draw bounding boxes and labels on an image.
    Args:
        image (np.ndarray): Image in RGB format (H, W, 3).
        boxes (torch.Tensor): Tensor of shape [N, 6] with [x_min, y_min, x_max, y_max, conf, class_id].
        labels (list, optional): List of class names (not used if boxes contain class_id).
        class_names (list): Names of classes.
        colors (list): List of colors per class (optional).
    Returns:
        np.ndarray: Image with drawn boxes.
    """
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Red, Blue, Yellow
    
    img = image.copy()
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = box.cpu().numpy()
        cls = int(cls)
        color = colors[cls % len(colors)]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.putText(img, label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def visualize_detections(rgb_image, ir_image, predictions, anchors, output_dir="visualizations", img_id=0):
    """
    Visualize predictions on RGB and IR images.
    Args:
        rgb_image (torch.Tensor): RGB image [3, H, W].
        ir_image (torch.Tensor): IR image [3, H, W] or None.
        predictions (list): Model predictions.
        anchors (list): Anchor boxes per scale.
        output_dir (str): Directory to save visualizations.
        img_id (int): Image ID for naming.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Decode predictions
    detections = decode_predictions(predictions, anchors)
    
    # Convert images to numpy
    rgb_img = (rgb_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ir_img = (ir_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) if ir_image is not None else None
    
    # Draw boxes
    rgb_with_boxes = draw_boxes(rgb_img, detections[0])
    if ir_img is not None:
        ir_with_boxes = draw_boxes(ir_img, detections[0])
    
    # Save images
    cv2.imwrite(os.path.join(output_dir, f"rgb_{img_id}.jpg"), cv2.cvtColor(rgb_with_boxes, cv2.COLOR_RGB2BGR))
    if ir_img is not None:
        cv2.imwrite(os.path.join(output_dir, f"ir_{img_id}.jpg"), cv2.cvtColor(ir_with_boxes, cv2.COLOR_RGB2BGR))

def visualize_heatmap(image, predictions, output_dir="visualizations", img_id=0):
    """
    Visualize objectness heatmap.
    Args:
        image (torch.Tensor): Input image [3, H, W].
        predictions (list): Model predictions.
        output_dir (str): Directory to save heatmap.
        img_id (int): Image ID for naming.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate objectness scores across scales
    heatmap = torch.zeros((image.shape[1], image.shape[2]))
    for pred, stride in zip(predictions, [8, 16, 32]):
        obj = torch.sigmoid(pred[..., 4])  # Objectness scores
        obj = obj.sum(dim=1).cpu()  # Sum over anchors
        obj = torch.nn.functional.interpolate(obj[None, None, ...], size=(image.shape[1], image.shape[2]), mode='bilinear')[0, 0]
        heatmap += obj
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    
    # Convert to numpy and visualize
    heatmap = (heatmap.numpy() * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay on image
    img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0.0)
    
    # Save
    cv2.imwrite(os.path.join(output_dir, f"heatmap_{img_id}.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # Example usage
    from models.fusion_yolov11 import FusionYOLOv11
    anchors = [np.array([[10, 13], [16, 30], [33, 23]]),
               np.array([[30, 61], [62, 45], [59, 119]]),
               np.array([[116, 90], [156, 198], [373, 326]])]
    model = FusionYOLOv11(num_classes=4)
    rgb = torch.randn(1, 3, 640, 640)
    ir = torch.randn(1, 3, 640, 640)
    predictions = model(rgb, ir)
    visualize_detections(rgb[0], ir[0], predictions, anchors, img_id=0)
    visualize_heatmap(rgb[0], predictions, img_id=0)