import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import numpy as np
from tqdm import tqdm
import torchvision
from datasets.multimodal_dataset import MultimodalDataset
from datasets.collate import collate_fn
from datasets.transforms import get_transforms
from models.fusion_yolov11 import FusionYOLOv11
from utils.metrics import evaluate_coco
from utils.anchors import assign_anchors_to_scales

def bbox_ciou(box1, box2, eps=1e-7):
    # box format: [x1,y1,x2,y2]
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1
    # IoU
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = b1_w * b1_h + b2_w * b2_h - inter_area + eps
    iou = inter_area / union
    # Center distance
    cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    center_dist = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    # Enclosing box
    enc_x1 = torch.min(b1_x1, b2_x1)
    enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2)
    enc_y2 = torch.max(b1_y2, b2_y2)
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps
    # Aspect ratio consistency
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(b1_w / (b1_h + eps)) - torch.atan(b2_w / (b2_h + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (center_dist / enc_diag + alpha * v)
    return 1 - ciou

def compute_yolo_loss(predictions, targets, anchors, strides=[8, 16, 32], img_size=640, num_classes=4, iou_type='ciou'):
    """
    Compute YOLOv11 loss: box (CIoU), objectness, and classification.
    Args:
        predictions (list): List of [batch_size, num_anchors, H, W, 5 + num_classes].
        targets (dict): Contains 'boxes' [batch_size, max_boxes, 4], 'labels' [batch_size, max_boxes].
        anchors (list): Anchor boxes per scale.
        strides (list): Strides per scale.
        img_size (int): Image size.
        num_classes (int): Number of classes.
        iou_type (str): IoU loss type (ciou, iou, giou, diou).
    Returns:
        tuple: (total_loss, box_loss, obj_loss, cls_loss).
    """
    device = predictions[0].device
    box_loss, obj_loss, cls_loss = 0.0, 0.0, 0.0
    bce_obj = nn.BCEWithLogitsLoss(reduction='sum')
    bce_cls = nn.BCEWithLogitsLoss(reduction='sum')
    
    for i, pred in enumerate(predictions):
        batch_size, num_anchors, h, w = pred.shape[:4]
        pred = pred.view(batch_size, num_anchors, h, w, -1)
        pred_xy = torch.sigmoid(pred[..., :2])  # Center coordinates
        pred_wh = pred[..., 2:4]  # Width and height
        pred_obj = pred[..., 4]  # Objectness
        pred_cls = pred[..., 5:]  # Class probabilities
        
        # Generate grid
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid_x = grid_x.to(device)
        grid_y = grid_y.to(device)
        
        # Decode predictions
        pred_xy = (pred_xy + torch.stack([grid_x, grid_y], dim=-1)) * strides[i]
        pred_wh = torch.exp(pred_wh) * torch.tensor(anchors[i], device=device)
        pred_boxes = torch.cat([pred_xy - pred_wh / 2, pred_xy + pred_wh / 2], dim=-1)  # [batch_size, num_anchors, H, W, 4]
        
        # Ground truth
        target_boxes = targets['boxes']  # [batch_size, max_boxes, 4]
        target_labels = targets['labels']  # [batch_size, max_boxes]
        
        # Assign targets to anchors
        obj_mask = torch.zeros(batch_size, num_anchors, h, w, device=device)
        noobj_mask = torch.ones(batch_size, num_anchors, h, w, device=device)
        target_box = torch.zeros_like(pred_boxes)
        target_cls = torch.zeros(batch_size, num_anchors, h, w, num_classes, device=device)
        
        for b in range(batch_size):
            valid_targets = target_labels[b] > -1
            if not valid_targets.any():
                continue
            gt_boxes = target_boxes[b][valid_targets]  # [num_targets, 4]
            gt_labels = target_labels[b][valid_targets]  # [num_targets]
            
            # Compute IoU between predicted and ground truth boxes
            ious = torchvision.ops.box_iou(pred_boxes[b].view(-1, 4), gt_boxes)
            max_ious, max_indices = ious.max(dim=1)
            
            # Assign anchors
            for idx, (iou, gt_idx) in enumerate(zip(max_ious, max_indices)):
                if iou > 0.5:
                    anchor_idx, y_idx, x_idx = np.unravel_index(idx, (num_anchors, h, w))
                    obj_mask[b, anchor_idx, y_idx, x_idx] = 1.0
                    noobj_mask[b, anchor_idx, y_idx, x_idx] = 0.0
                    target_box[b, anchor_idx, y_idx, x_idx] = gt_boxes[gt_idx]
                    target_cls[b, anchor_idx, y_idx, x_idx, gt_labels[gt_idx]] = 1.0
        
        # Box loss (CIoU)
        if obj_mask.sum() > 0:
            if iou_type == 'ciou':
                box_loss += bbox_ciou(pred_boxes[obj_mask.bool()], target_box[obj_mask.bool()]).mean()
        
        # Objectness loss
        obj_loss += bce_obj(pred_obj, obj_mask) / batch_size
        
        # Classification loss
        if obj_mask.sum() > 0:
            cls_loss += bce_cls(pred_cls[obj_mask.bool()], target_cls[obj_mask.bool()]) / (obj_mask.sum() + 1e-6)
    
    return box_loss, obj_loss, cls_loss

def train_one_epoch(model, dataloader, optimizer, config, device, epoch, writer):
    model.train()
    total_loss, total_box_loss, total_obj_loss, total_cls_loss = 0.0, 0.0, 0.0, 0.0
    num_batches = len(dataloader)
    anchors = assign_anchors_to_scales(np.array(config['model']['anchors']))
    
    try:
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            print(f"Processing batch {batch_idx}")  # Debug print
            rgb = batch['image_rgb'].to(device)
            ir = batch['image_ir'].to(device) if batch['image_ir'] is not None else None
            targets = {
                'boxes': batch['boxes'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            optimizer.zero_grad()
            predictions = model(rgb, ir)
            
            box_loss, obj_loss, cls_loss = compute_yolo_loss(
                predictions,
                targets,
                anchors,
                num_classes=config['model']['num_classes'],
                iou_type=config['loss']['iou_type']
            )
            
            loss = (config['loss']['box_weight'] * box_loss +
                    config['loss']['obj_weight'] * obj_loss +
                    config['loss']['cls_weight'] * cls_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            total_box_loss += box_loss.item()
            total_obj_loss += obj_loss.item()
            total_cls_loss += cls_loss.item()
            
            if batch_idx % config['logging']['log_freq'] == 0:
                writer.add_scalar('Loss/total', loss.item(), epoch * num_batches + batch_idx)
                writer.add_scalar('Loss/box', box_loss.item(), epoch * num_batches + batch_idx)
                writer.add_scalar('Loss/obj', obj_loss.item(), epoch * num_batches + batch_idx)
                writer.add_scalar('Loss/cls', cls_loss.item(), epoch * num_batches + batch_idx)
    except Exception as e:
        print(f"Error in train_one_epoch: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return total_loss / num_batches, total_box_loss / num_batches, total_obj_loss / num_batches, total_cls_loss / num_batches
def main():
    # Load configurations
    with open("experiments/configs/default.yaml", 'r') as f:
        config_default = yaml.safe_load(f)
    with open("experiments/configs/yolov11_fusion.yaml", 'r') as f:
        config_model = yaml.safe_load(f)
    
    # Merge configurations
    config = config_default.copy()
    config.update(config_model)
    
    # Initialize device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = FusionYOLOv11(num_classes=config['model']['num_classes'], img_size=config['data']['img_size'])
    model.to(device)
    
    if config['paths']['pretrained_weights']:
        model.load_state_dict(torch.load(config['paths']['pretrained_weights'], map_location=device))
    
    # Initialize datasets and dataloaders
    train_dataset = MultimodalDataset(
        root_dir=config['data']['root_dir'],
        dataset_name='smod',
        split='train',
        transforms=get_transforms(config, training=True)
    )
    val_dataset = MultimodalDataset(
        root_dir=config['data']['root_dir'],
        dataset_name='smod',
        split='val',
        transforms=get_transforms(config, training=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],  # Reduced to 2 for testing
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        collate_fn=lambda x: collate_fn(x, mosaic_prob=config['augmentation']['mosaic_prob'], img_size=config['data']['img_size'])
    )
    for batch in train_loader:
        print("First batch contents:", {k: v.shape if hasattr(v, 'shape') else v for k, v in batch.items()})
        break
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],  # Reduced to 2 for testing
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        collate_fn=lambda x: collate_fn(x, mosaic_prob=0.0, img_size=config['data']['img_size'])
    )
    
    # Initialize optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    if config['training']['lr_schedule'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    # Initialize logging
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(config['logging']['tensorboard_dir'])
    
    # Training loop
    best_map = 0.0
    for epoch in range(config['training']['epochs']):
        # Train
        avg_loss, avg_box_loss, avg_obj_loss, avg_cls_loss = train_one_epoch(
            model, train_loader, optimizer, config, device, epoch, writer
        )
        print(f"Epoch {epoch+1}/{config['training']['epochs']}: "
              f"Loss={avg_loss:.4f}, Box={avg_box_loss:.4f}, Obj={avg_obj_loss:.4f}, Cls={avg_cls_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % config['evaluation']['eval_freq'] == 0:
            metrics = evaluate_coco(
                model,
                val_loader,
                os.path.join(config['data']['root_dir'], config['data']['datasets'][0]['splits']['val']),
                config['model']['anchors'],
                device
            )
            writer.add_scalar('mAP@0.5', metrics['mAP@0.5'], epoch)
            writer.add_scalar('mAP@0.5:0.95', metrics['mAP@0.5:0.95'], epoch)
            print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}, mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
            
            # Save best model
            if metrics['mAP@0.5'] > best_map:
                best_map = metrics['mAP@0.5']
                torch.save(model.state_dict(), os.path.join(config['logging']['checkpoint_dir'], 'yolov11_fusion_best.pt'))
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(config['logging']['checkpoint_dir'], f'yolov11_fusion_epoch_{epoch+1}.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config['logging']['checkpoint_dir'], 'yolov11_fusion_last.pt'))
    writer.close()

if __name__ == "__main__":
    main()