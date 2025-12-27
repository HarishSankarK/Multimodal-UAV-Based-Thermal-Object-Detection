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
    Optimized YOLOv11 loss computation.
    """
    device = predictions[0].device
    box_loss, obj_loss, cls_loss = 0.0, 0.0, 0.0
    # Create loss functions once and reuse (faster than creating each time)
    if not hasattr(compute_yolo_loss, '_bce_obj'):
        compute_yolo_loss._bce_obj = nn.BCEWithLogitsLoss(reduction='mean')
        compute_yolo_loss._bce_cls = nn.BCEWithLogitsLoss(reduction='mean')
    bce_obj = compute_yolo_loss._bce_obj
    bce_cls = compute_yolo_loss._bce_cls
    
    for i, pred in enumerate(predictions):
        batch_size, num_anchors, h, w = pred.shape[:4]
        pred = pred.view(batch_size, num_anchors, h, w, -1)
        pred_xy = torch.sigmoid(pred[..., :2])  # Center coordinates
        pred_wh = pred[..., 2:4]  # Width and height
        pred_obj = pred[..., 4]  # Objectness
        pred_cls = pred[..., 5:]  # Class probabilities
        
        # Pre-compute grid (only once per scale) - cache this if same size
        # Use more efficient meshgrid
        yv, xv = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), 
                                torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
        grid = torch.stack([xv, yv], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]
        
        # Decode predictions efficiently
        pred_xy = (pred_xy + grid) * strides[i]  # [B, A, H, W, 2]
        
        # Convert anchors to tensor once - ensure correct shape [num_anchors, 2]
        # Cache anchor tensors to avoid repeated conversion
        if not hasattr(compute_yolo_loss, '_anchor_cache'):
            compute_yolo_loss._anchor_cache = {}
        cache_key = (i, num_anchors)
        if cache_key not in compute_yolo_loss._anchor_cache:
            anchor_array = np.array(anchors[i], dtype=np.float32)
            if anchor_array.ndim == 1:
                anchor_array = anchor_array.reshape(-1, 2)
            elif anchor_array.ndim == 2 and anchor_array.shape[1] != 2:
                anchor_array = anchor_array.reshape(-1, 2)
            if anchor_array.shape[0] != num_anchors:
                raise ValueError(f"Anchor count mismatch at scale {i}: expected {num_anchors}, got {anchor_array.shape[0]}. Anchors: {anchors[i]}")
            compute_yolo_loss._anchor_cache[cache_key] = torch.from_numpy(anchor_array).to(device)
        anchor_tensor = compute_yolo_loss._anchor_cache[cache_key]  # [num_anchors, 2]
        pred_wh = torch.exp(pred_wh) * anchor_tensor.view(1, num_anchors, 1, 1, 2)
        
        # Ground truth
        target_boxes = targets['boxes']  # [batch_size, max_boxes, 4]
        target_labels = targets['labels']  # [batch_size, max_boxes]
        
        # Initialize masks
        obj_mask = torch.zeros(batch_size, num_anchors, h, w, device=device, dtype=torch.bool)
        target_box = torch.zeros(batch_size, num_anchors, h, w, 4, device=device)
        target_cls = torch.zeros(batch_size, num_anchors, h, w, num_classes, device=device)
        
        # Fully vectorized anchor assignment - process all batches and GT boxes at once
        anchor_wh = anchor_tensor  # [num_anchors, 2]
        anchor_w = anchor_wh[:, 0:1]  # [num_anchors, 1] for broadcasting
        anchor_h = anchor_wh[:, 1:2]  # [num_anchors, 1]
        
        # Process all batches
        for b in range(batch_size):
            valid_mask = target_labels[b] >= 0
            if not valid_mask.any():
                continue
                
            gt_boxes = target_boxes[b][valid_mask]  # [num_gt, 4]
            gt_labels = target_labels[b][valid_mask].long()  # [num_gt]
            num_gt = gt_boxes.shape[0]
            
            if num_gt == 0:
                continue
            
            # Convert GT boxes to center format - vectorized for all GT boxes
            gt_xy = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) * 0.5  # [num_gt, 2]
            gt_wh = gt_boxes[:, 2:4] - gt_boxes[:, 0:2]  # [num_gt, 2]
            gt_w = gt_wh[:, 0:1]  # [num_gt, 1]
            gt_h = gt_wh[:, 1:2]  # [num_gt, 1]
            
            # Find grid cells for all GT boxes - vectorized
            gt_xy_scaled = gt_xy / strides[i]  # [num_gt, 2]
            grid_x = gt_xy_scaled[:, 0:1].long().clamp(0, w - 1)  # [num_gt, 1]
            grid_y = gt_xy_scaled[:, 1:2].long().clamp(0, h - 1)  # [num_gt, 1]
            
            # Vectorized anchor matching for all GT boxes at once
            # Compute ratios: [num_gt, num_anchors]
            ratio_w1 = gt_w / (anchor_w.T + 1e-8)  # [num_gt, num_anchors]
            ratio_w2 = anchor_w.T / (gt_w + 1e-8)  # [num_gt, num_anchors]
            ratio_w = torch.max(ratio_w1, ratio_w2)  # [num_gt, num_anchors]
            
            ratio_h1 = gt_h / (anchor_h.T + 1e-8)  # [num_gt, num_anchors]
            ratio_h2 = anchor_h.T / (gt_h + 1e-8)  # [num_gt, num_anchors]
            ratio_h = torch.max(ratio_h1, ratio_h2)  # [num_gt, num_anchors]
            
            ratio_max = torch.max(ratio_w, ratio_h)  # [num_gt, num_anchors]
            
            # Find best matching anchor for each GT box (ratio < 4.0)
            matching_mask = ratio_max < 4.0  # [num_gt, num_anchors]
            
            # For each GT box, find best anchor (smallest ratio among matching anchors)
            # Set non-matching anchors to large value
            ratio_max_masked = torch.where(matching_mask, ratio_max, torch.full_like(ratio_max, 1e6))
            best_anchor_indices = ratio_max_masked.argmin(dim=1)  # [num_gt]
            best_ratios = ratio_max.gather(1, best_anchor_indices.unsqueeze(1)).squeeze(1)  # [num_gt]
            
            # Only assign if ratio < 4.0
            valid_assignments = best_ratios < 4.0
            
            if valid_assignments.any():
                # Get valid indices
                valid_indices = valid_assignments.nonzero(as_tuple=True)[0]
                valid_anchors = best_anchor_indices[valid_indices]
                valid_gx = grid_x[valid_indices].squeeze(1)
                valid_gy = grid_y[valid_indices].squeeze(1)
                valid_gt_boxes = gt_boxes[valid_indices]
                valid_gt_labels = gt_labels[valid_indices]
                
                # Assign in one go (vectorized)
                obj_mask[b, valid_anchors, valid_gy, valid_gx] = True
                target_box[b, valid_anchors, valid_gy, valid_gx] = valid_gt_boxes
                target_cls[b, valid_anchors, valid_gy, valid_gx, valid_gt_labels] = 1.0
        
        # Box loss (CIoU) - only on positive anchors (optimized)
        if obj_mask.any():
            # Compute boxes only for positive anchors
            pred_xy_pos = pred_xy[obj_mask]  # [N_pos, 2]
            pred_wh_pos = pred_wh[obj_mask]  # [N_pos, 2]
            pred_boxes_pos = torch.cat([pred_xy_pos - pred_wh_pos / 2, pred_xy_pos + pred_wh_pos / 2], dim=-1)  # [N_pos, 4]
            target_box_pos = target_box[obj_mask]  # [N_pos, 4]
            box_loss += bbox_ciou(pred_boxes_pos, target_box_pos).mean()
        
        # Objectness loss - use in-place operations where possible
        obj_target = obj_mask.float()
        obj_loss += bce_obj(pred_obj.view(-1), obj_target.view(-1))
        
        # Classification loss - only on positive anchors (optimized)
        if obj_mask.any():
            pred_cls_pos = pred_cls[obj_mask]  # [N_pos, num_classes]
            target_cls_pos = target_cls[obj_mask]  # [N_pos, num_classes]
            cls_loss += bce_cls(pred_cls_pos, target_cls_pos)
    
    return box_loss, obj_loss, cls_loss

def train_one_epoch(model, dataloader, optimizer, config, device, epoch, writer):
    model.train()
    total_loss, total_box_loss, total_obj_loss, total_cls_loss = 0.0, 0.0, 0.0, 0.0
    num_batches = len(dataloader)
    # Anchors are already organized by scale in config: [[scale1_anchors], [scale2_anchors], [scale3_anchors]]
    # Convert to list of numpy arrays, one per scale
    anchors = [np.array(scale_anchors) for scale_anchors in config['model']['anchors']]
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    # Optimize CPU thread usage
    if device.type == 'cpu':
        import os
        # Use all available CPU cores
        num_threads = min(os.cpu_count() or 4, 8)  # Cap at 8 to avoid overhead
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        print(f"Using {num_threads} CPU threads for parallelization")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Optimized data transfer - use pin_memory if available
            rgb = batch['image_rgb'].to(device, non_blocking=True)
            ir = batch['image_ir'].to(device, non_blocking=True) if batch['image_ir'] is not None else None
            targets = {
                'boxes': batch['boxes'].to(device, non_blocking=True),
                'labels': batch['labels'].to(device, non_blocking=True)
            }
            
            # Zero gradients efficiently
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Forward pass
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
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            # Optimizer step
            optimizer.step()
            
            # Store loss values (extract before deleting)
            loss_val = float(loss.item())
            box_loss_val = float(box_loss.item())
            obj_loss_val = float(obj_loss.item())
            cls_loss_val = float(cls_loss.item())
            
            # Clean up
            del loss, box_loss, obj_loss, cls_loss, predictions
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            total_loss += loss_val
            total_box_loss += box_loss_val
            total_obj_loss += obj_loss_val
            total_cls_loss += cls_loss_val
            
            # Log periodically (reduce frequency for CPU)
            log_freq = config['logging']['log_freq'] * 2 if device.type == 'cpu' else config['logging']['log_freq']
            if batch_idx % log_freq == 0:
                writer.add_scalar('Loss/total', loss_val, epoch * num_batches + batch_idx)
                writer.add_scalar('Loss/box', box_loss_val, epoch * num_batches + batch_idx)
                writer.add_scalar('Loss/obj', obj_loss_val, epoch * num_batches + batch_idx)
                writer.add_scalar('Loss/cls', cls_loss_val, epoch * num_batches + batch_idx)
            
            # Update progress bar (less frequently for CPU to reduce overhead)
            update_freq = 50 if device.type == 'cpu' else 10
            if batch_idx % update_freq == 0 or batch_idx == num_batches - 1:
                pbar.set_postfix({
                    'loss': f'{loss_val:.4f}',
                    'box': f'{box_loss_val:.4f}',
                    'obj': f'{obj_loss_val:.4f}',
                    'cls': f'{cls_loss_val:.4f}'
                })
            
        except Exception as e:
            print(f"Error in train_one_epoch at batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return total_loss / num_batches, total_box_loss / num_batches, total_obj_loss / num_batches, total_cls_loss / num_batches

def main():
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Load configurations
    with open(os.path.join(project_root, "experiments/configs/default.yaml"), 'r') as f:
        config_default = yaml.safe_load(f)
    with open(os.path.join(project_root, "experiments/configs/yolov11_fusion.yaml"), 'r') as f:
        config_model = yaml.safe_load(f)
    
    # Merge configurations
    config = config_default.copy()
    config.update(config_model)
    
    # Resolve relative paths to absolute paths
    if not os.path.isabs(config['data']['root_dir']):
        config['data']['root_dir'] = os.path.join(project_root, config['data']['root_dir'])
    
    # Initialize device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model with fast mode for CPU training
    fast_mode = (device.type == 'cpu')  # Enable fast mode on CPU
    model = FusionYOLOv11(
        num_classes=config['model']['num_classes'], 
        img_size=config['data']['img_size'],
        fast_mode=fast_mode
    )
    model.to(device)
    
    # Compile model for faster execution (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile for faster execution...")
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully!")
    except Exception as e:
        print(f"Could not compile model (this is OK): {e}")
    
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
    
    # Use num_workers=0 on macOS to avoid multiprocessing issues, or use the config value
    import platform
    num_workers = 0 if platform.system() == 'Darwin' else config['data']['num_workers']
    if num_workers > 0:
        print(f"Using {num_workers} workers for data loading")
    else:
        print("Using single-threaded data loading (num_workers=0)")
    
    # Create collate function that's pickleable
    from functools import partial
    train_collate = partial(collate_fn, mosaic_prob=config['augmentation']['mosaic_prob'], img_size=config['data']['img_size'])
    val_collate = partial(collate_fn, mosaic_prob=0.0, img_size=config['data']['img_size'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate,
        pin_memory=(device.type == 'cuda'),  # Only pin memory for GPU
        persistent_workers=(num_workers > 0),  # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
        drop_last=True  # Drop last incomplete batch for consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Initialize optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = None
    if config['training']['lr_schedule'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    # Initialize logging
    log_dir = os.path.join(project_root, config['logging']['log_dir']) if not os.path.isabs(config['logging']['log_dir']) else config['logging']['log_dir']
    tensorboard_dir = os.path.join(project_root, config['logging']['tensorboard_dir']) if not os.path.isabs(config['logging']['tensorboard_dir']) else config['logging']['tensorboard_dir']
    checkpoint_dir = os.path.join(project_root, config['logging']['checkpoint_dir']) if not os.path.isabs(config['logging']['checkpoint_dir']) else config['logging']['checkpoint_dir']
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    
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
        if scheduler is not None:
            scheduler.step()
        
        # Evaluate
        if (epoch + 1) % config['evaluation']['eval_freq'] == 0:
            # Find the dataset being used (smod in this case)
            dataset_name = 'smod'
            val_annotation_path = os.path.join(config['data']['root_dir'], dataset_name, 'annotations', 'instances_val.json')
            metrics = evaluate_coco(
                model,
                val_loader,
                val_annotation_path,
                config['model']['anchors'],
                device
            )
            writer.add_scalar('mAP@0.5', metrics['mAP@0.5'], epoch)
            writer.add_scalar('mAP@0.5:0.95', metrics['mAP@0.5:0.95'], epoch)
            print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}, mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
            
            # Save best model
            if metrics['mAP@0.5'] > best_map:
                best_map = metrics['mAP@0.5']
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'yolov11_fusion_best.pt'))
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_freq'] == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'yolov11_fusion_epoch_{epoch+1}.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'yolov11_fusion_last.pt'))
    writer.close()

if __name__ == "__main__":
    main()
