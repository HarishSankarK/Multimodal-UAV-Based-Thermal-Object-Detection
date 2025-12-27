import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import torchvision
from datasets.multimodal_dataset import MultimodalDataset
from datasets.collate import collate_fn
from datasets.transforms import get_transforms
from models.fusion_yolov11 import FusionYOLOv11
from utils.metrics import evaluate_coco
from torch.cuda.amp import autocast, GradScaler

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
        # Cache grid tensors to avoid repeated computation
        if not hasattr(compute_yolo_loss, '_grid_cache'):
            compute_yolo_loss._grid_cache = {}
        grid_key = (i, h, w, device)
        if grid_key not in compute_yolo_loss._grid_cache:
            yv, xv = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), 
                                    torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
            grid = torch.stack([xv, yv], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]
            compute_yolo_loss._grid_cache[grid_key] = grid
        else:
            grid = compute_yolo_loss._grid_cache[grid_key]
        
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

def train_one_epoch(model, dataloader, optimizer, config, device, epoch, writer, scaler=None):
    model.train()
    # Use torch.set_grad_enabled(True) explicitly for optimization
    torch.set_grad_enabled(True)
    total_loss, total_box_loss, total_obj_loss, total_cls_loss = 0.0, 0.0, 0.0, 0.0
    num_batches = len(dataloader)
    # Anchors are already organized by scale in config: [[scale1_anchors], [scale2_anchors], [scale3_anchors]]
    # Convert to list of numpy arrays, one per scale
    anchors = [np.array(scale_anchors) for scale_anchors in config['model']['anchors']]
    
    # Disable tqdm for faster training (can re-enable for debugging)
    pbar = dataloader  # Use dataloader directly instead of tqdm for speed
    # pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=True)  # Disable progress bar for speed
    
    # Optimize CPU thread usage
    if device.type == 'cpu':
        # Use all available CPU cores
        num_threads = min(os.cpu_count() or 4, 8)  # Cap at 8 to avoid overhead
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        if epoch == 0:  # Only print once
            print(f"Using {num_threads} CPU threads for parallelization")
    
    import time
    start_time = time.time()
    for batch_idx, batch in enumerate(pbar):
        try:
            batch_start = time.time()
            # Optimized data transfer - use pin_memory if available
            rgb = batch['image_rgb'].to(device, non_blocking=True)
            ir = batch['image_ir'].to(device, non_blocking=True) if batch['image_ir'] is not None else None
            targets = {
                'boxes': batch['boxes'].to(device, non_blocking=True),
                'labels': batch['labels'].to(device, non_blocking=True)
            }
            
            # Debug: Print data loading time for first few batches
            if batch_idx < 3:
                data_time = time.time() - batch_start
                print(f"Batch {batch_idx}: Data loading took {data_time:.2f}s", flush=True)
            
            # Zero gradients efficiently (only at start of accumulation cycle)
            grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
            if batch_idx % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            # Profile timing for CPU
            if device.type == 'cpu' and batch_idx < 3:
                forward_start = time.time()

            # Forward pass with mixed precision (GPU only)
            if scaler is not None and device.type == 'cuda':
                # Mixed precision training for GPU (1.5-2x speedup)
                with autocast():
                    predictions = model(rgb, ir)
                    box_loss, obj_loss, cls_loss = compute_yolo_loss(
                        predictions,
                        targets,
                        anchors,
                        num_classes=config['model']['num_classes'],
                        iou_type=config['loss']['iou_type']
                    )
                    loss = (
                        config['loss']['box_weight'] * box_loss +
                        config['loss']['obj_weight'] * obj_loss +
                        config['loss']['cls_weight'] * cls_loss
                    )
                
                # Backward pass with mixed precision
                loss_scaled = loss / grad_accum_steps  # Scale for gradient accumulation
                scaler.scale(loss_scaled).backward()
                scaler.unscale_(optimizer)  # Unscale before clipping
                
                # Only update weights every N batches (gradient accumulation)
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # Standard precision for CPU
                predictions = model(rgb, ir)
                
                if device.type == 'cpu' and batch_idx < 3:
                    forward_time = time.time() - forward_start
                    loss_start = time.time()
                
                box_loss, obj_loss, cls_loss = compute_yolo_loss(
                    predictions,
                    targets,
                    anchors,
                    num_classes=config['model']['num_classes'],
                    iou_type=config['loss']['iou_type']
                )
                loss = (
                    config['loss']['box_weight'] * box_loss +
                    config['loss']['obj_weight'] * obj_loss +
                    config['loss']['cls_weight'] * cls_loss
                )
                
                if device.type == 'cpu' and batch_idx < 3:
                    loss_time = time.time() - loss_start
                    backward_start = time.time()
            
                # Backward pass with gradient accumulation
                loss_scaled = loss / grad_accum_steps  # Scale loss for gradient accumulation
                loss_scaled.backward()
                
                if device.type == 'cpu' and batch_idx < 3:
                    backward_time = time.time() - backward_start
                    print(f"Batch {batch_idx}: Forward={forward_time:.2f}s, Loss={loss_time:.2f}s, Backward={backward_time:.2f}s", flush=True)
                
                # Only update weights every N batches (gradient accumulation)
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    optimizer.step()
            
            # Store loss values (extract from original loss, not scaled version)
            # Loss values are extracted before scaling, so they represent the true loss
            if isinstance(loss, torch.Tensor):
                loss_val = float(loss.item())
            else:
                loss_val = float(loss)
            
            if isinstance(box_loss, torch.Tensor):
                box_loss_val = float(box_loss.item())
            else:
                box_loss_val = float(box_loss)
            
            if isinstance(obj_loss, torch.Tensor):
                obj_loss_val = float(obj_loss.item())
            else:
                obj_loss_val = float(obj_loss)
            
            if isinstance(cls_loss, torch.Tensor):
                cls_loss_val = float(cls_loss.item())
            else:
                cls_loss_val = float(cls_loss)
            
            # Clean up
            del loss, box_loss, obj_loss, cls_loss, predictions
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            total_loss += loss_val
            total_box_loss += box_loss_val
            total_obj_loss += obj_loss_val
            total_cls_loss += cls_loss_val
            
            # Debug: Print batch time for first few batches
            if batch_idx < 3:
                batch_time = time.time() - batch_start
                print(f"Batch {batch_idx}: Total time {batch_time:.2f}s", flush=True)
            
            # Force Python to flush output buffer
            if batch_idx % 5 == 0:
                import sys
                sys.stdout.flush()
            
            # Log periodically (reduce frequency for CPU, make non-blocking)
            log_freq = config['logging']['log_freq'] * 2 if device.type == 'cpu' else config['logging']['log_freq']
            if batch_idx % log_freq == 0:
                try:
                    writer.add_scalar('Loss/total', loss_val, epoch * num_batches + batch_idx)
                    writer.add_scalar('Loss/box', box_loss_val, epoch * num_batches + batch_idx)
                    writer.add_scalar('Loss/obj', obj_loss_val, epoch * num_batches + batch_idx)
                    writer.add_scalar('Loss/cls', cls_loss_val, epoch * num_batches + batch_idx)
                    writer.flush()  # Flush to prevent blocking
                except Exception as e:
                    # Don't let logging errors stop training
                    if batch_idx % (log_freq * 10) == 0:  # Only print error occasionally
                        print(f"Warning: TensorBoard logging error: {e}", flush=True)
            
            # Print progress less frequently for speed (with flush to see output immediately)
            print_freq = 10 if device.type == 'cpu' else 50  # More frequent on CPU for debugging
            if batch_idx % print_freq == 0 or batch_idx == num_batches - 1:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{num_batches}: "
                      f"loss={loss_val:.4f}, box={box_loss_val:.4f}, "
                      f"obj={obj_loss_val:.4f}, cls={cls_loss_val:.4f}", flush=True)
            
        except Exception as e:
            print(f"Error in train_one_epoch at batch {batch_idx}: {str(e)}", flush=True)
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
    
    # Detect Colab environment (multiple detection methods)
    is_colab = (
        'COLAB_GPU' in os.environ or 
        'COLAB_TPU' in os.environ or
        'google.colab' in str(os.environ.get('_', '')) or
        os.path.exists('/content') or
        'colab' in str(os.environ.get('HOME', '')).lower()
    )
    
    # Initialize device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Auto-optimize for Colab T4 GPU
    if is_colab and device.type == 'cuda':
        print("🎯 Colab T4 GPU detected! Auto-optimizing settings...")
        # Optimize batch size for T4 (16GB VRAM)
        if config['training']['batch_size'] < 16:
            config['training']['batch_size'] = 16
            print(f"  → Batch size set to {config['training']['batch_size']} for T4 GPU")
        # Disable gradient accumulation on GPU (not needed with larger batch size)
        if config['training'].get('gradient_accumulation_steps', 1) > 1:
            config['training']['gradient_accumulation_steps'] = 1
            print("  → Gradient accumulation disabled (using larger batch size instead)")
        # Optimize num_workers for Colab
        config['data']['num_workers'] = 4
        print(f"  → Num workers set to {config['data']['num_workers']} for Colab")
        # Use full resolution on GPU
        if config['data']['img_size'] < 640:
            config['data']['img_size'] = 640
            print(f"  → Image size set to {config['data']['img_size']} for better accuracy on GPU")
            # Update model config too
            config['model']['img_size'] = 640
            # Update anchors for 640
            config['model']['anchors'] = [
                [[10, 13], [16, 30], [33, 23]],  # Scale 1 (80x80)
                [[30, 61], [62, 45], [59, 119]],  # Scale 2 (40x40)
                [[116, 90], [156, 198], [373, 326]]  # Scale 3 (20x20)
            ]
        # Enable mosaic on GPU (affordable)
        config['augmentation']['mosaic'] = True
        config['augmentation']['mosaic_prob'] = 0.5
        print("  → Mosaic augmentation enabled for GPU")
        print("✅ Colab optimization complete!")
    
    # Initialize GradScaler for mixed precision training (GPU only)
    # Use new API for PyTorch 2.0+, fall back to old API for older versions
    if device.type == "cuda":
        try:
            # Try new API first (PyTorch 2.0+)
            scaler = GradScaler('cuda')
        except (TypeError, ValueError):
            # Fall back to old API for older PyTorch versions
            scaler = GradScaler(enabled=True)
    else:
        scaler = None

    # Initialize model with fast mode for CPU training (disable on GPU for better accuracy)
    fast_mode = (device.type == 'cpu')  # Enable fast mode on CPU, disable on GPU
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
    
    # Optimize num_workers based on platform and device
    import platform
    if platform.system() == 'Darwin':
        num_workers = 0  # macOS doesn't support fork, use 0
    elif is_colab and device.type == 'cuda':
        num_workers = 4  # Colab GPU: use 4 workers
    else:
        num_workers = config['data']['num_workers']
        # Optimize num_workers based on CPU count
        if num_workers > 0:
            cpu_count = os.cpu_count() or 4
            num_workers = min(num_workers, cpu_count, 8)  # Cap at 8 to avoid overhead
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
        prefetch_factor=4 if num_workers > 0 else None,  # Increased prefetch for faster loading
        generator=torch.Generator() if device.type == 'cpu' else None,  # Deterministic for CPU
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
    checkpoint_dir = config['logging']['checkpoint_dir']
    # If checkpoint_dir is absolute (Colab path), use it directly; otherwise make it relative to project root
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(project_root, checkpoint_dir)
    
    # Create directories (handle Colab paths that might not exist yet)
    try:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    except OSError as e:
        if '/content' in checkpoint_dir:
            # If Colab path doesn't exist, fall back to local checkpoint directory
            print(f"Warning: Could not create Colab checkpoint directory: {checkpoint_dir}")
            checkpoint_dir = os.path.join(project_root, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Using local checkpoint directory instead: {checkpoint_dir}")
        else:
            raise e
    writer = SummaryWriter(tensorboard_dir)
    
    # Training loop
    best_map = 0.0
    start_epoch = 0
    last_ckpt = os.path.join(checkpoint_dir, "yolov11_fusion_last.pt")
    if os.path.exists(last_ckpt):
        print("Resuming from last checkpoint")
        checkpoint = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        avg_loss, avg_box_loss, avg_obj_loss, avg_cls_loss = train_one_epoch(
            model, train_loader, optimizer, config, device, epoch, writer, scaler
        )
        print(f"Epoch {epoch+1}/{config['training']['epochs']}: "
              f"Loss={avg_loss:.4f}, Box={avg_box_loss:.4f}, Obj={avg_obj_loss:.4f}, Cls={avg_cls_loss:.4f}")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Evaluate (with no_grad for speed)
        if (epoch + 1) % config['evaluation']['eval_freq'] == 0:
            model.eval()  # Set to eval mode
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
            
            model.train()  # Set back to train mode
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_freq'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': best_map
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'yolov11_fusion_epoch_{epoch+1}.pt'))
    
    # Save final model
    final_checkpoint = {
        'epoch': config['training']['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_map': best_map
    }
    if scaler is not None:
        final_checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(final_checkpoint, os.path.join(checkpoint_dir, 'yolov11_fusion_last.pt'))
    writer.close()

if __name__ == "__main__":
    main()
