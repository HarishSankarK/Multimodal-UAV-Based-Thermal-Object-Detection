import torch
import numpy as np
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.visualize import decode_predictions

def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    Args:
        boxes1 (torch.Tensor): [N, 4] boxes [x_min, y_min, x_max, y_max].
        boxes2 (torch.Tensor): [M, 4] boxes [x_min, y_min, x_max, y_max].
    Returns:
        torch.Tensor: IoU matrix [N, M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

def evaluate_coco(model, dataloader, annotation_file, anchors, img_size, conf_thres, iou_thres, device='cuda'):
    """
    Evaluate model on a dataset using COCO mAP metrics.
    Args:
        model: YOLOv11 model.
        dataloader: DataLoader with MultimodalDataset.
        annotation_file (str): Path to COCO annotation JSON.
        anchors (list): Anchor boxes per scale.
        device (str): Device to run evaluation on.
    Returns:
        dict: COCO evaluation metrics (e.g., mAP@0.5, mAP@0.5:0.95).
    """
    model.eval()
    model.to(device)
    coco = COCO(annotation_file)
    coco_dt = []
    
    for batch in dataloader:
        rgb = batch['image_rgb'].to(device)
        ir = batch['image_ir'].to(device) if batch['image_ir'] is not None else None
        image_ids = batch['image_ids'].tolist()
        
        with torch.no_grad():
            predictions = model(rgb, ir)
        
        detections = decode_predictions(
          predictions,
          anchors,
          strides=[8,16,32],
          img_size=img_size,
          conf_thres=conf_thres,
          iou_thres=iou_thres
        )

        
        for img_id, dets in zip(image_ids, detections):
            if dets.shape[0] > 0:  # Check if there are any detections
                dets_np = dets.cpu().numpy()
                for det in dets_np:
                    x_min, y_min, x_max, y_max, conf, cls = det
                    coco_dt.append({
                        'image_id': int(img_id),
                        'category_id': int(cls) + 1,  # Assuming 1-based COCO category IDs
                        'bbox': [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                        'score': float(conf)
                    })
    # 🚨 SAFETY CHECK: no detections
    if len(coco_dt) == 0:
      print("⚠️ No detections produced for this epoch. Skipping COCO evaluation.")
      return {
        'mAP@0.5': 0.0,
        'mAP@0.5:0.95': 0.0
      }

    # Load detections into COCO format
    coco_dt = coco.loadRes(coco_dt)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    metrics = {
        'mAP@0.5': coco_eval.stats[1],
        'mAP@0.5:0.95': coco_eval.stats[0]
    }
    return metrics

if __name__ == "__main__":
    # Example usage
    from models.fusion_yolov11 import FusionYOLOv11
    from datasets.multimodal_dataset import MultimodalDataset
    from datasets.collate import collate_fn
    from torch.utils.data import DataLoader

    root_dir = "data"  # Relative path to data directory
    annotation_file = os.path.join(root_dir, "smod", "annotations", "instances_val.json")
    anchors = [np.array([[10, 13], [16, 30], [33, 23]]),
               np.array([[30, 61], [62, 45], [59, 119]]),
               np.array([[116, 90], [156, 198], [373, 326]])]
    
    dataset = MultimodalDataset(root_dir, "smod", "val", transforms=None)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = FusionYOLOv11(num_classes=4)
    
    metrics = evaluate_coco(model, dataloader, annotation_file, anchors)
    print("Evaluation metrics:", metrics)