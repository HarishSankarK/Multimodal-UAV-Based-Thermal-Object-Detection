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

def evaluate_coco(model, dataloader, annotation_file, anchors, device='cuda'):
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
        
        detections = decode_predictions(predictions, anchors, conf_thres=0.5, iou_thres=0.5)
        
        for img_id, dets in zip(image_ids, detections):
            for det in dets:
                x_min, y_min, x_max, y_max, conf, cls = det.cpu().numpy()
                coco_dt.append({
                    'image_id': int(img_id),
                    'category_id': int(cls) + 1,  # Assuming 1-based COCO category IDs
                    'bbox': [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                    'score': float(conf)
                })
    
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

    root_dir = r"D:\User\Videos\CE\project\Reference papers\multimodal_yolov11\data"
    annotation_file = os.path.join(root_dir, "smod", "annotations", "instances_val.json")
    anchors = [np.array([[10, 13], [16, 30], [33, 23]]),
               np.array([[30, 61], [62, 45], [59, 119]]),
               np.array([[116, 90], [156, 198], [373, 326]])]
    
    dataset = MultimodalDataset(root_dir, "smod", "val", transforms=None)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = FusionYOLOv11(num_classes=4)
    
    metrics = evaluate_coco(model, dataloader, annotation_file, anchors)
    print("Evaluation metrics:", metrics)