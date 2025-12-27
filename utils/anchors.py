import numpy as np
import json
import os
from sklearn.cluster import KMeans

def wh_iou(wh1, wh2):
    """Compute IoU between two sets of width-height pairs."""
    wh1 = wh1[:, None]  # [N, 1, 2]
    wh2 = wh2[None, :]  # [1, M, 2]
    inter = np.minimum(wh1, wh2).prod(2)  # [N, M]
    union = (wh1.prod(2) + wh2.prod(2) - inter)  # [N, M]
    return inter / (union + 1e-6)

def generate_anchors(annotation_files, num_anchors=9, img_size=640):
    """
    Generate anchor boxes using k-means clustering.
    Args:
        annotation_files (list): List of paths to COCO annotation JSON files.
        num_anchors (int): Total number of anchors (default: 9 for 3 per scale).
        img_size (int): Image size for normalization (default: 640).
    Returns:
        np.ndarray: Array of shape [num_anchors, 2] with anchor (width, height).
    """
    # Collect all bounding box sizes
    boxes = []
    for ann_file in annotation_files:
        with open(ann_file, 'r') as f:
            data = json.load(f)
        for ann in data['annotations']:
            w, h = ann['bbox'][2], ann['bbox'][3]
            boxes.append([w, h])
    
    boxes = np.array(boxes)
    if len(boxes) == 0:
        raise ValueError("No bounding boxes found in annotations.")
    
    # Normalize box sizes to image size
    boxes = boxes / img_size
    
    # K-means clustering
    kmeans = KMeans(n_clusters=num_anchors, random_state=0)
    kmeans.fit(boxes)
    anchors = kmeans.cluster_centers_
    
    # Scale back to image size
    anchors = anchors * img_size
    anchors = anchors.clip(min=1.0)  # Ensure anchors are positive
    
    # Sort anchors by area for assignment to scales (small to large)
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    
    return anchors

def assign_anchors_to_scales(anchors, num_scales=3):
    """
    Assign anchors to detection scales (e.g., 80x80, 40x40, 20x20).
    Args:
        anchors (np.ndarray): Array of shape [num_anchors, 2].
        num_scales (int): Number of detection scales (default: 3).
    Returns:
        list: List of [num_anchors_per_scale, 2] arrays for each scale.
    """
    num_anchors_per_scale = len(anchors) // num_scales
    return [anchors[i * num_anchors_per_scale:(i + 1) * num_anchors_per_scale] for i in range(num_scales)]

if __name__ == "__main__":
    # Example usage
    root_dir = "data"  # Relative path to data directory
    annotation_files = [
        os.path.join(root_dir, "smod", "annotations", "instances_train.json"),
        os.path.join(root_dir, "hit_uav", "annotations", "instances_train.json"),
        os.path.join(root_dir, "dronergbt", "annotations", "instances_train.json")
    ]
    anchors = generate_anchors(annotation_files, num_anchors=9)
    print("Generated anchors:", anchors)
    anchors_per_scale = assign_anchors_to_scales(anchors)
    for i, scale_anchors in enumerate(anchors_per_scale):
        print(f"Scale {i} anchors:", scale_anchors)