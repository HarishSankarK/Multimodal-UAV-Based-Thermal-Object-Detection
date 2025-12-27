import os
import json
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO

class MultimodalDataset(torch.utils.data.Dataset):
    """Dataset class for loading multimodal (RGB + IR) images and COCO annotations."""
    
    def __init__(self, root_dir, dataset_name, split, transforms=None, img_size=640):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'data/').
            dataset_name (str): Name of the dataset ('smod', 'hit_uav', 'dronergbt').
            split (str): Dataset split ('train', 'val', 'test').
            transforms: Transform function from transforms.py.
            img_size (int): Target image size for resizing.
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transforms = transforms
        self.img_size = img_size

        # Load modalities.json
        modalities_path = os.path.join(root_dir, 'modalities.json')
        with open(modalities_path, 'r') as f:
            self.modalities = json.load(f)

        # Define unified category mapping
        self.category_map = {
            'person': 0,
            'rider': 1,
            'bicycle': 2,
            'car': 3
        }

        # Load COCO annotations
        annotation_path = os.path.join(
            root_dir, dataset_name, 'annotations', f'instances_{split}.json'
        )
        self.coco = COCO(annotation_path)
        self.image_ids = self.coco.getImgIds()
        
        # Map dataset-specific category IDs to unified IDs
        self.dataset_category_map = {
            cat['id']: self.category_map[cat['name']]
            for cat in self.coco.loadCats(self.coco.getCatIds())
            if cat['name'] in self.category_map
        }

        # Image paths - use modalities.json to determine correct paths
        rgb_path = self.modalities[dataset_name].get(split, {}).get('rgb', '')
        if rgb_path:
            # Remove dataset_name prefix if present and construct full path
            if rgb_path.startswith(dataset_name + '/'):
                self.rgb_dir = os.path.join(root_dir, rgb_path.rstrip('/'))
            else:
                self.rgb_dir = os.path.join(root_dir, dataset_name, rgb_path.rstrip('/'))
        else:
            # Fallback to default structure
            self.rgb_dir = os.path.join(root_dir, dataset_name, 'images', split, 'rgb')
        
        # IR directory
        ir_path = self.modalities[dataset_name].get(split, {}).get('ir', '')
        if ir_path:
            if ir_path.startswith(dataset_name + '/'):
                self.ir_dir = os.path.join(root_dir, ir_path.rstrip('/'))
            else:
                self.ir_dir = os.path.join(root_dir, dataset_name, ir_path.rstrip('/'))
        else:
            self.ir_dir = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """Load RGB, IR (if available), and annotations for an image."""
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, img_info['file_name'])
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print(f"Warning: Failed to load RGB image {rgb_path}")
            rgb_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Load IR image (if available)
        ir_img = None
        if self.ir_dir:
            ir_path = os.path.join(self.ir_dir, img_info['file_name'])
            ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if ir_img is None:
                print(f"Warning: Failed to load IR image {ir_path}")
                ir_img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        for ann in anns:
            if ann['category_id'] in self.dataset_category_map:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
                labels.append(self.dataset_category_map[ann['category_id']])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # Apply transforms
        if self.transforms:
            data = {
                'image': rgb_img,
                'image_ir': ir_img,
                'bboxes': boxes.astype(np.float32),
                'labels': labels.astype(np.float32)
            }
            augmented = self.transforms(**data)
            rgb_img = augmented['image']
            ir_img = augmented['image_ir']
            boxes = augmented['bboxes']
            labels = augmented['labels']
        else:
            # Convert to tensors if no transforms
            rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
            ir_img = torch.from_numpy(ir_img).permute(2, 0, 1).float() / 255.0 if ir_img is not None else None
            boxes = torch.from_numpy(boxes).float()
            labels = torch.from_numpy(labels).long()

        return {
            'image_rgb': rgb_img,
            'image_ir': ir_img,
            'boxes': boxes,
            'labels': labels,
            'image_ids': img_id
        }