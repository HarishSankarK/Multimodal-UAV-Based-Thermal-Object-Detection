import torch
import random
import numpy as np
from datasets.transforms import apply_mosaic

def collate_fn(batch, mosaic_prob=0.5, img_size=640):
    """
    Custom collate function for multimodal dataset.
    Args:
        batch: List of samples from MultimodalDataset.
        mosaic_prob (float): Probability of applying mosaic augmentation.
        img_size (int): Target image size.
    Returns:
        dict: Batched data with padded annotations.
    """
    # Separate batch into components
    rgb_images = []
    ir_images = []
    boxes_list = []
    labels_list = []
    image_ids = []

    for sample in batch:
        rgb_images.append(sample['image_rgb'])
        ir_images.append(sample['image_ir'])
        boxes_list.append(sample['boxes'])
        labels_list.append(sample['labels'])
        image_ids.append(sample['image_ids'])  # Matches MultimodalDataset

    # Apply mosaic augmentation with probability
    if random.random() < mosaic_prob and len(batch) >= 4:
        # Select 4 samples for mosaic
        indices = random.sample(range(len(batch)), 4)
        mosaic_rgb, mosaic_ir, mosaic_boxes, mosaic_labels = apply_mosaic(
            [(rgb_images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for i in indices],
            [ir_images[i].permute(1, 2, 0).numpy() * 255 if ir_images[i] is not None else None for i in indices],
            [boxes_list[i].numpy() for i in indices],
            [labels_list[i].numpy() for i in indices],
            img_size=img_size
        )
        # Replace one sample with mosaic
        rgb_images[0] = torch.from_numpy(mosaic_rgb).permute(2, 0, 1).float()
        ir_images[0] = torch.from_numpy(mosaic_ir).permute(2, 0, 1).float() if mosaic_ir is not None else None
        boxes_list[0] = torch.from_numpy(mosaic_boxes).float()
        labels_list[0] = torch.from_numpy(mosaic_labels).long()
        image_ids[0] = -1  # Mosaic image doesn't correspond to a single ID

    # Stack RGB images
    rgb_images = torch.stack(rgb_images, dim=0)

    # Stack IR images if present (handle None values by creating zero tensors)
    if any(img is not None for img in ir_images):
        # Replace None with zero tensors matching the shape of non-None images
        ir_shape = None
        for img in ir_images:
            if img is not None:
                ir_shape = img.shape
                break
        if ir_shape is not None:
            ir_images = [img if img is not None else torch.zeros(ir_shape, dtype=rgb_images.dtype) for img in ir_images]
            ir_images = torch.stack(ir_images, dim=0)
        else:
            ir_images = None
    else:
        ir_images = None

    # Pad boxes and labels
    max_num_boxes = max(len(boxes) for boxes in boxes_list)
    padded_boxes = []
    padded_labels = []
    for boxes, labels in zip(boxes_list, labels_list):
        num_boxes = len(boxes)
        if num_boxes > 0:
            padded_boxes.append(torch.cat([
                boxes,
                torch.zeros((max_num_boxes - num_boxes, 4), dtype=torch.float32)
            ], dim=0))
            padded_labels.append(torch.cat([
                labels,
                torch.zeros(max_num_boxes - num_boxes, dtype=torch.int64)
            ], dim=0))
        else:
            padded_boxes.append(torch.zeros((max_num_boxes, 4), dtype=torch.float32))
            padded_labels.append(torch.zeros(max_num_boxes, dtype=torch.int64))

    padded_boxes = torch.stack(padded_boxes, dim=0)
    padded_labels = torch.stack(padded_labels, dim=0)

    # Create mask for valid boxes
    box_mask = torch.zeros((len(batch), max_num_boxes), dtype=torch.bool)
    for i, boxes in enumerate(boxes_list):
        box_mask[i, :len(boxes)] = 1

    return {
        'image_rgb': rgb_images,
        'image_ir': ir_images,
        'boxes': padded_boxes,
        'labels': padded_labels,
        'box_mask': box_mask,
        'image_ids': torch.tensor(image_ids, dtype=torch.int64)
    }