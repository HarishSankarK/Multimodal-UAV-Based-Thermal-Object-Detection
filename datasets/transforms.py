import torch
import numpy as np
import cv2
import random

def resize_image(image, target_size):
    """
    Resize image to target size while maintaining aspect ratio and padding.
    Args:
        image (np.ndarray): Input image (H, W, C).
        target_size (tuple): Target size (height, width).
    Returns:
        tuple: (resized_image, scale_factor, offset).
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    padded[top:top+new_h, left:left+new_w] = resized
    
    return padded, scale, (left, top)

def adjust_boxes(boxes, scale_factor, offset, img_size):
    """
    Adjust bounding boxes after resizing and padding.
    Args:
        boxes (np.ndarray): [N, 4] boxes [x_min, y_min, x_max, y_max].
        scale_factor (float): Scaling factor.
        offset (tuple): (x_offset, y_offset).
        img_size (int): Image size for clamping.
    Returns:
        np.ndarray: Adjusted boxes.
    """
    boxes = np.array(boxes, dtype=np.float32)
    if len(boxes) == 0:
        return boxes
    boxes[:, [0, 2]] *= scale_factor
    boxes[:, [1, 3]] *= scale_factor
    boxes[:, [0, 2]] += offset[0]
    boxes[:, [1, 3]] += offset[1]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_size)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_size)
    return boxes

def apply_mosaic(images, images_ir, bboxes, labels, img_size=640):
    """
    Apply mosaic augmentation to a batch of 4 images.
    Args:
        images (list): List of 4 RGB images (numpy arrays, H, W, 3).
        images_ir (list): List of 4 IR images (numpy arrays or None).
        bboxes (list): List of 4 [N, 4] numpy arrays [x_min, y_min, x_max, y_max].
        labels (list): List of 4 [N] numpy arrays.
        img_size (int): Target image size.
    Returns:
        tuple: (mosaic_rgb, mosaic_ir, mosaic_bboxes, mosaic_labels).
    """
    mosaic_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    mosaic_ir = np.zeros((img_size, img_size, 3), dtype=np.uint8) if images_ir[0] is not None else None
    all_bboxes = []
    all_labels = []
    
    # Random split points
    xc = random.randint(int(img_size * 0.3), int(img_size * 0.7))
    yc = random.randint(int(img_size * 0.3), int(img_size * 0.7))
    
    # Define quadrant sizes
    top_left_h, top_left_w = yc, xc
    top_right_h, top_right_w = yc, img_size - xc
    bottom_left_h, bottom_left_w = img_size - yc, xc
    bottom_right_h, bottom_right_w = img_size - yc, img_size - xc
    
    for i, (img, ir, box, lbl) in enumerate(zip(images, images_ir, bboxes, labels)):
        # Resize to match quadrant size
        if i == 0:  # Top-left
            target_size = (top_left_h, top_left_w)
        elif i == 1:  # Top-right
            target_size = (top_right_h, top_right_w)
        elif i == 2:  # Bottom-left
            target_size = (bottom_left_h, bottom_left_w)
        else:  # Bottom-right
            target_size = (bottom_right_h, bottom_right_w)
        
        img, scale_factor, offset = resize_image(img, target_size)
        box = adjust_boxes(box, scale_factor, offset, target_size[1] if i in [0, 2] else target_size[0])
        
        # Handle IR image
        if ir is not None:
            ir, _, _ = resize_image(ir, target_size)
        
        # Place in mosaic with proper slicing
        if i == 0:  # Top-left
            mosaic_rgb[:yc, :xc] = img
            if ir is not None:
                mosaic_ir[:yc, :xc] = ir
            mask = (box[:, 0] < xc) & (box[:, 1] < yc)
            box = box[mask]
            box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, xc)
            box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, yc)
            lbl = lbl[mask]
        elif i == 1:  # Top-right
            mosaic_rgb[:yc, xc:] = img
            if ir is not None:
                mosaic_ir[:yc, xc:] = ir
            box[:, [0, 2]] += xc
            mask = (box[:, 0] >= xc) & (box[:, 1] < yc)
            box = box[mask]
            box[:, [0, 2]] = np.clip(box[:, [0, 2]], xc, img_size)
            box[:, [1, 3]] = np.clip(box[:, [1, 3]], 0, yc)
            lbl = lbl[mask]
        elif i == 2:  # Bottom-left
            mosaic_rgb[yc:, :xc] = img
            if ir is not None:
                mosaic_ir[yc:, :xc] = ir
            box[:, [1, 3]] += yc
            mask = (box[:, 0] < xc) & (box[:, 1] >= yc)
            box = box[mask]
            box[:, [0, 2]] = np.clip(box[:, [0, 2]], 0, xc)
            box[:, [1, 3]] = np.clip(box[:, [1, 3]], yc, img_size)
            lbl = lbl[mask]
        else:  # Bottom-right
            mosaic_rgb[yc:, xc:] = img
            if ir is not None:
                mosaic_ir[yc:, xc:] = ir
            box[:, [0, 2]] += xc
            box[:, [1, 3]] += yc
            mask = (box[:, 0] >= xc) & (box[:, 1] >= yc)
            box = box[mask]
            box[:, [0, 2]] = np.clip(box[:, [0, 2]], xc, img_size)
            box[:, [1, 3]] = np.clip(box[:, [1, 3]], yc, img_size)
            lbl = lbl[mask]
        
        if len(box) > 0:
            all_bboxes.append(box)
            all_labels.append(lbl)
    
    if all_bboxes:
        all_bboxes = np.concatenate(all_bboxes, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_bboxes = np.zeros((0, 4), dtype=np.float32)
        all_labels = np.zeros((0,), dtype=np.int64)
    
    return mosaic_rgb, mosaic_ir, all_bboxes, all_labels

class TransformFunction:
    """Pickleable transform class for multiprocessing compatibility."""
    def __init__(self, config, training=True):
        """
        Args:
            config (dict): Configuration dictionary with 'data' key containing 'img_size'.
            training (bool): Whether to apply training augmentations.
        """
        self.img_size = config['data']['img_size']
        self.training = training
    
    def __call__(self, image=None, image_ir=None, bboxes=None, labels=None):
        image_rgb = np.array(image, dtype=np.uint8)
        image_ir = np.array(image_ir, dtype=np.uint8) if image_ir is not None else None
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Resize and normalize
        image_rgb, scale_factor, offset = resize_image(image_rgb, (self.img_size, self.img_size))
        bboxes = adjust_boxes(bboxes, scale_factor, offset, self.img_size)
        image_rgb = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        
        if image_ir is not None:
            image_ir, _, _ = resize_image(image_ir, (self.img_size, self.img_size))
            image_ir = torch.from_numpy(image_ir).permute(2, 0, 1).float() / 255.0
        
        if self.training:
            # Apply random horizontal flip
            if random.random() < 0.5:
                image_rgb = torch.flip(image_rgb, dims=[2])
                if image_ir is not None:
                    image_ir = torch.flip(image_ir, dims=[2])
                if len(bboxes) > 0:
                    bboxes[:, [0, 2]] = self.img_size - bboxes[:, [2, 0]]
            
            # Apply color jitter (simple brightness/contrast)
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                image_rgb *= factor
                image_rgb = torch.clamp(image_rgb, 0, 1)
                if image_ir is not None:
                    image_ir *= factor
                    image_ir = torch.clamp(image_ir, 0, 1)
            
            # Apply random rotation (up to 15 degrees)
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                h, w = image_rgb.shape[1:]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                image_rgb = torch.from_numpy(cv2.warpAffine(image_rgb.permute(1, 2, 0).numpy(), M, (w, h))/255.0).permute(2, 0, 1)
                if image_ir is not None:
                    image_ir = torch.from_numpy(cv2.warpAffine(image_ir.permute(1, 2, 0).numpy(), M, (w, h))/255.0).permute(2, 0, 1)
                if len(bboxes) > 0:
                    bboxes = rotate_boxes(bboxes, angle, w/2, h/2)
        
        return {
            'image': image_rgb,
            'image_ir': image_ir,
            'bboxes': torch.from_numpy(bboxes).float() if len(bboxes) > 0 else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.from_numpy(labels).long() if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        }

def get_transforms(config, training=True):
    """
    Get transforms based on configuration.
    Args:
        config (dict): Configuration dictionary.
        training (bool): Whether to apply training augmentations.
    Returns:
        callable: Transform function compatible with albumentations-style calls.
    """
    return TransformFunction(config, training=training)

def rotate_boxes(boxes, angle, cx, cy):
    """
    Rotate bounding boxes by angle around center (cx, cy).
    Args:
        boxes (np.ndarray): [N, 4] boxes [x_min, y_min, x_max, y_max].
        angle (float): Rotation angle in degrees.
        cx, cy (float): Center of rotation.
    Returns:
        np.ndarray: Rotated boxes.
    """
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    corners = np.stack([
        boxes[:, [0, 1]],  # Top-left
        boxes[:, [2, 1]],  # Top-right
        boxes[:, [0, 3]],  # Bottom-left
        boxes[:, [2, 3]]   # Bottom-right
    ], axis=1)  # [N, 4, 2]
    
    # Translate to origin
    corners[:, :, 0] -= cx
    corners[:, :, 1] -= cy
    
    # Rotate
    new_corners = np.zeros_like(corners)
    new_corners[:, :, 0] = corners[:, :, 0] * cos_a + corners[:, :, 1] * sin_a
    new_corners[:, :, 1] = -corners[:, :, 0] * sin_a + corners[:, :, 1] * cos_a
    
    # Translate back
    new_corners[:, :, 0] += cx
    new_corners[:, :, 1] += cy
    
    # Get new bounding box
    x_min = new_corners[:, :, 0].min(axis=1)
    y_min = new_corners[:, :, 1].min(axis=1)
    x_max = new_corners[:, :, 0].max(axis=1)
    y_max = new_corners[:, :, 1].max(axis=1)
    
    return np.stack([x_min, y_min, x_max, y_max], axis=1)