import os
import json
import glob
from pathlib import Path
import cv2
import shutil

def get_image_info(image_path, image_id):
    """Extract image metadata for COCO format."""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    file_name = os.path.basename(image_path)
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

def yolo_to_coco_bbox(bbox, img_width, img_height):
    """Convert YOLO bbox (normalized center-x, center-y, width, height) to COCO bbox (x_min, y_min, width, height)."""
    center_x, center_y, width, height = bbox
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height
    x_min = center_x - (width / 2)
    y_min = center_y - (height / 2)
    return [x_min, y_min, width, height]

def convert_split_to_coco(images_dir, labels_dir, output_images_dir, output_json_path, dataset_split, categories):
    """Convert a dataset split (train/val/test) to COCO format and copy images."""
    coco_format = {
        "info": {"description": f"HIT-UAV {dataset_split} dataset in COCO format"},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 0
    annotation_id = 0
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png"))

    for image_path in image_paths:
        # Get corresponding label file
        label_file = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        # Copy image to output directory
        output_image_path = os.path.join(output_images_dir, os.path.basename(image_path))
        shutil.copy(image_path, output_image_path)

        # Add image info
        coco_format["images"].append(get_image_info(image_path, image_id))

        # Read and process annotations
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                if class_id >= len(categories):  # Skip invalid class IDs
                    continue
                bbox = [float(x) for x in parts[1:5]]
                coco_bbox = yolo_to_coco_bbox(bbox, img_width, img_height)
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1

        image_id += 1

    # Save COCO JSON
    with open(output_json_path, "w") as f:
        json.dump(coco_format, f, indent=2)

def main():
    # Paths
    input_base_path = r"D:\User\Videos\CE\project\hit-uav"
    output_base_path = r"D:\User\Videos\CE\project\Reference papers\multimodal_yolov11\data\hit_uav"
    
    # Ensure output directories exist
    os.makedirs(os.path.join(output_base_path, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "images", "test"), exist_ok=True)

    # Categories for HIT-UAV
    categories = [
        {"id": 0, "name": "Person"},
        {"id": 1, "name": "Car"},
        {"id": 2, "name": "Bicycle"},
        {"id": 3, "name": "OtherVehicle"},
        {"id": 4, "name": "DontCare"}
    ]

    # Process each split
    splits = ["train", "val", "test"]
    for split in splits:
        images_dir = os.path.join(input_base_path, "images", split)
        labels_dir = os.path.join(input_base_path, "labels", split)
        output_images_dir = os.path.join(output_base_path, "images", split)
        output_json = os.path.join(output_base_path, "annotations", f"instances_{split}.json")
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            convert_split_to_coco(images_dir, labels_dir, output_images_dir, output_json, split, categories)
            print(f"Converted {split} split to {output_json} and copied images to {output_images_dir}")
        else:
            print(f"Warning: {split} split directories not found")

if __name__ == "__main__":
    main()