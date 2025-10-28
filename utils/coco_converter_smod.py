import os
import json
import glob
import random
import shutil
import cv2

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def remove_occlusion(annotations):
    for ann in annotations:
        if 'occlusion' in ann:
            del ann['occlusion']
    return annotations

def convert_split_to_coco(images, annotations, input_base_path, output_rgb_dir, output_ir_dir, output_json_path, dataset_split, categories):
    """Process a split, copy images, update file_names, remove occlusion."""
    coco_format = {
        "info": {"description": f"SMOD {dataset_split} dataset in COCO format"},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id_map = {}  # Old id to new id
    new_image_id = 0
    new_annotation_id = 0

    for img in images:
        old_image_id = img['id']
        file_name = img['file_name']  # e.g., "day/000000_rgb.jpg"
        base = file_name.split('/')[-1].replace('_rgb.jpg', '')  # e.g., "000000"
        shared_filename = base + '.jpg'

        rgb_path = os.path.join(input_base_path, file_name)
        ir_path = os.path.join(input_base_path, 'day', base + '_tir.jpg')

        # Verify both images exist
        if not os.path.exists(rgb_path):
            print(f"Skipping image {file_name}: RGB not found at {rgb_path}")
            continue
        if not os.path.exists(ir_path):
            print(f"Skipping image {file_name}: IR not found at {ir_path}")
            continue

        # Copy images
        output_rgb_path = os.path.join(output_rgb_dir, shared_filename)
        output_ir_path = os.path.join(output_ir_dir, shared_filename)
        shutil.copy(rgb_path, output_rgb_path)
        shutil.copy(ir_path, output_ir_path)

        # Update image info
        img['file_name'] = shared_filename
        img['id'] = new_image_id
        coco_format['images'].append(img)

        image_id_map[old_image_id] = new_image_id
        new_image_id += 1

    # Filter and update annotations
    annotations = remove_occlusion(annotations)
    for ann in annotations:
        old_image_id = ann['image_id']
        if old_image_id in image_id_map:
            ann['image_id'] = image_id_map[old_image_id]
            ann['id'] = new_annotation_id
            coco_format['annotations'].append(ann)
            new_annotation_id += 1

    # Save COCO JSON
    save_json(coco_format, output_json_path)
    print(f"Converted {dataset_split} split to {output_json_path} with {len(coco_format['images'])} images")

def split_train_to_train_val(train_data, split_ratio=0.8):
    """Split train into train and val."""
    images = train_data['images']
    annotations = train_data['annotations']

    # Shuffle images
    random.shuffle(images)
    num_train = int(len(images) * split_ratio)
    train_images = images[:num_train]
    val_images = images[num_train:]

    # Filter annotations for train
    train_image_ids = {img['id'] for img in train_images}
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]

    # Filter annotations for val
    val_image_ids = {img['id'] for img in val_images}
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]

    train_split = {
        'info': train_data['info'],
        'licenses': train_data['licenses'],
        'categories': train_data['categories'],
        'images': train_images,
        'annotations': train_annotations
    }

    val_split = {
        'info': train_data['info'],
        'licenses': train_data['licenses'],
        'categories': train_data['categories'],
        'images': val_images,
        'annotations': val_annotations
    }

    return train_split, val_split

def main():
    # Paths
    input_base_path = r"D:\User\Videos\CE\project\Reference papers\archive"
    output_base_path = r"D:\User\Videos\CE\project\Reference papers\multimodal_yolov11\data\smod"
    
    # Load JSONs
    train_json_path = os.path.join(input_base_path, 'anno', 'new_train_annotations_rgb.json')
    test_json_path = os.path.join(input_base_path, 'anno', 'new_test_annotations_rgb.json')
    
    train_data = load_json(train_json_path)
    test_data = load_json(test_json_path)

    # Split train into train and val
    train_split, val_split = split_train_to_train_val(train_data)

    # Ensure output directories exist
    os.makedirs(os.path.join(output_base_path, 'annotations'), exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base_path, 'images', split, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(output_base_path, 'images', split, 'ir'), exist_ok=True)

    # Categories (same for all)
    categories = train_data['categories']

    # Process train
    output_rgb_dir = os.path.join(output_base_path, 'images', 'train', 'rgb')
    output_ir_dir = os.path.join(output_base_path, 'images', 'train', 'ir')
    output_json = os.path.join(output_base_path, 'annotations', 'instances_train.json')
    convert_split_to_coco(train_split['images'], train_split['annotations'], input_base_path, output_rgb_dir, output_ir_dir, output_json, 'train', categories)

    # Process val
    output_rgb_dir = os.path.join(output_base_path, 'images', 'val', 'rgb')
    output_ir_dir = os.path.join(output_base_path, 'images', 'val', 'ir')
    output_json = os.path.join(output_base_path, 'annotations', 'instances_val.json')
    convert_split_to_coco(val_split['images'], val_split['annotations'], input_base_path, output_rgb_dir, output_ir_dir, output_json, 'val', categories)

    # Process test
    output_rgb_dir = os.path.join(output_base_path, 'images', 'test', 'rgb')
    output_ir_dir = os.path.join(output_base_path, 'images', 'test', 'ir')
    output_json = os.path.join(output_base_path, 'annotations', 'instances_test.json')
    convert_split_to_coco(test_data['images'], test_data['annotations'], input_base_path, output_rgb_dir, output_ir_dir, output_json, 'test', categories)

if __name__ == "__main__":
    main()