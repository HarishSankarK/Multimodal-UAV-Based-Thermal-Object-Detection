import os
import json
import glob
import random
import shutil
import cv2
import xml.etree.ElementTree as ET

def get_image_info(image_path, image_id, file_name):
    """Extract image metadata for COCO format."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    height, width = img.shape[:2]
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

def point_to_bbox(x, y, img_width, img_height, box_size=10.0):
    """Convert point to small COCO bbox (x_min, y_min, width, height)."""
    x_min = max(0, x - box_size / 2)
    y_min = max(0, y - box_size / 2)
    width = box_size
    height = box_size
    # Clip to image boundaries
    if x_min + width > img_width:
        width = img_width - x_min
    if y_min + height > img_height:
        height = img_height - y_min
    return [x_min, y_min, width, height]

def convert_split_to_coco(xml_list, input_rgb_dir, input_ir_dir, output_rgb_dir, output_ir_dir, output_json_path, dataset_split, categories):
    """Convert a dataset split (train/val/test) to COCO format, copy images, and handle point-to-bbox conversion."""
    coco_format = {
        "info": {"description": f"DroneRGBT {dataset_split} dataset in COCO format"},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 0
    annotation_id = 0

    for xml_path in xml_list:
        try:
            # Derive filenames from XML file name (e.g., '1R.xml' -> '1.jpg' for RGB, '1R.jpg' for IR)
            base_name = os.path.splitext(os.path.basename(xml_path))[0]  # e.g., '1R'
            rgb_filename = base_name.replace("R", "") + ".jpg"  # e.g., '1.jpg'
            ir_filename = base_name + ".jpg"  # e.g., '1R.jpg'
            rgb_path = os.path.join(input_rgb_dir, rgb_filename)
            ir_path = os.path.join(input_ir_dir, ir_filename)

            # Verify both images exist
            if not os.path.exists(rgb_path):
                print(f"Skipping {xml_path}: RGB image {rgb_path} not found")
                continue
            if not os.path.exists(ir_path):
                print(f"Skipping {xml_path}: IR image {ir_path} not found")
                continue

            # Copy images to output, using same filename for both modalities (e.g., '1.jpg')
            shared_filename = rgb_filename  # Use RGB filename (e.g., '1.jpg')
            output_rgb_path = os.path.join(output_rgb_dir, shared_filename)
            output_ir_path = os.path.join(output_ir_dir, shared_filename)
            shutil.copy(rgb_path, output_rgb_path)
            shutil.copy(ir_path, output_ir_path)

            # Get image dimensions from RGB (assume same for IR)
            img_info = get_image_info(rgb_path, image_id, shared_filename)
            if img_info is None:
                print(f"Skipping {xml_path}: Failed to read RGB image {rgb_path}")
                continue
            img_height, img_width = img_info["height"], img_info["width"]
            coco_format["images"].append(img_info)

            # Parse XML for annotations
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                point = obj.find('point')
                if point is None:
                    print(f"Skipping object in {xml_path}: No point tag found")
                    continue
                x_elem = point.find('x')
                y_elem = point.find('y')
                if x_elem is None or y_elem is None:
                    print(f"Skipping object in {xml_path}: Missing x or y in point tag")
                    continue
                try:
                    x = float(x_elem.text)
                    y = float(y_elem.text)
                except (TypeError, ValueError):
                    print(f"Skipping object in {xml_path}: Invalid x or y coordinates")
                    continue

                coco_bbox = point_to_bbox(x, y, img_width, img_height)
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,  # person
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1

            image_id += 1

        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            continue

    # Save COCO JSON
    with open(output_json_path, "w") as f:
        json.dump(coco_format, f, indent=2)
    print(f"Converted {dataset_split} split to {output_json_path} with {len(coco_format['images'])} images")

def main():
    # Paths
    input_base_path = r"D:\User\Videos\CE\project\Reference papers\DroneRGBT\DroneRGBT"
    output_base_path = r"D:\User\Videos\CE\project\Reference papers\multimodal_yolov11\data\dronergbt"
    
    # Ensure output directories exist
    os.makedirs(os.path.join(output_base_path, "annotations"), exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_base_path, "images", split, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(output_base_path, "images", split, "ir"), exist_ok=True)

    # Categories for DroneRGBT (crowd counting: person)
    categories = [
        {"id": 0, "name": "person"}
    ]

    # Process train and val from original Train
    train_dir = os.path.join(input_base_path, "Train")
    train_gt_dir = os.path.join(train_dir, "GT_")
    train_rgb_dir = os.path.join(train_dir, "RGB")
    train_ir_dir = os.path.join(train_dir, "Infrared")

    xml_paths = glob.glob(os.path.join(train_gt_dir, "*.xml"))
    if not xml_paths:
        print("No XML files found in Train/GT_")
        return

    random.shuffle(xml_paths)
    num_total = len(xml_paths)
    num_train = int(num_total * 0.8)
    train_xml = xml_paths[:num_train]
    val_xml = xml_paths[num_train:]

    # Train split
    output_rgb_dir = os.path.join(output_base_path, "images", "train", "rgb")
    output_ir_dir = os.path.join(output_base_path, "images", "train", "ir")
    output_json = os.path.join(output_base_path, "annotations", "instances_train.json")
    convert_split_to_coco(train_xml, train_rgb_dir, train_ir_dir, output_rgb_dir, output_ir_dir, output_json, "train", categories)

    # Val split
    output_rgb_dir = os.path.join(output_base_path, "images", "val", "rgb")
    output_ir_dir = os.path.join(output_base_path, "images", "val", "ir")
    output_json = os.path.join(output_base_path, "annotations", "instances_val.json")
    convert_split_to_coco(val_xml, train_rgb_dir, train_ir_dir, output_rgb_dir, output_ir_dir, output_json, "val", categories)

    # Process test from original Test
    test_dir = os.path.join(input_base_path, "Test")
    test_gt_dir = os.path.join(test_dir, "GT_")
    test_rgb_dir = os.path.join(test_dir, "RGB")
    test_ir_dir = os.path.join(test_dir, "Infrared")

    test_xml = glob.glob(os.path.join(test_gt_dir, "*.xml"))
    if not test_xml:
        print("No XML files found in Test/GT_")
    else:
        output_rgb_dir = os.path.join(output_base_path, "images", "test", "rgb")
        output_ir_dir = os.path.join(output_base_path, "images", "test", "ir")
        output_json = os.path.join(output_base_path, "annotations", "instances_test.json")
        convert_split_to_coco(test_xml, test_rgb_dir, test_ir_dir, output_rgb_dir, output_ir_dir, output_json, "test", categories)

if __name__ == "__main__":
    main()