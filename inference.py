import torch
import yaml
import os
import cv2
import numpy as np
from models.fusion_yolov11 import FusionYOLOv11
from utils.visualize import decode_predictions, draw_boxes

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
    
    # Initialize device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = FusionYOLOv11(num_classes=config['model']['num_classes'], img_size=config['data']['img_size'])
    model.to(device)
    
    # Resolve checkpoint path
    checkpoint_dir = os.path.join(project_root, config['logging']['checkpoint_dir']) if not os.path.isabs(config['logging']['checkpoint_dir']) else config['logging']['checkpoint_dir']
    checkpoint_path = os.path.join(checkpoint_dir, 'yolov11_fusion_best.pt')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load test image
    test_rgb_path = 'path/to/test_rgb.jpg'  # Replace with actual path
    test_ir_path = 'path/to/test_ir.jpg'  # Replace with actual path or None if no IR
    rgb_img = cv2.imread(test_rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
    rgb_img = rgb_img.unsqueeze(0).to(device)
    
    ir_img = None
    if test_ir_path:
        ir_img = cv2.imread(test_ir_path, cv2.IMREAD_GRAYSCALE)
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)
        ir_img = torch.from_numpy(ir_img).permute(2, 0, 1).float() / 255.0
        ir_img = ir_img.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        predictions = model(rgb_img, ir_img)
    
    # Decode predictions
    detections = decode_predictions(predictions, config['model']['anchors'], conf_thres=0.5, iou_thres=0.5)
    
    # Draw boxes on RGB image
    rgb_img = (rgb_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rgb_with_boxes = draw_boxes(rgb_img, detections[0])
    
    # Save output
    output_path = 'inference_output.jpg'
    cv2.imwrite(output_path, cv2.cvtColor(rgb_with_boxes, cv2.COLOR_RGB2BGR))
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()