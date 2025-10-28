import torch
import yaml
import os
from torch.utils.data import DataLoader
from datasets.multimodal_dataset import MultimodalDataset
from datasets.collate import collate_fn
from datasets.transforms import get_transforms
from models.fusion_yolov11 import FusionYOLOv11
from utils.metrics import evaluate_coco

def main():
    # Load configurations
    with open("experiments/configs/default.yaml", 'r') as f:
        config_default = yaml.safe_load(f)
    with open("experiments/configs/yolov11_fusion.yaml", 'r') as f:
        config_model = yaml.safe_load(f)
    
    # Merge configurations
    config = config_default.copy()
    config.update(config_model)
    
    # Initialize device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = FusionYOLOv11(num_classes=config['model']['num_classes'], img_size=config['data']['img_size'])
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], 'yolov11_fusion_best.pt')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Initialize validation dataset and dataloader
    val_dataset = MultimodalDataset(
        root_dir=config['data']['root_dir'],
        dataset_name='smod',
        split='val',
        transforms=get_transforms(config, training=False)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=lambda x: collate_fn(x, mosaic_prob=0.0, img_size=config['data']['img_size'])
    )
    
    # Evaluate on validation set
    metrics = evaluate_coco(
        model,
        val_loader,
        os.path.join(config['data']['root_dir'], config['data']['datasets'][0]['splits']['val']),
        config['model']['anchors'],
        device
    )
    
    print("Validation metrics:")
    print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")

if __name__ == "__main__":
    main()