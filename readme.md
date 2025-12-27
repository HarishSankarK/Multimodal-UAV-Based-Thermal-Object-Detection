# Multimodal UAV-Based Thermal Object Detection

A PyTorch implementation of a multimodal YOLOv11-based object detection system for UAV thermal imagery, supporting RGB and Infrared (IR) image fusion for improved detection accuracy.

## Features

- **Multimodal Fusion**: Supports RGB-only and RGB+IR image fusion for enhanced object detection
- **YOLOv11 Architecture**: Custom implementation with dual-stream backbone and PANet neck
- **Multiple Datasets**: Compatible with SMOD, HIT-UAV, and DroneRGB-T datasets
- **COCO Format**: Uses COCO annotation format for easy dataset integration
- **Comprehensive Training**: Includes training, validation, and inference scripts
- **Visualization Tools**: Built-in visualization and heatmap generation

## Dataset Support

The project supports three datasets:

1. **SMOD**: Multimodal dataset with RGB and IR images
2. **HIT-UAV**: RGB-only dataset
3. **DroneRGB-T**: Multimodal dataset with RGB and IR images

All datasets should be organized in COCO format with the following structure:

```
data/
├── modalities.json
├── {dataset_name}/
│   ├── annotations/
│   │   ├── instances_train.json
│   │   ├── instances_val.json
│   │   └── instances_test.json
│   └── images/
│       ├── train/
│       │   ├── rgb/  (or directly in train/ for RGB-only datasets)
│       │   └── ir/   (if available)
│       ├── val/
│       └── test/
```

## Installation

### Quick Setup (Recommended)

**On Linux/macOS:**
```bash
./setup.sh
```

**On Windows:**
```batch
setup.bat
```

### Manual Setup

#### 1. Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```batch
python -m venv venv
venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import yaml; print('PyYAML installed')"
python -c "from pycocotools.coco import COCO; print('pycocotools installed')"
```

## Usage

### Training

Train the model on a specific dataset:

```bash
python train.py
```

The training script will:
- Load configurations from `experiments/configs/`
- Train on the specified dataset (default: SMOD)
- Save checkpoints to `checkpoints/`
- Log metrics to TensorBoard in `experiments/tensorboard/`

### Configuration

Edit `experiments/configs/default.yaml` and `experiments/configs/yolov11_fusion.yaml` to customize:
- Training hyperparameters (batch size, learning rate, epochs)
- Model architecture
- Dataset paths
- Augmentation settings

### Validation

Evaluate the trained model:

```bash
python validate.py
```

### Inference

Run inference on test images:

```bash
python inference.py
```

Note: Update the image paths in `inference.py` before running.

## Project Structure

```
.
├── data/                    # Dataset directory
│   ├── modalities.json     # Dataset configuration
│   ├── smod/
│   ├── hit_uav/
│   └── dronergbt/
├── datasets/                # Dataset loading utilities
│   ├── multimodal_dataset.py
│   ├── collate.py
│   └── transforms.py
├── models/                  # Model architectures
│   ├── fusion_yolov11.py
│   ├── backbones.py
│   ├── necks.py
│   └── heads.py
├── utils/                   # Utility functions
│   ├── metrics.py
│   ├── visualize.py
│   ├── anchors.py
│   └── coco_converter_*.py
├── experiments/
│   ├── configs/            # Configuration files
│   └── tensorboard/        # TensorBoard logs
├── checkpoints/             # Model checkpoints
├── train.py                 # Training script
├── validate.py             # Validation script
├── inference.py             # Inference script
└── requirements.txt        # Python dependencies
```

## Model Architecture

The model consists of:

1. **Dual Backbone**: Separate SGGF-Net backbones for RGB and IR streams
2. **Fusion Layers**: Mid-level fusion by concatenating and reducing channel dimensions
3. **PANet Neck**: Path Aggregation Network for multi-scale feature fusion
4. **YOLOv11 Head**: Decoupled detection head with separate box, objectness, and classification predictions

## Training Details

- **Loss Function**: Combined CIoU box loss, objectness loss, and classification loss
- **Optimizer**: SGD with momentum
- **Learning Rate**: Cosine annealing schedule
- **Image Size**: 640x640 (configurable)
- **Classes**: Person, Rider, Bicycle, Car

## Evaluation Metrics

The model is evaluated using COCO metrics:
- mAP@0.5
- mAP@0.5:0.95

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM (recommended)
- 50GB+ disk space for datasets

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `default.yaml`
2. **Dataset Path Errors**: Verify paths in `modalities.json` and `default.yaml`
3. **Import Errors**: Ensure virtual environment is activated and all dependencies are installed

### Dataset Conversion

Use the provided COCO converter scripts in `utils/` to convert datasets to COCO format:
- `coco_converter_smod.py`
- `coco_converter_hit_uav.py`
- `coco_converter_drone.py`

## Citation

If you use this code in your research, please cite the original papers for:
- YOLOv11 architecture
- SMOD, HIT-UAV, and DroneRGB-T datasets

## License

[Add your license information here]

## Contributors

[Add contributor information here]

## Acknowledgments

- YOLOv11 architecture inspiration
- COCO evaluation tools
- Dataset providers (SMOD, HIT-UAV, DroneRGB-T)

