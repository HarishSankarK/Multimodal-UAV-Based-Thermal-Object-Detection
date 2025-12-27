#!/bin/bash

# Setup script for Multimodal UAV-Based Thermal Object Detection

echo "Setting up virtual environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'✓ OpenCV version: {cv2.__version__}')"
python -c "import yaml; print(f'✓ PyYAML installed')"
python -c "import numpy; print(f'✓ NumPy version: {numpy.__version__}')"
python -c "from pycocotools.coco import COCO; print(f'✓ pycocotools installed')"
python -c "import torchvision; print(f'✓ torchvision version: {torchvision.__version__}')"

echo ""
echo "Setup complete! Virtual environment is ready."
echo "To activate the virtual environment, run: source venv/bin/activate"


