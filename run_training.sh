#!/bin/bash

# Automated Training Setup and Execution Script
# This script will install all dependencies and start training automatically
# Perfect for users with no prior experience - just run: ./run_training.sh

set -e  # Exit on any error

echo "=========================================="
echo "🚀 Multimodal UAV-Based Thermal Object Detection"
echo "   Automated Training Setup & Execution"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed."
    echo "   Please install Python 3.8 or higher and try again."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "📦 Virtual environment already exists."
    echo "   Activating existing virtual environment..."
    source venv/bin/activate
else
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    echo "   Activating virtual environment..."
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Verify critical installations
echo ""
echo "🔍 Verifying installation..."
python -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || { echo "❌ PyTorch installation failed"; exit 1; }
python -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" || { echo "❌ OpenCV installation failed"; exit 1; }
python -c "import yaml; print(f'  ✓ PyYAML')" || { echo "❌ PyYAML installation failed"; exit 1; }
python -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')" || { echo "❌ NumPy installation failed"; exit 1; }
python -c "from pycocotools.coco import COCO; print(f'  ✓ pycocotools')" || { echo "❌ pycocotools installation failed"; exit 1; }

echo ""
echo "✅ All dependencies installed successfully!"
echo ""

# Check for CUDA availability
echo "🔍 Checking GPU availability..."
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "  Running on CPU"

echo ""
echo "=========================================="
echo "🎯 Starting Training..."
echo "=========================================="
echo ""
echo "💡 Tips:"
echo "   - Training progress will be shown below"
echo "   - Checkpoints will be saved automatically"
echo "   - Press Ctrl+C to stop training (checkpoints are saved after each epoch)"
echo "   - To resume training, just run this script again"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start training
python train.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Training completed!"
echo ""
echo "📁 Checkpoints saved in: checkpoints/"
echo "📊 TensorBoard logs: experiments/tensorboard/"
echo ""
echo "To view training progress with TensorBoard, run:"
echo "  tensorboard --logdir=experiments/tensorboard"
echo ""

