@echo off
REM Setup script for Multimodal UAV-Based Thermal Object Detection (Windows)

echo Setting up virtual environment...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; print('✓ PyTorch version:', torch.__version__)"
python -c "import cv2; print('✓ OpenCV version:', cv2.__version__)"
python -c "import yaml; print('✓ PyYAML installed')"
python -c "import numpy; print('✓ NumPy version:', numpy.__version__)"
python -c "from pycocotools.coco import COCO; print('✓ pycocotools installed')"
python -c "import torchvision; print('✓ torchvision version:', torchvision.__version__)"

echo.
echo Setup complete! Virtual environment is ready.
echo To activate the virtual environment, run: venv\Scripts\activate.bat

pause



