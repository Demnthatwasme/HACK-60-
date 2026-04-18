#!/bin/bash

echo "Checking for pip..."
if ! python3 -m pip --version &> /dev/null; then
    echo "pip not found. Downloading and installing pip locally..."
    wget -q https://bootstrap.pypa.io/get-pip.py
    python3 get-pip.py --user
    rm get-pip.py
    
    # Ensure local bin is in PATH so we can use python3 -m
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing virtualenv..."
python3 -m pip install virtualenv --user

echo "Creating virtual environment..."
python3 -m virtualenv romp_gmr_env

echo "Activating virtual environment..."
source romp_gmr_env/bin/activate

echo "Installing ROMP dependencies..."
pip install setuptools cython numpy scipy opencv-python torch torchvision

echo "Installing ROMP..."
pip install romp

echo "Installing GMR dependencies..."
pip install mink mujoco rich

echo "Installing GMR..."
cd GMR
pip install -e .
cd ..

echo "Setup Complete! You can now run the pipeline with:"
echo "source romp_gmr_env/bin/activate"
echo "python webcam_to_gmr.py"
