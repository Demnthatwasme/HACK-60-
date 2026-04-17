#!/bin/bash

# Stop execution immediately if any command fails
set -e

echo "========================================"
echo "🚀 Initializing Gesture AI Pipeline..."
echo "========================================"

echo -e "\n---> [1/4] Scrubbing raw dataset..."
python clean_data.py

echo -e "\n---> [2/4] Applying Rigid Bone Normalization & Scaling..."
python normalise_data.py

echo -e "\n---> [3/4] Training Deep Learning Model (Augmentation & Early Stopping)..."
python train_dl_model.py

echo -e "\n---> [4/4] Launching Live Explainable AI Inference..."
python live_demo.py

echo -e "\n✅ Pipeline execution complete!"
