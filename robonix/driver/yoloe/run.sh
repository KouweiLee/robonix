#!/bin/bash

# YOLO Object Detection Node Startup Script

#source /opt/ros/humble/setup.bash
source ../graspnet/install/setup.bash

# Set Python path for ROS2 packages
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH

# Add ultralytics package path if needed
# Uncomment if using conda/virtual environment
# export PYTHONPATH=/path/to/your/env/lib/python3.10/site-packages:$PYTHONPATH

# Set CUDA device (optional, adjust if needed)
export CUDA_VISIBLE_DEVICES=0

# Run YOLO detection node
echo "[*] Starting YOLO object detection node..."
echo "[*] Service will be available at: /yolo/detect_object"
echo "[*] Press Ctrl+C to stop"
echo ""
python3 object_detection_node.py
