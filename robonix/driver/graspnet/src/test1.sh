#!/bin/bash
cd graspnet-baseline

# export OPEN3D_HEADLESS=1

export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH

# Source ROS2 Humble setup
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "[*] Sourced ROS2 Humble setup"
fi

# Add ROS2 Humble Python packages to PYTHONPATH for conda environment
for PYTHON_SITE in /opt/ros/humble/lib/python*/site-packages; do
    if [ -d "$PYTHON_SITE" ]; then
        export PYTHONPATH="$PYTHON_SITE:$PYTHONPATH"
        echo "[*] Added ROS2 Python packages: $PYTHON_SITE"
    fi
done

# Also add workspace if it exists
if [ -d "$HOME/ros2_ws/install" ]; then
    for PYTHON_SITE in $HOME/ros2_ws/install/lib/python*/site-packages; do
        if [ -d "$PYTHON_SITE" ]; then
            export PYTHONPATH="$PYTHONPATH:$PYTHON_SITE"
            echo "[*] Added workspace Python packages: $PYTHON_SITE"
        fi
    done
fi

# Create output directory if it doesn't exist
OUTPUT_DIR="../output/visualization"
mkdir -p "$OUTPUT_DIR"

# Suppress ROS-related warnings
export PYTHONWARNINGS="ignore::UserWarning"

# quick test of ros2 packages, list the number of topics
python << 'EOF'
import rclpy
from rclpy.node import Node

# Initialize rclpy
rclpy.init()

# Create a node
node = Node('test_node')

# Get topics
topics_and_types = node.get_topic_names_and_types()

print('ROS2 Python packages loaded')
print(f'Number of topics: {len(topics_and_types)}')
if topics_and_types:
    print(f'First 5 topics: {topics_and_types[:5]}')

# Cleanup
node.destroy_node()
rclpy.shutdown()
EOF

CUDA_VISIBLE_DEVICES=0 \
    python -W ignore demo.py --checkpoint_path ../weights/checkpoint-kn.tar