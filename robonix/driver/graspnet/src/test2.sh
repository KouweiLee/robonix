#!/bin/bash

# run colcon build in this directory first
# colcon build

source install/setup.sh

cd graspnet-baseline

# pip install lark empy

# Kill any existing demo_ros2 processes before starting
echo "[*] Checking for existing demo_ros2 processes..."
DEMO_ROS2_PIDS=$(ps aux | grep -E "[d]emo_ros2\.py" | awk '{print $2}')
if [ -n "$DEMO_ROS2_PIDS" ]; then
    echo "[*] Found existing demo_ros2 processes, killing them..."
    echo "$DEMO_ROS2_PIDS" | xargs -r kill -9 2>/dev/null
    sleep 1
    echo "[*] Cleaned up existing processes"
else
    echo "[*] No existing demo_ros2 processes found"
fi

# to fix the error, manually do this once for conda environment to use system libstdc++.so.6
# mv /opt/miniconda3/envs/env1/lib/libstdc++.so.6 /opt/miniconda3/envs/env1/lib/libstdc++.so.6.bak

# ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/miniconda3/envs/env1/lib/libstdc++.so.6 (optional)

# export OPEN3D_HEADLESS=1

export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH

# Source ROS2 Humble setup
if [ -f "/opt/ros/humble/setup.bash" ]; then
    #source /opt/ros/humble/setup.bash
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


echo "[*] Building graspnet_msgs package..."
cd .
#source /opt/ros/humble/setup.bash

# Create output directory if it doesn't exist
OUTPUT_DIR="../output/visualization"
mkdir -p "$OUTPUT_DIR"

# Suppress ROS-related warnings
export PYTHONWARNINGS="ignore::UserWarning"

# Check ROS2 topics before running
echo "[*] Checking ROS2 topics..."
python << 'EOF'
import rclpy
from rclpy.node import Node

# Initialize rclpy
rclpy.init()

# Create a node
node = Node('test_node')

try:
    # Get topics
    topics_and_types = node.get_topic_names_and_types()
    
    print(f'[*] Found {len(topics_and_types)} ROS2 topics')
    
    # Show relevant camera topics
    camera_topics = [t[0] for t in topics_and_types if '/camera' in t[0]]
    if camera_topics:
        print('[*] Camera topics found:')
        for topic in camera_topics:
            print(f'    {topic}')
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
except Exception as e:
    print(f'Error: {e}')
    node.destroy_node()
    rclpy.shutdown()
EOF

# export OPEN3D_HEADLESS=1

# Run without timeout to allow proper Ctrl-C handling
# Note: timeout command interferes with signal handling, so it's removed
# Add --no_collision to disable collision detection for debugging
CUDA_VISIBLE_DEVICES=0 \
    python -W ignore demo_ros2.py \
    --checkpoint_path ../weights/checkpoint-kn.tar \
    --no_collision

echo ""
echo "[*] Done"