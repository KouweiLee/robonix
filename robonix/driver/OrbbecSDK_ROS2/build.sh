#!/bin/bash
# Build script for OrbbecSDK_ROS2

set -e

echo "Building OrbbecSDK_ROS2 packages..."
#source /opt/ros/humble/setup.bash
colcon build --packages-select orbbec_camera_msgs orbbec_camera orbbec_description --event-handlers  console_direct+  --cmake-args  -DCMAKE_BUILD_TYPE=Release
echo "OrbbecSDK_ROS2 build completed successfully!"
