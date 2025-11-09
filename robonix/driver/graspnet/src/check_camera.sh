#!/bin/bash
# Diagnostic script to check camera topics

echo "check if ROS2 camera topics are publishing data..."
echo ""

# Check if topics exist and get info
echo "[*] Topic List:"
ros2 topic list | grep camera
echo ""

echo "[*] Checking topic info..."

echo ""
echo "[1] /camera/color/image_raw:"
ros2 topic info /camera/color/image_raw

echo ""
echo "[2] /camera/depth/image_raw:"
ros2 topic info /camera/depth/image_raw

echo ""
echo "[3] /camera/color/camera_info:"
ros2 topic info /camera/color/camera_info

echo ""
echo "[*] Checking topic hz (frequency)..."
echo "[Note: Press Ctrl+C to stop]"
timeout 5 ros2 topic hz /camera/color/image_raw
timeout 5 ros2 topic hz /camera/depth/image_raw
timeout 5 ros2 topic hz /camera/color/camera_info

echo ""
echo "[*] Checking if topics have publishers..."
ros2 topic list | grep camera | while read topic; do
    has_pub=$(ros2 topic info $topic 2>/dev/null | grep -c "Publisher count: 1")
    if [ "$has_pub" -eq "1" ]; then
        echo "✓ $topic (has publisher)"
    else
        echo "✗ $topic (NO publisher)"
    fi
done

echo ""
echo "Done"

