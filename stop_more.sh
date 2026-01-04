#!/bin/bash

# Kill ROS2 nodes script
# This script finds and kills all ROS2 related processes

echo "[*] Searching for ROS2 processes..."

# Method 1: Kill processes by name pattern
ROS_PROCESSES=$(ps aux | grep -E 'ros|rcl|camera|yolo|detection|rviz' | grep -v grep | grep -v $$)
if [ -n "$ROS_PROCESSES" ]; then
    echo "[*] Found processes by name pattern:"
    echo "$ROS_PROCESSES"
    echo "$ROS_PROCESSES" | awk '{print $2}' | xargs sudo kill -9 2>/dev/null
fi

# Method 2: Kill processes by port usage (ROS2 typically uses specific ports)
# Find processes using typical ROS2 ports
for port in $(seq 11311 11320); do
    sudo lsof -ti:$port 2>/dev/null | xargs sudo kill -9 2>/dev/null
done

# Method 3: Kill all processes from ROS workspace
if [ -n "$ROS_WORKSPACE" ]; then
    echo "[*] Checking ROS workspace: $ROS_WORKSPACE"
    for pid in $(pgrep -f "$ROS_WORKSPACE"); do
        echo "  Killing PID $pid from workspace"
        sudo kill -9 $pid 2>/dev/null
    done
fi

# Method 4: Kill DDS discovery processes
# ROS2 uses DDS for communication, these processes might linger
DDS_PROCESSES=$(ps aux | grep -E 'fastdds|CycloneDDS|rmw' | grep -v grep | grep -v $$ | awk '{print $2}')
if [ -n "$DDS_PROCESSES" ]; then
    echo "[*] Killing DDS related processes"
    echo "$DDS_PROCESSES" | xargs sudo kill -9 2>/dev/null
fi

# Method 5: Special handling for duplicate nodes (like in your case)
# Kill all python processes that might be running ROS2 nodes
PYTHON_ROS_PROCESSES=$(ps aux | grep python | grep -E 'ros|yolo|rviz' | grep -v grep | grep -v $$ | awk '{print $2}')
if [ -n "$PYTHON_ROS_PROCESSES" ]; then
    echo "[*] Killing Python ROS processes"
    echo "$PYTHON_ROS_PROCESSES" | xargs sudo kill -9 2>/dev/null
fi

# Method 6: Check if any ROS2 nodes are still running
sleep 2  # Wait a moment for processes to terminate
echo "[*] Checking for remaining ROS2 nodes..."
NODES=$(timeout 5 ros2 node list 2>/dev/null || echo "")

if [ -n "$NODES" ]; then
    echo "[!] Some ROS2 nodes are still running:"
    echo "$NODES"
    
    # Try to kill them using ros2 lifecycle
    echo "[*] Attempting to shutdown nodes using ros2 lifecycle..."
    for node in $NODES; do
        # Remove leading slash if present
        node_clean=$(echo $node | sed 's|^/||')
        timeout 3 ros2 lifecycle set $node_clean shutdown 2>/dev/null
    done
    
    # Force kill any remaining processes
    echo "[*] Force killing any remaining ROS2 processes..."
    # Find all processes that have 'node' in their command
    NODE_PIDS=$(ps aux | grep -E '[n]ode|[r]os2' | grep -v $$ | awk '{print $2}')
    if [ -n "$NODE_PIDS" ]; then
        echo "$NODE_PIDS" | xargs sudo kill -9 2>/dev/null
    fi
else
    echo "[✓] No ROS2 nodes found running"
fi

# Method 7: Clean up shared memory (DDS uses shared memory)
echo "[*] Cleaning up DDS shared memory..."
sudo rm -rf /dev/shm/* 2>/dev/null || true

# Final check
sleep 1
echo "[*] Final system check..."
ROS_COUNT=$(ps aux | grep -E 'ros|rcl' | grep -v grep | grep -v $$ | wc -l)
if [ "$ROS_COUNT" -eq "0" ]; then
    echo "[✓] All ROS2 processes have been terminated successfully"
else
    echo "[!] WARNING: $ROS_COUNT ROS2 processes still detected"
    ps aux | grep -E 'ros|rcl' | grep -v grep | grep -v $$
fi