#!/bin/bash

# YOLO Object Detection Node Startup Script

source /opt/ros/humble/setup.bash
source /home/syswonder/lgw/robonix/robonix/driver/graspnet/install/setup.bash

python3 pick.py 'spray can'
