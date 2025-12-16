#source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch orbbec_camera dabai_dcw.launch.py depth_registration:=true enable_d2c_viewer:=true