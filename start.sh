bash init.sh
#source /opt/ros/humble/setup.sh

# ensure all ros2 processes are stopped
bash stop.sh

source robonix/driver/graspnet/install/setup.bash

# cd robonix
python3 robonix/manager/boot.py --config config/include/ranger_test.yml
# python -m robonix.manager.boot  --config ../config/include/ranger_test.yml
