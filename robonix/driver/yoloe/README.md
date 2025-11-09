# YOLO-World Object Detection Node

ROS2 node for **YOLO-World** open-vocabulary object detection integrated with GraspNet for robotic grasping.

## What is YOLO-World?

YOLO-World is an advanced **open-vocabulary** object detection model that allows detecting **any object by name**, not limited to pre-trained classes. Unlike traditional YOLO models that can only detect 80 COCO classes, YOLO-World can detect arbitrary objects described in natural language.

## Overview

This node provides object detection services that work with the GraspNet node to enable object-specific grasp pose generation. The workflow is:

1. **Pick Client** (pick.py) → Sends object name request
2. **YOLO Node** (this) → Detects object and computes 3D bounding box
3. **GraspNet Node** → Generates grasp poses within the detected region
4. **Response** → Returns grasp pose back through the service chain

## Prerequisites

### 1. Build graspnet_msgs

The service messages must be built first:

```bash
cd ../graspnet
bash build.sh
source install/setup.bash
```

### 2. Install Dependencies

Ensure you have the following Python packages:

```bash
pip install ultralytics opencv-python numpy rclpy
```

### 3. YOLO-World Model

The node automatically uses the YOLO-World model:
- **Default**: `yolov8s-world.pt` (downloads automatically on first run)
- **Custom**: Place your model at `robonix/skill/vision/models/yolov8s-world.pt`

**Available YOLO-World models**:
- `yolov8s-world.pt` - Small, fast (recommended)
- `yolov8m-world.pt` - Medium, balanced
- `yolov8l-world.pt` - Large, accurate

The model will be downloaded automatically from Ultralytics on first use.

## Usage

### Start the Node

```bash
bash run.sh
```

Or directly:

```bash
python3 object_detection_node.py
```

### Service Interface

**Service:** `/yolo/detect_object`

**Type:** `graspnet_msgs/srv/ObjectDetectionRequest`

**Request:**
```
string object_name  # Name of the object to detect (e.g., "bottle", "cup")
```

**Response:**
```
float64[] bbox_2d       # 2D bounding box [x_min, y_min, x_max, y_max]
float64[] bbox_3d       # 3D bounding box [x_min, y_min, z_min, x_max, y_max, z_max]
float64[] center_point  # 3D center [x, y, z] in camera frame
float32 confidence      # Detection confidence score
bool success           # Success flag
string message         # Status message
```

### Using from pick.py

```python
from robonix.skill.pick.pick import pick

# Request to pick any object - YOLO-World supports open vocabulary!
result = pick("bottle", timeout=30.0)  # or "red cup", "small box", etc.

if result:
    print(f"Grasp pose: {result['pose']}")
    print(f"Gripper width: {result['gripper_width']}")
```

Or from command line:

```bash
cd robonix/skill/pick
python3 pick.py bottle          # Basic object
python3 pick.py "red cup"       # With description
python3 pick.py "small box"     # Size description
python3 pick.py "toy car"       # Compound words
```

### Supported Objects

**YOLO-World supports ANY object!** Examples:
- Common objects: `bottle`, `cup`, `bowl`, `phone`, `book`
- With colors: `red bottle`, `blue cup`, `green apple`
- With sizes: `small box`, `large container`, `tiny screw`
- Tools: `hammer`, `screwdriver`, `wrench`, `pliers`
- Electronics: `laptop`, `keyboard`, `mouse`, `remote`
- And many more - try any object description!

## Configuration

### Camera Topics

The node subscribes to:
- `/camera/color/image_raw` - RGB image
- `/camera/depth/image_raw` - Depth image
- `/camera/color/camera_info` - Camera intrinsics

Make sure your camera node is publishing to these topics.

### YOLO Model Path

To use a custom YOLO model, modify the path in `object_detection_node.py`:

```python
model_path = "/path/to/your/yolo/model.pt"
```

## Architecture

```
┌─────────────┐
│  pick.py    │  Client requests object pick
└──────┬──────┘
       │ Service: /yolo/detect_object
       ▼
┌─────────────────────┐
│  YOLO Detection     │  Detects object, computes 3D bbox
│  Node (this)        │
└──────┬──────────────┘
       │ Service: /graspnet/grasp_request
       ▼
┌─────────────────────┐
│  GraspNet Node      │  Generates grasp pose
└──────┬──────────────┘
       │ Topic: /graspnet/grasps
       ▼
┌─────────────────────┐
│  pick.py            │  Receives result
└─────────────────────┘
```

## Troubleshooting

### Service not available

Make sure graspnet_msgs is built:
```bash
cd ../graspnet && bash build.sh
source install/setup.bash
```

### No camera topics

Start the camera node first:
```bash
# For RealSense
ros2 launch realsense2_camera rs_launch.py

# For Orbbec
cd ../OrbbecSDK_ROS2 && bash run.sh
```

### YOLO-World model download

The model downloads automatically on first use. If download fails:
```bash
# Manually download YOLO-World model
pip install ultralytics
python3 -c "from ultralytics import YOLOWorld; YOLOWorld('yolov8s-world.pt')"
```

### GraspNet service not available

Start the GraspNet node:
```bash
cd ../graspnet && bash run.sh
```

## Related Files

- **Service Definitions**: `../graspnet/src/graspnet_msgs/srv/`
  - `ObjectDetectionRequest.srv` - Detection service
  - `GraspRequest.srv` - Grasp service
- **Pick Client**: `../../skill/pick/pick.py`
- **GraspNet Node**: `../graspnet/src/graspnet-baseline/demo_ros2.py`

## Support

For issues or questions, please contact the syswonder team.


