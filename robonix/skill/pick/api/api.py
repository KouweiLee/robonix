import time

import rclpy
from rclpy.node import Node

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(root_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from robonix.manager.eaios_decorators import eaios

# 获取当前文件所在目录的上级目录（即 project 目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以导入 lib 模块
from pick import pick

@eaios.caller
@eaios.api
def skl_grasp_object(self_entity, object_name: str):
    try:
        result = pick(object_name)
        
        if result:
            print("\n" + "="*50)
            print("PICK OPERATION SUCCESSFUL")
            print("="*50)
            print(f"Object: {result['object_name']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Grasp pose:")
            print(f"  Position: ({result['pose'].pose.position.x:.3f}, "
                f"{result['pose'].pose.position.y:.3f}, "
                f"{result['pose'].pose.position.z:.3f})")
            print(f"  Orientation: ({result['pose'].pose.orientation.x:.3f}, "
                f"{result['pose'].pose.orientation.y:.3f}, "
                f"{result['pose'].pose.orientation.z:.3f}, "
                f"{result['pose'].pose.orientation.w:.3f})")
            print(f"Gripper width: {result['gripper_width']:.3f}m")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("PICK OPERATION FAILED")
            print("="*50)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if rclpy.ok():
            rclpy.shutdown()