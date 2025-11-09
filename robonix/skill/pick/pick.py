#!/usr/bin/env python3
"""
Pick skill - Request object detection and grasp pose generation
"""

import rclpy
from rclpy.node import Node
from graspnet_msgs.srv import ObjectDetectionRequest
from graspnet_msgs.msg import GraspPose
from geometry_msgs.msg import PoseStamped
import time


class PickClient:
    """Client for pick operation using YOLO detection and GraspNet."""
    
    def __init__(self):
        """Initialize ROS2 node and service client."""
        if not rclpy.ok():
            rclpy.init()
        
        self.node = Node('pick_client')
        
        # Create client for YOLO detection service
        self.detection_client = self.node.create_client(
            ObjectDetectionRequest, 
            '/yolo/detect_object')
        
        # Subscribe to grasp results
        self.grasp_result = None
        self.grasp_sub = self.node.create_subscription(
            GraspPose,
            '/graspnet/grasps',
            self.grasp_callback,
            10)
        
        self.node.get_logger().info('[*] Pick client initialized')
    
    def grasp_callback(self, msg):
        """Callback for grasp results."""
        self.grasp_result = msg
        self.node.get_logger().info(f'[*] Received grasp result: width={msg.gripper_width:.3f}m')
    
    def pick_object(self, object_name, timeout=30.0):
        """Request to pick a specific object.
        
        Args:
            object_name: Name of the object to pick
            timeout: Timeout in seconds
            
        Returns:
            dict: Grasp result with pose and gripper_width, or None if failed
        """
        self.node.get_logger().info(f'[*] Requesting to pick object: {object_name}')
        
        # Wait for detection service
        if not self.detection_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error('YOLO detection service not available')
            return None
        
        # Create request
        request = ObjectDetectionRequest.Request()
        request.object_name = object_name
        
        # Call detection service (which internally calls GraspNet)
        self.node.get_logger().info('[*] Calling YOLO detection service...')
        future = self.detection_client.call_async(request)
        
        # Wait for detection response
        start_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            if future.done():
                try:
                    response = future.result()
                    break
                except Exception as e:
                    self.node.get_logger().error(f'Service call failed: {e}')
                    return None
            
            if time.time() - start_time > timeout:
                self.node.get_logger().error('Detection service call timed out')
                return None
        
        if not response.success:
            self.node.get_logger().error(f'Detection failed: {response.message}')
            return None
        
        self.node.get_logger().info(f'[*] Object detected: {object_name}')
        self.node.get_logger().info(f'[*] 3D bbox: {response.bbox_3d}')
        self.node.get_logger().info(f'[*] Center: {response.center_point}')
        self.node.get_logger().info(f'[*] Confidence: {response.confidence:.3f}')
        
        # Wait for grasp result on topic
        self.node.get_logger().info('[*] Waiting for grasp pose...')
        self.grasp_result = None
        start_time = time.time()
        
        while rclpy.ok() and self.grasp_result is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            if time.time() - start_time > timeout:
                self.node.get_logger().error('Timeout waiting for grasp pose')
                return None
        
        if self.grasp_result is not None:
            result = {
                'success': True,
                'object_name': object_name,
                'pose': self.grasp_result.target_pose,
                'gripper_width': self.grasp_result.gripper_width,
                'bbox_3d': list(response.bbox_3d),
                'center_point': list(response.center_point),
                'confidence': response.confidence
            }
            
            self.node.get_logger().info(f'[*] Pick request completed successfully!')
            self.node.get_logger().info(f'[*] Grasp pose: x={result["pose"].pose.position.x:.3f}, '
                                       f'y={result["pose"].pose.position.y:.3f}, '
                                       f'z={result["pose"].pose.position.z:.3f}')
            self.node.get_logger().info(f'[*] Gripper width: {result["gripper_width"]:.3f}m')
            
            return result
        
        return None
    
    def shutdown(self):
        """Clean shutdown."""
        self.node.destroy_node()


def pick(object_name, timeout=30.0):
    """Convenience function to pick an object.
    
    Args:
        object_name: Name of the object to pick
        timeout: Timeout in seconds
        
    Returns:
        dict: Grasp result or None if failed
    """
    client = PickClient()
    try:
        result = client.pick_object(object_name, timeout)
        return result
    finally:
        client.shutdown()


# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pick.py <object_name>")
        print("Example: python pick.py bottle")
        sys.exit(1)
    
    object_name = sys.argv[1]
    
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
            sys.exit(1)
    
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
