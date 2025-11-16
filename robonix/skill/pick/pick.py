#!/usr/bin/env python3
"""
Pick skill - Request object detection and grasp pose generation
"""

import rclpy
from rclpy.node import Node
from graspnet_msgs.srv import ObjectDetectionRequest, GraspRequest
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
        
        # Create client for GraspNet service
        self.grasp_client = self.node.create_client(
            GraspRequest,
            '/graspnet/grasp_request')
        
        self.node.get_logger().info('[*] Pick client initialized')
    
    def pick_object(self, object_name, timeout=30.0):
        """Request to pick a specific object.
        
        Args:
            object_name: Name of the object to pick
            timeout: Timeout in seconds
            
        Returns:
            dict: Grasp result with pose and gripper_width, or None if failed
        """
        self.node.get_logger().info(f'[*] Requesting to pick object: {object_name}')
        
        # Step 1: Call YOLO detection service
        if not self.detection_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error('YOLO detection service not available')
            return None
        
        detection_request = ObjectDetectionRequest.Request()
        detection_request.object_name = object_name
        
        self.node.get_logger().info('[*] Calling YOLO detection service...')
        detection_future = self.detection_client.call_async(detection_request)
        
        # Wait for detection response
        start_time = time.time()
        detection_response = None
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            if detection_future.done():
                try:
                    detection_response = detection_future.result()
                    break
                except Exception as e:
                    self.node.get_logger().error(f'YOLO detection service call failed: {e}')
                    return None
            
            if time.time() - start_time > timeout:
                self.node.get_logger().error('YOLO detection service call timed out')
                return None
        
        if detection_response is None or not detection_response.success:
            self.node.get_logger().error(f'Detection failed: {detection_response.message if detection_response is not None else "No response received"}')
            return None
        
        self.node.get_logger().info(f'[*] Object detected: {object_name}')
        self.node.get_logger().info(f'[*] 2D bbox: {detection_response.bbox_2d}')
        self.node.get_logger().info(f'[*] 3D center: {detection_response.object_center_3d}')
        self.node.get_logger().info(f'[*] Confidence: {detection_response.confidence:.3f}')
        
        # Step 2: Call GraspNet service with detection results
        if not self.grasp_client.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error('GraspNet service not available')
            return None
        
        grasp_request = GraspRequest.Request()
        grasp_request.object_name = object_name
        grasp_request.bbox_2d = detection_response.bbox_2d
        grasp_request.object_center_3d = detection_response.object_center_3d if detection_response.object_center_3d else []
        
        self.node.get_logger().info('[*] Calling GraspNet service...')
        grasp_future = self.grasp_client.call_async(grasp_request)
        
        # Wait for grasp response
        start_time = time.time()
        grasp_response = None
        while rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            if grasp_future.done():
                try:
                    grasp_response = grasp_future.result()
                    break
                except Exception as e:
                    self.node.get_logger().error(f'GraspNet service call failed: {e}')
                    return None
            
            if time.time() - start_time > timeout:
                self.node.get_logger().error('GraspNet service call timed out')
                return None
        
        if grasp_response is None or not grasp_response.success:
            self.node.get_logger().error(f'Grasp generation failed: {grasp_response.message if grasp_response is not None else "No response received"}')
            return None
        
        # Build result dictionary
        result = {
            'success': True,
            'object_name': object_name,
            'pose': grasp_response.grasp_pose,
            'gripper_width': grasp_response.gripper_width,
            'score': grasp_response.score,
            'bbox_2d': list(detection_response.bbox_2d) if detection_response.bbox_2d else [],
            'object_center_3d': list(detection_response.object_center_3d) if detection_response.object_center_3d else [],
            'confidence': detection_response.confidence
        }
        
        self.node.get_logger().info(f'[*] Pick request completed successfully!')
        self.node.get_logger().info(f'[*] Grasp pose: x={result["pose"].pose.position.x:.3f}, '
                                   f'y={result["pose"].pose.position.y:.3f}, '
                                   f'z={result["pose"].pose.position.z:.3f}')
        self.node.get_logger().info(f'[*] Gripper width: {result["gripper_width"]:.3f}m')
        self.node.get_logger().info(f'[*] Grasp score: {result["score"]:.3f}')
        
        return result
    
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
