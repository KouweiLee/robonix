#!/usr/bin/env python3
"""
Pick skill - Request object detection and grasp pose generation
"""
import time

import rclpy
from rclpy.node import Node
from graspnet_msgs.srv import ObjectDetectionRequest, GraspRequest
from graspnet_msgs.msg import PiperStatusMsg
from geometry_msgs.msg import PoseStamped


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

        self.current_arm_status = None
        self.arm_status_sub = self.node.create_subscription(
            PiperStatusMsg,
            '/arm_status',
            self.arm_status_callback,
            10
        )
        
        self.node.get_logger().info('[*] Pick client initialized')

    def arm_status_callback(self, msg):
        """Callback for arm status."""
        print(f'[*] Arm status: {msg.arm_status}')
        self.current_arm_status = msg.arm_status
    
    def _call_service(self, client, request, service_name, timeout):
        """Helper method to call a service and wait for response."""
        future = client.call_async(request)
        try:
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
            if future.done():
                return future.result()
            else:
                self.node.get_logger().error(f'{service_name} service call timed out')
                return None
        except Exception as e:
            self.node.get_logger().error(f'{service_name} service call failed: {e}')
            return None
    
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
        detection_response = self._call_service(            self.detection_client, detection_request, 'YOLO detection', timeout)
        
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
        for retry in range(5):
            grasp_request = GraspRequest.Request()
            grasp_request.object_name = object_name
            grasp_request.bbox_2d = detection_response.bbox_2d
            grasp_request.object_center_3d = detection_response.object_center_3d if detection_response.object_center_3d else []
            grasp_request.retry = retry
            
            self.node.get_logger().info('[*] Calling GraspNet service...')
            grasp_response = self._call_service(
                self.grasp_client, grasp_request, 'GraspNet', timeout)
            
            if grasp_response is None or not grasp_response.success:
                self.node.get_logger().error(f'Grasp generation failed: {grasp_response.message if grasp_response is not None else "No response received"}')
                continue
            
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

            # ===== Wait for arm_status == 0 =====
            self.node.get_logger().info('[*] Waiting for arm_status == 0 ...')
            start_time = time.time()
            need_retry = False
            time.sleep(2.0)
            while rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if self.current_arm_status == 0:
                    self.node.get_logger().info('[*] Arm is normal (arm_status=0), stopping retry.')
                    return result
                else:
                    self.node.get_logger().info(f'[*] Current arm_status={self.current_arm_status}, waiting...')
                if time.time() - start_time > 3: # 3 seconds timeout
                    self.node.get_logger().warn('[*] Timeout waiting for arm_status==0, retrying next grasp...')
                    need_retry = True
                    break  # retry
            if need_retry:
                self.node.get_logger().info('[*] Retrying next grasp...')
                continue
            return result
        self.node.get_logger().error('All grasp generation attempts failed')
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
    # objects = [object_name,"neon light","keyboard", "skylight"]
    objects = [object_name,"toothbrush"]
    
    try:
        for object_name in objects:
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
