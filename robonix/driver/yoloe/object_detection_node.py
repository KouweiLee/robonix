#!/usr/bin/env python3
"""
YOLO Object Detection Node for ROS2
Provides object detection service and forwards grasp requests to GraspNet
"""

import os
import sys
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLOWorld
import threading
import time
import message_filters

# Import custom service messages
try:
    from graspnet_msgs.srv import ObjectDetectionRequest, GraspRequest
except Exception as e:
    print("[!] Missing ROS2 service types 'graspnet_msgs/ObjectDetectionRequest' or 'graspnet_msgs/GraspRequest'.")
    print("    Please build the graspnet_msgs package before running:")
    print("    1) cd robonix/driver/graspnet && bash build.sh")
    print("    2) source install/setup.bash")
    raise e


class YOLODetectionNode(Node):
    """ROS2 node for YOLO-based object detection with GraspNet integration."""
    
    def __init__(self, model_path=None):
        super().__init__('yolo_detection_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Camera data (synchronized)
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_camera_info = None
        self.data_lock = threading.Lock()
        
        # Load YOLO-World model
        if model_path is None:
            # Try to find custom model in vision skill directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            custom_model_path = os.path.join(script_dir, "..", "..", "skill", "vision", "models", "yolov8s-world.pt")
            
            # Use default YOLO-World model if custom model not found
            if os.path.exists(custom_model_path):
                model_path = custom_model_path
                self.get_logger().info(f'Using custom YOLO-World model: {model_path}')
            else:
                # Will download automatically if not present
                model_path = 'yolov8s-world.pt'
                self.get_logger().info('Using default YOLO-World model (will download if needed)')
        
        self.get_logger().info(f'Loading YOLO-World model from: {model_path}')
        try:
            self.yolo_model = YOLOWorld(model_path)
            self.get_logger().info('YOLO-World model loaded successfully')
            self.get_logger().info('Model supports open-vocabulary detection')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO-World model: {e}')
            self.get_logger().error('Please ensure ultralytics is installed: pip install ultralytics')
            raise e
        
        # Subscribe to camera topics with message_filters for synchronization
        self.sub_color = message_filters.Subscriber(
            self,
            Image,
            '/camera/color/image_raw')
        
        self.sub_depth = message_filters.Subscriber(
            self,
            Image,
            '/camera/depth/image_raw')
        
        self.sub_camera_info = message_filters.Subscriber(
            self,
            CameraInfo,
            '/camera/color/camera_info')
        
        # Synchronize messages
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_camera_info],
            queue_size=10,
            slop=0.1)  # 100ms tolerance
        self.sync.registerCallback(self.camera_callback)
        
        # Create object detection service
        self.detection_srv = self.create_service(
            ObjectDetectionRequest,
            '/yolo/detect_object',
            self.handle_detection_request)
        
        # Create client for GraspNet service
        self.grasp_client = self.create_client(GraspRequest, '/graspnet/grasp_request')
        
        self.get_logger().info('[*] YOLO Detection Node started')
        self.get_logger().info('[*] Service available at: /yolo/detect_object')
        self.get_logger().info('[*] Waiting for GraspNet service at: /graspnet/grasp_request')
    
    def camera_callback(self, color_msg, depth_msg, camera_info_msg):
        """Synchronized callback for camera data."""
        try:
            # Convert color image
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Store synchronized data
            with self.data_lock:
                self.latest_color_image = color_image
                self.latest_depth_image = depth_image
                self.latest_camera_info = camera_info_msg
                
        except Exception as e:
            self.get_logger().error(f'Error in camera callback: {e}')
    
    def handle_detection_request(self, request, response):
        """Handle object detection service request."""
        object_name = request.object_name
        self.get_logger().info(f'[*] Received detection request for object: {object_name}')
        
        try:
            # Get current camera data (synchronized)
            with self.data_lock:
                if self.latest_color_image is None or self.latest_depth_image is None or self.latest_camera_info is None:
                    response.success = False
                    response.message = 'Camera data not available (waiting for synchronized messages)'
                    self.get_logger().error(response.message)
                    return response
                
                color_img = self.latest_color_image.copy()
                depth_img = self.latest_depth_image.copy()
                cam_info = self.latest_camera_info
            
            # Detect object using YOLO
            detection_result = self.detect_object(object_name, color_img, depth_img, cam_info)
            
            if not detection_result['success']:
                response.success = False
                response.message = detection_result['message']
                self.get_logger().warning(f'Detection failed: {response.message}')
                return response
            
            # Populate detection response
            response.bbox_2d = detection_result['bbox_2d']
            response.confidence = detection_result['confidence']
            response.success = True
            response.message = f"Object '{object_name}' detected successfully"
            
            self.get_logger().info(f'[*] Object detected: {object_name}, confidence: {response.confidence:.3f}')
            self.get_logger().info(f'[*] 2D bbox: {response.bbox_2d}')
            
            # Now call GraspNet service with 2D bbox
            self.get_logger().info('[*] Requesting grasp pose from GraspNet...')
            grasp_response = self.request_grasp(object_name, response.bbox_2d)
            
            if grasp_response is not None and grasp_response.success:
                self.get_logger().info(f'[*] Grasp pose received: score={grasp_response.score:.3f}, width={grasp_response.gripper_width:.3f}m')
                # Note: The grasp pose is already published by GraspNet node to /graspnet/grasps
                # We just log it here, the response from detection service doesn't include grasp
            else:
                self.get_logger().warning('[*] Failed to get grasp pose from GraspNet')
            
            return response
            
        except Exception as e:
            response.success = False
            response.message = f'Error during detection: {str(e)}'
            self.get_logger().error(response.message)
            import traceback
            self.get_logger().error(traceback.format_exc())
            return response
    
    def detect_object(self, object_name, color_img, depth_img, cam_info):
        """Detect specific object and return its 2D bounding box."""
        result = {
            'success': False,
            'message': '',
            'bbox_2d': [],
            'confidence': 0.0
        }
        
        try:
            # Set detection classes dynamically for YOLO-World (open-vocabulary)
            # This allows detecting any object by name, not limited to pre-trained classes
            self.get_logger().info(f'[*] Setting YOLO-World to detect: {object_name}')
            self.yolo_model.set_classes([object_name])
            
            # Run YOLO-World inference (only detects the specified object)
            self.get_logger().info('[*] Running YOLO-World inference...')
            results = self.yolo_model(source=color_img, device="cuda:0", verbose=False)
            detection = results[0]
            
            if detection is None or len(detection.boxes) == 0:
                result['message'] = f"Object '{object_name}' not detected in image"
                return result
            
            # Extract detection results
            # Note: YOLO-World only returns detections for the specified class
            boxes = detection.boxes.xyxy.cpu().numpy()
            confidences = detection.boxes.conf.cpu().numpy()
            
            # Get the detection with highest confidence
            # (YOLO-World already filtered to only the requested object)
            best_idx = confidences.argmax()
            best_conf = float(confidences[best_idx])
            
            self.get_logger().info(f'[*] Found {len(boxes)} instance(s) of "{object_name}", best confidence: {best_conf:.3f}')
            
            # Check confidence threshold
            if best_conf < 0.3:  # Lower threshold for YOLO-World as it's open-vocabulary
                result['message'] = f"Object '{object_name}' detected with low confidence: {best_conf:.3f} (threshold: 0.3)"
                return result
            
            # Get 2D bounding box in pixel coordinates
            x1, y1, x2, y2 = boxes[best_idx]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            result['success'] = True
            result['message'] = f"Object '{object_name}' detected successfully"
            result['bbox_2d'] = [float(x1), float(y1), float(x2), float(y2)]
            result['confidence'] = float(best_conf)
            
            self.get_logger().info(f'[*] 2D bbox (pixels): x=[{x1}, {x2}], y=[{y1}, {y2}]')
            
            return result
            
        except Exception as e:
            result['message'] = f'Detection error: {str(e)}'
            return result
    
    def request_grasp(self, object_name, bbox_2d):
        """Request grasp pose from GraspNet service."""
        try:
            # Wait for service to be available
            if not self.grasp_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warning('GraspNet service not available')
                return None
            
            # Create request with 2D bounding box
            request = GraspRequest.Request()
            request.object_name = object_name
            request.bbox_2d = bbox_2d
            
            # Call service synchronously
            future = self.grasp_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
            
            if future.done():
                response = future.result()
                return response
            else:
                self.get_logger().error('GraspNet service call timed out')
                return None
                
        except Exception as e:
            self.get_logger().error(f'Error calling GraspNet service: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = YOLODetectionNode()
        node.get_logger().info('[*] YOLO Detection Node ready, spinning...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

