""" Demo for ROS2 to show prediction results from real-time camera input.
    Based on demo.py by chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import signal
import rclpy
from rclpy.node import Node
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32, Header

try:
    from graspnet_msgs.msg import GraspPose
    from graspnet_msgs.srv import GraspRequest
except Exception as import_err:
    print("[!] Missing ROS2 message types from 'graspnet_msgs'.")
    print("    Please build and source the message package before running:")
    print("    1) cd src && colcon build")
    print("    2) source src/install/setup.bash")
    raise import_err
from cv_bridge import CvBridge
import threading
import time
from scipy.spatial.transform import Rotation as R

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo as GrCameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--no_collision', action='store_true', help='Disable collision detection for debugging')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--headless', action='store_true', help='Run in headless mode (no interactive visualization)')
parser.add_argument('--color_topic', type=str, default='/camera/color/image_raw', help='Color image topic')
parser.add_argument('--depth_topic', type=str, default='/camera/depth/image_raw', help='Depth image topic')
parser.add_argument('--camera_info_topic', type=str, default='/camera/color/camera_info', help='Camera info topic')
parser.add_argument('--grasp_topic', type=str, default='/graspnet/grasps', help='Topic to publish grasp results')
parser.add_argument('--processing_interval', type=float, default=2.0, help='Interval between processing frames in seconds [default: 2.0]')
cfgs = parser.parse_args()


class GraspNetRos2Node(Node):
    """ROS2 node for GraspNet real-time prediction."""
    
    def __init__(self, net):
        super().__init__('graspnet_ros2_node')
        
        self.net = net
        self.bridge = CvBridge()
        
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self.data_lock = threading.Lock()
        
        self.sub_color = self.create_subscription(
            Image,
            cfgs.color_topic,
            self.color_callback,
            10)
        
        self.sub_depth = self.create_subscription(
            Image,
            cfgs.depth_topic,
            self.depth_callback,
            10)
        
        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            cfgs.camera_info_topic,
            self.camera_info_callback,
            10)
        
        self.grasp_pub = self.create_publisher(GraspPose, cfgs.grasp_topic, 10)
        
        # Create grasp request service
        self.grasp_service = self.create_service(
            GraspRequest,
            '/graspnet/grasp_request',
            self.handle_grasp_request)
        
        self.get_logger().info('[*] GraspNet ROS2 node started')
        self.get_logger().info(f'[*] Subscribing to: {cfgs.color_topic}, {cfgs.depth_topic}, {cfgs.camera_info_topic}')
        self.get_logger().info(f'[*] Publishing grasps to: {cfgs.grasp_topic}')
        self.get_logger().info(f'[*] Service available at: /graspnet/grasp_request')
        self.get_logger().info('[*] Node will process frames only when service is called')
    
    def color_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            with self.data_lock:
                self.color_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {e}')
    
    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.data_lock:
                self.depth_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    def camera_info_callback(self, msg):
        with self.data_lock:
            self.camera_info = msg
    
    def handle_grasp_request(self, request, response):
        """Handle grasp request service call."""
        object_name = request.object_name
        bbox_2d = list(request.bbox_2d) if len(request.bbox_2d) == 4 else None
        object_center_3d = list(request.object_center_3d) if len(request.object_center_3d) == 3 else None
        
        self.get_logger().info(f'[*] Received grasp request for object: {object_name}')
        if bbox_2d:
            self.get_logger().info(f'[*] 2D bbox constraint (pixels): {bbox_2d}')
        if object_center_3d:
            self.get_logger().info(f'[*] 3D center constraint (meters): [{object_center_3d[0]:.3f}, {object_center_3d[1]:.3f}, {object_center_3d[2]:.3f}]')
        
        try:
            # Get current camera data
            with self.data_lock:
                if self.color_image is None or self.depth_image is None or self.camera_info is None:
                    response.success = False
                    response.message = 'Camera data not available'
                    self.get_logger().error(response.message)
                    return response
                
                color_img = self.color_image.copy()
                depth_img = self.depth_image.copy()
                cam_info = self.camera_info
            
            # Process frame with 2D bbox and 3D center constraints
            grasp_result = self.process_frame(color_img, depth_img, cam_info, bbox_2d=bbox_2d, object_center_3d=object_center_3d)
            
            if grasp_result is None:
                response.success = False
                response.message = 'Failed to generate grasp pose'
                self.get_logger().warning(response.message)
                return response
            
            best_grasp = grasp_result['grasp']
            
            # Populate response
            response.grasp_pose = self.grasp_to_pose_stamped(best_grasp)
            response.gripper_width = float(best_grasp.width)
            response.score = float(best_grasp.score)
            response.success = True
            response.message = f"Grasp pose generated for '{object_name}'"
            
            # Also publish to topic (maintaining existing behavior)
            grasp_pose_msg = GraspPose()
            grasp_pose_msg.target_pose = response.grasp_pose
            grasp_pose_msg.gripper_width = response.gripper_width
            self.grasp_pub.publish(grasp_pose_msg)
            
            self.get_logger().info(f'[*] Grasp pose generated: score={response.score:.3f}, width={response.gripper_width:.3f}m')
            
            return response
            
        except Exception as e:
            response.success = False
            response.message = f'Error during grasp generation: {str(e)}'
            self.get_logger().error(response.message)
            import traceback
            self.get_logger().error(traceback.format_exc())
            return response
    
    def grasp_to_pose_stamped(self, grasp):
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        
        trans = grasp.translation.astype(float)
        msg.pose.position.x = float(trans[0])
        msg.pose.position.y = float(trans[1])
        msg.pose.position.z = float(trans[2])
        
        rotation_matrix = grasp.rotation_matrix.reshape(3, 3)
        quat = R.from_matrix(rotation_matrix).as_quat()
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        
        return msg
    
    def process_frame(self, color_img, depth_img, cam_info, bbox_2d=None, object_center_3d=None):
        """Process one frame and return grasp results.
        
        Args:
            color_img: RGB image
            depth_img: Depth image
            cam_info: Camera info
            bbox_2d: Optional 2D bounding box [x_min, y_min, x_max, y_max] in pixels to filter grasps
            object_center_3d: Optional 3D center position [x, y, z] in meters to filter grasps by distance
        """
        start_time = time.time()
        
        try:
            color = color_img.astype(np.float32) / 255.0
            depth = depth_img.astype(np.float32)
            
            K = cam_info.k
            fx, fy = K[0], K[4]
            cx, cy = K[2], K[5]
            height, width = depth.shape
            
            depth_max = depth.max()
            scale_factor = 1000.0 if depth_max > 10 else 1.0
            
            depth_meters = depth / scale_factor
            valid_depth = (depth_meters > 0.3) & (depth_meters < 1.5)
            workspace_mask = valid_depth
            
            camera = GrCameraInfo(width, height, fx, fy, cx, cy, scale=scale_factor)
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
            
            mask = (workspace_mask & (depth > 0))
            cloud_masked = cloud[mask]
            color_masked = color[mask]
            
            if len(cloud_masked) == 0:
                self.get_logger().warning('No valid points in point cloud')
                return None
            
            # Apply table filtering to whole scene (no 3D bbox filtering)
            if len(cloud_masked) > 0:
                z_coords = cloud_masked[:, 2]
                z_median = np.median(z_coords)
                z_min = max(z_median - 0.5, 0.3)
                z_max = min(z_median + 0.5, 2.0)
                
                x_coords = cloud_masked[:, 0]
                y_coords = cloud_masked[:, 1]
                x_median, y_median = np.median(x_coords), np.median(y_coords)
                x_std, y_std = np.std(x_coords), np.std(y_coords)
                
                x_threshold = 0.8 * x_std
                y_threshold = 2.0 * y_std
                xy_outlier_mask = (np.abs(x_coords - x_median) <= x_threshold) & (np.abs(y_coords - y_median) <= y_threshold)
                
                xy_distances = np.sqrt(x_coords**2 + y_coords**2)
                xy_distance_mask = xy_distances <= 0.8
                
                x_q1, x_q3 = np.percentile(x_coords, [25, 75])
                y_q1, y_q3 = np.percentile(y_coords, [25, 75])
                x_iqr, y_iqr = x_q3 - x_q1, y_q3 - y_q1
                x_iqr_mask = (x_coords >= x_q1 - 1.5*x_iqr) & (x_coords <= x_q3 + 1.5*x_iqr)
                y_iqr_mask = (y_coords >= y_q1 - 1.5*y_iqr) & (y_coords <= y_q3 + 1.5*y_iqr)
                iqr_mask = x_iqr_mask & y_iqr_mask
                
                table_mask = (z_coords >= z_min) & (z_coords <= z_max) & xy_outlier_mask & xy_distance_mask & iqr_mask
                cloud_masked = cloud_masked[table_mask]
                color_masked = color_masked[table_mask]
            
            if len(cloud_masked) == 0:
                self.get_logger().warning('No valid points in point cloud after filtering')
                return None
            
            if len(cloud_masked) >= cfgs.num_point:
                idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            cloud_sampled = cloud_masked[idxs]
            color_sampled = color_masked[idxs]
            
            end_points = dict()
            cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            cloud_sampled_tensor = cloud_sampled_tensor.to(device)
            
            end_points['point_clouds'] = cloud_sampled_tensor
            end_points['cloud_colors'] = color_sampled
            
            try:
                vis_cloud = o3d.geometry.PointCloud()
                vis_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
                vis_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
                output_dir = '../output'
                os.makedirs(output_dir, exist_ok=True)
                ply_filename = f'{output_dir}/input_pointcloud_{int(time.time())}.ply'
                o3d.io.write_point_cloud(ply_filename, vis_cloud)
            except Exception as e:
                self.get_logger().warning(f'Error creating visualization point cloud: {e}')
            
            inference_start_time = time.time()
            with torch.no_grad():
                end_points = self.net(end_points)
                grasp_preds = pred_decode(end_points)
            inference_time = time.time() - inference_start_time
            
            gg_array = grasp_preds[0].detach().cpu().numpy()
            gg = GraspGroup(gg_array)
            
            if not cfgs.no_collision and cfgs.collision_thresh > 0:
                mfcdetector = ModelFreeCollisionDetector(cloud_masked, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]
            
            # Filter grasps by 3D distance to object center
            # This is more robust and lenient than 2D projection filtering
            if object_center_3d is not None and len(object_center_3d) == 3 and len(gg) > 0:
                obj_center = np.array(object_center_3d)
                self.get_logger().info(f'[*] Filtering grasps by 3D distance to object center: [{obj_center[0]:.3f}, {obj_center[1]:.3f}, {obj_center[2]:.3f}]m')
                
                # Distance threshold: 0.2 meters (lenient)
                distance_threshold = 0.2
                
                # Filter grasps whose 3D position is close to object center
                valid_grasps = []
                for i in range(len(gg)):
                    grasp = gg[i]
                    trans = grasp.translation  # 3D position in camera frame
                    
                    # Calculate Euclidean distance between grasp center and object center
                    distance = np.linalg.norm(trans - obj_center)
                    
                    if distance <= distance_threshold:
                        valid_grasps.append(i)
                        self.get_logger().debug(f'[*] Grasp {i}: distance={distance:.3f}m (valid)')
                
                if len(valid_grasps) > 0:
                    gg = gg[valid_grasps]
                    self.get_logger().info(f'[*] Grasps after 3D distance filter: {len(gg)} (threshold: {distance_threshold}m)')
                else:
                    self.get_logger().warning(f'[*] No grasps found within {distance_threshold}m of object center, trying lenient threshold...')
                    # Try a more lenient threshold if no grasps found
                    distance_threshold = 0.3
                    valid_grasps = []
                    for i in range(len(gg)):
                        grasp = gg[i]
                        trans = grasp.translation
                        distance = np.linalg.norm(trans - obj_center)
                        if distance <= distance_threshold:
                            valid_grasps.append(i)
                    
                    if len(valid_grasps) > 0:
                        gg = gg[valid_grasps]
                        self.get_logger().info(f'[*] Grasps after lenient 3D filter: {len(gg)} (threshold: {distance_threshold}m)')
                    else:
                        self.get_logger().warning(f'[*] Still no grasps found within {distance_threshold}m of object center')
            
            gg.nms()
            gg.sort_by_score()
            
            # Keep only the best grasp for visualization
            if not cfgs.headless and len(gg) > 0:
                self.visualize_grasps(gg[:1], vis_cloud)
            
            processing_time = time.time() - start_time
            
            if len(gg) == 0:
                self.get_logger().warning('No grasps found after filtering')
                return None
            
            best_grasp = gg[0]
            grasp_data = {
                'grasp': best_grasp,
                'processing_time': processing_time,
                'inference_time': inference_time,
                'timestamp': time.time()
            }

            self.get_logger().info(f'Grasp found: score={best_grasp.score:.3f}, width={best_grasp.width:.3f}m, inference={inference_time:.2f}s, total={processing_time:.2f}s best grasp = {best_grasp}')
            return grasp_data
            
        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def visualize_grasps(self, gg, cloud):
        """Visualize predicted grasps."""
        grippers = gg.to_open3d_geometry_list()
        
        # Check for headless mode
        headless = cfgs.headless or os.environ.get('OPEN3D_HEADLESS', '').lower() in ('1', 'true', 'yes')
        
        if not headless:
            try:
                o3d.visualization.draw_geometries([cloud, *grippers])
                pass
            except Exception as e:
                self.get_logger().warning(f'Visualization error: {e}')


def get_net():
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, checkpoint['epoch']))
    net.eval()
    return net


_node_instance = None

def signal_handler(sig, frame):
    global _node_instance
    print("\n[*] Ctrl-C detected, shutting down gracefully...", flush=True)
    try:
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
    except:
        pass


def main():
    global _node_instance
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    rclpy.init()
    
    node = None
    try:
        print("[*] Loading GraspNet model...")
        net = get_net()
        node = GraspNetRos2Node(net)
        _node_instance = node
        
        print("[*] GraspNet node ready, waiting for service calls...")
        print("[*] Service available at: /graspnet/grasp_request")
        print("[*] Press Ctrl+C to stop")
        
        # Spin and wait for service calls
        rclpy.spin(node)
    
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    except rclpy.executors.ExternalShutdownException:
        print("\n[*] ROS2 shutdown requested externally...")
    except Exception as e:
        print(f"\n[*] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        _node_instance = None
        
        if node is not None:
            try:
                node.destroy_node()
            except Exception as e:
                print(f"[*] Error destroying node: {e}")
        
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
        
        print("[*] Shutdown complete.")


if __name__=='__main__':
    import cv2
    main()

