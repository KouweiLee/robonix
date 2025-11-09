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
except Exception as import_err:
    print("[!] Missing ROS2 message type 'graspnet_msgs/GraspPose'.")
    print("    Please build and source the message package before running:")
    print("    1) cd src && colcon build")
    print("    2) source src/install/setup.bash")
    raise import_err
from cv_bridge import CvBridge
import threading
import time
import json
import queue
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
        self.grasp_result_queue = queue.Queue(maxsize=5)
        self.running = True
        self.processing_thread = None
        self.publishing_thread = None
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
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.publishing_thread = threading.Thread(target=self.publishing_loop, daemon=True)
        self.processing_thread.start()
        self.publishing_thread.start()
        
        self.get_logger().info('[*] GraspNet ROS2 node started')
        self.get_logger().info(f'[*] Subscribing to: {cfgs.color_topic}, {cfgs.depth_topic}, {cfgs.camera_info_topic}')
        self.get_logger().info(f'[*] Publishing grasps to: {cfgs.grasp_topic}')
        self.get_logger().info(f'[*] Processing interval: {cfgs.processing_interval}s')
    
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
    
    def processing_loop(self):
        while self.running:
            try:
                with self.data_lock:
                    if self.color_image is None or self.depth_image is None or self.camera_info is None:
                        for _ in range(10):
                            if not self.running:
                                break
                            time.sleep(0.01)
                        continue
                    color_img = self.color_image.copy()
                    depth_img = self.depth_image.copy()
                    cam_info = self.camera_info
                
                if not self.running:
                    break
                
                grasp_result = self.process_frame(color_img, depth_img, cam_info)
                
                if grasp_result is not None:
                    try:
                        self.grasp_result_queue.put_nowait(grasp_result)
                    except queue.Full:
                        self.get_logger().warn('Grasp result queue full, dropping oldest result')
                        try:
                            self.grasp_result_queue.get_nowait()
                            self.grasp_result_queue.put_nowait(grasp_result)
                        except queue.Empty:
                            pass
                
                elapsed = 0.0
                check_interval = 0.1
                while elapsed < cfgs.processing_interval and self.running:
                    time.sleep(min(check_interval, cfgs.processing_interval - elapsed))
                    elapsed += check_interval
                
            except Exception as e:
                if self.running:
                    self.get_logger().error(f'Error in processing loop: {e}')
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                for _ in range(10):
                    if not self.running:
                        break
                    time.sleep(0.1)
    
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
    
    def publishing_loop(self):
        while self.running:
            try:
                try:
                    grasp_data = self.grasp_result_queue.get(timeout=0.1)
                    
                    if not self.running:
                        break
                    
                    best_grasp = grasp_data['grasp']
                    pose_msg = self.grasp_to_pose_stamped(best_grasp)
                    gripper_width = float(best_grasp.width)
                    
                    grasp_pose_msg = GraspPose()
                    grasp_pose_msg.target_pose = pose_msg
                    grasp_pose_msg.gripper_width = gripper_width
                    self.grasp_pub.publish(grasp_pose_msg)
                except queue.Empty:
                    continue
                    
            except Exception as e:
                if self.running:
                    self.get_logger().error(f'Error in publishing loop: {e}')
                    import traceback
                    self.get_logger().error(traceback.format_exc())
                for _ in range(10):
                    if not self.running:
                        break
                    time.sleep(0.01)
    
    def process_frame(self, color_img, depth_img, cam_info):
        """Process one frame and return grasp results."""
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
                self.get_logger().warning('No valid points in point cloud after table filtering')
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
            
            gg = gg[:1] # only keep the best grasp
            if not cfgs.headless:
                self.visualize_grasps(gg, vis_cloud)
            
            gg.nms()
            gg.sort_by_score()
            # gg = gg[:50] 
            
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
_shutdown_flag = threading.Event()

def signal_handler(sig, frame):
    global _node_instance, _shutdown_flag
    print("\n[*] Ctrl-C detected, shutting down gracefully...", flush=True)
    _shutdown_flag.set()
    if _node_instance is not None:
        _node_instance.running = False
    try:
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()
    except:
        pass


def main():
    global _node_instance, _shutdown_flag
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    rclpy.init()
    
    node = None
    ros2_thread = None
    executor = None
    try:
        print("[*] Loading GraspNet model...")
        net = get_net()
        node = GraspNetRos2Node(net)
        _node_instance = node
        
        print("[*] Starting continuous processing (press Ctrl+C to stop)...")
        print(f"[*] Processing frame every {cfgs.processing_interval} seconds")
        print("[*] Grasp results will be published to:", cfgs.grasp_topic)
        
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        
        def ros2_spin_thread():
            try:
                while rclpy.ok() and node.running and not _shutdown_flag.is_set():
                    executor.spin_once(timeout_sec=0.1)
            except Exception:
                pass
        
        ros2_thread = threading.Thread(target=ros2_spin_thread, daemon=True)
        ros2_thread.start()
        
        try:
            import select
            use_select = hasattr(select, 'select')
            
            while rclpy.ok() and node.running and not _shutdown_flag.is_set():
                if use_select:
                    try:
                        select.select([], [], [], 0.01)
                    except (KeyboardInterrupt, SystemExit):
                        print("\n[*] Interrupt detected...", flush=True)
                        _shutdown_flag.set()
                        node.running = False
                        break
                else:
                    try:
                        time.sleep(0.01)
                    except KeyboardInterrupt:
                        print("\n[*] KeyboardInterrupt...", flush=True)
                        _shutdown_flag.set()
                        node.running = False
                        break
        except (KeyboardInterrupt, SystemExit):
            print("\n[*] Interrupt caught...", flush=True)
            _shutdown_flag.set()
            node.running = False
    
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        if node is not None:
            node.running = False
    except rclpy.executors.ExternalShutdownException:
        print("\n[*] ROS2 shutdown requested externally...")
        if node is not None:
            node.running = False
    except Exception as e:
        print(f"\n[*] Error: {e}")
        import traceback
        traceback.print_exc()
        if node is not None:
            node.running = False
    finally:
        _shutdown_flag.set()
        _node_instance = None
        
        if node is not None:
            print("[*] Stopping threads...", flush=True)
            node.running = False
            
            if ros2_thread is not None and ros2_thread.is_alive():
                ros2_thread.join(timeout=0.5)
            
            if node.processing_thread is not None and node.processing_thread.is_alive():
                node.processing_thread.join(timeout=0.5)
            
            if node.publishing_thread is not None and node.publishing_thread.is_alive():
                node.publishing_thread.join(timeout=0.5)
            
            try:
                node.destroy_node()
            except Exception as e:
                print(f"[*] Error destroying node: {e}")
        
        if executor is not None:
            try:
                executor.shutdown()
            except:
                pass
        
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
        
        print("[*] Shutdown complete.")


if __name__=='__main__':
    import cv2
    main()

