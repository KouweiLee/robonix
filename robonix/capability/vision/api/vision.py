import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import threading
import message_filters

import numpy as np
import cv2


def _safe_imgmsg_to_cv2(bridge: CvBridge, msg: Image, want: str = "rgb"):
    """
    Convert ROS Image -> numpy array in a way that avoids cv_bridge internal cvtColor.

    want:
      - "rgb": return RGB uint8 image if possible
      - "bgr": return BGR uint8 image if possible
      - "passthrough": return raw decoded array without color conversion

    Notes:
      - Always uses desired_encoding='passthrough' to avoid cv_bridge C++ cvtColor,
        then converts in Python based on msg.encoding.
    """
    enc = (msg.encoding or "").lower()

    # 1) Always passthrough: avoid cv_bridge's internal color conversion (core-dump path)
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    # 2) Make an owned copy to avoid lifetime issues (esp. with message_filters + threading)
    #    This is cheap relative to stability and is a "temporary safe" measure.
    img = img.copy()

    if want == "passthrough":
        return img

    # 3) Convert color in Python, based on actual encoding
    #    Handle common encodings; otherwise return as-is with a warning at call site.
    if enc == "rgb8":
        if want == "rgb":
            return img
        else:  # want == "bgr"
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if enc == "bgr8":
        if want == "bgr":
            return img
        else:  # want == "rgb"
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if enc in ("rgba8",):
        if want == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    if enc in ("bgra8",):
        if want == "bgr":
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    if enc in ("mono8", "8uc1"):
        # Return 3-channel image for RGB/BGR requests (often what downstream expects)
        if want == "rgb":
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Unknown encoding: return raw; caller may handle it.
    return img


class CameraImageGetter(Node):
    def __init__(self, topic_name):
        super().__init__('camera_image_getter')
        self.image = None
        self._event = threading.Event()
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10
        )
        self.cv_bridge = CvBridge()

    def image_callback(self, msg: Image):
        try:
            # Temporary safe: do NOT request 'bgr8' directly from cv_bridge.
            # Return RGB by default (adjust to "bgr" if your downstream expects BGR).
            self.image = _safe_imgmsg_to_cv2(self.cv_bridge, msg, want="bgr8")
            self.get_logger().info(f"Got camera image. encoding={msg.encoding}")
            self._event.set()
            self.destroy_subscription(self.subscription)
        except Exception as e:
            self.get_logger().error(f"Error converting color image: {e}")


class CameraRGBDGetter(Node):
    def __init__(self, rgb_topic, depth_topic):
        super().__init__('camera_rgbd_getter')
        self.rgb_image = None
        self.depth_image = None
        self._event = threading.Event()
        self.cv_bridge = CvBridge()
        self.rgb_sub = message_filters.Subscriber(self, Image, rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 3, 0.02
        )
        self.sync.registerCallback(self.callback)

    def callback(self, rgb_msg: Image, depth_msg: Image):
        try:
            # RGB: safe conversion (passthrough + python-side conversion)
            self.rgb_image = _safe_imgmsg_to_cv2(self.cv_bridge, rgb_msg, want="bgr8")

            # Depth: passthrough is correct; also copy to be safe
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").copy()

            self.get_logger().info(
                f"Got synchronized RGB and depth images. rgb_enc={rgb_msg.encoding}, depth_enc={depth_msg.encoding}"
            )
            self._event.set()

            # unsubscribe once we have data
            self.destroy_subscription(self.rgb_sub.sub)
            self.destroy_subscription(self.depth_sub.sub)
        except Exception as e:
            self.get_logger().error(f"Error converting RGBD images: {e}")


class CameraInfoGetter(Node):
    def __init__(self, topic_name):
        super().__init__('camera_info_getter')
        self.camera_info = None
        self._event = threading.Event()
        self.subscription = self.create_subscription(
            CameraInfo,
            topic_name,
            self.camera_info_callback,
            10
        )

    def camera_info_callback(self, msg: CameraInfo):
        """
        Get camera parameters through a callback function.
        """
        self.camera_info = {
            'k': msg.k,           # 3x3 camera intrinsic matrix
            'p': msg.p,           # 3x4 projection matrix
            'd': msg.d,           # distortion coefficients
            'r': msg.r,           # 3x3 rotation matrix
            'width': msg.width,   # image width
            'height': msg.height, # image height
            'roi': msg.roi        # region of interest
        }
        self.get_logger().info("Got camera info.")
        self._event.set()
        self.destroy_subscription(self.subscription)
