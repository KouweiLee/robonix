# SPDX-License-Identifier: MulanPSL-2.0
"""
Dependency topic mapping for benchmark data flow.

Maps std_name (prm::*, srv::*) from benchmark.yaml dependencies to ROS2 topics
that background workers publish. Skills subscribe to these topics and wait for
fresh data before each workload iteration, modeling real data dependencies.
"""

from typing import List

# std_name -> ROS2 topic (published by background workers)
DEPENDENCY_TOPICS = {
    # SLAM Worker outputs
    "srv::bench_slam": "/robot1/bench/slam/map",
    "prm::base.pose.cov": "/robot1/bench/slam/pose",

    # Perception Worker outputs
    "srv::bench_perception": "/robot1/bench/perception/detections",
    "prm::camera.rgb": "/robot1/bench/camera/rgb",
    "prm::camera.depth": "/robot1/bench/camera/depth",

    # Motion Plan Worker outputs
    "srv::bench_motion_plan": "/robot1/bench/motion/plan",
    # Usually prm::base.navigate is handled by the nav stack itself, 
    # but let's assume it gets global path hints or map updates from SLAM/Nav service.
    "prm::base.navigate": "/robot1/bench/nav/goal", 

    # Speech Worker outputs
    "srv::bench_speech": "/robot1/bench/speech/command",
}


def get_dependency_topics(dependencies: list) -> list:
    """
    Get ROS2 topics for skill dependencies that have publishers.

    Args:
        dependencies: List of std_name strings from benchmark config.

    Returns:
        List of topic paths the skill should subscribe to.
    """
    topics = []
    for dep in dependencies or []:
        if dep in DEPENDENCY_TOPICS:
            topics.append(DEPENDENCY_TOPICS[dep])
    return list(dict.fromkeys(topics))  # preserve order, deduplicate


class DependencyWaiter:
    """
    Waits for fresh data from dependency topics before each skill iteration.
    Subscribes to topics and blocks until at least one message received from each.
    """

    def __init__(self, node, topics: List[str]):
        """
        Args:
            node: rclpy Node for creating subscriptions.
            topics: List of ROS2 topic paths to wait on.
        """
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

        self._node = node
        self._topics = topics
        self._received: dict = {t: False for t in topics}
        self._subs = []
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        for topic in topics:
            sub = node.create_subscription(
                String, topic,
                lambda msg, t=topic: self._cb(msg, t),
                qos,
            )
            self._subs.append(sub)

    def _cb(self, msg, topic: str):
        self._received[topic] = True

    def wait_for_dependencies(self, timeout_sec: float = 2.0) -> bool:
        """
        Block until we have received at least one message from each topic,
        or timeout. Resets received flags after success for next iteration.
        
        NOTE: This must be called from a separate thread, NOT the main ROS2 thread,
        as it relies on the main thread to spin and process callbacks.

        Returns:
            True if all dependencies received, False on timeout.
        """
        import time

        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            # Do NOT spin here; let the main node spin loop handle callbacks.
            if all(self._received.values()):
                # Reset for next iteration - we need fresh data each time
                self._received = {t: False for t in self._topics}
                return True
            time.sleep(0.005) # Yield to other threads

        return False
