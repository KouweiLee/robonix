# SPDX-License-Identifier: MulanPSL-2.0
"""
Background contention processes for the scheduling benchmark.

These processes simulate always-running robot subsystems that consume CPU and/or
GPU resources, creating realistic resource contention with the active skill.
Each background worker runs in its own process and can be started/stopped
by the benchmark runner.

Background workers that are skill dependencies publish their output to ROS2 
topics after each iteration. Skills subscribe and wait for this data before 
each workload iteration, modeling real data flow.
"""

import argparse
import json
import os
import signal
import sys
import time
import logging
import threading
from typing import Optional, List, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from scheduler_benchmark.workloads import (
    HeavyPerceptionWorkload,
    CameraStreamWorkload,
    LidarStreamWorkload,
    SLAMWorkload,
    LidarSLAMWorkload,
    SpeechWorkload,
    MotionPlanWorkload,
    CPUGemm,
    GPUGemm,
    CPUPointCloud,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark.bg")


class PublishingBackgroundWorker:
    """
    Background worker that publishes workload output to ROS2 topics.
    Each iteration runs the workload and then publishes to all configured topics.
    """

    def __init__(self, name: str, workload, target_rate_hz: float,
                 output_topics: List[str]):
        """
        Args:
            name: Worker name
            workload: Workload instance
            target_rate_hz: Loop rate
            output_topics: List of topic strings to publish to
        """
        self.name = name
        self.workload = workload
        self.target_period = 1.0 / target_rate_hz if target_rate_hz > 0 else 0
        self.output_topics = output_topics
        self._running = True
        self._iteration = 0
        self._node: Optional[Node] = None
        self._pubs: List[Any] = []

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"{self.name}: Received signal {signum}, shutting down")
        self._running = False

    def _init_ros2(self):
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(f"{self.name}_node")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        for topic in self.output_topics:
            pub = self._node.create_publisher(String, topic, qos)
            self._pubs.append(pub)
        
        # Brief spin to allow discovery
        for _ in range(20):
            rclpy.spin_once(self._node, timeout_sec=0.05)
        logger.info(f"{self.name}: Publishing to {self.output_topics}")

    def _publish_output(self):
        if not self._pubs or self._node is None:
            return
        
        # Publish to all topics
        msg = String()
        # Create a payload to simulate realistic data transfer size (e.g. 50KB)
        payload = "x" * 50000 
        
        msg.data = json.dumps({
            "source": self.name,
            "iteration": self._iteration,
            "timestamp": time.perf_counter(),
            "payload": payload, 
        })
        for pub in self._pubs:
            pub.publish(msg)

    def run(self):
        """Main loop: run workload, publish output, rate limit."""
        self._init_ros2()
        logger.info(f"{self.name}: Started (PID={os.getpid()}, period={self.target_period:.3f}s)")
        while self._running:
            start = time.perf_counter()
            try:
                self.workload.run()
                self._iteration += 1
                self._publish_output()
            except Exception as e:
                logger.warning(f"{self.name}: Workload error: {e}")

            remaining = self.target_period - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)

        if self._node is not None:
            self._node.destroy_node()
        logger.info(f"{self.name}: Stopped after {self._iteration} iterations")


def run_perception_bg(rate_hz: float = 30.0):
    """Run background perception pipeline (camera + preprocessing)."""
    worker = PublishingBackgroundWorker(
        "perception_bg", CameraStreamWorkload(), rate_hz,
        output_topics=[
            "/robot1/bench/camera/rgb",
            "/robot1/bench/camera/depth",
        ],
    )
    worker.run()


def run_lidar_slam_bg(rate_hz: float = 10.0):
    """Run background merged LiDAR driver + SLAM pipeline."""
    worker = PublishingBackgroundWorker(
        "lidar_slam_bg", LidarSLAMWorkload(), rate_hz,
        output_topics=[
            "/robot1/bench/lidar/points",
            "/robot1/bench/slam/map",
            "/robot1/bench/slam/pose",
        ],
    )
    worker.run()


def run_speech_bg(rate_hz: float = 4.0):
    """Run background speech processing."""
    worker = PublishingBackgroundWorker(
        "speech_bg", SpeechWorkload(), rate_hz,
        output_topics=["/robot1/bench/speech/command"],
    )
    worker.run()


def run_motion_plan_bg(rate_hz: float = 8.0):
    """Run background motion planning."""
    worker = PublishingBackgroundWorker(
        "motion_plan_bg", MotionPlanWorkload(), rate_hz,
        output_topics=["/robot1/bench/motion/plan"],
    )
    worker.run()


def run_cpu_noise_bg(rate_hz: float = 10.0):
    """Run background CPU-only noise workload."""
    worker = PublishingBackgroundWorker(
        "cpu_noise_bg", CPUGemm(size=512), rate_hz,
        output_topics=[],
    )
    worker.run()


def run_gpu_noise_bg(rate_hz: float = 10.0):
    """Run background GPU-only noise workload."""
    worker = PublishingBackgroundWorker(
        "gpu_noise_bg", GPUGemm(size=1024), rate_hz,
        output_topics=[],
    )
    worker.run()


# Entry points for subprocess spawning
WORKERS = {
    "perception": run_perception_bg,
    "lidar_slam": run_lidar_slam_bg,
    "speech": run_speech_bg,
    "motion_plan": run_motion_plan_bg,
    "cpu_noise": run_cpu_noise_bg,
    "gpu_noise": run_gpu_noise_bg,
}


def main():
    """CLI entry point for running a background worker."""
    parser = argparse.ArgumentParser(description="Background contention worker")
    parser.add_argument("worker", choices=WORKERS.keys(), help="Worker type")
    parser.add_argument("--rate", type=float, default=0,
                        help="Target rate in Hz (0 = use default)")
    args = parser.parse_args()

    fn = WORKERS[args.worker]
    if args.rate > 0:
        fn(rate_hz=args.rate)
    else:
        fn()


if __name__ == "__main__":
    main()
