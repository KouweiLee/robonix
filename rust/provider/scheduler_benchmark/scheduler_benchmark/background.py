# SPDX-License-Identifier: MulanPSL-2.0
"""
Background contention processes for the scheduling benchmark.

These processes simulate always-running robot subsystems that consume CPU and/or
GPU resources, creating realistic resource contention with the active skill.
Each background worker runs in its own process and can be started/stopped
by the benchmark runner.

Background workers:
  - PerceptionBG: Camera image processing + object detection (CPU+GPU)
  - SLAMBG: Scan matching + pose graph optimization (CPU-heavy)
  - SpeechBG: Audio feature extraction + model inference (CPU+GPU)
  - MotionPlanBG: Collision checking + trajectory optimization (CPU-heavy)
"""

import argparse
import os
import signal
import sys
import time
import logging
from typing import Optional

from scheduler_benchmark.workloads import (
    PerceptionWorkload,
    SLAMWorkload,
    SpeechWorkload,
    MotionPlanWorkload,
    CPUGemm,
    CPUPointCloud,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark.bg")


class BackgroundWorker:
    """
    Base class for background contention workers.
    Runs a workload in a tight loop at a configurable rate (iterations/sec).
    Gracefully shuts down on SIGTERM/SIGINT.
    """

    def __init__(self, name: str, workload, target_rate_hz: float = 10.0):
        self.name = name
        self.workload = workload
        self.target_period = 1.0 / target_rate_hz if target_rate_hz > 0 else 0
        self._running = True
        self._iteration = 0

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"{self.name}: Received signal {signum}, shutting down")
        self._running = False

    def run(self):
        """Main loop: run workload at target rate until stopped."""
        logger.info(f"{self.name}: Started (PID={os.getpid()}, period={self.target_period:.3f}s)")
        while self._running:
            start = time.perf_counter()
            try:
                elapsed = self.workload.run()
            except Exception as e:
                logger.warning(f"{self.name}: Workload error: {e}")
                elapsed = 0

            self._iteration += 1

            # Rate limiting: sleep to maintain target period
            remaining = self.target_period - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)

        logger.info(f"{self.name}: Stopped after {self._iteration} iterations")


def run_perception_bg(rate_hz: float = 10.0):
    """Run background perception pipeline (camera + detection + pointcloud)."""
    worker = BackgroundWorker("perception_bg", PerceptionWorkload(), rate_hz)
    worker.run()


def run_slam_bg(rate_hz: float = 5.0):
    """Run background SLAM (scan matching + graph optimization)."""
    worker = BackgroundWorker("slam_bg", SLAMWorkload(), rate_hz)
    worker.run()


def run_speech_bg(rate_hz: float = 3.0):
    """Run background speech processing (mel spectrogram + transformer)."""
    worker = BackgroundWorker("speech_bg", SpeechWorkload(), rate_hz)
    worker.run()


def run_motion_plan_bg(rate_hz: float = 8.0):
    """Run background motion planning (collision check + trajectory opt)."""
    worker = BackgroundWorker("motion_plan_bg", MotionPlanWorkload(), rate_hz)
    worker.run()


# Entry points for subprocess spawning
WORKERS = {
    "perception": run_perception_bg,
    "slam": run_slam_bg,
    "speech": run_speech_bg,
    "motion_plan": run_motion_plan_bg,
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
