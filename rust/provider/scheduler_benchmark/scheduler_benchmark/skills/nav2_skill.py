# SPDX-License-Identifier: MulanPSL-2.0
"""
Nav2 Navigation Benchmark Skill (skl::bench_nav)

Simulates a Nav2-based navigation skill that performs path planning and
localization updates in a loop. Each iteration represents one planning cycle
(costmap wavefront expansion + scan matching for localization).

This skill is CPU-intensive and stresses the CPU scheduler. When running
alongside background processes (SLAM, perception, etc.), the robonix-scheduler
should boost this skill's priority to reduce latency and jitter.
"""

import json
import os
import sys
import logging
import time
from typing import Optional

import rclpy
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from scheduler_benchmark.workloads import NavigationWorkload
from scheduler_benchmark.metrics import MetricsCollector
from scheduler_benchmark.dependency_topics import get_dependency_topics, DependencyWaiter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench_nav")

# Default benchmark parameters
DEFAULT_ITERATIONS = 200
DEFAULT_WARMUP = 20
DEFAULT_GRID_SIZE = 300
DEFAULT_SCAN_GRID = 200


class BenchNavSkill(Node):
    """
    Navigation benchmark skill node.

    Listens on start_topic for a start command (JSON with benchmark params),
    runs the navigation workload for N iterations, measures per-iteration
    latency, and publishes results via status_topic.
    """

    def __init__(self):
        super().__init__("bench_nav_skill")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.start_sub = self.create_subscription(
            String,
            "/robot1/skill/bench_nav/start",
            self._on_start,
            qos,
        )
        self.status_pub = self.create_publisher(
            String,
            "/robot1/skill/bench_nav/status",
            qos,
        )
        self._running = False
        logger.info("BenchNavSkill initialized (PID=%d)", os.getpid())

    def _publish_status(self, skill_id: str, state: str,
                        result: Optional[dict] = None, errno: int = 0,
                        error: str = "", message: str = ""):
        msg = String()
        msg.data = json.dumps({
            "skill_id": skill_id,
            "state": state,
            "result": result or {},
            "errno": errno,
            "error": error,
            "message": message,
        })
        self.status_pub.publish(msg)

    def _on_start(self, msg: String):
        """Handle start command."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in start message")
            return

        # Check for terminate signal
        if data.get("terminate", False):
            logger.info("Received terminate signal")
            self._running = False
            return

        if self._running:
            logger.warning("Already running, ignoring start")
            return

        skill_id = data.get("skill_id", "bench_nav_0")
        params = data.get("params", {})

        iterations = params.get("iterations", DEFAULT_ITERATIONS)
        warmup = params.get("warmup", DEFAULT_WARMUP)
        grid_size = params.get("grid_size", DEFAULT_GRID_SIZE)
        scan_grid = params.get("scan_grid", DEFAULT_SCAN_GRID)
        output_file = params.get("output_file", "")
        scheduler_enabled = params.get("scheduler_enabled", False)
        dependencies = params.get("dependencies", [])

        logger.info(
            "Starting benchmark: iterations=%d, warmup=%d, grid=%d, deps=%s, scheduler=%s",
            iterations, warmup, grid_size, dependencies, scheduler_enabled,
        )

        self._running = True
        self._publish_status(skill_id, "running", message="Benchmark started")

        # Run benchmark in a separate thread to avoid blocking ROS2 callbacks
        threading.Thread(
            target=self._run_benchmark_thread,
            args=(skill_id, iterations, warmup, grid_size, scan_grid,
                  output_file, scheduler_enabled, dependencies),
            daemon=True
        ).start()

    def _run_benchmark_thread(self, *args):
        """Wrapper to run benchmark in thread and handle results."""
        skill_id = args[0]
        try:
            result = self._run_benchmark(*args)
            self._publish_status(skill_id, "finished", result=result)
        except Exception as e:
            logger.error("Benchmark failed: %s", e, exc_info=True)
            self._publish_status(skill_id, "error", errno=1, error=str(e))
        finally:
            self._running = False

    def _run_benchmark(self, skill_id: str, iterations: int, warmup: int,
                       grid_size: int, scan_grid: int, output_file: str,
                       scheduler_enabled: bool, dependencies: list) -> dict:
        """Execute the navigation benchmark workload."""
        workload = NavigationWorkload(grid_size=grid_size, scan_grid=scan_grid)
        dep_topics = get_dependency_topics(dependencies)
        dep_waiter = DependencyWaiter(self, dep_topics) if dep_topics else None

        collector = MetricsCollector(
            skill_name="skl::bench_nav",
            scheduler_enabled=scheduler_enabled,
            total_iterations=iterations,
            warmup_iterations=warmup,
        )

        collector.start()

        for i in range(iterations):
            if not self._running:
                logger.info("Benchmark interrupted at iteration %d", i)
                break

            # Wait for dependency data (srv::bench_slam) before each iteration
            if dep_waiter and not dep_waiter.wait_for_dependencies(timeout_sec=2.0):
                logger.warning("Iteration %d: dependency timeout, proceeding anyway", i + 1)

            t_start = collector.begin_iteration()
            workload.run()
            collector.end_iteration(t_start)

            # Publish progress every 50 iterations
            if (i + 1) % 50 == 0:
                tag = "warmup" if i < warmup else "measuring"
                logger.info("Iteration %d/%d (%s)", i + 1, iterations, tag)
                self._publish_status(
                    skill_id, "running",
                    message=f"Progress: {i + 1}/{iterations} ({tag})",
                )

        metrics = collector.finish()
        stats = metrics.compute_stats()

        # Save raw data to file
        if output_file:
            data = metrics.to_dict()
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Results saved to %s", output_file)

        logger.info(
            "Benchmark complete: mean=%.2fms, p99=%.2fms, jitter_cv=%.4f, throughput=%.1f iter/s",
            stats["latency"]["mean_ms"],
            stats["latency"]["p99_ms"],
            stats["stability"]["coefficient_of_variation"],
            stats["throughput"]["iterations_per_sec"],
        )

        return stats


def main():
    rclpy.init()
    node = BenchNavSkill()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
