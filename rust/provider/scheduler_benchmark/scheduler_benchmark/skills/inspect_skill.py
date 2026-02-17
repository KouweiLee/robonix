# SPDX-License-Identifier: MulanPSL-2.0
"""
Visual Inspection Benchmark Skill (skl::bench_inspect)

Simulates a visual inspection task that combines perception (GPU-accelerated
object detection via CNN) with analysis (CPU-based report generation).
This represents a mixed CPU+GPU workload pattern common in quality inspection,
environment monitoring, and safety checking tasks.

The skill's resource profile creates contention with both CPU-heavy (SLAM, nav2)
and GPU-heavy (VLA, speech) background processes, making it a good stress test
for the scheduler's ability to prioritize mixed workloads.
"""

import json
import os
import logging
import time
from typing import Optional

import rclpy
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from scheduler_benchmark.workloads import PerceptionWorkload, gpu_available
from scheduler_benchmark.metrics import MetricsCollector
from scheduler_benchmark.dependency_topics import get_dependency_topics, DependencyWaiter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench_inspect")

DEFAULT_ITERATIONS = 150
DEFAULT_WARMUP = 15


class BenchInspectSkill(Node):
    """
    Visual inspection benchmark skill node.

    Combines image processing + CNN detection + point cloud analysis,
    representing a full perception pipeline for inspection tasks.
    """

    def __init__(self):
        super().__init__("bench_inspect_skill")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.start_sub = self.create_subscription(
            String,
            "/robot1/skill/bench_inspect/start",
            self._on_start,
            qos,
        )
        self.status_pub = self.create_publisher(
            String,
            "/robot1/skill/bench_inspect/status",
            qos,
        )
        self._running = False
        logger.info(
            "BenchInspectSkill initialized (PID=%d, GPU=%s)",
            os.getpid(), gpu_available(),
        )

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
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in start message")
            return

        if data.get("terminate", False):
            logger.info("Received terminate signal")
            self._running = False
            return

        if self._running:
            logger.warning("Already running, ignoring start")
            return

        skill_id = data.get("skill_id", "bench_inspect_0")
        params = data.get("params", {})

        iterations = params.get("iterations", DEFAULT_ITERATIONS)
        warmup = params.get("warmup", DEFAULT_WARMUP)
        output_file = params.get("output_file", "")
        scheduler_enabled = params.get("scheduler_enabled", False)
        dependencies = params.get("dependencies", [])

        logger.info(
            "Starting inspection benchmark: iterations=%d, warmup=%d, "
            "GPU=%s, scheduler=%s",
            iterations, warmup, gpu_available(), scheduler_enabled,
        )

        self._running = True
        self._publish_status(skill_id, "running", message="Inspection benchmark started")

        threading.Thread(
            target=self._run_benchmark_thread,
            args=(skill_id, iterations, warmup, output_file,
                  scheduler_enabled, dependencies),
            daemon=True
        ).start()

    def _run_benchmark_thread(self, *args):
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
                       output_file: str, scheduler_enabled: bool,
                       dependencies: list) -> dict:
        """Execute the visual inspection benchmark workload."""
        workload = PerceptionWorkload()
        dep_topics = get_dependency_topics(dependencies)
        dep_waiter = DependencyWaiter(self, dep_topics) if dep_topics else None

        collector = MetricsCollector(
            skill_name="skl::bench_inspect",
            scheduler_enabled=scheduler_enabled,
            total_iterations=iterations,
            warmup_iterations=warmup,
        )

        collector.start()

        for i in range(iterations):
            if not self._running:
                logger.info("Benchmark interrupted at iteration %d", i)
                break

            # Wait for dependency data (prm::camera.*, srv::bench_perception)
            if dep_waiter and not dep_waiter.wait_for_dependencies(timeout_sec=2.0):
                logger.warning("Iteration %d: dependency timeout, proceeding anyway", i + 1)

            t_start = collector.begin_iteration()
            workload.run()
            collector.end_iteration(t_start)

            if (i + 1) % 30 == 0:
                tag = "warmup" if i < warmup else "measuring"
                logger.info("Iteration %d/%d (%s)", i + 1, iterations, tag)
                self._publish_status(
                    skill_id, "running",
                    message=f"Progress: {i + 1}/{iterations} ({tag})",
                )

        metrics = collector.finish()
        stats = metrics.compute_stats()

        if output_file:
            data = metrics.to_dict()
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Results saved to %s", output_file)

        logger.info(
            "Inspection benchmark complete: mean=%.2fms, p99=%.2fms, "
            "jitter_cv=%.4f, throughput=%.1f iter/s",
            stats["latency"]["mean_ms"],
            stats["latency"]["p99_ms"],
            stats["stability"]["coefficient_of_variation"],
            stats["throughput"]["iterations_per_sec"],
        )

        return stats


def main():
    rclpy.init()
    node = BenchInspectSkill()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
