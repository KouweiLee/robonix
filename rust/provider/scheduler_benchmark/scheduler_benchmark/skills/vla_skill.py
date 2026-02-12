# SPDX-License-Identifier: MulanPSL-2.0
"""
VLA (Vision-Language-Action) Grasping Benchmark Skill (skl::bench_grasp)

Simulates a Vision-Language-Action model used for robotic grasping. Each
iteration represents one inference cycle: image preprocessing on CPU,
transformer-based inference on GPU, and action decoding on CPU.

This skill is GPU-intensive (transformer inference) and demonstrates the
benefit of GPU scheduling via xsched. When the scheduler boosts this skill,
GPU inference should experience lower latency due to priority-based CUDA
stream scheduling.
"""

import json
import os
import logging
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from scheduler_benchmark.workloads import VLAWorkload, gpu_available
from scheduler_benchmark.metrics import MetricsCollector

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bench_grasp")

# Default benchmark parameters
DEFAULT_ITERATIONS = 150
DEFAULT_WARMUP = 15
DEFAULT_IMAGE_SIZE = 224
DEFAULT_LAYERS = 12
DEFAULT_HIDDEN = 768
DEFAULT_SEQ_LEN = 256


class BenchGraspSkill(Node):
    """
    VLA grasping benchmark skill node.

    Simulates a vision-language-action model inference loop for grasp planning.
    Heavy on GPU for transformer inference, with CPU pre/post-processing.
    """

    def __init__(self):
        super().__init__("bench_grasp_skill")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.start_sub = self.create_subscription(
            String,
            "/robot1/skill/bench_grasp/start",
            self._on_start,
            qos,
        )
        self.status_pub = self.create_publisher(
            String,
            "/robot1/skill/bench_grasp/status",
            qos,
        )
        self._running = False
        logger.info(
            "BenchGraspSkill initialized (PID=%d, GPU=%s)",
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

        skill_id = data.get("skill_id", "bench_grasp_0")
        params = data.get("params", {})

        iterations = params.get("iterations", DEFAULT_ITERATIONS)
        warmup = params.get("warmup", DEFAULT_WARMUP)
        image_size = params.get("image_size", DEFAULT_IMAGE_SIZE)
        layers = params.get("layers", DEFAULT_LAYERS)
        hidden = params.get("hidden", DEFAULT_HIDDEN)
        seq_len = params.get("seq_len", DEFAULT_SEQ_LEN)
        output_file = params.get("output_file", "")
        scheduler_enabled = params.get("scheduler_enabled", False)

        logger.info(
            "Starting VLA benchmark: iterations=%d, warmup=%d, layers=%d, "
            "hidden=%d, seq_len=%d, GPU=%s, scheduler=%s",
            iterations, warmup, layers, hidden, seq_len,
            gpu_available(), scheduler_enabled,
        )

        self._running = True
        self._publish_status(skill_id, "running", message="VLA benchmark started")

        try:
            result = self._run_benchmark(
                skill_id, iterations, warmup, image_size, layers,
                hidden, seq_len, output_file, scheduler_enabled,
            )
            self._publish_status(skill_id, "finished", result=result)
        except Exception as e:
            logger.error("Benchmark failed: %s", e, exc_info=True)
            self._publish_status(skill_id, "error", errno=1, error=str(e))
        finally:
            self._running = False

    def _run_benchmark(self, skill_id: str, iterations: int, warmup: int,
                       image_size: int, layers: int, hidden: int,
                       seq_len: int, output_file: str,
                       scheduler_enabled: bool) -> dict:
        """Execute the VLA benchmark workload."""
        workload = VLAWorkload(
            image_size=image_size,
            layers=layers,
            hidden=hidden,
            seq_len=seq_len,
        )
        collector = MetricsCollector(
            skill_name="skl::bench_grasp",
            scheduler_enabled=scheduler_enabled,
            total_iterations=iterations,
            warmup_iterations=warmup,
        )

        collector.start()

        for i in range(iterations):
            if not self._running:
                logger.info("Benchmark interrupted at iteration %d", i)
                break

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
            "VLA benchmark complete: mean=%.2fms, p99=%.2fms, jitter_cv=%.4f, "
            "throughput=%.1f iter/s",
            stats["latency"]["mean_ms"],
            stats["latency"]["p99_ms"],
            stats["stability"]["coefficient_of_variation"],
            stats["throughput"]["iterations_per_sec"],
        )

        return stats


def main():
    rclpy.init()
    node = BenchGraspSkill()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
