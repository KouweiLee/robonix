# SPDX-License-Identifier: MulanPSL-2.0
"""
Benchmark Runner - Orchestrates the full scheduling benchmark.

Responsibilities:
  1. Start background contention processes (perception, SLAM, speech, motion plan)
  2. Register their PIDs with the scheduler via ROS2 service (in-memory, no file I/O)
  3. Start skill processes (nav2, VLA grasp, visual inspection)
  4. For each skill: call scheduler to boost/restore priorities
  5. Send start command via ROS2 topic, wait for completion
  6. Collect results from output files
  7. Run with and without scheduler for A/B comparison
  8. Generate comparison report

Usage:
  python -m scheduler_benchmark.runner --config config/benchmark.yaml
  python -m scheduler_benchmark.runner --no-scheduler  # baseline only
  python -m scheduler_benchmark.runner --scheduler-only # scheduler only
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark.runner")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SkillConfig:
    """Configuration for a single skill benchmark."""
    name: str               # e.g. "skl::bench_nav"
    module: str             # e.g. "scheduler_benchmark.skills.nav2_skill"
    topic_prefix: str       # e.g. "/robot1/skill/bench_nav"
    iterations: int = 200
    warmup: int = 20
    params: dict = field(default_factory=dict)
    # Components that the scheduler should boost for this skill
    dependencies: List[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Full benchmark configuration."""
    output_dir: str = "benchmark_results"
    # Background worker configuration
    background_workers: Dict[str, dict] = field(default_factory=lambda: {
        "perception": {"rate_hz": 30.0, "std_name": "srv::bench_perception"},
        "lidar_slam": {"rate_hz": 10.0, "std_name": "srv::bench_lidar_slam"},
        "speech": {"rate_hz": 4.0, "std_name": "srv::bench_speech"},
    })
    # Skill benchmarks to run
    skills: List[SkillConfig] = field(default_factory=lambda: [
        SkillConfig(
            name="skl::bench_nav",
            module="scheduler_benchmark.skills.nav2_skill",
            topic_prefix="/robot1/skill/bench_nav",
            iterations=100,
            warmup=20,
            params={"grid_size": 500, "scan_grid": 256},
            dependencies=["srv::bench_lidar_slam"],
        ),
        SkillConfig(
            name="skl::bench_grasp",
            module="scheduler_benchmark.skills.vla_skill",
            topic_prefix="/robot1/skill/bench_grasp",
            iterations=100,
            warmup=15,
            params={
                "image_size": 224, "layers": 12,
                "hidden": 768, "seq_len": 256,
            },
            dependencies=["srv::bench_perception"],
        ),
        SkillConfig(
            name="skl::bench_inspect",
            module="scheduler_benchmark.skills.inspect_skill",
            topic_prefix="/robot1/skill/bench_inspect",
            iterations=100,
            warmup=15,
            params={},
            dependencies=["srv::bench_perception", "srv::bench_lidar_slam"],
        ),
    ])
    # Settle time after starting background workers (seconds)
    settle_time: float = 3.0
    # Time to wait for skill node to initialize (seconds)
    skill_init_time: float = 2.0
    # Maximum time to wait for a skill to complete (seconds)
    skill_timeout: float = 600.0
    # Number of benchmark runs for statistical significance
    num_runs: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()
        config.output_dir = data.get("output_dir", config.output_dir)
        config.settle_time = data.get("settle_time", config.settle_time)
        config.skill_init_time = data.get("skill_init_time", config.skill_init_time)
        config.skill_timeout = data.get("skill_timeout", config.skill_timeout)
        config.num_runs = data.get("num_runs", config.num_runs)

        if "background_workers" in data:
            config.background_workers = data["background_workers"]

        if "skills" in data:
            config.skills = []
            for sk in data["skills"]:
                config.skills.append(SkillConfig(
                    name=sk["name"],
                    module=sk["module"],
                    topic_prefix=sk["topic_prefix"],
                    iterations=sk.get("iterations", 200),
                    warmup=sk.get("warmup", 20),
                    params=sk.get("params", {}),
                    dependencies=sk.get("dependencies", []),
                ))

        return config


# ---------------------------------------------------------------------------
# Process Manager
# ---------------------------------------------------------------------------

class ProcessManager:
    """Manages background worker and skill processes."""

    def __init__(self, scheduler_registrar: "SchedulerRegistrar", xsched_enabled: bool = False):
        self._registrar = scheduler_registrar
        self._xsched_enabled = xsched_enabled
        self._bg_procs: Dict[str, subprocess.Popen] = {}
        # std_name -> PID mapping for all processes we've started
        self._registered: Dict[str, int] = {}
        self._skill_proc: Optional[subprocess.Popen] = None
        self._skill_std_name: Optional[str] = None

    def _get_subprocess_env(self) -> Dict[str, str]:
        """Get environment variables for subprocesses, including xsched if enabled."""
        env = os.environ.copy()
        if not self._xsched_enabled:
            return env

        home = os.path.expanduser("~")
        robonix_dir = os.path.join(home, ".robonix")
        lib_dir = os.path.join(robonix_dir, "lib")

        if os.path.isdir(lib_dir):
            # Prepend to LD_LIBRARY_PATH (matches xsched_env.sh)
            lp = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{lp}" if lp else lib_dir

            # Set xsched variables from xsched_env.sh
            env["XSCHED_SCHEDULER"] = "GLB"
            env["XSCHED_AUTO_XQUEUE"] = "ON"
            env["XSCHED_AUTO_XQUEUE_LEVEL"] = "1"
            env["XSCHED_AUTO_XQUEUE_PRIORITY"] = "0"
            env["XSCHED_AUTO_XQUEUE_THRESHOLD"] = "16"
            env["XSCHED_AUTO_XQUEUE_BATCH_SIZE"] = "8"
            
            logger.debug("Applied xsched environment variables to subprocess")
        return env

    def start_background_workers(self, workers: Dict[str, dict]):
        """Start all background contention workers as subprocesses."""
        env = self._get_subprocess_env()
        for name, cfg in workers.items():
            rate = cfg.get("rate_hz", 5.0)
            std_name = cfg.get("std_name", f"srv::bench_{name}")

            cmd = [
                sys.executable, "-m", "scheduler_benchmark.background",
                name, "--rate", str(rate),
            ]
            logger.info("Starting background worker: %s (rate=%.1f Hz)", name, rate)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            self._bg_procs[name] = proc
            self._registered[std_name] = proc.pid
            logger.info("  -> PID %d for %s (new session)", proc.pid, std_name)

        # Register all PIDs with scheduler (with retry)
        self._registrar.register_all(self._registered)

    def start_skill(self, module: str, skill_name: str) -> subprocess.Popen:
        """Start a skill process."""
        env = self._get_subprocess_env()
        cmd = [sys.executable, "-m", module]
        logger.info("Starting skill process: %s (%s)", skill_name, module)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        self._skill_proc = proc
        self._skill_std_name = skill_name

        # Register skill PID with scheduler
        self._registrar.register_one(skill_name, proc.pid)
        logger.info("  -> PID %d for %s (new session)", proc.pid, skill_name)
        return proc

    def stop_skill(self):
        """Stop the current skill process."""
        if self._skill_proc and self._skill_proc.poll() is None:
            logger.info("Stopping skill (PID %d)", self._skill_proc.pid)
            self._skill_proc.terminate()
            try:
                self._skill_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._skill_proc.kill()
                self._skill_proc.wait()

        # Unregister skill from scheduler
        if self._skill_std_name:
            self._registrar.unregister_one(self._skill_std_name)
        self._skill_proc = None
        self._skill_std_name = None

    def stop_all_background(self):
        """Stop all background workers."""
        for name, proc in self._bg_procs.items():
            if proc.poll() is None:
                logger.info("Stopping background worker: %s (PID %d)", name, proc.pid)
                proc.terminate()
        for name, proc in self._bg_procs.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # Unregister all from scheduler
        self._registrar.unregister_all(self._registered)
        self._bg_procs.clear()
        self._registered.clear()


# ---------------------------------------------------------------------------
# ROS2 Communication Bridge (persistent node, zero subprocess overhead)
# ---------------------------------------------------------------------------

# Maximum retries for service calls that must succeed
_MAX_RETRIES = 5
_RETRY_DELAY = 1.0  # seconds between retries


class ROS2Bridge:
    """
    Persistent rclpy node for efficient ROS2 IPC.

    Replaces subprocess-based ``ros2 service call`` / ``ros2 topic pub`` with
    in-process rclpy service clients and publishers.  Eliminates per-call
    overhead of fork + exec + Python startup + DDS discovery (~1-3 s each)
    down to < 5 ms per call.
    """

    def __init__(self):
        if not _RCLPY_AVAILABLE:
            raise RuntimeError(
                "rclpy is not available. Source your ROS 2 workspace first:\n"
                "  source /opt/ros/$ROS_DISTRO/setup.bash\n"
                "  source <ws>/install/setup.bash"
            )

        # Lazy import: types only available after workspace is built & sourced
        from robonix_sdk.srv import AdjustPriority, RegisterProcess
        from std_msgs.msg import String as StringMsg

        self._AdjustPriority = AdjustPriority
        self._RegisterProcess = RegisterProcess
        self._StringMsg = StringMsg

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self._node = rclpy.create_node('benchmark_runner')

        self._register_cli = self._node.create_client(
            RegisterProcess, '/rbnx/scheduler_register')
        self._policy_cli = self._node.create_client(
            AdjustPriority, '/rbnx/scheduler_policy')

        self._publishers: Dict[str, Any] = {}
        logger.info("ROS2Bridge: persistent node 'benchmark_runner' created")

    # -- service helpers ---------------------------------------------------

    def wait_for_services(self, timeout_sec: float = 10.0) -> bool:
        """Block until both scheduler services are discoverable."""
        logger.info("Waiting for scheduler services (timeout=%.1fs)...", timeout_sec)
        if not self._register_cli.wait_for_service(timeout_sec=timeout_sec):
            logger.warning("scheduler_register service not available")
            return False
        if not self._policy_cli.wait_for_service(timeout_sec=timeout_sec):
            logger.warning("scheduler_policy service not available")
            return False
        logger.info("Scheduler services discovered")
        return True

    def call_register(self, std_name: str, pid: int, register: bool,
                      timeout_sec: float = 10.0) -> Optional[bool]:
        """Call RegisterProcess service.  Returns ``ok`` field or *None* on failure."""
        req = self._RegisterProcess.Request()
        req.std_name = std_name
        req.pid = pid
        req.do_register = register
        resp = self._call_service(self._register_cli, req, timeout_sec)
        return resp.ok if resp is not None else None

    def call_policy(self, skill_name: str, high_priority: bool,
                    timeout_sec: float = 10.0) -> Optional[bool]:
        """Call AdjustPriority service.  Returns ``ok`` field or *None* on failure."""
        req = self._AdjustPriority.Request()
        req.skill_name = skill_name
        req.high_priority = high_priority
        resp = self._call_service(self._policy_cli, req, timeout_sec)
        return resp.ok if resp is not None else None

    def _call_service(self, client, request, timeout_sec: float):
        """Internal: synchronous service call with timeout."""
        assert self._node is not None, "ROS2Bridge already destroyed"
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=min(timeout_sec, 5.0)):
                logger.warning("Service not ready: %s", client.srv_name)
                return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(
            self._node, future, timeout_sec=timeout_sec)
        if future.done():
            return future.result()
        future.cancel()
        logger.warning("Service call timed out: %s", client.srv_name)
        return None

    # -- topic helpers -----------------------------------------------------

    def publish_string(self, topic: str, data: str,
                       wait_for_sub_sec: float = 5.0) -> bool:
        """Publish a ``std_msgs/String`` message, waiting for subscriber discovery."""
        assert self._node is not None, "ROS2Bridge already destroyed"
        if topic not in self._publishers:
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self._publishers[topic] = self._node.create_publisher(
                self._StringMsg, topic, qos)
            logger.debug("Created publisher for %s", topic)

        pub = self._publishers[topic]

        # Wait for at least one subscriber (DDS discovery)
        deadline = time.time() + wait_for_sub_sec
        while pub.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            time.sleep(0.05)

        if pub.get_subscription_count() == 0:
            logger.warning("No subscribers on %s after %.1fs", topic, wait_for_sub_sec)
            return False

        msg = self._StringMsg()
        msg.data = data
        pub.publish(msg)
        return True

    # -- lifecycle ---------------------------------------------------------

    def destroy(self):
        """Destroy node; optionally shut down rclpy if we own it."""
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
        logger.info("ROS2Bridge destroyed")


class SchedulerRegistrar:
    """
    Registers / unregisters process PIDs with the scheduler via the
    scheduler_register ROS2 service.  Uses the persistent ROS2Bridge for
    zero-overhead IPC (no subprocess fork per call).
    """

    def __init__(self, bridge: ROS2Bridge, enabled: bool = True):
        self._bridge = bridge
        self.enabled = enabled

    def register_one(self, std_name: str, pid: int):
        """Register a single process. Retries until success."""
        if not self.enabled:
            return
        for attempt in range(1, _MAX_RETRIES + 1):
            result = self._bridge.call_register(std_name, pid, True)
            if result is not None:
                return
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "Register %s (PID %d): attempt %d/%d failed, retrying in %.0fs...",
                    std_name, pid, attempt, _MAX_RETRIES, _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
        raise RuntimeError(
            f"Register {std_name} (PID {pid}): all {_MAX_RETRIES} attempts failed. "
            f"Is robonix-scheduler running?"
        )

    def unregister_one(self, std_name: str):
        """Unregister a single process. Best-effort (no retry)."""
        if not self.enabled:
            return
        self._bridge.call_register(std_name, 0, False)

    def register_all(self, entries: Dict[str, int]):
        """Register multiple processes. Each retries until success."""
        if not self.enabled:
            return
        for std_name, pid in entries.items():
            self.register_one(std_name, pid)

    def unregister_all(self, entries: Dict[str, int]):
        """Unregister multiple processes. Best-effort."""
        if not self.enabled:
            return
        for std_name in entries:
            self.unregister_one(std_name)


class SchedulerClient:
    """
    Calls the robonix-scheduler's scheduler_policy service via
    the persistent ROS2Bridge (in-process, no subprocess overhead).
    """

    def __init__(self, bridge: ROS2Bridge, enabled: bool = True):
        self._bridge = bridge
        self.enabled = enabled
        if not enabled:
            logger.info("Scheduler client DISABLED (baseline mode)")

    def escalate(self, skill_name: str) -> bool:
        """Request high priority for a skill's dependencies."""
        if not self.enabled:
            return True
        return self._call(skill_name, True)

    def de_escalate(self, skill_name: str) -> bool:
        """Restore normal priority for a skill's dependencies."""
        if not self.enabled:
            return True
        return self._call(skill_name, False)

    def _call(self, skill_name: str, high_priority: bool) -> bool:
        action = "escalate" if high_priority else "de-escalate"
        logger.info("Scheduler %s: %s", action, skill_name)
        result = self._bridge.call_policy(skill_name, high_priority)
        if result is not None:
            logger.info("Scheduler %s OK for %s", action, skill_name)
            return True
        logger.warning("Scheduler %s failed for %s", action, skill_name)
        return False


# ---------------------------------------------------------------------------
# Skill Trigger (via ROS2 topic publish through persistent bridge)
# ---------------------------------------------------------------------------

class SkillTrigger:
    """Publishes start/terminate commands to skill topics via ROS2Bridge."""

    def __init__(self, bridge: ROS2Bridge):
        self._bridge = bridge

    def start_skill(self, topic: str, skill_id: str, params: dict,
                    output_file: str, scheduler_enabled: bool) -> bool:
        """Publish a start command to the skill's start topic."""
        msg_data = json.dumps({
            "skill_id": skill_id,
            "params": {
                **params,
                "output_file": output_file,
                "scheduler_enabled": scheduler_enabled,
            },
        })
        return self._bridge.publish_string(f"{topic}/start", msg_data)

    def terminate_skill(self, topic: str, skill_id: str) -> bool:
        """Publish a terminate command to the skill's start topic."""
        msg_data = json.dumps({"terminate": True, "skill_id": skill_id})
        return self._bridge.publish_string(f"{topic}/start", msg_data)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Main benchmark orchestrator.
    Runs all skills with and without the scheduler, collecting metrics.

    Owns a persistent ``ROS2Bridge`` that is reused across all benchmark
    phases (baseline / scheduler), eliminating repeated node creation and
    DDS discovery overhead.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._bridge = ROS2Bridge()

    def run(self, scheduler_enabled: bool) -> List[dict]:
        """
        Run the complete benchmark suite for one condition.

        Args:
            scheduler_enabled: Whether to use the robonix-scheduler.

        Returns:
            List of per-skill result dictionaries.
        """
        condition = "scheduler" if scheduler_enabled else "baseline"
        logger.info("=" * 70)
        logger.info("BENCHMARK RUN: %s", condition.upper())
        logger.info("=" * 70)

        # Wait for scheduler services once at the start of a scheduler run
        if scheduler_enabled:
            if not self._bridge.wait_for_services(timeout_sec=10.0):
                raise RuntimeError(
                    "Scheduler services not available. "
                    "Is robonix-scheduler running?"
                )

        registrar = SchedulerRegistrar(self._bridge, enabled=scheduler_enabled)
        scheduler = SchedulerClient(self._bridge, enabled=scheduler_enabled)
        trigger = SkillTrigger(self._bridge)
        proc_mgr = ProcessManager(registrar, xsched_enabled=scheduler_enabled)
        results = []

        # Start background workers
        logger.info("Starting %d background workers...",
                     len(self.config.background_workers))
        proc_mgr.start_background_workers(self.config.background_workers)

        # Allow workers to settle
        logger.info("Settling for %.1fs...", self.config.settle_time)
        time.sleep(self.config.settle_time)

        try:
            for skill_cfg in self.config.skills:
                for run_idx in range(self.config.num_runs):
                    result = self._run_single_skill(
                        skill_cfg, scheduler, trigger,
                        proc_mgr, condition, run_idx,
                    )
                    if result:
                        results.append(result)
        finally:
            # Cleanup
            logger.info("Stopping all background workers...")
            proc_mgr.stop_all_background()

        return results

    def _run_single_skill(self, skill_cfg: SkillConfig,
                          scheduler: SchedulerClient,
                          trigger: SkillTrigger,
                          proc_mgr: ProcessManager,
                          condition: str, run_idx: int) -> Optional[dict]:
        """Run a single skill benchmark."""
        skill_name = skill_cfg.name
        run_label = f"{condition}_run{run_idx}"
        skill_id = f"{skill_name}_{run_label}"

        logger.info("-" * 50)
        logger.info("Skill: %s [%s]", skill_name, run_label)
        logger.info("-" * 50)

        # Prepare output file
        safe_name = skill_name.replace("::", "_").replace("/", "_")
        output_file = os.path.join(
            self.config.output_dir, condition,
            f"{safe_name}_run{run_idx}.json",
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Start skill process
        skill_proc = proc_mgr.start_skill(skill_cfg.module, skill_name)
        time.sleep(self.config.skill_init_time)

        # Escalate scheduler priorities
        scheduler.escalate(skill_name)

        # Send start command
        logger.info("Triggering skill %s (iterations=%d, warmup=%d)",
                     skill_name, skill_cfg.iterations, skill_cfg.warmup)
        trigger.start_skill(
            skill_cfg.topic_prefix,
            skill_id,
            {**skill_cfg.params, "iterations": skill_cfg.iterations,
             "warmup": skill_cfg.warmup, "dependencies": skill_cfg.dependencies},
            output_file,
            scheduler.enabled,
        )

        # Wait for skill to complete (poll output file)
        logger.info("Waiting for skill to complete (timeout=%ds)...",
                     int(self.config.skill_timeout))
        start_wait = time.time()
        while time.time() - start_wait < self.config.skill_timeout:
            # Check if skill process exited
            if skill_proc.poll() is not None:
                logger.info("Skill process exited (code=%d)", skill_proc.returncode)
                break
            # Check if output file exists and has content
            if os.path.exists(output_file):
                try:
                    with open(output_file) as f:
                        data = json.load(f)
                    if data.get("stats", {}).get("num_samples", 0) > 0:
                        logger.info("Skill completed, results available")
                        break
                except (json.JSONDecodeError, IOError):
                    pass
            time.sleep(2.0)
        else:
            logger.warning("Skill %s timed out!", skill_name)
            trigger.terminate_skill(skill_cfg.topic_prefix, skill_id)
            time.sleep(2.0)

        # De-escalate scheduler
        scheduler.de_escalate(skill_name)

        # Stop skill process
        proc_mgr.stop_skill()

        # Read results
        if os.path.exists(output_file):
            with open(output_file) as f:
                result = json.load(f)
            stats = result.get("stats", {})
            logger.info(
                "Result: mean=%.2fms, p99=%.2fms, cv=%.4f, throughput=%.1f/s",
                stats.get("latency", {}).get("mean_ms", 0),
                stats.get("latency", {}).get("p99_ms", 0),
                stats.get("stability", {}).get("coefficient_of_variation", 0),
                stats.get("throughput", {}).get("iterations_per_sec", 0),
            )
            return result
        else:
            logger.error("No results file found for %s", skill_name)
            return None

    def run_comparison(self) -> Tuple[List[dict], List[dict]]:
        """
        Run the full A/B comparison: baseline (no scheduler) vs scheduler.
        Returns (baseline_results, scheduler_results).
        """
        logger.info("=" * 70)
        logger.info("SCHEDULING BENCHMARK - A/B COMPARISON")
        logger.info("=" * 70)

        # Phase 1: Baseline (no scheduler)
        logger.info("\n>>> PHASE 1: BASELINE (Linux default CFS) <<<\n")
        baseline = self.run(scheduler_enabled=False)

        # Brief pause between phases
        logger.info("\nCooldown between phases (5s)...\n")
        time.sleep(5.0)

        # Phase 2: With scheduler
        logger.info("\n>>> PHASE 2: WITH ROBONIX-SCHEDULER <<<\n")
        scheduler = self.run(scheduler_enabled=True)

        return baseline, scheduler

    def cleanup(self):
        """Destroy the ROS2Bridge and release resources."""
        self._bridge.destroy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Robonix Scheduling Benchmark Runner",
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "config", "benchmark.yaml"),
        help="Path to benchmark configuration YAML",
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help="Override output directory",
    )
    parser.add_argument(
        "--scheduler-only", action="store_true",
        help="Only run with scheduler (skip baseline)",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only run baseline (no scheduler)",
    )
    parser.add_argument(
        "--xsched-overhead", action="store_true",
        help="Quantify xsched GPU scheduling overhead (isolated, no contention)",
    )
    parser.add_argument(
        "--comparison", action="store_true", default=True,
        help="Run full A/B comparison (default)",
    )
    parser.add_argument(
        "--runs", type=int, default=0,
        help="Override number of runs per condition",
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if os.path.exists(config_path):
        config = BenchmarkConfig.from_yaml(config_path)
        logger.info("Loaded config from %s", config_path)
    else:
        config = BenchmarkConfig()
        logger.info("Using default configuration")

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.runs > 0:
        config.num_runs = args.runs

    # Add timestamp to output dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    config.output_dir = os.path.join(config.output_dir, ts)
    os.makedirs(config.output_dir, exist_ok=True)

    # Save config used
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump({
            "output_dir": config.output_dir,
            "settle_time": config.settle_time,
            "skill_init_time": config.skill_init_time,
            "num_runs": config.num_runs,
            "skills": [
                {"name": s.name, "iterations": s.iterations, "warmup": s.warmup,
                "dependencies": s.dependencies}
                for s in config.skills
            ],
            "background_workers": config.background_workers,
        }, f, indent=2)

    runner = BenchmarkRunner(config)
    try:
        if args.xsched_overhead:
            from scheduler_benchmark.xsched_overhead import run_overhead_benchmark
            output_path = os.path.join(config.output_dir, "xsched_overhead.json")
            run_overhead_benchmark(
                iterations=200,
                warmup=20,
                output_file=output_path,
            )
            logger.info("XSched overhead benchmark complete. Results in: %s", config.output_dir)
            return
        if args.baseline_only:
            results = runner.run(scheduler_enabled=False)
            _save_results(config.output_dir, "baseline", results)
        elif args.scheduler_only:
            results = runner.run(scheduler_enabled=True)
            _save_results(config.output_dir, "scheduler", results)
        else:
            baseline, scheduler = runner.run_comparison()
            _save_results(config.output_dir, "baseline", baseline)
            _save_results(config.output_dir, "scheduler", scheduler)

            # Generate comparison report
            from scheduler_benchmark.report import generate_comparison_report
            generate_comparison_report(
                baseline, scheduler, config.output_dir,
            )

        logger.info("Benchmark complete. Results in: %s", config.output_dir)
    finally:
        runner.cleanup()


def _save_results(output_dir: str, condition: str, results: List[dict]):
    """Save aggregated results for a condition."""
    path = os.path.join(output_dir, f"{condition}_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s results to %s", condition, path)


if __name__ == "__main__":
    main()
