# SPDX-License-Identifier: MulanPSL-2.0
"""
Benchmark Runner - Orchestrates the full scheduling benchmark.

Responsibilities:
  1. Start background contention processes (perception, SLAM, speech, motion plan)
  2. Register their PIDs in ~/.robonix/processes.json (for the scheduler)
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
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

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
        "perception": {"rate_hz": 10.0, "std_name": "srv::bench_perception"},
        "slam": {"rate_hz": 5.0, "std_name": "srv::bench_slam"},
        "speech": {"rate_hz": 3.0, "std_name": "srv::bench_speech"},
        "motion_plan": {"rate_hz": 8.0, "std_name": "srv::bench_motion_plan"},
    })
    # Skill benchmarks to run
    skills: List[SkillConfig] = field(default_factory=lambda: [
        SkillConfig(
            name="skl::bench_nav",
            module="scheduler_benchmark.skills.nav2_skill",
            topic_prefix="/robot1/skill/bench_nav",
            iterations=200,
            warmup=20,
            params={"grid_size": 300, "scan_grid": 200},
            dependencies=[
                "prm::base.navigate", "prm::base.pose.cov",
                "srv::bench_slam",
            ],
        ),
        SkillConfig(
            name="skl::bench_grasp",
            module="scheduler_benchmark.skills.vla_skill",
            topic_prefix="/robot1/skill/bench_grasp",
            iterations=150,
            warmup=15,
            params={
                "image_size": 224, "layers": 12,
                "hidden": 768, "seq_len": 256,
            },
            dependencies=[
                "prm::camera.rgb", "prm::camera.depth",
                "srv::bench_perception",
            ],
        ),
        SkillConfig(
            name="skl::bench_inspect",
            module="scheduler_benchmark.skills.inspect_skill",
            topic_prefix="/robot1/skill/bench_inspect",
            iterations=150,
            warmup=15,
            params={},
            dependencies=[
                "prm::camera.rgb", "prm::camera.depth",
                "srv::bench_perception",
            ],
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

    def __init__(self, processes_json: str):
        self.processes_json = processes_json
        self._bg_procs: Dict[str, subprocess.Popen] = {}
        self._skill_proc: Optional[subprocess.Popen] = None
        self._process_entries: List[dict] = []

    def start_background_workers(self, workers: Dict[str, dict]):
        """Start all background contention workers as subprocesses."""
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
            )
            self._bg_procs[name] = proc
            self._process_entries.append({
                "package_name": "scheduler_benchmark",
                "std_name": std_name,
                "package_type": "cap",
                "pid": proc.pid,
                "log_file": "",
                "hostname": os.uname().nodename,
            })
            logger.info("  -> PID %d for %s", proc.pid, std_name)

        self._write_processes_json()

    def start_skill(self, module: str, skill_name: str) -> subprocess.Popen:
        """Start a skill process."""
        cmd = [sys.executable, "-m", module]
        logger.info("Starting skill process: %s (%s)", skill_name, module)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._skill_proc = proc

        # Add to processes.json
        self._process_entries.append({
            "package_name": "scheduler_benchmark",
            "std_name": skill_name,
            "package_type": "skl",
            "pid": proc.pid,
            "log_file": "",
            "hostname": os.uname().nodename,
        })
        self._write_processes_json()
        logger.info("  -> PID %d for %s", proc.pid, skill_name)
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

        # Remove skill entry from processes list
        if self._skill_proc:
            self._process_entries = [
                e for e in self._process_entries
                if e.get("pid") != self._skill_proc.pid
            ]
            self._write_processes_json()
        self._skill_proc = None

    def stop_all_background(self):
        """Stop all background workers."""
        for name, proc in self._bg_procs.items():
            if proc.poll() is None:
                logger.info("Stopping background worker: %s (PID %d)", name, proc.pid)
                proc.terminate()
        # Wait for all to finish
        for name, proc in self._bg_procs.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._bg_procs.clear()
        self._process_entries = []
        self._write_processes_json()

    def _write_processes_json(self):
        """Write current process entries to ~/.robonix/processes.json."""
        # Merge with existing entries (from other Robonix components)
        existing = []
        if os.path.exists(self.processes_json):
            try:
                with open(self.processes_json) as f:
                    existing = json.load(f)
                # Filter out stale benchmark entries
                our_pids = {e["pid"] for e in self._process_entries}
                existing = [
                    e for e in existing
                    if e.get("package_name") != "scheduler_benchmark"
                    or e.get("pid") in our_pids
                ]
            except (json.JSONDecodeError, KeyError):
                existing = []

        # Merge: existing non-benchmark + our entries
        merged = [
            e for e in existing
            if e.get("package_name") != "scheduler_benchmark"
        ] + self._process_entries

        os.makedirs(os.path.dirname(self.processes_json), exist_ok=True)
        with open(self.processes_json, "w") as f:
            json.dump(merged, f, indent=2)


# ---------------------------------------------------------------------------
# Scheduler Client (via ROS2 CLI or direct ROS2 call)
# ---------------------------------------------------------------------------

class SchedulerClient:
    """
    Calls the robonix-scheduler's scheduler_policy service.
    Uses ros2 service call via CLI for simplicity and independence.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            logger.info("Scheduler client DISABLED (baseline mode)")

    def escalate(self, skill_name: str) -> bool:
        """Request high priority for a skill's dependencies."""
        if not self.enabled:
            return True
        return self._call_service(skill_name, True)

    def de_escalate(self, skill_name: str) -> bool:
        """Restore normal priority for a skill's dependencies."""
        if not self.enabled:
            return True
        return self._call_service(skill_name, False)

    def _call_service(self, skill_name: str, high_priority: bool) -> bool:
        """Call scheduler_policy service via ros2 CLI."""
        action = "escalate" if high_priority else "de-escalate"
        logger.info("Scheduler %s: %s", action, skill_name)
        try:
            cmd = [
                "ros2", "service", "call",
                "/rbnx/scheduler_policy",
                "robonix_sdk/srv/AdjustPriority",
                json.dumps({
                    "skill_name": skill_name,
                    "high_priority": high_priority,
                }),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                logger.info("Scheduler %s OK for %s", action, skill_name)
                return True
            else:
                logger.warning(
                    "Scheduler %s failed for %s: %s",
                    action, skill_name, result.stderr.strip(),
                )
                return False
        except subprocess.TimeoutExpired:
            logger.warning("Scheduler call timed out for %s", skill_name)
            return False
        except FileNotFoundError:
            logger.warning("ros2 CLI not found, scheduler integration disabled")
            self.enabled = False
            return False


# ---------------------------------------------------------------------------
# Skill Trigger (via ROS2 topic publish)
# ---------------------------------------------------------------------------

class SkillTrigger:
    """Publishes start/terminate commands to skill topics via ros2 CLI."""

    @staticmethod
    def start_skill(topic: str, skill_id: str, params: dict,
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
        try:
            cmd = [
                "ros2", "topic", "pub", "--once",
                f"{topic}/start",
                "std_msgs/msg/String",
                json.dumps({"data": msg_data}),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("Failed to publish start: %s", e)
            return False

    @staticmethod
    def terminate_skill(topic: str, skill_id: str) -> bool:
        """Publish a terminate command to the skill's start topic."""
        msg_data = json.dumps({"terminate": True, "skill_id": skill_id})
        try:
            cmd = [
                "ros2", "topic", "pub", "--once",
                f"{topic}/start",
                "std_msgs/msg/String",
                json.dumps({"data": msg_data}),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Main benchmark orchestrator.
    Runs all skills with and without the scheduler, collecting metrics.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        home = os.path.expanduser("~")
        self.processes_json = os.path.join(home, ".robonix", "processes.json")
        self.proc_mgr = ProcessManager(self.processes_json)

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

        scheduler = SchedulerClient(enabled=scheduler_enabled)
        results = []

        # Start background workers
        logger.info("Starting %d background workers...",
                     len(self.config.background_workers))
        self.proc_mgr.start_background_workers(self.config.background_workers)

        # Allow workers to settle
        logger.info("Settling for %.1fs...", self.config.settle_time)
        time.sleep(self.config.settle_time)

        try:
            for skill_cfg in self.config.skills:
                for run_idx in range(self.config.num_runs):
                    result = self._run_single_skill(
                        skill_cfg, scheduler, condition, run_idx,
                    )
                    if result:
                        results.append(result)
        finally:
            # Cleanup
            logger.info("Stopping all background workers...")
            self.proc_mgr.stop_all_background()

        return results

    def _run_single_skill(self, skill_cfg: SkillConfig,
                          scheduler: SchedulerClient,
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
        skill_proc = self.proc_mgr.start_skill(skill_cfg.module, skill_name)
        time.sleep(self.config.skill_init_time)

        # Escalate scheduler priorities
        scheduler.escalate(skill_name)

        # Send start command
        logger.info("Triggering skill %s (iterations=%d, warmup=%d)",
                     skill_name, skill_cfg.iterations, skill_cfg.warmup)
        SkillTrigger.start_skill(
            skill_cfg.topic_prefix,
            skill_id,
            {**skill_cfg.params, "iterations": skill_cfg.iterations,
             "warmup": skill_cfg.warmup},
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
            SkillTrigger.terminate_skill(skill_cfg.topic_prefix, skill_id)
            time.sleep(2.0)

        # De-escalate scheduler
        scheduler.de_escalate(skill_name)

        # Stop skill process
        self.proc_mgr.stop_skill()

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
                {"name": s.name, "iterations": s.iterations, "warmup": s.warmup}
                for s in config.skills
            ],
            "background_workers": config.background_workers,
        }, f, indent=2)

    runner = BenchmarkRunner(config)

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


def _save_results(output_dir: str, condition: str, results: List[dict]):
    """Save aggregated results for a condition."""
    path = os.path.join(output_dir, f"{condition}_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s results to %s", condition, path)


if __name__ == "__main__":
    main()
