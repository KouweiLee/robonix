# SPDX-License-Identifier: MulanPSL-2.0
"""
XSched Overhead Benchmark - Quantify xsched interception and scheduling overhead.

Runs a pure GPU workload (VLA) in isolation with and without xsched LD_PRELOAD,
then compares latency to isolate xsched overhead from CPU scheduler effects.

Usage:
  python -m scheduler_benchmark.xsched_overhead [--iterations N] [--warmup W]
  python -m scheduler_benchmark.xsched_overhead --worker  # subprocess mode

Prerequisites:
  - CUDA available
  - For xsched run: xserver must be running, LD_PRELOAD set by parent
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

from scheduler_benchmark.workloads import VLAWorkload, gpu_available
from scheduler_benchmark.metrics import MetricsCollector, SkillMetrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("xsched_overhead")

DEFAULT_ITERATIONS = 200
DEFAULT_WARMUP = 20


def _get_xsched_env() -> dict:
    """Build environment with xsched LD_PRELOAD (matches runner.py)."""
    env = os.environ.copy()
    home = os.path.expanduser("~")
    lib_dir = os.path.join(home, ".robonix", "lib")
    if not os.path.isdir(lib_dir):
        return None
    lp = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{lp}" if lp else lib_dir
    env["XSCHED_SCHEDULER"] = "GLB"
    env["XSCHED_AUTO_XQUEUE"] = "ON"
    env["XSCHED_AUTO_XQUEUE_LEVEL"] = "1"
    env["XSCHED_AUTO_XQUEUE_PRIORITY"] = "0"
    env["XSCHED_AUTO_XQUEUE_THRESHOLD"] = "16"
    env["XSCHED_AUTO_XQUEUE_BATCH_SIZE"] = "8"
    return env


def run_workload_in_process(
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
    image_size: int = 224,
    layers: int = 12,
    hidden: int = 768,
    seq_len: int = 256,
) -> dict:
    """Run VLA workload in current process, return stats dict."""
    workload = VLAWorkload(
        image_size=image_size,
        layers=layers,
        hidden=hidden,
        seq_len=seq_len,
    )
    collector = MetricsCollector(
        skill_name="xsched_overhead",
        scheduler_enabled=False,
        total_iterations=iterations,
        warmup_iterations=warmup,
    )
    collector.start()
    for i in range(iterations):
        t_start = collector.begin_iteration()
        workload.run()
        collector.end_iteration(t_start)
        if (i + 1) % 50 == 0:
            logger.info("Iteration %d/%d", i + 1, iterations)
    metrics = collector.finish()
    return metrics.compute_stats()


def _get_clean_env() -> dict:
    """Environment without xsched (remove LD_PRELOAD and xsched lib from path)."""
    env = os.environ.copy()
    # Remove xsched lib from LD_LIBRARY_PATH to avoid loading shim
    lp = env.get("LD_LIBRARY_PATH", "")
    if lp:
        home = os.path.expanduser("~")
        robonix_lib = os.path.join(home, ".robonix", "lib")
        parts = [p for p in lp.split(":") if p.rstrip("/") != robonix_lib.rstrip("/")]
        env["LD_LIBRARY_PATH"] = ":".join(parts) if parts else ""
    env.pop("LD_PRELOAD", None)
    for k in list(env.keys()):
        if k.startswith("XSCHED_"):
            env.pop(k)
    return env


def run_as_subprocess(xsched_enabled: bool, iterations: int, warmup: int) -> dict:
    """Run workload in subprocess with or without xsched env. Returns stats dict."""
    cmd = [
        sys.executable, "-m", "scheduler_benchmark.xsched_overhead",
        "--worker",
        "--iterations", str(iterations),
        "--warmup", str(warmup),
    ]
    env = _get_xsched_env() if xsched_enabled else _get_clean_env()
    if xsched_enabled and env is None:
        raise RuntimeError(
            "xsched env not found (~/.robonix/lib). "
            "Run: make init-xsched && make build-xsched && make install-xsched"
        )
    label = "xsched" if xsched_enabled else "no_xsched"
    logger.info("Starting subprocess (%s)...", label)
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        logger.error("Subprocess failed (code=%d): %s", proc.returncode, proc.stderr)
        err = proc.stderr[:500]
        if "cannot send event to server" in err or "no receiver" in err.lower():
            raise RuntimeError(
                f"xsched worker failed (xserver not running?): {err}\n"
                "Start xserver first: ~/.robonix/bin/xserver HPF 50000"
            )
        raise RuntimeError(f"Worker failed: {err}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from worker: %s", proc.stdout[:200])
        raise RuntimeError(f"Worker output parse error: {e}") from e


def print_overhead_report(no_xsched: dict, with_xsched: dict) -> None:
    """Print human-readable overhead comparison."""
    def get_lat(name: str, d: dict) -> float:
        return d.get("latency", {}).get(name, 0) or 0

    mean_a = get_lat("mean_ms", no_xsched)
    mean_b = get_lat("mean_ms", with_xsched)
    p50_a = get_lat("p50_ms", no_xsched)
    p50_b = get_lat("p50_ms", with_xsched)
    p99_a = get_lat("p99_ms", no_xsched)
    p99_b = get_lat("p99_ms", with_xsched)

    if mean_a <= 0:
        logger.error("No baseline data")
        return

    overhead_mean_pct = 100.0 * (mean_b - mean_a) / mean_a
    overhead_p50_pct = 100.0 * (p50_b - p50_a) / p50_a if p50_a else 0
    overhead_p99_pct = 100.0 * (p99_b - p99_a) / p99_a if p99_a else 0
    overhead_mean_abs_ms = mean_b - mean_a

    print()
    print("=" * 70)
    print("  XSCHED OVERHEAD REPORT")
    print("  Pure GPU workload (VLA), no background contention")
    print("=" * 70)
    print()
    print("  Condition              Mean (ms)   P50 (ms)   P99 (ms)")
    print("  " + "-" * 50)
    print(f"  No xsched (baseline)    {mean_a:>8.2f}   {p50_a:>8.2f}   {p99_a:>8.2f}")
    print(f"  With xsched            {mean_b:>8.2f}   {p50_b:>8.2f}   {p99_b:>8.2f}")
    print()
    print("  XSched Overhead:")
    print(f"    Mean latency:        +{overhead_mean_abs_ms:.2f} ms  ({overhead_mean_pct:+.1f}%)")
    print(f"    Median (P50):        {overhead_p50_pct:+.1f}%")
    print(f"    P99:                 {overhead_p99_pct:+.1f}%")
    print()
    if overhead_mean_pct > 5:
        print("  Verdict: xsched adds noticeable overhead (>5% mean latency)")
    elif overhead_mean_pct > 0:
        print("  Verdict: xsched adds modest overhead (<5% mean latency)")
    else:
        print("  Verdict: xsched overhead negligible or within noise")
    print("=" * 70)


def run_overhead_benchmark(
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
    output_file: str = "",
) -> dict:
    """
    Run xsched overhead benchmark and return comparison data.
    Callable from runner or other code.
    """
    if not gpu_available():
        raise RuntimeError("CUDA not available. This benchmark requires GPU.")

    logger.info("XSched overhead benchmark: iterations=%d, warmup=%d", iterations, warmup)
    logger.info("Ensure xserver is running for xsched: ~/.robonix/bin/xserver HPF 50000")

    no_xsched = run_as_subprocess(xsched_enabled=False, iterations=iterations, warmup=warmup)
    logger.info("Baseline (no xsched): mean=%.2f ms", no_xsched.get("latency", {}).get("mean_ms", 0))

    time.sleep(2.0)  # Brief cooldown

    with_xsched = run_as_subprocess(xsched_enabled=True, iterations=iterations, warmup=warmup)
    logger.info("With xsched: mean=%.2f ms", with_xsched.get("latency", {}).get("mean_ms", 0))

    print_overhead_report(no_xsched, with_xsched)

    mean_a = no_xsched.get("latency", {}).get("mean_ms", 0) or 0
    mean_b = with_xsched.get("latency", {}).get("mean_ms", 0) or 0
    data = {
        "no_xsched": no_xsched,
        "with_xsched": with_xsched,
        "overhead_pct": 100.0 * (mean_b - mean_a) / max(mean_a, 1e-6),
    }
    if output_file:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved to %s", output_file)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Quantify xsched GPU scheduling overhead",
    )
    parser.add_argument(
        "--worker", action="store_true",
        help="Run as worker subprocess (internal use)",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--output", "-o", type=str, default="",
        help="Save comparison JSON to file",
    )
    args = parser.parse_args()

    if not gpu_available():
        logger.error("CUDA not available. This benchmark requires GPU.")
        sys.exit(1)

    if args.worker:
        # Subprocess mode: run workload and print JSON to stdout
        stats = run_workload_in_process(
            iterations=args.iterations,
            warmup=args.warmup,
        )
        print(json.dumps(stats))
        return

    run_overhead_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
