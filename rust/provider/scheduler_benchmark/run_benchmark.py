#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""
Robonix Scheduling Benchmark - Main Entry Point

This script orchestrates a complete scheduling benchmark that compares
the robonix-scheduler against default Linux CFS (Completely Fair Scheduler).

The benchmark:
  1. Starts background contention processes (perception, SLAM, speech, motion planning)
  2. Runs each benchmark skill (nav2, VLA grasp, visual inspection) sequentially
  3. Measures per-iteration latency, throughput, and stability (jitter)
  4. Runs the same suite with and without the scheduler
  5. Generates a comparison report

Prerequisites:
  - ROS2 environment sourced
  - robonix-scheduler built and optionally running (for scheduler mode)
  - Python packages: numpy, pyyaml (pip install numpy pyyaml)
  - Optional: torch with CUDA for GPU benchmarks

Quick Start:
  # Full A/B comparison (recommended)
  python3 run_benchmark.py

  # Baseline only (no scheduler)
  python3 run_benchmark.py --baseline-only

  # With scheduler only
  python3 run_benchmark.py --scheduler-only

  # Custom configuration
  python3 run_benchmark.py --config config/benchmark.yaml --runs 3

  # Regenerate report from existing results
  python3 -m scheduler_benchmark.report benchmark_results/<timestamp>
"""

import os
import sys

# Ensure the package is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from scheduler_benchmark.runner import main

if __name__ == "__main__":
    main()
