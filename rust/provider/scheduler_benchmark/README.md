# Scheduler Benchmark

A benchmark suite for evaluating the performance of **robonix-scheduler** (priority-based CPU/GPU scheduling) against **default Linux CFS** (Completely Fair Scheduler) in embodied AI workloads.

## Motivation

In a robotic system, multiple subsystems run concurrently — perception, SLAM, speech recognition, motion planning — but at any given moment, only one **skill** (high-level task) is actively executing. Under Linux CFS, all processes receive equal CPU time regardless of importance. The robonix-scheduler addresses this by dynamically boosting the active skill and its dependencies via `nice` / `SCHED_RR` (CPU) and xsched priority hints (GPU).

This benchmark quantifies the scheduling gap by running realistic synthetic workloads with and without the scheduler, measuring:

- **Latency** — mean, P50, P90, P95, P99 per-iteration latency
- **Throughput** — iterations per second
- **Stability** — coefficient of variation (CV), tail ratio (P99/P50), consecutive jitter

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Benchmark Runner                          │
│  (orchestrates processes, triggers skills, collects metrics) │
└────────┬────────────────────────────────┬────────────────────┘
         │                                │
         ▼                                ▼
┌─────────────────────┐      ┌─────────────────────────────┐
│  Background Workers │      │  Foreground Skill (1 at a   │
│  (always running)   │      │  time, measured)             │
│                     │      │                              │
│  • Perception (C+G) │      │  • bench_nav    (CPU-heavy)  │
│  • SLAM       (CPU) │ ◄──► │  • bench_grasp  (GPU-heavy)  │
│  • Speech     (C+G) │      │  • bench_inspect (mixed)     │
│  • Motion Plan(CPU) │      │                              │
└─────────────────────┘      └──────────────────────────────┘
         │                                │
         ▼                                ▼
┌──────────────────────────────────────────────────────────────┐
│              robonix-scheduler (optional)                     │
│  • PIDs registered in-memory via scheduler_register service  │
│  • Boosts active skill's dependencies (nice / SCHED_RR)      │
│  • Sends xsched GPU priority hints for CUDA workloads        │
│  • Zero file I/O during benchmark (no processes.json reads)  │
└──────────────────────────────────────────────────────────────┘
```

### Background Contention Workers

These processes simulate always-running robot subsystems that compete for CPU/GPU:

| Worker | Workload Profile | Default Rate |
|--------|-----------------|--------------|
| **Perception** | Camera image processing + CNN detection + point cloud analysis | 10 Hz |
| **SLAM** | LiDAR scan matching (FFT cross-correlation) + pose graph optimization | 5 Hz |
| **Speech** | Mel spectrogram extraction + transformer decoder inference | 3 Hz |
| **Motion Planning** | Collision checking + trajectory optimization (GEMM) | 8 Hz |

### Benchmark Skills

Foreground skills executed one at a time while background workers run:

| Skill | Type | Workload |
|-------|------|----------|
| `skl::bench_nav` | CPU-heavy | Nav2-like path planning (costmap wavefront expansion) + scan matching |
| `skl::bench_grasp` | GPU-heavy | VLA model inference (image preprocessing + transformer + action decoding) |
| `skl::bench_inspect` | CPU+GPU mixed | Full perception pipeline (image + CNN + point cloud) |

GPU workloads use **PyTorch CUDA** when available; automatic **numpy CPU fallback** when not.

## Prerequisites

- **Python 3.8+**
- **numpy** (`pip install numpy`)
- **PyYAML** (`pip install pyyaml`)
- **ROS 2** environment sourced (for skill ROS2 nodes)
- **robonix-scheduler** built and running (for scheduler mode)
- Optional: **PyTorch with CUDA** for GPU benchmarks (`pip install torch`)
- Optional: **scipy** for optimized image processing

## Quick Start

Follow these steps to run a full A/B comparison benchmark (CFS vs. Robonix).

### 1. Build and Install Components

From the `rust/` directory (root of the workspace):

```bash
# Build and install core, cli, and scheduler
make build-install

# Build and install SDK (ROS2 interface)
make build-sdk

# Build and install xsched (GPU scheduler - requires CUDA)
make init-xsched && make build-xsched && make install-xsched
```

### 2. Prepare Configuration

Copy the benchmark-specific scheduler configuration to your home directory:

```bash
cp provider/scheduler_benchmark/config/scheduler.yaml ~/.robonix/scheduler.yaml
```

### 3. Start Background Services

You will need three terminals (or run in background):

**Terminal 1: Start robonix-scheduler (requires sudo)**
```bash
sudo ~/.cargo/bin/robonix-scheduler
```

**Terminal 2: Start xsched server (for GPU scheduling)**
```bash
~/.robonix/bin/xserver HPF 50000
```

**Terminal 3: Run the Benchmark**
```bash
# Source ROS2 and Robonix SDK
source /opt/ros/humble/setup.bash  # or your distro
eval $(make source-sdk)

# Run the full A/B comparison
make bench-scheduler
```

## Usage

### Running the benchmark script directly

If you want more control, you can run the Python script directly from the `provider/scheduler_benchmark` directory:

```bash
cd provider/scheduler_benchmark

# Full comparison (default)
python3 run_benchmark.py

# Baseline only (no scheduler)
python3 run_benchmark.py --baseline-only

# Scheduler only
python3 run_benchmark.py --scheduler-only

# Custom config and runs
python3 run_benchmark.py --config config/benchmark.yaml --runs 3 --output-dir ./my_results
```

### Advanced GPU Scheduling Options

The benchmark automatically detects if `xsched` is running. To ensure it's used:

1.  Verify `xserver` is running on port 50000.
2.  Ensure `xsched: enabled: true` is set in `~/.robonix/scheduler.yaml`.
3.  Always `source ~/.robonix/xsched_env.sh` before running the benchmark to intercept CUDA calls.

## Configuration

All benchmark parameters are configurable via `config/benchmark.yaml`:

```yaml
output_dir: benchmark_results
settle_time: 3.0          # Seconds to let background workers stabilize
num_runs: 1               # Runs per skill per condition

background_workers:
  perception:
    rate_hz: 10.0          # Iteration rate (higher = more contention)
  slam:
    rate_hz: 5.0
  speech:
    rate_hz: 3.0
  motion_plan:
    rate_hz: 8.0

skills:
  - name: "skl::bench_nav"
    iterations: 200        # Total iterations
    warmup: 20             # Iterations to discard (warm-up)
    params:
      grid_size: 300       # Larger = heavier CPU load
```

### Tuning for your hardware

- **Increase contention**: Raise `rate_hz` on background workers, or increase workload sizes
- **More stable statistics**: Increase `iterations` and `num_runs`
- **Faster runs**: Decrease `iterations` and `warmup`, reduce workload sizes
- **GPU stress**: Increase `layers`, `hidden`, `seq_len` for `bench_grasp`

## Output

Results are saved to `benchmark_results/<timestamp>/`:

```
benchmark_results/20260212_110000/
├── config.json                # Configuration used
├── baseline/
│   ├── skl__bench_nav_run0.json      # Per-skill raw data
│   ├── skl__bench_grasp_run0.json
│   └── skl__bench_inspect_run0.json
├── scheduler/
│   ├── skl__bench_nav_run0.json
│   ├── skl__bench_grasp_run0.json
│   └── skl__bench_inspect_run0.json
├── baseline_results.json      # Aggregated baseline
├── scheduler_results.json     # Aggregated scheduler
├── report.txt                 # Human-readable comparison
└── report.json                # Machine-readable comparison
```

### Sample report output

```
================================================================================
  ROBONIX SCHEDULING BENCHMARK REPORT
  Comparing: Linux CFS (baseline) vs robonix-scheduler
================================================================================

  skl::bench_nav
  ============================================================================
  Latency:
    Mean Latency (ms)                      3.53         2.89      -18.1% v
    P99 Latency (ms)                       5.21         3.12      -40.1% v
  Throughput:
    Iterations/sec                       283.29       345.95      +22.1% ^
  Stability:
    Coeff of Variation                   0.1520       0.0340      -77.6% v
    Consecutive Jitter (ms)              0.4200       0.0800      -81.0% v
```

## Project Structure

```
scheduler_benchmark/
├── scheduler_benchmark/
│   ├── workloads.py       # CPU/GPU synthetic workload generators
│   ├── metrics.py         # Per-iteration timing + statistical analysis
│   ├── background.py      # Background contention worker processes
│   ├── runner.py          # Benchmark orchestrator
│   ├── report.py          # Results comparison & report generation
│   └── skills/
│       ├── nav2_skill.py      # Navigation benchmark (ROS2 node)
│       ├── vla_skill.py       # VLA grasping benchmark (ROS2 node)
│       └── inspect_skill.py   # Visual inspection benchmark (ROS2 node)
├── config/
│   ├── benchmark.yaml     # Benchmark parameters
│   └── scheduler.yaml     # Scheduler config for benchmark skills
├── rbnx/                  # Start/stop scripts for rbnx integration
├── rbnx_manifest.yaml     # Robonix package manifest
├── run_benchmark.py       # Entry point
├── setup.py
└── package.xml
```
