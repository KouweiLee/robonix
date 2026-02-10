# Robonix Scheduler

Robonix Scheduler is a **Policy Governor Service** designed to manage the runtime priorities of various robot components based on the active tasks (skills) being performed.

## Overview

The scheduler monitors "skill" activation requests via a ROS 2 service and adjusts the Linux scheduling policies and nice values of dependent processes. This ensures that latency-critical components (like motion control) receive higher priority than background tasks when needed.

### Key Features

- **Dynamic Priority Adjustment**: Escalates or de-escalates process priorities in real-time.
- **Support for Multiple Scheduling Policies**:
  - **Latency Critical**: Uses `SCHED_RR` (Real-Time Round Robin) with fallback to high-priority CFS (Nice -15).
  - **Throughput Critical**: Uses high-priority CFS (Nice -10).
- **Process Tracking**: Automatically maps skill dependencies to PIDs using a shared state file (`~/.robonix/processes.json`).
- **ROS 2 Integration**: Provides the `scheduler_policy` service for skill-based priority management.

## Prerequisites

- **ROS 2**: A working ROS 2 installation (configured via `ros2-client`).
- **Process State File**: The scheduler expects a JSON file at `~/.robonix/processes.json` containing information about running Robonix processes (typically managed by the Robonix CLI).
- **Permissions**: Root privileges or `CAP_SYS_NICE` capabilities are required to adjust the scheduling policy of other processes.

## Usage

### Building

Build the project using Cargo:

```bash
cargo build --release
```

### Running

To run the scheduler with default logging:

```bash
sudo cargo run
```

### Debugging

To run the scheduler and see all debug output:

```bash
sudo RUST_LOG=debug cargo run
# or
sudo ./target/debug/robonix-scheduler
```

## Service API

The scheduler exposes a ROS 2 service:

- **Service Name**: `scheduler_policy`
- **Service Type**: `robonix_sdk/AdjustPriority`
- **Request**:
  - `skill_name` (string): The name of the skill (e.g., `move_to_object`).
  - `high_priority` (bool): `true` to escalate dependencies, `false` to restore them.
