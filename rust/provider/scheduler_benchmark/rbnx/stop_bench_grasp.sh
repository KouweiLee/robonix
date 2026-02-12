#!/bin/bash
# Stop the VLA grasping benchmark skill
pkill -f "scheduler_benchmark.skills.vla_skill" 2>/dev/null || true
