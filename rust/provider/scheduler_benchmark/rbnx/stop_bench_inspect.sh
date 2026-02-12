#!/bin/bash
# Stop the visual inspection benchmark skill
pkill -f "scheduler_benchmark.skills.inspect_skill" 2>/dev/null || true
