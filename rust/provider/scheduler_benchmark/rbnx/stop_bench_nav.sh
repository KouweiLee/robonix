#!/bin/bash
# Stop the navigation benchmark skill
pkill -f "scheduler_benchmark.skills.nav2_skill" 2>/dev/null || true
