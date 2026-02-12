#!/bin/bash
# Start the navigation benchmark skill
cd "$(dirname "$0")/.." || exit 1
python3 -m scheduler_benchmark.skills.nav2_skill &
echo $!
