#!/bin/bash
# Start the VLA grasping benchmark skill
cd "$(dirname "$0")/.." || exit 1
python3 -m scheduler_benchmark.skills.vla_skill &
echo $!
