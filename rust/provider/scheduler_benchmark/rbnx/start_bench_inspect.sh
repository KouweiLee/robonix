#!/bin/bash
# Start the visual inspection benchmark skill
cd "$(dirname "$0")/.." || exit 1
python3 -m scheduler_benchmark.skills.inspect_skill &
echo $!
