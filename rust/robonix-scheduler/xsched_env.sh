#!/usr/bin/env bash
# xsched environment for GPU process preload
# Source this before running GPU-using skills/capabilities to enable xsched scheduling.
# Usage: source xsched_env.sh  (or . xsched_env.sh)
#
# Prerequisites:
# 1. Run 'make build-xsched' and 'make install-xsched'
# 2. Start xserver: ~/.robonix/bin/xserver HPF 50000

XSCHED_LIB="${HOME}/.robonix/lib"
XSCHED_BIN="${HOME}/.robonix/bin"

if [ -d "$XSCHED_LIB" ]; then
    export LD_LIBRARY_PATH="${XSCHED_LIB}:${LD_LIBRARY_PATH}"
    export XSCHED_SCHEDULER=GLB
    export XSCHED_AUTO_XQUEUE=ON
    export XSCHED_AUTO_XQUEUE_LEVEL=1
    export XSCHED_AUTO_XQUEUE_PRIORITY=0
    export XSCHED_AUTO_XQUEUE_THRESHOLD=16
    export XSCHED_AUTO_XQUEUE_BATCH_SIZE=8
    echo "xsched env: LD_LIBRARY_PATH includes $XSCHED_LIB"
else
    echo "xsched env: $XSCHED_LIB not found. Run 'make init-xsched && make build-xsched && make install-xsched'"
fi
