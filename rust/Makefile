# SPDX-License-Identifier: MulanPSL-2.0
# Robonix Makefile
#
# Convenient commands for building and running Robonix components
#
# Tab completion: bash-completion is installed in Docker container
# If tab completion doesn't work, ensure bash-completion is installed and enabled

.PHONY: help build build-install build-cli build-core build-scheduler build-sdk build-xsched debug release install install-cli install-core install-scheduler install-xsched source-sdk env clean test test-all test-rust test-python test-cpp test-ros2 benchmark check setup-dev fmt test-framework init-xsched bench-scheduler bench-scheduler-baseline bench-scheduler-full bench-report
.DEFAULT_GOAL := help

# Default ROS2 distribution (can be overridden)
ROS2_DISTRO ?= humble
ROS2_SETUP ?= /opt/ros/$(ROS2_DISTRO)/setup.bash

# Build mode: debug (default) or release
BUILD_MODE ?= debug

# Paths
RUST_DIR := $(shell pwd)
CLI_DIR := $(RUST_DIR)/robonix-cli
CORE_DIR := $(RUST_DIR)/robonix-core
SDK_DIR := $(RUST_DIR)/robonix-sdk
SCHEDULER_DIR := $(RUST_DIR)/robonix-scheduler
PROVIDER_DIR := $(RUST_DIR)/provider
XSCHED_DIR := $(RUST_DIR)/lib/xsched

# Default target
help:
	@echo "Robonix Makefile - Available commands:"
	@echo ""
	@echo "Build commands:"
	@echo "  make build          - Build robonix-cli and robonix-core (default: debug mode)"
	@echo "  make build-install  - Build and install all binaries (build + install)"
	@echo "  make debug          - Build in debug mode (default)"
	@echo "  make release        - Build in release mode"
	@echo "  make build-cli      - Build robonix-cli only (uses BUILD_MODE, default: debug)"
	@echo "  make build-core     - Build robonix-core only (uses BUILD_MODE, default: debug)"
	@echo "  make build-scheduler - Build robonix-scheduler only (uses BUILD_MODE, default: debug)"
	@echo "  make build-sdk      - Build robonix-sdk ROS2 interface package"
	@echo "  make build-xsched   - Build xsched (GPU scheduler, requires CUDA). Run 'make init-xsched' first."
	@echo "  make init-xsched    - Initialize xsched submodules (required before build-xsched)"
	@echo "  Note: Set BUILD_MODE=release to build in release mode, or use 'make release'"
	@echo ""
	@echo "Install commands:"
	@echo "  make install        - Install robonix-cli and robonix-core binaries via cargo install"
	@echo "  make install-cli    - Install robonix-cli binaries (rbnx, rbnx-daemon) only"
	@echo "  make install-core   - Install robonix-core binary only"
	@echo "  make install-scheduler - Install robonix-scheduler binary only"
	@echo "  make install-xsched   - Install xsched (xcli, xserver) to ~/.robonix/bin/"
	@echo "  Note: Binaries are installed to ~/.cargo/bin (make sure it's in your PATH)"
	@echo ""
	@echo "Environment commands:"
	@echo "  make source-sdk     - Source robonix-sdk environment (use: eval \$$(make source-sdk))"
	@echo "  make env            - Show environment information"
	@echo ""
	@echo "Development commands:"
	@echo "  make setup-dev      - Setup development environment (link provider directory)"
	@echo "  make fmt            - Format code using cargo fmt"
	@echo "  make check          - Run cargo check on all Rust projects"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "Test commands:"
	@echo "  make test           - Run all tests (Rust, Python)"
	@echo "  make test-rust      - Run Rust CLI stress tests"
	@echo "  make test-python    - Run Python stress tests"
	@echo "  make benchmark      - Run benchmark suite with different concurrency levels"
	@echo ""
	@echo "Scheduler benchmark commands:"
	@echo "  make bench-scheduler       - Run full A/B comparison (baseline vs scheduler)"
	@echo "  make bench-scheduler-baseline - Run baseline only (no scheduler)"
	@echo "  make bench-scheduler-full  - Run with custom iterations (BENCH_ITERS=300)"
	@echo "  make bench-report DIR=<dir> - Regenerate report from existing results"
	@echo ""
	@echo "Examples:"
	@echo "  make build"
	@echo "  make build-install   # build + install in one step"
	@echo "  make install"
	@echo "  rbnx config --show"
	@echo "  rbnx package list"
	@echo "  eval \$$(make source-sdk)"
	@echo ""
	@echo "Tab Completion:"
	@echo "  Tab completion for make targets is enabled automatically in Docker container"
	@echo "  If it doesn't work, ensure bash-completion package is installed"

# Build mode helper: determine cargo build flags
ifeq ($(BUILD_MODE),release)
CARGO_BUILD_FLAGS = --release
BUILD_MODE_DESC = release
else
CARGO_BUILD_FLAGS = 
BUILD_MODE_DESC = debug
endif

# Build commands
build: build-cli build-core build-scheduler
	@echo "✓ All components built successfully ($(BUILD_MODE_DESC) mode)"

build-install: build install
	@echo "✓ Build and install complete"

debug:
	@$(MAKE) BUILD_MODE=debug build

release:
	@$(MAKE) BUILD_MODE=release build

build-cli:
	@echo "Building robonix-cli ($(BUILD_MODE_DESC) mode)..."
	cd $(CLI_DIR) && cargo build $(CARGO_BUILD_FLAGS)
	@echo "✓ robonix-cli built successfully ($(BUILD_MODE_DESC) mode)"

build-core:
	@echo "Building robonix-core ($(BUILD_MODE_DESC) mode)..."
	cd $(CORE_DIR) && cargo build $(CARGO_BUILD_FLAGS)
	@echo "✓ robonix-core built successfully ($(BUILD_MODE_DESC) mode)"

build-scheduler:
	@echo "Building robonix-scheduler ($(BUILD_MODE_DESC) mode)..."
	cd $(SCHEDULER_DIR) && cargo build $(CARGO_BUILD_FLAGS)
	@echo "✓ robonix-scheduler built successfully ($(BUILD_MODE_DESC) mode)"

init-xsched:
	@echo "Initializing xsched submodules..."
	@cd $(XSCHED_DIR) && for sub in CLI11 cpp-httplib cpp-ipc cuxtra jsoncpp tabulate; do \
		git submodule update --init "3rdparty/$$sub" 2>/dev/null || true; \
	done
	@echo "✓ xsched submodules initialized"

build-xsched:
	@echo "Building xsched (CUDA platform)..."
	@if [ ! -d "$(XSCHED_DIR)" ]; then \
		echo "Error: xsched not found. Run 'git submodule update --init rust/lib/xsched'"; \
		exit 1; \
	fi
	cd $(XSCHED_DIR) && make cuda
	@echo "✓ xsched built successfully (output: $(XSCHED_DIR)/output)"

build-sdk:
	@echo "Building robonix-sdk ROS2 interface package..."
	@if [ ! -f $(ROS2_SETUP) ]; then \
		echo "Error: ROS2 setup file not found at $(ROS2_SETUP)"; \
		echo "Please set ROS2_DISTRO environment variable or ensure ROS2 is installed"; \
		exit 1; \
	fi
	cd $(SDK_DIR) && bash build_ros2.sh
	@echo "✓ robonix-sdk built successfully"

# Alias for backward compatibility
buildsdk: build-sdk

# Install commands
install: install-cli install-core install-scheduler
	@echo "✓ All binaries installed successfully"
	@echo ""
	@echo "Binaries installed via cargo install to ~/.cargo/bin"
	@echo "Make sure ~/.cargo/bin is in your PATH"
	@echo ""
	@echo "You can now run:"
	@echo "  rbnx <command>"
	@echo "  rbnx-daemon <command>"
	@echo "  robonix-core"

install-cli:
	@echo "Installing robonix-cli binaries via cargo install..."
	cd $(CLI_DIR) && cargo install --path . --bin rbnx --bin rbnx-daemon
	@echo "✓ Installed rbnx and rbnx-daemon"

install-core:
	@echo "Installing robonix-core binary via cargo install..."
	cd $(CORE_DIR) && cargo install --path .
	@echo "✓ Installed robonix-core"

install-scheduler:
	@echo "Installing robonix-scheduler binary via cargo install..."
	cd $(SCHEDULER_DIR) && cargo install --path .
	@mkdir -p ~/.robonix
	cp $(SCHEDULER_DIR)/scheduler.yaml ~/.robonix/scheduler.yaml
	@echo "✓ Installed scheduler.yaml to ~/.robonix/"
	@$(MAKE) install-xsched
	@echo "✓ Installed robonix-scheduler"

install-xsched:
	@echo "Installing xsched (xcli, xserver) to ~/.robonix/bin/..."
	@mkdir -p ~/.robonix/bin ~/.robonix/lib
	@cp $(SCHEDULER_DIR)/xsched_env.sh ~/.robonix/
	@if [ -f "$(XSCHED_DIR)/output/bin/xcli" ]; then \
		cp $(XSCHED_DIR)/output/bin/xcli $(XSCHED_DIR)/output/bin/xserver ~/.robonix/bin/; \
		cp -a $(XSCHED_DIR)/output/lib/*.so* ~/.robonix/lib/ 2>/dev/null || true; \
		echo "✓ Installed xcli and xserver to ~/.robonix/bin/"; \
		echo "  Start xserver: ~/.robonix/bin/xserver HPF 50000"; \
		echo "  Source xsched env for GPU processes: . ~/.robonix/xsched_env.sh"; \
	else \
		echo "Note: xsched not built. Run 'make init-xsched && make build-xsched' to enable GPU scheduling"; \
	fi

# Environment commands
source-sdk:
	@# Output commands to source SDK environment
	@# Usage: eval $$(make source-sdk)
	@SDK_PATH="$(SDK_DIR)"; \
	if [ -f "$$SDK_PATH/install/setup.bash" ]; then \
		echo "source $$SDK_PATH/install/setup.bash"; \
		echo "export ROBONIX_SDK_PATH=$$SDK_PATH"; \
		echo "echo '✓ robonix-sdk environment sourced'"; \
	else \
		echo "echo 'Error: robonix-sdk not built. Run: make build-sdk'"; \
		exit 1; \
	fi

env:
	@echo "Robonix Environment Information:"
	@echo "================================"
	@echo "RUST_DIR: $(RUST_DIR)"
	@echo "CLI_DIR: $(CLI_DIR)"
	@echo "CORE_DIR: $(CORE_DIR)"
	@echo "SDK_DIR: $(SDK_DIR)"
	@echo "PROVIDER_DIR: $(PROVIDER_DIR)"
	@echo ""
	@echo "Build Configuration:"
	@echo "  BUILD_MODE: $(BUILD_MODE)"
	@echo ""
	@echo "ROS2 Configuration:"
	@echo "  ROS2_DISTRO: $(ROS2_DISTRO)"
	@echo "  ROS2_SETUP: $(ROS2_SETUP)"
	@if [ -f "$(ROS2_SETUP)" ]; then \
		echo "  ROS2 setup: ✓ Found"; \
	else \
		echo "  ROS2 setup: ✗ Not found"; \
	fi
	@echo ""
	@echo "robonix-sdk Status:"
	@if [ -f "$(SDK_DIR)/install/setup.bash" ]; then \
		echo "  Built: ✓ Yes"; \
		echo "  ROBONIX_SDK_PATH: $(SDK_DIR)"; \
	else \
		echo "  Built: ✗ No (run: make build-sdk)"; \
	fi
	@echo ""
	@echo "Rust Projects:"
	@if [ -f "$(CLI_DIR)/Cargo.toml" ]; then \
		echo "  robonix-cli: ✓ Found"; \
	else \
		echo "  robonix-cli: ✗ Not found"; \
	fi
	@if [ -f "$(CORE_DIR)/Cargo.toml" ]; then \
		echo "  robonix-core: ✓ Found"; \
	else \
		echo "  robonix-core: ✗ Not found"; \
	fi
	@echo ""
	@echo "Development Setup:"
	@if [ -L ~/.robonix/packages ] && [ "$(PROVIDER_DIR)" = "$$(readlink -f ~/.robonix/packages)" ]; then \
		echo "  Provider link: ✓ Configured"; \
	else \
		echo "  Provider link: ✗ Not configured (run: make setup-dev)"; \
	fi

# Development setup
setup-dev:
	@echo "Setting up development environment..."
	@mkdir -p ~/.robonix
	@if [ -e ~/.robonix/packages ]; then \
		echo "Removing existing packages path..."; \
		rm -rf ~/.robonix/packages; \
	fi
	@ln -sf "$(PROVIDER_DIR)" ~/.robonix/packages
	@echo "✓ Linked $(PROVIDER_DIR) to ~/.robonix/packages"
	@echo "✓ Development environment setup complete"

# Format code
fmt:
	@echo "Formatting code with cargo fmt..."
	@echo "Formatting robonix-cli..."
	cd $(CLI_DIR) && cargo fmt
	@echo "Formatting robonix-core..."
	cd $(CORE_DIR) && cargo fmt
	@echo "Formatting robonix-scheduler..."
	cd $(SCHEDULER_DIR) && cargo fmt
	@echo "✓ Code formatted"

# Check and test
check:
	@echo "Running cargo check on all Rust projects..."
	@echo "Checking robonix-cli..."
	cd $(CLI_DIR) && cargo check
	@echo "Checking robonix-core..."
	cd $(CORE_DIR) && cargo check
	@echo "Checking robonix-scheduler..."
	cd $(SCHEDULER_DIR) && cargo check
	@echo "✓ All checks passed"

# Test framework paths
TEST_FRAMEWORK_DIR := $(RUST_DIR)/tools/test_framework
TEST_RUST_DIR := $(TEST_FRAMEWORK_DIR)/rust_tests
TEST_PYTHON_DIR := $(TEST_FRAMEWORK_DIR)/python_tests
TEST_CPP_DIR := $(TEST_FRAMEWORK_DIR)/cpp_tests
TEST_ROS2_DIR := $(TEST_FRAMEWORK_DIR)/ros2_tests

# Test parameters (can be overridden)
CONCURRENCY ?= 3
REQUESTS ?= 1000
RATE ?= 100
DURATION ?= 0

# Test commands
test: test-all

test-all: test-rust test-python
	@echo "✓ All tests completed"

test-rust:
	@echo "Running Rust CLI stress tests (concurrency=$(CONCURRENCY), requests=$(REQUESTS))..."
	@if [ -f "$(TEST_FRAMEWORK_DIR)/test.sh" ]; then \
		bash $(TEST_FRAMEWORK_DIR)/test.sh rust rustdds $(CONCURRENCY) $(REQUESTS) $(RATE) $(DURATION); \
	else \
		echo "Test framework not found. Run 'make test-framework' first."; \
		exit 1; \
	fi

test-python:
	@echo "Running Python stress tests (concurrency=$(CONCURRENCY), requests=$(REQUESTS))..."
	@if [ -f "$(TEST_FRAMEWORK_DIR)/test.sh" ]; then \
		bash $(TEST_FRAMEWORK_DIR)/test.sh python rustdds $(CONCURRENCY) $(REQUESTS) $(RATE) $(DURATION); \
	else \
		echo "Test framework not found. Run 'make test-framework' first."; \
		exit 1; \
	fi


benchmark:
	@echo "Running benchmark suite with different concurrency levels..."
	@if [ -f "$(TEST_FRAMEWORK_DIR)/benchmark.sh" ]; then \
		bash $(TEST_FRAMEWORK_DIR)/benchmark.sh; \
	else \
		echo "Benchmark script not found."; \
		exit 1; \
	fi

test-framework:
	@echo "Setting up test framework..."
	@echo "✓ Test framework ready"
	@echo "Run 'make test' to execute all tests"

# Clean commands
clean:
	@echo "Cleaning build artifacts..."
	cd $(CLI_DIR) && cargo clean
	cd $(CORE_DIR) && cargo clean
	cd $(SCHEDULER_DIR) && cargo clean
	@echo "✓ Clean complete"

clean-all: clean
	@echo "Cleaning SDK build artifacts..."
	@if [ -d "$(SDK_DIR)/build" ]; then rm -rf $(SDK_DIR)/build; fi
	@if [ -d "$(SDK_DIR)/install" ]; then rm -rf $(SDK_DIR)/install; fi
	@if [ -d "$(SDK_DIR)/log" ]; then rm -rf $(SDK_DIR)/log; fi
	@echo "✓ All artifacts cleaned"

# ---------------------------------------------------------------------------
# Scheduler Benchmark
# ---------------------------------------------------------------------------
BENCH_DIR := $(PROVIDER_DIR)/scheduler_benchmark
BENCH_ITERS ?= 0

bench-scheduler:
	@echo "Running scheduler benchmark (A/B comparison)..."
	cd $(BENCH_DIR) && python3 run_benchmark.py --config config/benchmark.yaml
	@echo "✓ Benchmark complete. Results in $(BENCH_DIR)/benchmark_results/"

bench-scheduler-baseline:
	@echo "Running scheduler benchmark (baseline only)..."
	cd $(BENCH_DIR) && python3 run_benchmark.py --config config/benchmark.yaml --baseline-only
	@echo "✓ Baseline benchmark complete."

bench-scheduler-full:
	@echo "Running scheduler benchmark (full comparison, $(if $(filter-out 0,$(BENCH_ITERS)),iters=$(BENCH_ITERS),default iters))..."
	cd $(BENCH_DIR) && python3 run_benchmark.py --config config/benchmark.yaml --runs 3
	@echo "✓ Full benchmark complete."

bench-report:
	@if [ -z "$(DIR)" ]; then \
		echo "Usage: make bench-report DIR=benchmark_results/<timestamp>"; \
		exit 1; \
	fi
	cd $(BENCH_DIR) && python3 -m scheduler_benchmark.report "$(DIR)"
