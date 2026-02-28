# SPDX-License-Identifier: MulanPSL-2.0
"""
Metrics collection and statistical analysis for scheduling benchmarks.

Collects per-iteration timing data, computes latency statistics (mean, median,
percentiles, jitter), throughput, and supports serialization to JSON for
cross-run comparison.
"""

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class IterationSample:
    """A single iteration timing sample."""
    iteration: int
    latency_s: float  # Wall-clock seconds for this iteration (computation)
    cycle_s: float    # Total time since last iteration completion
    timestamp: float   # Unix timestamp when iteration ended


@dataclass
class SkillMetrics:
    """Aggregated metrics for one skill benchmark run."""
    skill_name: str
    scheduler_enabled: bool
    total_iterations: int = 0
    warmup_iterations: int = 0
    samples: List[IterationSample] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def add_sample(self, iteration: int, latency_s: float, cycle_s: float, timestamp: float):
        self.samples.append(IterationSample(iteration, latency_s, cycle_s, timestamp))

    @property
    def effective_samples(self) -> List[IterationSample]:
        """Samples after warmup period."""
        return [s for s in self.samples if s.iteration >= self.warmup_iterations]

    @property
    def latencies(self) -> List[float]:
        """Latency values (seconds) for effective samples."""
        return [s.latency_s for s in self.effective_samples]

    @property
    def cycle_times(self) -> List[float]:
        """Cycle times (seconds) for effective samples."""
        return [s.cycle_s for s in self.effective_samples]

    def compute_stats(self) -> Dict[str, Any]:
        """Compute comprehensive statistics from collected samples."""
        lats = self.latencies
        cycles = self.cycle_times
        if not lats:
            return {"error": "no samples"}

        n = len(lats)
        lats_sorted = sorted(lats)
        cycles_sorted = sorted(cycles)
        
        # Throughput calculation: strictly based on the measured period (excluding warmup)
        measured_duration = sum(cycles)
        throughput = n / measured_duration if measured_duration > 0 else 0

        # Latency metrics (P50, P95, P99)
        mean_lat = sum(lats) / n
        # Median: average of two middle values for even n
        if n % 2 == 1:
            p50_lat = lats_sorted[(n - 1) // 2]
        else:
            p50_lat = (lats_sorted[n // 2 - 1] + lats_sorted[n // 2]) / 2
        p95_lat = lats_sorted[max(0, int(n * 0.95) - 1)]
        p99_lat = lats_sorted[max(0, int(n * 0.99) - 1)]

        # Completion Interval Stability (Cycle Jitter)
        # Measures the variation in time between consecutive iteration completions.
        mean_cycle = sum(cycles) / n
        stddev_cycle = math.sqrt(sum((x - mean_cycle) ** 2 for x in cycles) / n)
        cv_cycle = stddev_cycle / mean_cycle if mean_cycle > 0 else 0
        p95_cycle = cycles_sorted[max(0, int(n * 0.95) - 1)]

        return {
            "skill_name": self.skill_name,
            "scheduler_enabled": self.scheduler_enabled,
            "num_samples": n,
            "measured_duration_s": round(measured_duration, 4),
            "latency": {
                "mean_ms": round(mean_lat * 1000, 3),
                "p50_ms": round(p50_lat * 1000, 3),
                "p95_ms": round(p95_lat * 1000, 3),
                "p99_ms": round(p99_lat * 1000, 3),
            },
            "throughput": {
                "iterations_per_sec": round(throughput, 2),
            },
            "stability": {
                "interval_cv": round(cv_cycle, 4),
                "interval_stddev_ms": round(stddev_cycle * 1000, 3), # Absolute jitter
                "p95_interval_ms": round(p95_cycle * 1000, 3),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization including raw samples."""
        return {
            "stats": self.compute_stats(),
            "raw_samples": [
                {
                    "iteration": s.iteration,
                    "latency_ms": round(s.latency_s * 1000, 4),
                    "timestamp": s.timestamp,
                }
                for s in self.effective_samples
            ],
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result across all skills and conditions."""
    timestamp: str = ""
    hostname: str = ""
    gpu_available: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    skill_results: List[Dict[str, Any]] = field(default_factory=list)

    def add_skill_result(self, metrics: SkillMetrics):
        self.skill_results.append(metrics.to_dict())

    def save(self, path: str):
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkResult":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class MetricsCollector:
    """
    Live metrics collector for a single skill benchmark run.
    Handles warmup, per-iteration timing, and final aggregation.
    """

    def __init__(self, skill_name: str, scheduler_enabled: bool,
                 total_iterations: int, warmup_iterations: int = 10):
        self.metrics = SkillMetrics(
            skill_name=skill_name,
            scheduler_enabled=scheduler_enabled,
            total_iterations=total_iterations,
            warmup_iterations=warmup_iterations,
        )
        self._iteration = 0
        self._t_last_end = 0.0

    def start(self):
        """Mark the start of the benchmark run."""
        self.metrics.start_time = time.time()
        self._t_last_end = time.perf_counter()

    def begin_iteration(self) -> float:
        """Begin timing an iteration. Returns the start timestamp."""
        return time.perf_counter()

    def end_iteration(self, start_time: float):
        """
        End timing an iteration.
        Args:
            start_time: Value returned by begin_iteration().
        """
        now = time.perf_counter()
        elapsed = now - start_time
        # Cycle time is total time between iteration completions
        cycle = now - self._t_last_end if self._t_last_end > 0 else elapsed
        
        self.metrics.add_sample(
            iteration=self._iteration,
            latency_s=elapsed,
            cycle_s=cycle,
            timestamp=time.time(),
        )
        self._iteration += 1
        self._t_last_end = now

    def finish(self) -> SkillMetrics:
        """Mark the end and return completed metrics."""
        self.metrics.end_time = time.time()
        self.metrics.total_iterations = self._iteration
        return self.metrics

    @property
    def is_warmup(self) -> bool:
        """Whether current iteration is in warmup phase."""
        return self._iteration < self.metrics.warmup_iterations

    @property
    def current_iteration(self) -> int:
        return self._iteration
