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
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class IterationSample:
    """A single iteration timing sample."""
    iteration: int
    latency_s: float  # Wall-clock seconds for this iteration
    timestamp: float   # Unix timestamp when iteration started


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

    def add_sample(self, iteration: int, latency_s: float, timestamp: float):
        self.samples.append(IterationSample(iteration, latency_s, timestamp))

    @property
    def effective_samples(self) -> List[IterationSample]:
        """Samples after warmup period."""
        return [s for s in self.samples if s.iteration >= self.warmup_iterations]

    @property
    def latencies(self) -> List[float]:
        """Latency values (seconds) for effective samples."""
        return [s.latency_s for s in self.effective_samples]

    def compute_stats(self) -> Dict[str, Any]:
        """Compute comprehensive statistics from collected samples."""
        lats = self.latencies
        if not lats:
            return {"error": "no samples"}

        lats_sorted = sorted(lats)
        n = len(lats_sorted)
        total_time = self.end_time - self.start_time if self.end_time > 0 else 0

        mean = sum(lats) / n
        variance = sum((x - mean) ** 2 for x in lats) / n
        stddev = math.sqrt(variance)

        # Percentiles
        def percentile(p):
            k = (n - 1) * p / 100.0
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return lats_sorted[int(k)]
            return lats_sorted[f] * (c - k) + lats_sorted[c] * (k - f)

        p50 = percentile(50)
        p90 = percentile(90)
        p95 = percentile(95)
        p99 = percentile(99)

        # Throughput: iterations per second
        throughput = n / total_time if total_time > 0 else 0

        # Jitter: coefficient of variation (stddev / mean)
        cv = stddev / mean if mean > 0 else 0

        # Tail ratio: P99 / P50 (higher = more tail latency, less stable)
        tail_ratio = p99 / p50 if p50 > 0 else 0

        # Consecutive jitter: mean absolute difference between adjacent iterations
        consec_diffs = [abs(lats[i + 1] - lats[i]) for i in range(len(lats) - 1)]
        consec_jitter = sum(consec_diffs) / len(consec_diffs) if consec_diffs else 0

        return {
            "skill_name": self.skill_name,
            "scheduler_enabled": self.scheduler_enabled,
            "num_samples": n,
            "warmup_iterations": self.warmup_iterations,
            "total_wall_time_s": round(total_time, 4),
            "latency": {
                "mean_ms": round(mean * 1000, 3),
                "median_ms": round(p50 * 1000, 3),
                "stddev_ms": round(stddev * 1000, 3),
                "min_ms": round(lats_sorted[0] * 1000, 3),
                "max_ms": round(lats_sorted[-1] * 1000, 3),
                "p50_ms": round(p50 * 1000, 3),
                "p90_ms": round(p90 * 1000, 3),
                "p95_ms": round(p95 * 1000, 3),
                "p99_ms": round(p99 * 1000, 3),
            },
            "throughput": {
                "iterations_per_sec": round(throughput, 2),
            },
            "stability": {
                "coefficient_of_variation": round(cv, 4),
                "tail_ratio_p99_p50": round(tail_ratio, 3),
                "consecutive_jitter_ms": round(consec_jitter * 1000, 3),
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

    def start(self):
        """Mark the start of the benchmark run."""
        self.metrics.start_time = time.time()

    def begin_iteration(self) -> float:
        """Begin timing an iteration. Returns the start timestamp."""
        return time.perf_counter()

    def end_iteration(self, start_time: float):
        """
        End timing an iteration.
        Args:
            start_time: Value returned by begin_iteration().
        """
        elapsed = time.perf_counter() - start_time
        self.metrics.add_sample(
            iteration=self._iteration,
            latency_s=elapsed,
            timestamp=time.time(),
        )
        self._iteration += 1

    def record_iteration(self, latency_s: float):
        """Directly record an iteration with known latency."""
        self.metrics.add_sample(
            iteration=self._iteration,
            latency_s=latency_s,
            timestamp=time.time(),
        )
        self._iteration += 1

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
