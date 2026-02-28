# SPDX-License-Identifier: MulanPSL-2.0
"""
Benchmark Report Generator - Compares scheduler vs baseline results.

Generates a text-based comparison report showing:
  - Per-skill latency comparison (mean, P50, P95, P99)
  - Throughput comparison
  - Stability comparison (interval CV, interval stddev, P95 interval)
  - Per-dimension summary (scheduler vs baseline)
  - Overall verdict
"""

import json
import os
import sys
import logging
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark.report")


def _pct_change(baseline: float, scheduler: float) -> float:
    """Compute percentage change: negative = improvement (lower is better)."""
    if baseline == 0:
        return 0.0
    return ((scheduler - baseline) / baseline) * 100.0


def _pct_change_higher_better(baseline: float, scheduler: float) -> float:
    """Compute percentage change where higher is better (throughput)."""
    if baseline == 0:
        return 0.0
    return ((scheduler - baseline) / baseline) * 100.0


def _extract_skill_stats(results: List[dict]) -> Dict[str, dict]:
    """Extract stats indexed by skill name from result list."""
    stats_by_skill = {}
    for r in results:
        if r is None:
            continue
        stats = r.get("stats", r)
        name = stats.get("skill_name", "unknown")
        stats_by_skill[name] = stats
    return stats_by_skill


def generate_comparison_report(
    baseline_results: List[dict],
    scheduler_results: List[dict],
    output_dir: str,
) -> str:
    """
    Generate a comparison report between baseline and scheduler results.

    Returns the report as a string and saves it to output_dir/report.txt.
    """
    baseline_stats = _extract_skill_stats(baseline_results)
    scheduler_stats = _extract_skill_stats(scheduler_results)

    all_skills = sorted(set(baseline_stats.keys()) | set(scheduler_stats.keys()))

    lines = []
    lines.append("=" * 80)
    lines.append("  ROBONIX SCHEDULING BENCHMARK REPORT")
    lines.append("  Comparing: Linux CFS (baseline) vs robonix-scheduler")
    lines.append("=" * 80)
    lines.append("")

    # Summary table header
    lines.append("-" * 80)
    lines.append(f"{'Metric':<35} {'Baseline':>12} {'Scheduler':>12} {'Change':>12}")
    lines.append("-" * 80)

    # Per-dimension improvement tracking (for summary)
    # Negative = improvement for latency/stability; for throughput we store -pct
    latency_changes: List[float] = []
    throughput_changes: List[float] = []
    stability_changes: List[float] = []
    # For overall avg: mean_latency, p95_latency, throughput, interval_stddev_ms
    # Use stddev (not CV) for stability: within-skill comparison, absolute jitter in ms is more meaningful
    overall_changes: List[float] = []

    for skill_name in all_skills:
        bl = baseline_stats.get(skill_name)
        sc = scheduler_stats.get(skill_name)
        if not bl or not sc:
            lines.append(f"\n  {skill_name}: SKIPPED (missing data)")
            continue

        lines.append(f"\n  {skill_name}")
        lines.append(f"  {'=' * 76}")

        bl_lat = bl.get("latency", {})
        sc_lat = sc.get("latency", {})
        bl_tp = bl.get("throughput", {})
        sc_tp = sc.get("throughput", {})
        bl_stab = bl.get("stability", {})
        sc_stab = sc.get("stability", {})

        # Latency metrics (lower is better)
        latency_metrics = [
            ("Mean Latency (ms)", "mean_ms"),
            ("P50 Latency (ms)", "p50_ms"),
            ("P95 Latency (ms)", "p95_ms"),
            ("P99 Latency (ms)", "p99_ms"),
        ]

        lines.append(f"  {'Latency:'}")
        for label, key in latency_metrics:
            bv = bl_lat.get(key, 0)
            sv = sc_lat.get(key, 0)
            pct = _pct_change(bv, sv)
            arrow = "v" if pct < 0 else "^" if pct > 0 else "="
            lines.append(
                f"    {label:<33} {bv:>10.2f}   {sv:>10.2f}   "
                f"{pct:>+8.1f}% {arrow}"
            )
            if key in ("mean_ms", "p95_ms"):
                latency_changes.append(pct)
                overall_changes.append(pct)

        # Throughput (higher is better) - captures scheduling/wait time, not redundant with latency
        lines.append(f"  {'Throughput (Excl. Warmup):'}")
        bv = bl_tp.get("iterations_per_sec", 0)
        sv = sc_tp.get("iterations_per_sec", 0)
        pct = _pct_change_higher_better(bv, sv)
        arrow = "^" if pct > 0 else "v" if pct < 0 else "="
        lines.append(
            f"    {'Iterations/sec':<33} {bv:>10.2f}   {sv:>10.2f}   "
            f"{pct:>+8.1f}% {arrow}"
        )
        throughput_changes.append(-pct)  # Negate so negative = improvement
        overall_changes.append(-pct)

        # Stability metrics (lower is better)
        # StdDev: absolute jitter in ms; preferred for within-skill baseline vs scheduler comparison.
        # CV: relative; can be distorted when mean cycle changes significantly.
        stability_metrics = [
            ("Interval CV (coeff. of var.)", "interval_cv"),
            ("Interval Jitter (StdDev ms)", "interval_stddev_ms"),
            ("P95 Interval (ms)", "p95_interval_ms"),
        ]

        lines.append(f"  {'Stability (Completion Intervals):'}")
        for label, key in stability_metrics:
            bv = bl_stab.get(key, 0)
            sv = sc_stab.get(key, 0)
            pct = _pct_change(bv, sv)
            arrow = "v" if pct < 0 else "^" if pct > 0 else "="
            lines.append(
                f"    {label:<33} {bv:>10.4f}   {sv:>10.4f}   "
                f"{pct:>+8.1f}% {arrow}"
            )
            if key == "interval_stddev_ms":
                stability_changes.append(pct)
                overall_changes.append(pct)

        lines.append("")

    # Overall summary
    lines.append("=" * 80)
    lines.append("  SUMMARY")
    lines.append("=" * 80)

    if overall_changes:
        # Per-dimension averages
        avg_latency = sum(latency_changes) / len(latency_changes) if latency_changes else 0
        avg_throughput = sum(throughput_changes) / len(throughput_changes) if throughput_changes else 0
        avg_stability = sum(stability_changes) / len(stability_changes) if stability_changes else 0
        avg_overall = sum(overall_changes) / len(overall_changes)

        def _dim_line(label: str, val: float, lower_better: bool) -> str:
            if abs(val) < 0.1:
                return f"  {label:<35} scheduler ~0% change  (comparable)"
            if lower_better:
                direction = "lower" if val < 0 else "higher"
                verdict = "better" if val < 0 else "worse"
            else:
                direction = "higher" if val < 0 else "lower"
                verdict = "better" if val < 0 else "worse"
            return f"  {label:<35} scheduler {abs(val):.1f}% {direction:6}  ({verdict})"

        lines.append("  Per-dimension (scheduler vs baseline):")
        lines.append(_dim_line("Latency (mean + P95):", avg_latency, lower_better=True))
        lines.append(_dim_line("Throughput:", avg_throughput, lower_better=False))  # stored as -pct
        lines.append(_dim_line("Stability (interval StdDev):", avg_stability, lower_better=True))
        lines.append("")
        lines.append(
            f"  Overall average: {avg_overall:+.1f}% "
            "(mean latency, P95 latency, throughput, interval stddev)"
        )

        # Verdict: >= 2 per-dimension metrics improve -> scheduler better; any >20% -> mention regression
        dim_metrics = [
            ("Latency", avg_latency),
            ("Throughput", avg_throughput),
            ("Stability", avg_stability),
        ]
        good_count = sum(1 for _, v in dim_metrics if v < 0)
        regressed = [(name, v) for name, v in dim_metrics if v > 20]

        if good_count >= 2:
            verdict = "Verdict: robonix-scheduler performs better"
        else:
            verdict = "Verdict: baseline (Linux CFS) performs better"

        if regressed:
            regressed_str = ", ".join(f"{name} (+{v:.0f}%)" for name, v in regressed)
            verdict += f"; {regressed_str} regressed significantly (>20%)"
        verdict = "  " + verdict
        lines.append(verdict)
    lines.append("")
    lines.append("  Legend: v = decrease (better for latency/stability)")
    lines.append("          ^ = increase (better for throughput, worse for stability)")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Save report
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)

    # Also save as JSON for programmatic access
    avg_overall_pct = sum(overall_changes) / len(overall_changes) if overall_changes else 0
    json_report = {
        "skills": {},
        "overall_avg_change_pct": avg_overall_pct,
        "summary": {
            "avg_latency_pct": sum(latency_changes) / len(latency_changes) if latency_changes else 0,
            "avg_throughput_pct": sum(throughput_changes) / len(throughput_changes) if throughput_changes else 0,
            "avg_stability_pct": sum(stability_changes) / len(stability_changes) if stability_changes else 0,
            "overall_metrics": "mean_latency, p95_latency, throughput, interval_stddev_ms",
        },
    }
    for skill_name in all_skills:
        bl = baseline_stats.get(skill_name, {})
        sc = scheduler_stats.get(skill_name, {})
        if bl and sc:
            json_report["skills"][skill_name] = {
                "baseline": bl,
                "scheduler": sc,
                "changes": {
                    "mean_latency_pct": _pct_change(
                        bl.get("latency", {}).get("mean_ms", 0),
                        sc.get("latency", {}).get("mean_ms", 0),
                    ),
                    "p99_latency_pct": _pct_change(
                        bl.get("latency", {}).get("p99_ms", 0),
                        sc.get("latency", {}).get("p99_ms", 0),
                    ),
                    "throughput_pct": _pct_change_higher_better(
                        bl.get("throughput", {}).get("iterations_per_sec", 0),
                        sc.get("throughput", {}).get("iterations_per_sec", 0),
                    ),
                    "jitter_stddev_pct": _pct_change(
                        bl.get("stability", {}).get("interval_stddev_ms", 0),
                        sc.get("stability", {}).get("interval_stddev_ms", 0),
                    ),
                    "p95_interval_pct": _pct_change(
                        bl.get("stability", {}).get("p95_interval_ms", 0),
                        sc.get("stability", {}).get("p95_interval_ms", 0),
                    ),
                },
            }

    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    # Print report to stdout
    print(report)
    return report


def main():
    """CLI entry point for regenerating reports from saved results."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report")
    parser.add_argument("result_dir", help="Directory containing baseline/scheduler results")
    args = parser.parse_args()

    baseline_path = os.path.join(args.result_dir, "baseline_results.json")
    scheduler_path = os.path.join(args.result_dir, "scheduler_results.json")

    if not os.path.exists(baseline_path) or not os.path.exists(scheduler_path):
        logger.error("Missing result files in %s", args.result_dir)
        sys.exit(1)

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(scheduler_path) as f:
        scheduler = json.load(f)

    generate_comparison_report(baseline, scheduler, args.result_dir)


if __name__ == "__main__":
    main()
