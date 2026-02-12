# SPDX-License-Identifier: MulanPSL-2.0
"""
Benchmark Report Generator - Compares scheduler vs baseline results.

Generates a text-based comparison report showing:
  - Per-skill latency comparison (mean, P50, P95, P99)
  - Throughput comparison
  - Stability comparison (CV, tail ratio, consecutive jitter)
  - Improvement percentages
  - Summary verdict
"""

import json
import os
import sys
import logging
from typing import Dict, List, Optional, Any

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

    overall_improvements = []

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
            ("Median Latency (ms)", "median_ms"),
            ("P90 Latency (ms)", "p90_ms"),
            ("P95 Latency (ms)", "p95_ms"),
            ("P99 Latency (ms)", "p99_ms"),
            ("Min Latency (ms)", "min_ms"),
            ("Max Latency (ms)", "max_ms"),
            ("Std Dev (ms)", "stddev_ms"),
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
            if key == "mean_ms":
                overall_improvements.append(pct)

        # Throughput (higher is better)
        lines.append(f"  {'Throughput:'}")
        bv = bl_tp.get("iterations_per_sec", 0)
        sv = sc_tp.get("iterations_per_sec", 0)
        pct = _pct_change_higher_better(bv, sv)
        arrow = "^" if pct > 0 else "v" if pct < 0 else "="
        lines.append(
            f"    {'Iterations/sec':<33} {bv:>10.2f}   {sv:>10.2f}   "
            f"{pct:>+8.1f}% {arrow}"
        )

        # Stability metrics (lower is better)
        stability_metrics = [
            ("Coeff of Variation", "coefficient_of_variation"),
            ("Tail Ratio (P99/P50)", "tail_ratio_p99_p50"),
            ("Consecutive Jitter (ms)", "consecutive_jitter_ms"),
        ]

        lines.append(f"  {'Stability:'}")
        for label, key in stability_metrics:
            bv = bl_stab.get(key, 0)
            sv = sc_stab.get(key, 0)
            pct = _pct_change(bv, sv)
            arrow = "v" if pct < 0 else "^" if pct > 0 else "="
            lines.append(
                f"    {label:<33} {bv:>10.4f}   {sv:>10.4f}   "
                f"{pct:>+8.1f}% {arrow}"
            )
            if key == "coefficient_of_variation":
                overall_improvements.append(pct)

        lines.append("")

    # Overall summary
    lines.append("=" * 80)
    lines.append("  SUMMARY")
    lines.append("=" * 80)
    if overall_improvements:
        avg_improvement = sum(overall_improvements) / len(overall_improvements)
        lines.append(
            f"  Average metric change: {avg_improvement:+.1f}%"
        )
        if avg_improvement < -5:
            lines.append(
                "  Verdict: robonix-scheduler provides SIGNIFICANT improvement"
            )
        elif avg_improvement < -1:
            lines.append(
                "  Verdict: robonix-scheduler provides MODERATE improvement"
            )
        elif avg_improvement < 1:
            lines.append(
                "  Verdict: Results are COMPARABLE (within noise)"
            )
        else:
            lines.append(
                "  Verdict: Baseline performs better (check configuration)"
            )
    lines.append("")
    lines.append("  Legend: v = lower (better for latency/jitter)")
    lines.append("          ^ = higher (better for throughput)")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Save report
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)

    # Also save as JSON for programmatic access
    json_report = {
        "skills": {},
        "overall_avg_change_pct": sum(overall_improvements) / len(overall_improvements)
        if overall_improvements else 0,
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
                    "cv_pct": _pct_change(
                        bl.get("stability", {}).get("coefficient_of_variation", 0),
                        sc.get("stability", {}).get("coefficient_of_variation", 0),
                    ),
                    "tail_ratio_pct": _pct_change(
                        bl.get("stability", {}).get("tail_ratio_p99_p50", 0),
                        sc.get("stability", {}).get("tail_ratio_p99_p50", 0),
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
