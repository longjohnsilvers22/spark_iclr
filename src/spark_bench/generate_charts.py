#!/usr/bin/env python3
"""
Generate LIBERO-PRO comparison charts.

Compares SPARK (GT + SAM3D) vs CaP-X (18%) vs VLA baselines.
Reads results from ~/spark/videos/libero_pro/results/*.json
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path.home() / 'spark' / 'videos' / 'libero_pro' / 'results'
CHARTS_DIR = Path.home() / 'spark' / 'videos' / 'libero_pro' / 'charts'


# CaP-X baselines from paper (Table 2, arXiv:2603.22435)
CAP_X_RESULTS = {
    'CaP-Agent0': {'position': 0.20, 'task': 0.16, 'language': 0.18},
    'Pi0.5': {'position': 0.21, 'task': 0.01, 'language': 0.96},
    'Pi0': {'position': 0.00, 'task': 0.00, 'language': 0.91},
    'OpenVLA': {'position': 0.00, 'task': 0.00, 'language': 0.97},
}


def load_spark_results(results_dir: Path = RESULTS_DIR) -> dict:
    """
    Load SPARK results from JSON files.
    """
    results = {}
    for suite in ['object', 'spatial', 'goal', '10']:
        path = results_dir / f'{suite}.json'
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results[suite] = data.get('perturbations', {})
    return results


def aggregate_results(results: dict) -> dict:
    """
    Aggregate per-suite results into overall metrics.
    """
    agg = {}
    for ptype in ['language', 'position', 'task']:
        total_pass = 0
        total_tasks = 0
        for suite, perturbs in results.items():
            if ptype in perturbs:
                total_pass += perturbs[ptype]['pass']
                total_tasks += perturbs[ptype]['total']
        if total_tasks > 0:
            agg[ptype] = total_pass / total_tasks
        else:
            agg[ptype] = 0.0
    return agg


def plot_overall_comparison(spark_gt: dict, spark_sam3d: dict = None):
    """
    Bar chart comparing SPARK vs baselines across perturbation types.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    ptypes = ['language', 'position', 'task']
    ptype_labels = ['Semantic\n(Language)', 'Position\n(Swap)', 'Task\n(New Goal)']

    methods = ['SPARK (GT)', 'CaP-Agent0', 'Pi0.5', 'Pi0', 'OpenVLA']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']

    data = {
        'SPARK (GT)': [spark_gt.get(p, 0) for p in ptypes],
        'CaP-Agent0': [CAP_X_RESULTS['CaP-Agent0'].get(p, 0) for p in ptypes],
        'Pi0.5': [CAP_X_RESULTS['Pi0.5'].get(p, 0) for p in ptypes],
        'Pi0': [CAP_X_RESULTS['Pi0'].get(p, 0) for p in ptypes],
        'OpenVLA': [CAP_X_RESULTS['OpenVLA'].get(p, 0) for p in ptypes],
    }

    if spark_sam3d:
        methods.insert(1, 'SPARK (SAM3D)')
        colors.insert(1, '#03A9F4')
        data['SPARK (SAM3D)'] = [spark_sam3d.get(p, 0) for p in ptypes]

    x = np.arange(len(ptypes))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (method, color) in enumerate(zip(methods, colors)):
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, data[method], width, label=method, color=color,
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, data[method]):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.0%}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Success Rate', fontsize=14)
    ax.set_title('LIBERO-PRO: SPARK vs Baselines (3 Perturbation Types)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(ptype_labels, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.18, color='gray', linestyle='--', alpha=0.5, label='CaP-X 18%')

    plt.tight_layout()
    plt.savefig(str(CHARTS_DIR / 'libero_pro_overall.png'), dpi=150)
    print(f"Saved: {CHARTS_DIR / 'libero_pro_overall.png'}")
    plt.close()


def plot_per_suite(spark_gt: dict):
    """
    Per-suite breakdown for SPARK.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    suites = list(spark_gt.keys())
    ptypes = ['language', 'position', 'task']
    ptype_labels = ['Semantic', 'Position', 'Task']
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(suites))
    width = 0.25

    for i, (ptype, label, color) in enumerate(zip(ptypes, ptype_labels, colors)):
        values = []
        for suite in suites:
            if suite in spark_gt and ptype in spark_gt[suite]:
                values.append(spark_gt[suite][ptype]['rate'])
            else:
                values.append(0)
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color,
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.0%}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Success Rate', fontsize=14)
    ax.set_title('SPARK LIBERO-PRO: Per-Suite Breakdown', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in suites], fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(CHARTS_DIR / 'libero_pro_per_suite.png'), dpi=150)
    print(f"Saved: {CHARTS_DIR / 'libero_pro_per_suite.png'}")
    plt.close()


def main():
    gt_results = load_spark_results(RESULTS_DIR)
    if not gt_results:
        print("No results found. Run the benchmark first.")
        return

    print("SPARK LIBERO-PRO Results")
    for suite, perturbs in gt_results.items():
        print(f"\n{suite}:")
        for ptype, data in perturbs.items():
            print(f"{ptype}: {data['pass']}/{data['total']} ({data['rate']:.0%})")

    agg = aggregate_results(gt_results)
    print(f"\nOverall: {agg}")

    # Load SAM3D results if available
    sam3d_dir = RESULTS_DIR.parent / 'sam3d_results'
    sam3d_results = load_spark_results(sam3d_dir) if sam3d_dir.exists() else {}
    sam3d_agg = aggregate_results(sam3d_results) if sam3d_results else None

    plot_overall_comparison(agg, sam3d_agg)
    plot_per_suite(gt_results)

    print("\nCharts saved to:", CHARTS_DIR)


if __name__ == '__main__':
    main()
