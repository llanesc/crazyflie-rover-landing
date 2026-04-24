#!/usr/bin/env python3
"""Generic YAML-driven parameter sweep for evaluation scripts.

Reads a sweep config YAML that specifies which eval script to run,
the base eval arguments, and which LandingEnvConfig parameters to sweep.
Builds a Cartesian product grid, runs each combo via subprocess, and
collects results into a resumable CSV.

Usage:
    python scripts/sweep_eval.py sweeps/mass_sweep.yaml
    python scripts/sweep_eval.py sweeps/mass_sweep.yaml --dry-run
    python scripts/sweep_eval.py sweeps/mass_sweep.yaml --output-dir results/sweeps/

Sweep config format:
    eval_script: scripts/eval_mappo_acmpc.py
    experiment: X3
    run: run_20260401120000          # or: checkpoint: path/to/file.pt
    n_episodes: 500
    n_worlds: 100
    deterministic: true
    level: 6
    no_domain_rand: true
    no_disturbance: true

    sweep:
      mass:
        min: 0.0306
        max: 0.0506
        step: 0.002
      rover_vx_max:
        values: [0.25, 0.5, 0.75, 1.0]
"""

import argparse
import csv
import itertools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


METRIC_COLS = [
    "landing_rate", "crash_rate", "oob_rate", "timeout_rate",
    "landings", "crashes", "oob", "timeouts", "n_episodes",
]


def build_param_grid(sweep_cfg: dict) -> tuple[list[str], list[tuple]]:
    """Build Cartesian product grid from sweep config.

    Returns (param_names, grid) where grid is a list of value tuples.
    """
    param_names = sorted(sweep_cfg.keys())
    param_values = []
    for name in param_names:
        spec = sweep_cfg[name]
        if "values" in spec:
            param_values.append([v for v in spec["values"]])
        else:
            vals = np.arange(
                spec["min"], spec["max"] + spec["step"] / 2, spec["step"]
            )
            param_values.append(np.round(vals, 10).tolist())
    return param_names, list(itertools.product(*param_values))


def build_eval_command(
    cfg: dict, override_parts: list[str], output_path: str
) -> list[str]:
    """Build the subprocess command for a single eval run."""
    cmd = [sys.executable, cfg["eval_script"],
           "--experiment", cfg["experiment"]]
    if "run" in cfg:
        cmd += ["--run", cfg["run"]]
    elif "checkpoint" in cfg:
        cmd += ["--checkpoint", cfg["checkpoint"]]
    cmd += ["--n-episodes", str(cfg.get("n_episodes", 100))]
    if cfg.get("n_worlds"):
        cmd += ["--n-worlds", str(cfg["n_worlds"])]
    if cfg.get("deterministic", False):
        cmd.append("--deterministic")
    if cfg.get("level") is not None:
        cmd += ["--level", str(cfg["level"])]
    if cfg.get("no_domain_rand", False):
        cmd.append("--no-domain-rand")
    if cfg.get("no_disturbance", False):
        cmd.append("--no-disturbance")
    cmd += ["--override"] + override_parts
    cmd += ["--output", output_path]
    return cmd


def load_completed(csv_path: Path, param_names: list[str]) -> set[tuple]:
    """Load already-completed parameter combos from an existing CSV."""
    completed = set()
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = tuple(round(float(row[p]), 10) for p in param_names)
                    completed.add(key)
                except (KeyError, ValueError):
                    continue
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep evaluations from a YAML config",
    )
    parser.add_argument("config", type=str, help="Path to sweep YAML config")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for CSV output (default: next to config)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the parameter grid without running")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sweep_cfg = cfg["sweep"]
    param_names, grid = build_param_grid(sweep_cfg)

    print(f"Sweep parameters: {param_names}")
    print(f"Grid size: {len(grid)} combinations")

    if args.dry_run:
        for i, combo in enumerate(grid):
            labels = [f"{n}={v}" for n, v in zip(param_names, combo)]
            print(f"  [{i + 1}/{len(grid)}] {', '.join(labels)}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.config).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_name = Path(args.config).stem + "_results.csv"
    csv_path = output_dir / csv_name

    completed = load_completed(csv_path, param_names)
    if completed:
        print(f"Resuming: {len(completed)}/{len(grid)} already completed")

    header = param_names + METRIC_COLS
    write_header = not csv_path.exists() or len(completed) == 0

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)

        for i, combo in enumerate(grid):
            key = tuple(round(v, 10) for v in combo)
            if key in completed:
                continue

            override_parts = [f"{n}={v}" for n, v in zip(param_names, combo)]

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                tmp_json = tmp.name

            cmd = build_eval_command(cfg, override_parts, tmp_json)

            labels = ", ".join(override_parts)
            print(f"\n[{i + 1}/{len(grid)}] {labels}")
            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  FAILED (exit {result.returncode})")
                if result.stderr:
                    print(result.stderr[-500:])
                Path(tmp_json).unlink(missing_ok=True)
                continue

            elapsed = time.time() - t0
            try:
                with open(tmp_json) as f:
                    metrics = json.load(f)

                row = [f"{v:.6g}" for v in combo]
                row += [str(metrics.get(col, "")) for col in METRIC_COLS]
                writer.writerow(row)
                csvfile.flush()

                lr = metrics.get("landing_rate", 0)
                cr = metrics.get("crash_rate", 0)
                tr = metrics.get("timeout_rate", 0)
                print(f"  landing={lr * 100:.1f}% crash={cr * 100:.1f}% timeout={tr * 100:.1f}% ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  ERROR parsing results: {e}")
            finally:
                Path(tmp_json).unlink(missing_ok=True)

    print(f"\nSweep complete. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
