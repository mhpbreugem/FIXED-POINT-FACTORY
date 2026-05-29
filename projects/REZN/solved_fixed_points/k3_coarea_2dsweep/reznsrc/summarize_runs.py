"""Summarise a directory of staggered-run .npz files into a table.

Usage: python -m code.summarize_runs <dir>

Reads every .npz in the directory and writes a simple Markdown table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--filter", type=str, default="",
                   help="only include filenames containing this substring")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for npz in sorted(args.directory.glob("*.npz")):
        if args.filter and args.filter not in npz.name:
            continue
        d = np.load(npz, allow_pickle=False)
        try:
            stages = list(d["elapsed_s"])
            row = {
                "file": npz.stem,
                "K": int(d["K"]),
                "G_inner": int(d["G_inner"]),
                "pad": int(d["pad"]),
                "G_full": int(d["G_full"]),
                "deficit": float(d["deficit_final"]),
                "F_inf": float(d["F_inner_inf_final"]),
                "stages": len(stages) - 1,
                "wall_s": float(stages[-1]),
            }
            # also record stage trajectory
            row["stage_F_inf"] = list(map(float, d["stage_F_inf"]))
            row["stage_deficit"] = list(map(float, d["stage_deficit"]))
            rows.append(row)
        except KeyError as e:
            print(f"# skip {npz.name}: missing key {e}", file=sys.stderr)

    if not rows:
        print("# no .npz files found", file=sys.stderr)
        return

    print("| file | G | pad | stages | wall (s) | ||F||inf | 1-R^2 |")
    print("|------|---|-----|--------|----------|----------|-------|")
    for r in rows:
        print(f"| `{r['file']}` | {r['G_inner']} | {r['pad']} | "
              f"{r['stages']} | {r['wall_s']:.0f} | "
              f"{r['F_inf']:.4e} | {r['deficit']:.4e} |")

    print()
    print("### Stage trajectories")
    for r in rows:
        print(f"- **{r['file']}**")
        for s, (f, d) in enumerate(zip(r["stage_F_inf"], r["stage_deficit"])):
            print(f"  - stage {s}: ||F|| = {f:.4e}  1-R^2 = {d:.4e}")


if __name__ == "__main__":
    main()
