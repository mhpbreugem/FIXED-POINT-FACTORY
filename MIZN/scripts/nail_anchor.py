"""Nail one (model, grid, params) configuration to machine precision and
log the result via tools.runner.

Default: CARA default at G=9 — used as the canonical scaffolding test.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

# Make `tools.runner` importable when invoked directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mizn.config import Params
from mizn.main import solve
from tools.runner import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='cara_default')
    ap.add_argument('--G', type=int, default=9)
    ap.add_argument('--model', default='cara')
    ap.add_argument('--gamma', type=float, default=2.0)
    args = ap.parse_args()
    p = Params(G=args.G, model=args.model, gamma=args.gamma)
    rid = run(args.scenario, p, solver_fn=solve)
    print(f"Run ID: {rid}")


if __name__ == '__main__':
    main()
