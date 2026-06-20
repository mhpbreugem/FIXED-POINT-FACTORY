"""K=4 contour-method REE solver for the noise-traders-not-a-primitive paper.

All numerical work is float64. The hot path (the Phi map) is JIT-compiled
with numba in serial mode; pin the process to one core via
``taskset -c 0`` and cap thread libraries with
``OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMBA_NUM_THREADS=1``.

Module map:

    config        parameter dataclasses
    signals       Lambda, logit, f_v, T*, ex-ante weights
    demand        CRRA/CARA demand and per-realisation market clearing
    contour_K4    Phi map and no-learning initialisation
    symmetry      S_K averaging
    solver        Picard and Anderson fixed-point solvers
    metrics       1 - R^2 and trade-volume diagnostics
    run           CLI entry point
    smoke         cheap correctness checks (G=5)
"""

__all__ = [
    "config", "signals", "demand", "contour_K4",
    "symmetry", "solver", "metrics",
]
