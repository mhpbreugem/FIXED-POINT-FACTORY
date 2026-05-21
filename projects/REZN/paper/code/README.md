# `code/` — K=4 contour-method REE solver

Single-thread, float64 implementation of the contour fixed-point map for
the model in `theory.md`. Designed to be pinned to one CPU core; large
sweeps are expected to be launched on external compute.

## File index

| file              | purpose |
|-------------------|---------|
| `config.py`       | `Config` and `SolverConfig` dataclasses; `DTYPE = np.float64` |
| `signals.py`      | `lam`, `logit`, signal density `f_signal`, `t_star`, ex-ante `weights` |
| `demand.py`       | `x_crra`, `x_cara`, `clear_crra` (bisection), `clear_cara` (closed form) |
| `contour_K4.py`   | the hot kernel: `init_no_learning`, `phi_K4`, `residual_inf` |
| `symmetry.py`     | `symmetrize` averages over the 24 axis permutations of S<sub>4</sub> |
| `solver.py`       | `picard`, `anderson`, dispatcher `solve` |
| `metrics.py`      | `revelation_deficit` (1 − R²), `trade_volume`, `summary` |
| `run.py`          | CLI entry — argparse, runs solver, writes `.npz` + `.log` |
| `smoke.py`        | cheap correctness checks at `G=5` |

## Run pinned to CPU 0

The kernel is intentionally serial; do not let BLAS or numba spawn
threads. Set thread caps and CPU affinity:

```sh
taskset -c 0 env \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    python -m code.run --G 10 --gamma 0.5 --tau 2.0 --solver anderson
```

## Smoke tests

```sh
taskset -c 0 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    python -m code.smoke
```

Expected: `=== all smoke tests passed ===`.

Checks:
- CRRA demand is monotone-decreasing in `p`
- Homogeneous posteriors clear at `p = mu`
- No-learning `P` is invariant under axis permutations
- CARA no-learning is fully revealing (1 − R² ≈ 0)
- One Φ step runs and (after symmetrise) is in S<sub>4</sub>-symmetric form
- Under CARA, Λ(T*/K) is a fixed point of Φ
- Anderson at `G=5` converges in well under 30 iterations

## Cost (rough)

Per Φ evaluation, with K=4, the inner work is `G^4` grid points × four
agents × three passes × `G^2` sweep cells. Anderson reaches `1e-7` in
~10–20 iterations on the symmetric homogeneous problem. Rough
single-core wall times:

| `G` | grid points | per-Φ time | tol-1e-7 |
|-----|-------------|-----------|---------|
| 5   | 625         | <1 s      | seconds |
| 10  | 10 000      | seconds   | minutes |
| 15  | 50 625      | tens of s | tens of minutes |
| 20  | 160 000     | minutes   | tens of minutes to hours |

Numbers in this table are approximate; rerun the timer in
`run.py` for definitive figures on the target machine.
