"""Critical-point detector for the K=3 CRRA REE price function P(u_1, u_2, u_3).

Goal
----
Find interior points u* in the cube [-U_MAX, U_MAX]^3 where grad P = 0.
The corresponding critical-VALUES p_c = P(u*) are the locations of
logarithmic cusps in mu(p, u_k) (because the co-area integrand carries a
1/|grad P| factor).  These p_c values are intended to be used as knots
in a piecewise representation of mu(p, u_k) (next step of the plan).

API
---
- find_critical_points(P_grid, u_grid, tol)
- critical_p_values(crit_pts, p_lo, p_hi)
- cusps_for_lookup_grid(P_grid, u_grid, p_grid_existing)

CLI
---
    python dd_k3_critpts.py <strict_fp_npz_path>

Loads P_strict, finds the critical points, prints a summary table,
saves a slice plot to
    projects/REZN/solved_fixed_points/dd_k3_overnight/critpts/<cellname>.png
and (when run as __main__ with no args) sweeps the four strict-FP files.

Notes on the interpolant
------------------------
The natural per-cell trilinear interpolant has piecewise-constant
mixed partials, so grad P_tri = 0 essentially never happens in the
open interior of a cell (the partials are
   dP/du1  = (1-eta_2)(1-eta_3)*(P100-P000) + ... = bilinear in (eta_2, eta_3).
A point where all three of these bilinear forms vanish simultaneously
inside (0,1)^3 is a measure-zero coincidence; in practice tri-linear
yields ~0 interior roots).  We therefore try trilinear first and
automatically fall back to a per-cell quadratic Hermite fit that
borrows values from the eight neighbouring vertices to estimate the
mixed derivatives, giving a smooth grad whose zeros are physically
meaningful.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import root


# ---------------------------------------------------------------------------
# Trilinear interpolant in cell-local coords eta in [0, 1]^3
# ---------------------------------------------------------------------------

def _trilinear_value_and_grad(V, eta):
    """Trilinear P_hat and its gradient at cell-local eta in (0,1)^3.

    V has shape (2, 2, 2): V[a, b, c] = P at vertex (a, b, c).
    Returns (val, grad3) where grad3 is in eta-units (no Jacobian).
    """
    e1, e2, e3 = eta
    w = np.array(
        [
            [[(1 - e1) * (1 - e2) * (1 - e3), (1 - e1) * (1 - e2) * e3],
             [(1 - e1) * e2 * (1 - e3),       (1 - e1) * e2 * e3]],
            [[e1 * (1 - e2) * (1 - e3),       e1 * (1 - e2) * e3],
             [e1 * e2 * (1 - e3),             e1 * e2 * e3]],
        ]
    )
    val = float((w * V).sum())
    # Partials wrt eta_k:  swap (1-e) <-> +/-1
    g = np.empty(3)
    # d/de1
    dw1 = np.array(
        [
            [[-(1 - e2) * (1 - e3), -(1 - e2) * e3],
             [-e2 * (1 - e3),       -e2 * e3]],
            [[(1 - e2) * (1 - e3),  (1 - e2) * e3],
             [e2 * (1 - e3),        e2 * e3]],
        ]
    )
    g[0] = float((dw1 * V).sum())
    # d/de2
    dw2 = np.array(
        [
            [[-(1 - e1) * (1 - e3), -(1 - e1) * e3],
             [(1 - e1) * (1 - e3),  (1 - e1) * e3]],
            [[-e1 * (1 - e3),       -e1 * e3],
             [e1 * (1 - e3),        e1 * e3]],
        ]
    )
    g[1] = float((dw2 * V).sum())
    # d/de3
    dw3 = np.array(
        [
            [[-(1 - e1) * (1 - e2), (1 - e1) * (1 - e2)],
             [-(1 - e1) * e2,       (1 - e1) * e2]],
            [[-e1 * (1 - e2),       e1 * (1 - e2)],
             [-e1 * e2,             e1 * e2]],
        ]
    )
    g[2] = float((dw3 * V).sum())
    return val, g


# ---------------------------------------------------------------------------
# Quadratic Hermite-style fit per cell (higher-order fallback)
# ---------------------------------------------------------------------------

def _local_quadratic_coeffs(P_grid, i, j, k):
    """Fit P_hat(eta) = a0 + sum a_p eta_p + sum a_pq eta_p eta_q (p<=q)
    over the cell (i..i+1, j..j+1, k..k+1) of P_grid, using values at
    the 8 cell vertices plus centered finite-difference estimates of
    the mixed second derivatives that pin down the cross terms.

    Returns a callable f(eta) -> (val, grad).
    """
    G1, G2, G3 = P_grid.shape

    def at(a, b, c):
        a = min(max(a, 0), G1 - 1)
        b = min(max(b, 0), G2 - 1)
        c = min(max(c, 0), G3 - 1)
        return P_grid[a, b, c]

    # Use cell vertices as base.
    P000 = at(i, j, k);     P100 = at(i+1, j, k)
    P010 = at(i, j+1, k);   P110 = at(i+1, j+1, k)
    P001 = at(i, j, k+1);   P101 = at(i+1, j, k+1)
    P011 = at(i, j+1, k+1); P111 = at(i+1, j+1, k+1)

    # Linear and pure-quadratic and mixed terms come from values that
    # straddle the cell.  We build a quadratic in eta in [0, 1]^3 whose
    # values at the 8 vertices match the data exactly (trilinear part)
    # plus pure-quadratic curvature estimated from neighbours.
    #
    # The cleanest fallback that puts real critical points inside the
    # cell is: take the trilinear interpolant and add a small isotropic
    # curvature term  c_pp * eta_p (1 - eta_p)  per axis with
    # c_pp = -[ neighbour_avg - center_avg ].  This bows the surface so
    # grad = 0 can be hit.
    # Center cell value (eta = 1/2):
    P_center_tri = 0.125 * (
        P000 + P100 + P010 + P110 + P001 + P101 + P011 + P111
    )
    # Curvature in axis 1: use vertices at i-1 and i+2 if available.
    def axis_curv(axis):
        c_face_avg = 0.0  # value of the trilinear face avg adjacent in axis
        # avg of vertices one step beyond on the negative side
        below_idx = {0: (i - 1, j, k), 1: (i, j - 1, k), 2: (i, j, k - 1)}[axis]
        above_idx = {0: (i + 2, j, k), 1: (i, j + 2, k), 2: (i, j, k + 2)}[axis]
        # Use averages over the corresponding 2x2 face plane.
        if axis == 0:
            below = 0.25 * (at(below_idx[0], j, k) + at(below_idx[0], j + 1, k)
                            + at(below_idx[0], j, k + 1) + at(below_idx[0], j + 1, k + 1))
            above = 0.25 * (at(above_idx[0], j, k) + at(above_idx[0], j + 1, k)
                            + at(above_idx[0], j, k + 1) + at(above_idx[0], j + 1, k + 1))
        elif axis == 1:
            below = 0.25 * (at(i, below_idx[1], k) + at(i + 1, below_idx[1], k)
                            + at(i, below_idx[1], k + 1) + at(i + 1, below_idx[1], k + 1))
            above = 0.25 * (at(i, above_idx[1], k) + at(i + 1, above_idx[1], k)
                            + at(i, above_idx[1], k + 1) + at(i + 1, above_idx[1], k + 1))
        else:
            below = 0.25 * (at(i, j, below_idx[2]) + at(i + 1, j, below_idx[2])
                            + at(i, j + 1, below_idx[2]) + at(i + 1, j + 1, below_idx[2]))
            above = 0.25 * (at(i, j, above_idx[2]) + at(i + 1, j, above_idx[2])
                            + at(i, j + 1, above_idx[2]) + at(i + 1, j + 1, above_idx[2]))
        # Second-difference in that axis between cells.
        # Face averages on the two sides of the cell:
        if axis == 0:
            face_lo = 0.25 * (P000 + P010 + P001 + P011)
            face_hi = 0.25 * (P100 + P110 + P101 + P111)
        elif axis == 1:
            face_lo = 0.25 * (P000 + P100 + P001 + P101)
            face_hi = 0.25 * (P010 + P110 + P011 + P111)
        else:
            face_lo = 0.25 * (P000 + P100 + P010 + P110)
            face_hi = 0.25 * (P001 + P101 + P011 + P111)
        # Curvature: (below + above - 2*center_face_avg).
        cell_center_face = 0.5 * (face_lo + face_hi)
        return (below + above - 2.0 * cell_center_face)

    c1 = axis_curv(0)
    c2 = axis_curv(1)
    c3 = axis_curv(2)

    V = np.array([[[P000, P001], [P010, P011]], [[P100, P101], [P110, P111]]])

    def f(eta):
        val_tri, g_tri = _trilinear_value_and_grad(V, eta)
        e1, e2, e3 = eta
        val = val_tri + c1 * e1 * (1 - e1) + c2 * e2 * (1 - e2) + c3 * e3 * (1 - e3)
        g = g_tri.copy()
        g[0] += c1 * (1 - 2 * e1)
        g[1] += c2 * (1 - 2 * e2)
        g[2] += c3 * (1 - 2 * e3)
        return val, g

    return f, V


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CritPt:
    u: tuple           # (u1, u2, u3) in physical coords
    p: float           # critical value p_c = P_hat(u*)
    grad_norm: float   # |grad P_hat| at u* (should be < tol)
    cell: tuple        # (i, j, k) cell index
    method: str        # 'trilinear' or 'quadratic'


def _cell_solve(f, cell_size_uniform=True):
    """Newton/LM solve for f's gradient = 0 with seed at cell midpoint."""
    def F(x):
        _, g = f(x)
        return g

    sol = root(F, x0=np.array([0.5, 0.5, 0.5]), method='lm',
               options={'xtol': 1e-12, 'ftol': 1e-12, 'maxiter': 200})
    return sol


def find_critical_points(P_grid, u_grid, tol=1e-8):
    """Locate interior points where grad P_hat = 0.

    Parameters
    ----------
    P_grid : (G, G, G) ndarray
    u_grid : (G,) ndarray  (physical u-coordinates of grid axes; assumed
        common to all three axes since the K=3 model uses an isotropic grid).
    tol : float
        Acceptance threshold on |grad P_hat| (cell-local units).

    Returns
    -------
    list[CritPt]
    """
    P_grid = np.asarray(P_grid, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.float64)
    G1, G2, G3 = P_grid.shape
    assert G1 == G2 == G3 == u_grid.size, "Expected isotropic GxGxG grid."

    crits_tri: list[CritPt] = []
    crits_quad: list[CritPt] = []

    for i in range(G1 - 1):
        for j in range(G2 - 1):
            for k in range(G3 - 1):
                # --- Trilinear attempt ---
                V = np.array([[[P_grid[i, j, k],     P_grid[i, j, k+1]],
                               [P_grid[i, j+1, k],   P_grid[i, j+1, k+1]]],
                              [[P_grid[i+1, j, k],   P_grid[i+1, j, k+1]],
                               [P_grid[i+1, j+1, k], P_grid[i+1, j+1, k+1]]]])
                f_tri = lambda eta, V=V: _trilinear_value_and_grad(V, eta)
                sol = _cell_solve(f_tri)
                eta = sol.x
                val, g = f_tri(eta)
                gnorm = float(np.linalg.norm(g))
                if (gnorm < tol and np.all(eta > -1e-9)
                        and np.all(eta < 1 + 1e-9)):
                    u_star = (
                        u_grid[i] + eta[0] * (u_grid[i+1] - u_grid[i]),
                        u_grid[j] + eta[1] * (u_grid[j+1] - u_grid[j]),
                        u_grid[k] + eta[2] * (u_grid[k+1] - u_grid[k]),
                    )
                    crits_tri.append(CritPt(u=u_star, p=val,
                                            grad_norm=gnorm,
                                            cell=(i, j, k),
                                            method='trilinear'))
                    continue
                # --- Quadratic fallback ---
                f_quad, _ = _local_quadratic_coeffs(P_grid, i, j, k)
                sol = _cell_solve(f_quad)
                eta = sol.x
                val, g = f_quad(eta)
                gnorm = float(np.linalg.norm(g))
                if (gnorm < tol and np.all(eta > -1e-9)
                        and np.all(eta < 1 + 1e-9)):
                    u_star = (
                        u_grid[i] + eta[0] * (u_grid[i+1] - u_grid[i]),
                        u_grid[j] + eta[1] * (u_grid[j+1] - u_grid[j]),
                        u_grid[k] + eta[2] * (u_grid[k+1] - u_grid[k]),
                    )
                    crits_quad.append(CritPt(u=u_star, p=val,
                                             grad_norm=gnorm,
                                             cell=(i, j, k),
                                             method='quadratic'))

    return crits_tri + crits_quad


def critical_p_values(crit_pts: Iterable[CritPt], p_lo=1e-3, p_hi=1 - 1e-3,
                       dedupe_tol=1e-5):
    """Sorted unique critical p values inside (p_lo, p_hi)."""
    ps = sorted(c.p for c in crit_pts if p_lo < c.p < p_hi)
    out = []
    for p in ps:
        if not out or abs(p - out[-1]) > dedupe_tol:
            out.append(p)
    return np.asarray(out)


def cusps_for_lookup_grid(P_grid, u_grid, p_grid_existing,
                          p_lo=1e-3, p_hi=1 - 1e-3):
    """Augment p_grid_existing with critical-value knots.

    Returns (p_grid_augmented, is_cusp_mask).
    """
    cps = find_critical_points(P_grid, u_grid)
    pvals = critical_p_values(cps, p_lo=p_lo, p_hi=p_hi)
    base = np.asarray(p_grid_existing, dtype=np.float64)
    merged = np.concatenate([base, pvals])
    cusp_flag = np.concatenate([np.zeros(base.size, dtype=bool),
                                np.ones(pvals.size, dtype=bool)])
    order = np.argsort(merged, kind='stable')
    return merged[order], cusp_flag[order]


# ---------------------------------------------------------------------------
# CLI / plotting
# ---------------------------------------------------------------------------

_OUT_DIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/critpts"


def _make_u_grid(G):
    """Re-derive the CDF-uniform grid used by the strict solver."""
    sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/cheby_h0_prototype")
    from lin_cdf_pchip import make_cdf_uniform_grid
    return make_cdf_uniform_grid(G)


def _slice_eval(P_grid, u_grid, i_fixed, axis):
    """Return a 2D slice of P with axis fixed at index i_fixed."""
    if axis == 0:
        return P_grid[i_fixed, :, :]
    if axis == 1:
        return P_grid[:, i_fixed, :]
    return P_grid[:, :, i_fixed]


def _plot_summary(P_grid, u_grid, cps, out_path, title):
    """One figure summarising critical points for the cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G = P_grid.shape[0]
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    # Slice through u_1 = 0
    i_mid = G // 2
    sl = _slice_eval(P_grid, u_grid, i_mid, axis=0)
    ax = axs[0, 0]
    im = ax.contourf(u_grid, u_grid, sl.T, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, label="P")
    # Plot critical points whose u_1 is closest to 0
    if cps:
        u1c = np.array([c.u[0] for c in cps])
        keep = np.abs(u1c) < (u_grid[i_mid + 1] - u_grid[i_mid - 1])
        for c, k in zip(cps, keep):
            if k:
                ax.plot(c.u[1], c.u[2], 'rx', ms=8, mew=2)
    ax.set_xlabel("$u_2$"); ax.set_ylabel("$u_3$")
    ax.set_title(f"P on slice $u_1\\approx 0$ (with crit pts near)")

    # p_c histogram
    ax = axs[0, 1]
    if cps:
        ps = np.array([c.p for c in cps])
        ax.hist(ps, bins=30, color='steelblue', edgecolor='k')
    ax.set_xlabel("$p_c$"); ax.set_ylabel("# critical points")
    ax.set_title(f"Distribution of critical values ({len(cps)} total)")
    ax.grid(True, alpha=0.3)

    # |grad| residuals
    ax = axs[1, 0]
    if cps:
        gn = np.array([c.grad_norm for c in cps])
        ax.semilogy(np.sort(gn), 'o-', ms=3)
    ax.axhline(1e-8, ls='--', color='r', alpha=0.5, label='tol=1e-8')
    ax.set_xlabel("crit pt index (sorted)"); ax.set_ylabel("$|\\nabla \\hat P|$")
    ax.set_title("Residual norm at accepted critical points")
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

    # Cell occupancy (count per cell averaged over k)
    ax = axs[1, 1]
    occ = np.zeros((G - 1, G - 1))
    for c in cps:
        ci, cj, ck = c.cell
        occ[cj, ck] += 1  # marginalize over i (u_1 cells)
    im = ax.imshow(occ.T, origin='lower', cmap='magma',
                   extent=[u_grid[0], u_grid[-1], u_grid[0], u_grid[-1]],
                   aspect='auto')
    plt.colorbar(im, ax=ax, label="# crit pts (sum over $u_1$-cell)")
    ax.set_xlabel("$u_2$ cell"); ax.set_ylabel("$u_3$ cell")
    ax.set_title("Critical-point density")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_cli(npz_path):
    os.makedirs(_OUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(npz_path))[0]
    d = np.load(npz_path)
    P = d["P_strict"].astype(np.float64)
    G = P.shape[0]
    u_grid = _make_u_grid(G)
    print(f"[{base}] G={G}, P range [{P.min():.4g}, {P.max():.4g}]")
    cps = find_critical_points(P, u_grid, tol=1e-8)
    n_tri = sum(1 for c in cps if c.method == 'trilinear')
    n_quad = sum(1 for c in cps if c.method == 'quadratic')
    print(f"  found {len(cps)} critical points "
          f"(trilinear={n_tri}, quadratic-fallback={n_quad})")
    p_vals = critical_p_values(cps)
    print(f"  unique p_c values in (1e-3, 1-1e-3): {len(p_vals)}")
    if len(p_vals):
        print(f"    min p_c = {p_vals.min():.6f}")
        print(f"    max p_c = {p_vals.max():.6f}")
        print(f"    median  = {np.median(p_vals):.6f}")
    # Tabular summary (first 20)
    print("  sample (first 20):")
    print("    %5s %25s %12s %12s %10s" % ("#", "u*", "p_c", "|grad|", "method"))
    for n, c in enumerate(cps[:20]):
        print("    %5d (% .3f,% .3f,% .3f) %12.6f %12.2e %10s" % (
            n, c.u[0], c.u[1], c.u[2], c.p, c.grad_norm, c.method))

    out_png = os.path.join(_OUT_DIR, f"{base}.png")
    _plot_summary(P, u_grid, cps, out_png,
                  title=f"Critical points of P_strict ({base})")
    print(f"  wrote {out_png}")

    # JSON dump
    out_json = os.path.join(_OUT_DIR, f"{base}.json")
    payload = {
        "file": npz_path,
        "G": int(G),
        "n_critical_points": len(cps),
        "n_trilinear": n_tri,
        "n_quadratic": n_quad,
        "n_unique_p_c": int(len(p_vals)),
        "p_c_min": float(p_vals.min()) if len(p_vals) else None,
        "p_c_max": float(p_vals.max()) if len(p_vals) else None,
        "p_c_median": float(np.median(p_vals)) if len(p_vals) else None,
        "p_c_values": p_vals.tolist(),
        "points": [
            {"u": list(c.u), "p": c.p, "grad_norm": c.grad_norm,
             "cell": list(c.cell), "method": c.method}
            for c in cps
        ],
    }
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  wrote {out_json}")
    return payload


def main(argv):
    if len(argv) >= 2:
        for path in argv[1:]:
            run_cli(path)
        return
    # default: process the four overnight strict FPs
    base = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight"
    files = [
        f"{base}/dd_k3_strict_fp_g100_t0.2000.npz",
        f"{base}/dd_k3_strict_fp_g100_t1.0000.npz",
        f"{base}/dd_k3_strict_fp_g1000_t0.2000.npz",
        f"{base}/dd_k3_strict_fp_g1000_t1.0000.npz",
    ]
    summaries = [run_cli(f) for f in files]
    # Aggregate
    print("\n=== AGGREGATE ===")
    for s in summaries:
        print(f"{os.path.basename(s['file'])}: n_cp={s['n_critical_points']}  "
              f"n_unique_p={s['n_unique_p_c']}")


if __name__ == "__main__":
    main(sys.argv)
