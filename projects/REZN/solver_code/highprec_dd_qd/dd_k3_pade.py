"""Pade extrapolation of mu(p, u_k) in kernel-bandwidth h.

For each (p, u_k), compute mu_h at several bandwidths h, then fit a Pade
approximant in h^2 to extrapolate to h=0. Compare to:
  - strict-h=0 (ground truth)
  - Richardson R2, R3, R4 (polynomial extrapolation)

Pade rationale: kernel-band integral A_h has non-analytic dependence on h
when band edges sweep across kinks. Polynomial extrapolation breaks down;
rational (Pade) can fit poles/cusps.

Pade(0,1):  mu_pade(h) = a / (1 + b*h^2)             (2 unknowns from 2 data)
Pade(1,1):  mu_pade(h) = (a + b*h^2) / (1 + c*h^2)   (3 unknowns from 3 data)
"""
import os, sys, time, json, glob
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from lin_cdf_strict import (build_mu_table_lin_strict, make_cdf_uniform_grid,
                                  make_p_grid, make_gl_for_u)
from lin_cdf_kern_tab import build_mu_table_lin_kern
from lin_cdf_richardson import richardson_weights


def pade_01_h0(h_arr, A_arr):
    """Pade(0,1) extrapolation from 2 points to h=0. f(h)=a/(1+b*h^2)."""
    h1, h2 = h_arr; A1, A2 = A_arr
    H1, H2 = h1*h1, h2*h2
    denom = A1*H1 - A2*H2
    if abs(denom) < 1e-30: return 0.5*(A1+A2)
    b = (A2 - A1) / denom
    if 1 + b*H1 == 0: return A1
    a = A1 * (1 + b*H1)
    return a    # f(0) = a/(1) = a


def pade_11_h0(h_arr, A_arr):
    """Pade(1,1) from 3 points. f(h)=(a+b*h^2)/(1+c*h^2)."""
    h1, h2, h3 = h_arr; A1, A2, A3 = A_arr
    H1, H2, H3 = h1*h1, h2*h2, h3*h3
    # a + b*H_i - c*A_i*H_i = A_i  for i=1,2,3
    # Linear system in (a, b, c)
    M = np.array([[1, H1, -A1*H1], [1, H2, -A2*H2], [1, H3, -A3*H3]])
    rhs = np.array([A1, A2, A3])
    try:
        a, b, c = np.linalg.solve(M, rhs)
        return a    # f(0) = a / 1 = a
    except Exception:
        return np.mean(A_arr)


def build_mu_pade_table(P, u_grid, p_grid, gl_u, gl_du, tau, NQK, hs, pade_kind):
    G = u_grid.size; G_p = p_grid.size
    mu_h_tables = []
    for h in hs:
        mu_h = build_mu_table_lin_kern(P, u_grid, p_grid, gl_u, gl_du, tau,
                                            G, NQK, float(h))
        mu_h_tables.append(mu_h)
    mu_h = np.stack(mu_h_tables, axis=0)        # (n_h, G_p, G)
    out = np.empty((G_p, G))
    for ip in range(G_p):
        for k in range(G):
            A_arr = mu_h[:, ip, k]
            if pade_kind == "01":
                out[ip, k] = pade_01_h0(hs, A_arr)
            elif pade_kind == "11":
                out[ip, k] = pade_11_h0(hs, A_arr)
            else:
                raise ValueError(pade_kind)
    return np.clip(out, 0.0, 1.0)


def build_mu_richardson(P, u_grid, p_grid, gl_u, gl_du, tau, NQK, hs):
    G = u_grid.size; G_p = p_grid.size
    w = richardson_weights(tuple(hs))
    mu = np.zeros((G_p, G))
    for hi, h in enumerate(hs):
        mu_h = build_mu_table_lin_kern(P, u_grid, p_grid, gl_u, gl_du, tau,
                                            G, NQK, float(h))
        mu += float(w[hi]) * mu_h
    return mu


G = 11; G_p = 121; NQK_KERN = 16; NQK_STRICT = 16


def run():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u_k, gl_du_k = make_gl_for_u(u_grid[0], u_grid[-1], NQK_KERN)
    gl_u_s, gl_du_s = make_gl_for_u(u_grid[0], u_grid[-1], NQK_STRICT)

    HS_R2 = (0.5, 0.2)
    HS_R3 = (0.5, 0.35, 0.2)
    HS_R4 = (0.5, 0.4, 0.3, 0.2)
    HS_PADE01 = (0.4, 0.2)
    HS_PADE11 = (0.5, 0.3, 0.15)

    print("JIT warmup...", flush=True); t0 = time.time()
    P0 = np.full((G,G,G), 0.5)
    build_mu_table_lin_kern(P0, u_grid, p_grid, gl_u_k, gl_du_k, 1.0, G, NQK_KERN, 0.3)
    build_mu_table_lin_strict(P0, u_grid, p_grid, gl_u_s, gl_du_s, 1.0, G, NQK_STRICT)
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    fps = sorted(glob.glob("/tmp/dd_k3_sweep_fps/*.npz"))
    results = {}
    for fp in fps[:16]:
        d = np.load(fp)
        if d["mu_hi"].shape[0] != G: continue
        P = (d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"])).astype(np.float64)
        gamma = float(d["gamma"]); tau = float(d["tau"])
        key = os.path.basename(fp).replace(".npz", "")
        print(f"\n=== {key} ===", flush=True)
        mu_strict = build_mu_table_lin_strict(P, u_grid, p_grid, gl_u_s, gl_du_s,
                                                       tau, G, NQK_STRICT)
        cell = dict(gamma=gamma, tau=tau,
                      mu_strict_range=[float(mu_strict.min()), float(mu_strict.max())])
        # Richardson R2, R3, R4
        for label, hs in [("R2", HS_R2), ("R3", HS_R3), ("R4", HS_R4)]:
            mu_R = build_mu_richardson(P, u_grid, p_grid, gl_u_k, gl_du_k, tau,
                                              NQK_KERN, hs)
            d_max = float(np.max(np.abs(mu_R - mu_strict)))
            d_med = float(np.median(np.abs(mu_R - mu_strict)))
            cell[label] = dict(hs=list(hs), max=d_max, med=d_med)
            print(f"  Rich {label}: max|d|={d_max:.3e} med={d_med:.3e}", flush=True)
        # Pade(0,1) with 2 h values
        mu_P01 = build_mu_pade_table(P, u_grid, p_grid, gl_u_k, gl_du_k, tau,
                                            NQK_KERN, HS_PADE01, "01")
        d_max = float(np.max(np.abs(mu_P01 - mu_strict)))
        d_med = float(np.median(np.abs(mu_P01 - mu_strict)))
        cell["Pade01"] = dict(hs=list(HS_PADE01), max=d_max, med=d_med)
        print(f"  Pade(0,1): max|d|={d_max:.3e} med={d_med:.3e}", flush=True)
        # Pade(1,1) with 3 h values
        mu_P11 = build_mu_pade_table(P, u_grid, p_grid, gl_u_k, gl_du_k, tau,
                                            NQK_KERN, HS_PADE11, "11")
        d_max = float(np.max(np.abs(mu_P11 - mu_strict)))
        d_med = float(np.median(np.abs(mu_P11 - mu_strict)))
        cell["Pade11"] = dict(hs=list(HS_PADE11), max=d_max, med=d_med)
        print(f"  Pade(1,1): max|d|={d_max:.3e} med={d_med:.3e}", flush=True)
        results[key] = cell
        json.dump(results, open("/tmp/dd_k3_pade.json", "w"), indent=2)
    print(f"\nDONE: {len(results)} cells -> /tmp/dd_k3_pade.json", flush=True)


if __name__ == "__main__":
    run()
