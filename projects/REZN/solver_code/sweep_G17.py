#!/usr/bin/env python3
"""
Self-contained gamma sweep for G17/umax4 (K=3 agents, tau=2.5, CRRA).

Economic model
──────────────
Three agents observe binary state v ∈ {0,1} via Gaussian signals s ~ N(±½, 1/tau).
At equilibrium each agent's posterior mu_k = P(v=1 | own signal, observed price p).
Market-clearing price p* satisfies sum_k x_k(mu_k, p*, gamma) = 0 where
  x_k = W_k * (R_k - 1) / ((1-p) + R_k*p),  R_k = exp((logit mu_k - logit p) / gamma).

Fixed-point operator phi(P)[i,j,l] computes the clearing price when each agent
updates beliefs using a Gaussian kernel over the 2-D price slice through their
own signal coordinate, then clears the CRRA market.

Sweep algorithm
────────────────
For each gamma step:
  1. Quadratic Lagrange predictor from last three converged points.
  2. Anderson mixing (AA-I) warm-up — skipped if it diverges.
  3. Jacobian-free Newton-Krylov (LGMRES) to machine precision (< 1e-12).
Step size adapts on Newton iteration count; advances past stiff points.

Usage:  python sweep_G17.py left     # gamma 1.0 → 0.01
        python sweep_G17.py right    # gamma 1.0 → 5.0
"""
import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import lgmres, LinearOperator

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path("/home/user/FIXED-POINT-FACTORY")
CKPT = REPO / "projects/REZN/checkpoints"
OUT  = REPO / "projects/REZN/overnight"
OUT.mkdir(parents=True, exist_ok=True)

DIRECTION = (sys.argv[1].lower() if len(sys.argv) > 1 else "left")
assert DIRECTION in ("left", "right"), "Usage: sweep_G17.py left|right"

def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Economic model primitives  (Numba JIT — compiled once, cached)
# ══════════════════════════════════════════════════════════════════════════════

EPS = 1e-12  # price floor / ceiling

@njit(cache=True)
def f_signal(u, v, tau):
    """Gaussian signal density: sqrt(tau/2pi) * exp(-tau/2 * (u - mean)^2)."""
    mean = 0.5 if v == 1 else -0.5
    d = u - mean
    return (tau / (2.0 * 3.141592653589793)) ** 0.5 * np.exp(-0.5 * tau * d * d)

@njit(cache=True)
def x_crra(mu, p, gamma, W):
    """CRRA demand: W*(R-1)/((1-p)+R*p), R = exp((logit mu - logit p)/gamma)."""
    z = (np.log(mu) - np.log(1.0 - mu) - np.log(p) + np.log(1.0 - p)) / gamma
    if z >= 0.0:
        e = np.exp(-z)
        return W * (1.0 - e) / ((1.0 - p) * e + p)
    e = np.exp(z)
    return W * (e - 1.0) / ((1.0 - p) + p * e)

@njit(cache=True)
def clear_crra(mu0, mu1, mu2, gamma, W):
    """Bisection: find p in (EPS, 1-EPS) with x_crra(mu0)+x_crra(mu1)+x_crra(mu2)=0."""
    a, b = EPS, 1.0 - EPS
    fa = x_crra(mu0, a, gamma, W) + x_crra(mu1, a, gamma, W) + x_crra(mu2, a, gamma, W)
    if fa <= 0.0: return a
    if x_crra(mu0, b, gamma, W) + x_crra(mu1, b, gamma, W) + x_crra(mu2, b, gamma, W) >= 0.0:
        return b
    for _ in range(60):
        c  = 0.5 * (a + b)
        fc = x_crra(mu0, c, gamma, W) + x_crra(mu1, c, gamma, W) + x_crra(mu2, c, gamma, W)
        if fc >= 0.0: a = c
        else:         b = c
    return 0.5 * (a + b)

@njit(cache=True, parallel=True)
def phi_K3(P, u, lo, hi, tau, gamma, W, h):
    """
    Fixed-point operator phi for K=3 symmetric agents.

    For each inner cell (i,j,l):
      Agent k uses the 2-D price slice perpendicular to their own signal axis.
      Evidence: A_v = sum_{a,b} K_h(P[slice] - p) * f_v(u_a) * f_v(u_b)
      Posterior: mu_k = f_1(u_k) * A_1 / (f_0(u_k)*A_0 + f_1(u_k)*A_1)
      New price:  phi(P)[i,j,l] = clear_crra(mu0, mu1, mu2, gamma, W)

    Gaussian kernel K_h(d) = exp(-d^2 / (2h^2)); h = kernel bandwidth.
    """
    G     = u.size
    inv2h2 = 1.0 / (2.0 * h * h)
    P_new  = P.copy()

    for i in prange(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P[i, j, l]

                # Agent 0 — slice P[i,:,:], observers have tau[1], tau[2]
                A0 = A1 = 0.0
                for a in range(G):
                    f0a = f_signal(u[a], 0, tau[1]); f1a = f_signal(u[a], 1, tau[1])
                    for b in range(G):
                        w = np.exp(-(P[i, a, b] - p)**2 * inv2h2)
                        A0 += w * f0a * f_signal(u[b], 0, tau[2])
                        A1 += w * f1a * f_signal(u[b], 1, tau[2])
                n = f_signal(u[i], 1, tau[0]) * A1
                d = f_signal(u[i], 0, tau[0]) * A0 + n
                mu0 = max(EPS, min(1.0 - EPS, n / d)) if d > 0.0 else 0.5

                # Agent 1 — slice P[:,j,:], observers have tau[0], tau[2]
                A0 = A1 = 0.0
                for a in range(G):
                    f0a = f_signal(u[a], 0, tau[0]); f1a = f_signal(u[a], 1, tau[0])
                    for b in range(G):
                        w = np.exp(-(P[a, j, b] - p)**2 * inv2h2)
                        A0 += w * f0a * f_signal(u[b], 0, tau[2])
                        A1 += w * f1a * f_signal(u[b], 1, tau[2])
                n = f_signal(u[j], 1, tau[1]) * A1
                d = f_signal(u[j], 0, tau[1]) * A0 + n
                mu1 = max(EPS, min(1.0 - EPS, n / d)) if d > 0.0 else 0.5

                # Agent 2 — slice P[:,:,l], observers have tau[0], tau[1]
                A0 = A1 = 0.0
                for a in range(G):
                    f0a = f_signal(u[a], 0, tau[0]); f1a = f_signal(u[a], 1, tau[0])
                    for b in range(G):
                        w = np.exp(-(P[a, b, l] - p)**2 * inv2h2)
                        A0 += w * f0a * f_signal(u[b], 0, tau[1])
                        A1 += w * f1a * f_signal(u[b], 1, tau[1])
                n = f_signal(u[l], 1, tau[2]) * A1
                d = f_signal(u[l], 0, tau[2]) * A0 + n
                mu2 = max(EPS, min(1.0 - EPS, n / d)) if d > 0.0 else 0.5

                P_new[i, j, l] = clear_crra(mu0, mu1, mu2, gamma, W)
    return P_new

def revelation_deficit(P_inner, u_inner, tau):
    """
    1 - R² where R² is the squared correlation of logit(P) with T* = sum_k tau_k * u_k.
    Measures information revelation: 0 = full revelation, 1 = no revelation.
    """
    T   = sum(tau[k] * u_inner for k in range(3))  # T* signal
    # Flat signal-density weight: w ∝ prod f_1 + prod f_0 (symmetric prior)
    f1  = np.array([np.exp(-0.5 * tau[0] * (u_inner - 0.5)**2) for _ in [0]])[0]
    f0  = np.array([np.exp(-0.5 * tau[0] * (u_inner + 0.5)**2) for _ in [0]])[0]
    # For 3-D cube, weight = outer product
    w1  = np.einsum('i,j,k', f1, f1, f1) + np.einsum('i,j,k', f0, f0, f0)
    w   = np.where((P_inner > 1e-4) & (P_inner < 1.0 - 1e-4), w1, 0.0)
    Ws  = w.sum()
    if Ws <= 0.0: return float("nan")
    L   = np.log(P_inner / (1.0 - P_inner))  # logit
    T3  = T[:, None, None] + T[None, :, None] + T[None, None, :]
    Lm  = (w * L).sum() / Ws;  Tm = (w * T3).sum() / Ws
    vL  = (w * (L - Lm)**2).sum() / Ws
    vT  = (w * (T3 - Tm)**2).sum() / Ws
    cov = (w * (L - Lm) * (T3 - Tm)).sum() / Ws
    return float(1.0 - cov**2 / (vL * vT)) if vL > 0 and vT > 0 else float("nan")

# ══════════════════════════════════════════════════════════════════════════════
# Solver
# ══════════════════════════════════════════════════════════════════════════════

def anderson_mix(phi_fn, P0, lo, hi, G, m=10, n_iter=30, tol=1e-7, tag=""):
    """
    AA-I warm-up. Skips to returning P0 if diverging (spectral radius > 1).
    x_{k+1} = (x_k + f_k) − (dX + dF) @ argmin ||f_k + dF @ gamma||²
    """
    sl = slice(lo, hi); P = P0.copy()
    X_hist, F_hist = [], []
    res0 = None
    for it in range(n_iter):
        phi_P = phi_fn(P)
        f_k   = (phi_P - P)[sl, sl, sl].ravel()
        res   = float(np.max(np.abs(f_k)))
        if res0 is None: res0 = res
        if it % 10 == 0 or res < tol:
            log(f"    AA {tag} it={it}  ||F||={res:.3e}")
        if res < tol:
            return P, res, it
        # Abort if residual has grown ×10 — Picard spectral radius > 1
        if res > 10.0 * res0:
            log(f"    AA {tag} diverged at it={it} — skipping to Newton")
            return P0, res0, 0
        x_k = P[sl, sl, sl].ravel().copy()
        X_hist.append(x_k); F_hist.append(f_k.copy())
        n_diff = min(m, len(X_hist) - 1)
        if n_diff == 0:
            x_new = x_k + f_k
        else:
            i0 = len(X_hist) - n_diff - 1
            dX = np.column_stack([X_hist[i0+j+1] - X_hist[i0+j] for j in range(n_diff)])
            dF = np.column_stack([F_hist[i0+j+1] - F_hist[i0+j] for j in range(n_diff)])
            try:
                gam, _, _, _ = np.linalg.lstsq(dF, -f_k, rcond=None)
                x_new = (x_k + f_k) - (dX + dF) @ gam
            except Exception:
                x_new = x_k + f_k
        P_new = P.copy(); P_new[sl, sl, sl] = x_new.reshape(hi-lo, hi-lo, hi-lo)
        P = np.clip(P_new, EPS, 1.0 - EPS)
    phi_P = phi_fn(P)
    res   = float(np.max(np.abs((phi_P - P)[sl, sl, sl])))
    return P, res, n_iter

def newton_krylov(phi_fn, P0, lo, hi, G, tol=1e-12, max_iter=10, tag=""):
    """JFNK via LGMRES. Finite-diff step eps = 1.5e-8*(1+||P||)/||w||."""
    sl = slice(lo, hi); ni = (hi - lo) ** 3
    P     = P0.copy()
    phi_P = phi_fn(P)
    F     = (phi_P - P)[sl, sl, sl].ravel()
    res   = float(np.max(np.abs(F)))
    for it in range(max_iter):
        if res < tol:
            log(f"    NK {tag} it={it}  ||F||={res:.3e}  [converged]"); return P, res, it
        nP = np.linalg.norm(P[sl, sl, sl]); fp = phi_P[sl, sl, sl].ravel()
        def mv(w, _n=nP, _fp=fp, _P=P):
            wn = np.linalg.norm(w)
            if wn == 0.0: return w
            eps = 1.5e-8 * (1.0 + _n) / wn
            Pp  = _P.copy(); Pp[sl, sl, sl] += eps * w.reshape(hi-lo, hi-lo, hi-lo)
            return w - (phi_fn(Pp)[sl, sl, sl].ravel() - _fp) / eps
        A = LinearOperator((ni, ni), matvec=mv, dtype=np.float64)
        t0 = time.time()
        try:    dv, _ = lgmres(A, F, rtol=1e-4, maxiter=40, inner_m=25)
        except TypeError: dv, _ = lgmres(A, F, tol=1e-4, maxiter=40, inner_m=25)
        P[sl, sl, sl] += dv.reshape(hi-lo, hi-lo, hi-lo)
        P     = np.clip(P, EPS, 1.0 - EPS)
        phi_P = phi_fn(P); F = (phi_P - P)[sl, sl, sl].ravel()
        res   = float(np.max(np.abs(F)))
        log(f"    NK {tag} it={it}  ||F||={res:.3e}  t={time.time()-t0:.0f}s")
    return P, res, max_iter

def quad_predict(history, g_next):
    """Quadratic Lagrange extrapolation from last 3 converged (gamma, P) pairs."""
    if len(history) < 3: return history[-1][1].copy()
    (g0,P0),(g1,P1),(g2,P2) = history[-3], history[-2], history[-1]
    d01,d02,d12 = g1-g0, g2-g0, g2-g1
    if abs(d01) < 1e-12 or abs(d12) < 1e-12: return P2.copy()
    t  = g_next
    L0 = (t-g1)*(t-g2)/(d01*d02)
    L1 = (t-g0)*(t-g2)/(-d01*d12)
    L2 = (t-g0)*(t-g1)/(d02*-d12)
    return np.clip(L0*P0 + L1*P1 + L2*P2, EPS, 1.0 - EPS)

# ══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ══════════════════════════════════════════════════════════════════════════════
d        = np.load(CKPT / "g100_t0250_G17.npz", allow_pickle=True)
G_inner  = int(d["G_inner"]); pad = int(d["pad"])
lo, hi   = pad, pad + G_inner
u_full   = d["u_full"].astype(np.float64)
tau      = d["tau_vec"].astype(np.float64)
W        = float(d["W_vec"][0])
P_anchor = d["P_full"].astype(np.float64)
h        = max(0.005, 0.05 * float(u_full[1] - u_full[0]))
sl       = slice(lo, hi)

log(f"G17  tau={tau[0]}  h={h:.4f}  dir={DIRECTION}")

def make_phi(g):
    gv = np.full(3, g)
    return lambda P: phi_K3(P, u_full, lo, hi, tau, g, W, h)

# JIT warm-up (compiles phi_K3 on first call)
log("Compiling Numba kernel...")
_ = make_phi(1.0)(P_anchor)
log("Done.  Starting sweep.")
log("=" * 60)

# Sweep parameters
STEP_INIT = 0.005; STEP_MIN = 1e-5; STEP_MAX = 0.05
GROW = 1.5; SHRINK = 0.5; TOL = 1e-12
g_lim = 0.01 if DIRECTION == "left" else 5.0

history = [(1.0, P_anchor.copy())]
g_cur, P_cur, step = 1.0, P_anchor.copy(), STEP_INIT
results, n_ok = [], 0

while (g_cur > g_lim + 1e-9) if DIRECTION == "left" else (g_cur < g_lim - 1e-9):
    g_next = max(g_cur - step, g_lim) if DIRECTION == "left" else min(g_cur + step, g_lim)
    phi    = make_phi(g_next)
    P_pred = quad_predict(history, g_next)

    log(f"--- gamma={g_next:.6f}  step={step:.5f} ---")
    # Check predictor quality; only run AA if starting residual is small enough
    phi_pred = phi(P_pred)
    res_pred = float(np.max(np.abs((phi_pred - P_pred)[sl, sl, sl])))
    if res_pred < 0.1:
        P_aa, _, aa_it = anderson_mix(phi, P_pred, lo, hi, G_inner,
                                       m=10, n_iter=30, tol=1e-7,
                                       tag=f"g={g_next:.5f}")
        # If AA made things worse, revert to predictor
        phi_aa = phi(P_aa)
        res_aa = float(np.max(np.abs((phi_aa - P_aa)[sl, sl, sl])))
        P_for_nk = P_aa if res_aa < res_pred else P_pred
    else:
        log(f"    predictor ||F||={res_pred:.2e} — skipping AA, using P_prev")
        P_for_nk = P_cur   # last converged P is safer than bad predictor
        aa_it = 0
    P_new, res, nk_it = newton_krylov(phi, P_for_nk, lo, hi, G_inner,
                                       tol=TOL, max_iter=15,
                                       tag=f"g={g_next:.5f}")

    if res < TOL:
        def1 = revelation_deficit(P_new[sl, sl, sl], u_full[lo:hi], tau)
        log(f"  ✓ gamma={g_next:.6f}  ||F||={res:.2e}  1-R²={def1*100:.4f}%"
            f"  AA:{aa_it}+NK:{nk_it}")
        history.append((g_next, P_new.copy()))
        if len(history) > 4: history.pop(0)
        g_cur, P_cur = g_next, P_new.copy()
        results.append({"gamma": g_next, "F_inf": res, "deficit": def1,
                        "aa": aa_it, "nk": nk_it})
        n_ok += 1
        step = min(step * GROW, STEP_MAX) if nk_it <= 4 else (
               max(step * SHRINK, STEP_MIN) if nk_it >= 8 else step)
        if n_ok % 5 == 0:
            tag_s = f"g{int(round(g_next*1000)):04d}_t{int(round(tau[0]*100)):04d}_G{G_inner}_{DIRECTION}"
            np.savez_compressed(CKPT / f"{tag_s}.npz",
                P_full=P_new, P_inner=P_new[sl,sl,sl],
                u_full=u_full, u_grid_inner=u_full[lo:hi],
                gamma_vec=np.full(3, g_next), tau_vec=tau,
                W_vec=np.full(3, W), G_inner=G_inner, pad=pad, K=3,
                stage_F_inf=res, stage_deficit=def1)
            log(f"  checkpoint: {tag_s}.npz")
    else:
        log(f"  ✗ gamma={g_next:.6f}  ||F||={res:.2e}  AA:{aa_it}+NK:{nk_it}")
        step = max(step * SHRINK, STEP_MIN)
        if step <= STEP_MIN * 1.5:
            log("  step at minimum — advancing")
            g_cur = g_next

    with open(OUT / f"sweep_G17_{DIRECTION}.json", "w") as fh:
        json.dump(results, fh, indent=2)

log("=" * 60)
log(f"Done ({DIRECTION}): {n_ok} points, last gamma={g_cur:.6f}")
