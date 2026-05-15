#!/usr/bin/env python3
"""
Self-contained sweep for K=3 CRRA model: G11/umax2/tau=1.0
───────────────────────────────────────────────────────────
1. Build anchor P*(gamma=1, tau=1) from no-learning init via Newton.
2. Right sweep: gamma 1.0 → 5.0
3. Left  sweep: gamma 1.0 → 0.2
All machine precision (||F||_inf < 1e-12).

Grid: G_inner=11, pad=4, umax=2 → u_inner in [-2,2], u_full in [-3.6,3.6],
      du=0.4, kernel_h=0.02, n_inner=11^3=1331, G_full=19.

Economic model:
  State v∈{0,1}, signal s_k ~ N(±½, 1/tau). Agent k's posterior mu_k inferred
  from own signal and Gaussian-smoothed price slice. CRRA market clearing.
"""
import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import lgmres, LinearOperator

REPO = Path("/home/user/FIXED-POINT-FACTORY")
CKPT = REPO / "projects/REZN/checkpoints"
OUT  = REPO / "projects/REZN/overnight"
OUT.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# ── Grid parameters ───────────────────────────────────────────────────────────
G_INNER = 11
UMAX    = 2.0
PAD     = 4
TAU     = 1.0          # signal precision (same for all 3 agents)
W       = 1.0          # wealth (same for all 3 agents)
GAMMA0  = 1.0          # anchor gamma
GAMMA_R = 5.0          # right sweep limit
GAMMA_L = 0.2          # left  sweep limit

lo   = PAD
hi   = PAD + G_INNER
du   = 2.0 * UMAX / (G_INNER - 1)
u_in = np.linspace(-UMAX, UMAX, G_INNER)
u_full = np.concatenate([u_in[0] - du * np.arange(PAD, 0, -1),
                         u_in,
                         u_in[-1] + du * np.arange(1, PAD + 1)])
G_FULL = len(u_full)   # 19
h      = max(0.005, 0.05 * du)   # kernel bandwidth  = 0.02
sl     = slice(lo, hi)
n_in   = G_INNER ** 3

TAU_VEC = np.full(3, TAU)
GAMMA_VEC0 = np.full(3, GAMMA0)

log(f"Grid: G_inner={G_INNER} umax={UMAX} pad={PAD} du={du:.3f} h={h:.4f}")
log(f"tau={TAU}  W={W}  n_inner={n_in}  G_full={G_FULL}")

# ══════════════════════════════════════════════════════════════════════════════
# Economic model  (Numba JIT)
# ══════════════════════════════════════════════════════════════════════════════
EPS = 1e-12

@njit(cache=True)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z)) if z >= 0.0 else np.exp(z) / (1.0 + np.exp(z))

@njit(cache=True)
def f_sig(u, v, tau):
    """Gaussian signal density N(mean=±0.5, precision=tau)."""
    mean = 0.5 if v == 1 else -0.5
    d = u - mean
    return (tau / 6.283185307179586) ** 0.5 * np.exp(-0.5 * tau * d * d)

@njit(cache=True)
def x_crra(mu, p, gamma, W):
    """CRRA demand.  R = exp((logit mu − logit p) / gamma)."""
    z = (np.log(mu) - np.log(1.0 - mu) - np.log(p) + np.log(1.0 - p)) / gamma
    if z >= 0.0:
        e = np.exp(-z); return W * (1.0 - e) / ((1.0 - p) * e + p)
    e = np.exp(z);     return W * (e - 1.0) / ((1.0 - p) + p * e)

@njit(cache=True)
def clear(mu0, mu1, mu2, gamma, W):
    """Bisection: find p so x_crra(mu0)+x_crra(mu1)+x_crra(mu2)=0."""
    a, b = EPS, 1.0 - EPS
    if x_crra(mu0, a, gamma, W) + x_crra(mu1, a, gamma, W) + x_crra(mu2, a, gamma, W) <= 0.0:
        return a
    if x_crra(mu0, b, gamma, W) + x_crra(mu1, b, gamma, W) + x_crra(mu2, b, gamma, W) >= 0.0:
        return b
    for _ in range(64):
        c  = 0.5 * (a + b)
        fc = x_crra(mu0, c, gamma, W) + x_crra(mu1, c, gamma, W) + x_crra(mu2, c, gamma, W)
        if fc >= 0.0: a = c
        else:         b = c
    return 0.5 * (a + b)

@njit(cache=True, parallel=True)
def phi(P, u, lo, hi, tau, gamma, W, h):
    """
    Fixed-point operator phi(P)[i,j,l].

    For each inner cell (i,j,l) with current price p = P[i,j,l]:
      Agent k uses the 2-D price-slice through their own signal coordinate.
      Evidence A_v = Σ_{a,b} K_h(P_slice[a,b] − p) · f_v(u_a) · f_v(u_b)
        where K_h(d) = exp(−d²/(2h²))  [Gaussian kernel].
      Posterior mu_k = f_1(u_k)·A_1 / (f_0(u_k)·A_0 + f_1(u_k)·A_1).
      New price phi(P)[i,j,l] = clear_crra(mu_0, mu_1, mu_2, gamma, W).
    """
    inv2h2 = 1.0 / (2.0 * h * h)
    G      = u.size
    P_new  = P.copy()
    for i in prange(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P[i, j, l]

                # Agent 0 — slice P[i,:,:]
                A0 = A1 = 0.0
                for a in range(G):
                    f0a = f_sig(u[a], 0, tau); f1a = f_sig(u[a], 1, tau)
                    for b in range(G):
                        w = np.exp(-(P[i, a, b] - p) ** 2 * inv2h2)
                        A0 += w * f0a * f_sig(u[b], 0, tau)
                        A1 += w * f1a * f_sig(u[b], 1, tau)
                num = f_sig(u[i], 1, tau) * A1; den = f_sig(u[i], 0, tau) * A0 + num
                mu0 = max(EPS, min(1.0 - EPS, num / den)) if den > 0.0 else 0.5

                # Agent 1 — slice P[:,j,:]
                A0 = A1 = 0.0
                for a in range(G):
                    f0a = f_sig(u[a], 0, tau); f1a = f_sig(u[a], 1, tau)
                    for b in range(G):
                        w = np.exp(-(P[a, j, b] - p) ** 2 * inv2h2)
                        A0 += w * f0a * f_sig(u[b], 0, tau)
                        A1 += w * f1a * f_sig(u[b], 1, tau)
                num = f_sig(u[j], 1, tau) * A1; den = f_sig(u[j], 0, tau) * A0 + num
                mu1 = max(EPS, min(1.0 - EPS, num / den)) if den > 0.0 else 0.5

                # Agent 2 — slice P[:,:,l]
                A0 = A1 = 0.0
                for a in range(G):
                    f0a = f_sig(u[a], 0, tau); f1a = f_sig(u[a], 1, tau)
                    for b in range(G):
                        w = np.exp(-(P[a, b, l] - p) ** 2 * inv2h2)
                        A0 += w * f0a * f_sig(u[b], 0, tau)
                        A1 += w * f1a * f_sig(u[b], 1, tau)
                num = f_sig(u[l], 1, tau) * A1; den = f_sig(u[l], 0, tau) * A0 + num
                mu2 = max(EPS, min(1.0 - EPS, num / den)) if den > 0.0 else 0.5

                P_new[i, j, l] = clear(mu0, mu1, mu2, gamma, W)
    return P_new

@njit(cache=True, parallel=True)
def init_no_learning(u, lo, hi, tau, gamma, W):
    """
    No-learning equilibrium: agents ignore the price and use only own signal.
    mu_k = sigmoid(tau * u_k),  p = clear_crra(mu_0, mu_1, mu_2).
    Good starting point for Newton at any gamma.
    """
    G = u.size
    P = np.empty((G, G, G))
    for i in prange(G):
        mu0 = sigmoid(tau * u[i])
        for j in range(G):
            mu1 = sigmoid(tau * u[j])
            for l in range(G):
                mu2 = sigmoid(tau * u[l])
                P[i, j, l] = clear(mu0, mu1, mu2, gamma, W)
    return P

def deficit(P_inner, u_inner, tau):
    """1 − R²: revelation deficit (0 = full info, 1 = no learning)."""
    T  = tau * u_inner
    f1 = np.exp(-0.5 * tau * (u_inner - 0.5) ** 2)
    f0 = np.exp(-0.5 * tau * (u_inner + 0.5) ** 2)
    w  = np.einsum('i,j,k', f1, f1, f1) + np.einsum('i,j,k', f0, f0, f0)
    w  = np.where((P_inner > 1e-4) & (P_inner < 1 - 1e-4), w, 0.0)
    Ws = w.sum()
    if Ws <= 0: return float('nan')
    L  = np.log(P_inner / (1 - P_inner))
    T3 = T[:, None, None] + T[None, :, None] + T[None, None, :]
    Lm = (w * L).sum() / Ws;  Tm = (w * T3).sum() / Ws
    vL = (w * (L - Lm) ** 2).sum() / Ws
    vT = (w * (T3 - Tm) ** 2).sum() / Ws
    cv = (w * (L - Lm) * (T3 - Tm)).sum() / Ws
    return float(1 - cv ** 2 / (vL * vT)) if vL > 0 and vT > 0 else float('nan')

# ══════════════════════════════════════════════════════════════════════════════
# Solver
# ══════════════════════════════════════════════════════════════════════════════

def newton(phi_fn, P0, tol=1e-12, max_iter=20, tag=""):
    """JFNK via LGMRES.  eps = 1.5e-8*(1+||P||)/||w||."""
    P = P0.copy()
    phiP = phi_fn(P); F = (phiP - P)[sl, sl, sl].ravel(); res = float(np.max(np.abs(F)))
    for it in range(max_iter):
        if res < tol:
            log(f"  NK {tag} it={it}  ||F||={res:.2e}  [ok]"); return P, res, it
        nP = np.linalg.norm(P[sl, sl, sl]); fp = phiP[sl, sl, sl].ravel()
        def mv(w, _n=nP, _fp=fp, _P=P):
            wn = np.linalg.norm(w)
            if wn == 0: return w
            eps = 1.5e-8 * (1 + _n) / wn
            Pp = _P.copy(); Pp[sl, sl, sl] += eps * w.reshape(G_INNER, G_INNER, G_INNER)
            return w - (phi_fn(Pp)[sl, sl, sl].ravel() - _fp) / eps
        A = LinearOperator((n_in, n_in), matvec=mv, dtype=np.float64)
        t0 = time.time()
        try:    dv, _ = lgmres(A, F, rtol=1e-4, maxiter=40, inner_m=25)
        except TypeError: dv, _ = lgmres(A, F, tol=1e-4, maxiter=40, inner_m=25)
        P[sl, sl, sl] += dv.reshape(G_INNER, G_INNER, G_INNER)
        P = np.clip(P, EPS, 1 - EPS)
        phiP = phi_fn(P); F = (phiP - P)[sl, sl, sl].ravel(); res = float(np.max(np.abs(F)))
        log(f"  NK {tag} it={it}  ||F||={res:.2e}  t={time.time()-t0:.0f}s")
    return P, res, max_iter

def quad_predict(hist, g_next):
    """Quadratic Lagrange predictor from last 3 converged (gamma, P) pairs."""
    if len(hist) < 3: return hist[-1][1].copy()
    (g0,P0),(g1,P1),(g2,P2) = hist[-3], hist[-2], hist[-1]
    d01,d02,d12 = g1-g0, g2-g0, g2-g1
    if abs(d01)<1e-12 or abs(d12)<1e-12: return P2.copy()
    t = g_next
    L0=(t-g1)*(t-g2)/(d01*d02); L1=(t-g0)*(t-g2)/(-d01*d12); L2=(t-g0)*(t-g1)/(d02*-d12)
    return np.clip(L0*P0+L1*P1+L2*P2, EPS, 1-EPS)

def make_phi(g): return lambda P: phi(P, u_full, lo, hi, TAU, g, W, h)

def sweep(P_start, g_start, g_end, label, results_all):
    """Adaptive gamma continuation from g_start toward g_end."""
    is_right = g_end > g_start
    STEP_INIT=0.01; STEP_MIN=1e-5; STEP_MAX=0.1; GROW=1.5; SHRINK=0.5; TOL=1e-12
    hist = [(g_start, P_start.copy())]
    g, P, step = g_start, P_start.copy(), STEP_INIT
    n_ok = 0
    log(f"=== {label} sweep: {g_start:.3f} → {g_end:.3f} ===")
    while (g < g_end - 1e-9) if is_right else (g > g_end + 1e-9):
        g_next = min(g + step, g_end) if is_right else max(g - step, g_end)
        phi_fn = make_phi(g_next)
        P_pred = quad_predict(hist, g_next)
        # Check predictor quality; fall back to last P if bad
        res_pred = float(np.max(np.abs((phi_fn(P_pred) - P_pred)[sl, sl, sl])))
        P0_nk = P_pred if res_pred < 0.1 else P
        if res_pred >= 0.1:
            log(f"  pred ||F||={res_pred:.2e} — using P_prev")
        log(f"--- gamma={g_next:.5f}  step={step:.5f} ---")
        P_new, res, nk_it = newton(phi_fn, P0_nk, tol=TOL, max_iter=15,
                                    tag=f"g={g_next:.4f}")
        if res < TOL:
            d1 = deficit(P_new[sl,sl,sl], u_in, TAU)
            log(f"  ✓ gamma={g_next:.5f}  ||F||={res:.2e}  1-R²={d1*100:.4f}%  NK:{nk_it}")
            hist.append((g_next, P_new.copy()))
            if len(hist) > 4: hist.pop(0)
            g, P = g_next, P_new.copy(); n_ok += 1
            step = min(step*GROW,STEP_MAX) if nk_it<=4 else (max(step*SHRINK,STEP_MIN) if nk_it>=8 else step)
            results_all.append({"gamma":g_next,"F_inf":res,"deficit":d1,"nk":nk_it,"dir":label})
            # Checkpoint every 10 points
            if n_ok % 10 == 0:
                tag_s = f"g{int(round(g_next*100)):04d}_t{int(round(TAU*100)):04d}_G{G_INNER}"
                np.savez_compressed(CKPT/f"{tag_s}.npz", P_full=P_new,
                    P_inner=P_new[sl,sl,sl], u_full=u_full, u_grid_inner=u_in,
                    gamma_vec=np.full(3,g_next), tau_vec=TAU_VEC, W_vec=np.full(3,W),
                    G_inner=G_INNER, pad=PAD, K=3, stage_F_inf=res, stage_deficit=d1)
                log(f"  checkpoint: {tag_s}.npz")
        else:
            log(f"  ✗ gamma={g_next:.5f}  ||F||={res:.2e}")
            step = max(step*SHRINK, STEP_MIN)
            if step <= STEP_MIN*1.5:
                log("  step at minimum — fold boundary reached"); g = g_next
        with open(OUT/f"sweep_G{G_INNER}.json","w") as fh: json.dump(results_all,fh,indent=2)
    log(f"=== {label} done: {n_ok} points, last gamma={g:.5f} ===")
    return P, g

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
ANCHOR_FILE = CKPT / f"g100_t{int(TAU*100):04d}_G{G_INNER}.npz"

log("Compiling Numba kernels...")
_P_tmp = np.full((G_FULL, G_FULL, G_FULL), 0.5)
_ = phi(_P_tmp, u_full, lo, hi, TAU, GAMMA0, W, h)
log("Done.")

# ── Build or load anchor ─────────────────────────────────────────────────────
if ANCHOR_FILE.exists():
    log(f"Loading existing anchor: {ANCHOR_FILE.name}")
    d = np.load(ANCHOR_FILE); P_anchor = d["P_full"]
else:
    log("Building anchor from no-learning initialisation...")
    P_nl = init_no_learning(u_full, lo, hi, TAU, GAMMA0, W)
    log(f"  No-learning init done.  Computing phi to check quality...")
    res_nl = float(np.max(np.abs((phi(P_nl, u_full, lo, hi, TAU, GAMMA0, W, h) - P_nl)[sl,sl,sl])))
    log(f"  ||F||_nl = {res_nl:.3e}  — running Newton to converge...")
    phi0 = make_phi(GAMMA0)
    P_anchor, res_anc, nit = newton(phi0, P_nl, tol=1e-12, max_iter=20, tag="anchor")
    if res_anc >= 1e-12:
        log(f"  WARNING: anchor Newton did not converge (||F||={res_anc:.2e})")
    d_anc = deficit(P_anchor[sl,sl,sl], u_in, TAU)
    log(f"  Anchor: ||F||={res_anc:.2e}  1-R²={d_anc*100:.4f}%  iters={nit}")
    np.savez_compressed(ANCHOR_FILE, P_full=P_anchor, P_inner=P_anchor[sl,sl,sl],
        u_full=u_full, u_grid_inner=u_in, gamma_vec=np.full(3,GAMMA0),
        tau_vec=TAU_VEC, W_vec=np.full(3,W), G_inner=G_INNER, pad=PAD, K=3,
        stage_F_inf=res_anc, stage_deficit=d_anc)
    log(f"  Anchor saved: {ANCHOR_FILE.name}")

# ── Sweeps ───────────────────────────────────────────────────────────────────
results = []
_, _ = sweep(P_anchor, GAMMA0, GAMMA_R, "right", results)
_, _ = sweep(P_anchor, GAMMA0, GAMMA_L, "left",  results)

log("All done.")
log(f"Total converged points: {len(results)}")
log(f"Results → {OUT}/sweep_G{G_INNER}.json")
