#!/usr/bin/env python3
"""Test whether the fold at gamma≈0.689 (tau=3.0, G21) is a float64 artifact
or a genuine mathematical fold, by running Anderson-accelerated fixed-point
iteration using mpmath at 40 decimal places.

Steps:
1. Float64 Newton from anchor to get P*(gamma=0.6889)
2. Convert to mpmath, verify ||F_mp|| < 1e-35
3. Try gamma=0.6885 with mpmath Anderson iteration
4. Report whether it converges or stalls
"""
import sys, time, json
from pathlib import Path
from datetime import datetime

import numpy as np
import mpmath as mp

mp.mp.dps = 40

REPO   = Path("/home/user/FIXED-POINT-FACTORY")
CKPT   = REPO / "projects/REZN/checkpoints"
OUT    = REPO / "projects/REZN/overnight"
SOLVER = REPO / "projects/REZN/solver_code"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOLVER))
sys.path.insert(0, "/home/user/rezn-src")

from scipy.sparse.linalg import lgmres, LinearOperator
from code.contour_K3_halo import phi_K3_halo_smooth
from code.metrics import revelation_deficit

def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

# ── Load G21 tau=3.0 anchor ───────────────────────────────────────────────────
ANCHOR = CKPT / "g100_t0300_G21.npz"
d = np.load(ANCHOR, allow_pickle=True)
G_inner = int(d["G_inner"]); pad = int(d["pad"])
lo, hi  = pad, pad + G_inner
u_full  = d["u_full"].astype(np.float64)
tau     = d["tau_vec"].astype(np.float64)
W       = d["W_vec"].astype(np.float64)
P_anchor = d["P_full"].astype(np.float64)
tau_fixed = float(tau[0])
du = float(u_full[1] - u_full[0])
kernel_h = max(0.005, 0.05 * du)
n_inner = G_inner ** 3
sl = slice(lo, hi)

mp.mp.dps = 50
if "P_inner_mp_str" in d.files:
    s = d["P_inner_mp_str"]
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
                P_anchor[lo+i, lo+j, lo+l] = float(mp.mpf(str(s[i, j, l])))
mp.mp.dps = 40

log(f"Anchor: G_inner={G_inner} tau={tau_fixed} kernel_h={kernel_h}")

def phi64_factory(g):
    gv = np.full(3, g)
    def fn(P):
        return phi_K3_halo_smooth(P, u_full, lo, hi, tau, gv, W, kernel_h)
    return fn

def newton64(phi_fn, P0, tol=1e-12, max_iter=12, tag=""):
    P = P0.copy()
    phi_P = phi_fn(P)
    F_in = (phi_P - P)[sl, sl, sl].ravel()
    res = float(np.max(np.abs(F_in)))
    for it in range(max_iter):
        if res < tol:
            log(f"  {tag} it={it}  ||F||={res:.3e}  [converged]")
            return P, res, it
        normP = np.linalg.norm(P[sl, sl, sl])
        phi_P_in = phi_P[sl, sl, sl].ravel()
        def mv(w, _n=normP, _fp=phi_P_in, _P=P):
            wn = np.linalg.norm(w)
            if wn == 0.0: return w
            dlt = 1.5e-8 * (1.0 + _n) / wn
            Pp = _P.copy()
            Pp[sl, sl, sl] += dlt * w.reshape(G_inner, G_inner, G_inner)
            return w - (phi_fn(Pp)[sl, sl, sl].ravel() - _fp) / dlt
        A = LinearOperator((n_inner, n_inner), matvec=mv, dtype=np.float64)
        t0 = time.time()
        try:
            dv, info = lgmres(A, F_in, rtol=1e-4, maxiter=40, inner_m=25)
        except TypeError:
            dv, info = lgmres(A, F_in, tol=1e-4, maxiter=40, inner_m=25)
        P[sl, sl, sl] += dv.reshape(G_inner, G_inner, G_inner)
        P = np.clip(P, 1e-12, 1.0 - 1e-12)
        phi_P = phi_fn(P)
        F_in = (phi_P - P)[sl, sl, sl].ravel()
        res = float(np.max(np.abs(F_in)))
        log(f"  {tag} it={it}  ||F||={res:.3e}  t={time.time()-t0:.0f}s")
    return P, res, max_iter

# ── Step 1: float64 Newton to reach gamma=0.6889 ─────────────────────────────
log("="*60)
log("Step 1: float64 Newton sweep to gamma=0.6889")

gammas_to_reach = [0.9, 0.8, 0.75, 0.72, 0.71, 0.70, 0.695, 0.690, 0.6889]
P_cur = P_anchor.copy()
g_cur = 1.0

for g_next in gammas_to_reach:
    phi = phi64_factory(g_next)
    P_cur, res, nit = newton64(phi, P_cur, tol=1e-12, max_iter=12,
                               tag=f"f64 g={g_next:.4f}")
    if res < 1e-12:
        log(f"  -> converged at gamma={g_next:.4f}  ||F||={res:.2e}  iters={nit}")
        g_cur = g_next
    else:
        log(f"  -> FAILED at gamma={g_next:.4f}  ||F||={res:.2e} — using last good P")
        break

log(f"Float64 reference point: gamma={g_cur:.5f}")

# ── mpmath phi implementation ─────────────────────────────────────────────────
log("="*60)
log("Building mpmath phi (40 dps)")

_tau_mp  = [mp.mpf(str(tau[k]))  for k in range(3)]
_gam_mp  = [mp.mpf(str(tau_fixed / tau_fixed))]   # will be set per call
_W_mp    = [mp.mpf(str(W[k]))    for k in range(3)]
_u_mp    = [mp.mpf(str(u_full[i])) for i in range(len(u_full))]
_h_mp    = mp.mpf(str(kernel_h))
_inv2h2  = mp.mpf(1) / (2 * _h_mp**2)
EPS_MP   = mp.mpf('1e-12')

def f_mp(u_val, v, k):
    """f_v(u; tau[k]) in mpmath."""
    mean = mp.mpf('0.5') if v == 1 else mp.mpf('-0.5')
    d = u_val - mean
    return mp.sqrt(_tau_mp[k] / (2 * mp.pi)) * mp.exp(-_tau_mp[k] / 2 * d * d)

# Pre-compute f values at all grid points (doesn't change)
log("Pre-computing f_signal values...")
G_full = len(u_full)
fv = {}   # fv[k][v] = list of mp values at each grid point
for k in range(3):
    fv[k] = {}
    for v in range(2):
        fv[k][v] = [f_mp(_u_mp[i], v, k) for i in range(G_full)]
log("f_signal precomputed.")

def logit_mp(p):
    return mp.log(p) - mp.log(1 - p)

def x_crra_mp(mu, p, gamma_mp, W_mp):
    z = (logit_mp(mu) - logit_mp(p)) / gamma_mp
    if z >= 0:
        e = mp.exp(-z)
        return W_mp * (1 - e) / ((1 - p) * e + p)
    else:
        e = mp.exp(z)
        return W_mp * (e - 1) / ((1 - p) + p * e)

def clear_crra_mp(mu_vec, gamma_mp, W_mp_vec):
    a, b = EPS_MP, 1 - EPS_MP
    fa = sum(x_crra_mp(mu_vec[k], a, gamma_mp, W_mp_vec[k]) for k in range(3))
    fb = sum(x_crra_mp(mu_vec[k], b, gamma_mp, W_mp_vec[k]) for k in range(3))
    if fa <= 0: return a
    if fb >= 0: return b
    for _ in range(80):
        c = (a + b) / 2
        fc = sum(x_crra_mp(mu_vec[k], c, gamma_mp, W_mp_vec[k]) for k in range(3))
        if fc >= 0:
            a = c
        else:
            b = c
        if b - a < mp.mpf('1e-44'):
            break
    return (a + b) / 2

def phi_mp(P_mp, gamma_mp):
    """Evaluate phi at 40 dps. P_mp is 3D list of mpf values (full grid).
    Only updates inner cells; halo unchanged."""
    t_phi = time.time()
    G = G_full
    P_new = [[[P_mp[i][j][l] for l in range(G)] for j in range(G)] for i in range(G)]

    n_cells = (hi - lo)**3
    done = 0
    t0 = time.time()

    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P_mp[i][j][l]

                # Agent 0: slice P[i,:,:], tau_o0=tau[1], tau_o1=tau[2]
                A0, A1 = mp.mpf(0), mp.mpf(0)
                for ia in range(G):
                    fa0 = fv[1][0][ia]; fa1 = fv[1][1][ia]
                    P_row = P_mp[i][ia]
                    for ib in range(G):
                        diff = P_row[ib] - p
                        w = mp.exp(-diff * diff * _inv2h2)
                        A0 += w * fa0 * fv[2][0][ib]
                        A1 += w * fa1 * fv[2][1][ib]
                mu0_num = fv[0][1][i] * A1
                mu0_den = fv[0][0][i] * A0 + mu0_num
                mu0 = (mu0_num / mu0_den) if mu0_den > 0 else mp.mpf('0.5')
                mu0 = max(EPS_MP, min(1 - EPS_MP, mu0))

                # Agent 1: slice P[:,j,:], tau_o0=tau[0], tau_o1=tau[2]
                A0, A1 = mp.mpf(0), mp.mpf(0)
                for ia in range(G):
                    fa0 = fv[0][0][ia]; fa1 = fv[0][1][ia]
                    for ib in range(G):
                        diff = P_mp[ia][j][ib] - p
                        w = mp.exp(-diff * diff * _inv2h2)
                        A0 += w * fa0 * fv[2][0][ib]
                        A1 += w * fa1 * fv[2][1][ib]
                mu1_num = fv[1][1][j] * A1
                mu1_den = fv[1][0][j] * A0 + mu1_num
                mu1 = (mu1_num / mu1_den) if mu1_den > 0 else mp.mpf('0.5')
                mu1 = max(EPS_MP, min(1 - EPS_MP, mu1))

                # Agent 2: slice P[:,:,l], tau_o0=tau[0], tau_o1=tau[1]
                A0, A1 = mp.mpf(0), mp.mpf(0)
                for ia in range(G):
                    fa0 = fv[0][0][ia]; fa1 = fv[0][1][ia]
                    P_ia = P_mp[ia]
                    for ib in range(G):
                        diff = P_ia[ib][l] - p
                        w = mp.exp(-diff * diff * _inv2h2)
                        A0 += w * fa0 * fv[1][0][ib]
                        A1 += w * fa1 * fv[1][1][ib]
                mu2_num = fv[2][1][l] * A1
                mu2_den = fv[2][0][l] * A0 + mu2_num
                mu2 = (mu2_num / mu2_den) if mu2_den > 0 else mp.mpf('0.5')
                mu2 = max(EPS_MP, min(1 - EPS_MP, mu2))

                P_new[i][j][l] = clear_crra_mp([mu0, mu1, mu2], gamma_mp, _W_mp)

                done += 1
                if done % 500 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (n_cells - done) / rate
                    log(f"    phi_mp: {done}/{n_cells} cells  eta={eta:.0f}s")

    log(f"  phi_mp done in {time.time()-t_phi:.0f}s")
    return P_new

def f64_to_mp(P_f64):
    """Convert float64 numpy array to nested list of mpf."""
    G = P_f64.shape[0]
    return [[[mp.mpf(str(P_f64[i, j, l])) for l in range(G)]
              for j in range(G)] for i in range(G)]

def residual_mp(P_mp, P_new_mp):
    """Max absolute difference over inner cells."""
    res = mp.mpf(0)
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                d = abs(P_new_mp[i][j][l] - P_mp[i][j][l])
                if d > res:
                    res = d
    return res

def picard_mp(gamma_val, P_start_f64, n_iter=15, tag=""):
    """Run Picard fixed-point iteration P <- phi(P) at 40 dps."""
    gamma_mp = mp.mpf(str(gamma_val))
    P_mp = f64_to_mp(P_start_f64)
    log(f"  Picard_mp {tag}: gamma={gamma_val}  starting {n_iter} iterations at 40 dps")
    residuals = []
    for it in range(n_iter):
        t0 = time.time()
        P_new = phi_mp(P_mp, gamma_mp)
        res = residual_mp(P_mp, P_new)
        residuals.append(float(res))
        log(f"  {tag} iter={it}  ||P_new-P||={res:.3e}  t={time.time()-t0:.0f}s")
        P_mp = P_new
        if res < mp.mpf('1e-35'):
            log(f"  {tag} CONVERGED at iter={it}  ||F||={res:.3e}")
            break
    return P_mp, residuals

# ── Step 2: verify float64 solution in mpmath ─────────────────────────────────
log("="*60)
log(f"Step 2: verify float64 P*(gamma={g_cur:.4f}) in mpmath")
gamma_mp_cur = mp.mpf(str(g_cur))
P_mp_cur = f64_to_mp(P_cur)
log("Evaluating phi_mp once to get residual...")
P_mp_phi = phi_mp(P_mp_cur, gamma_mp_cur)
res_verify = residual_mp(P_mp_cur, P_mp_phi)
log(f"  Verification: ||phi_mp(P) - P||_inf = {res_verify:.6e}  (should be ~1e-12 or better)")

# ── Step 3: test at gamma just below fold ─────────────────────────────────────
test_gammas = [0.6887, 0.6885, 0.688, 0.685, 0.680]

results = {"anchor_gamma": g_cur,
           "verification_residual": str(res_verify),
           "fold_tests": []}

for g_test in test_gammas:
    log("="*60)
    log(f"Step 3: testing gamma={g_test} with mpmath Picard (40 dps)")
    P_test, resids = picard_mp(g_test, P_cur, n_iter=20, tag=f"g={g_test}")
    verdict = "CONVERGING" if len(resids) > 2 and resids[-1] < resids[0] * 0.01 else \
              "STALLED/DIVERGING"
    log(f"  Verdict at gamma={g_test}: {verdict}")
    log(f"  Residuals: {[f'{r:.2e}' for r in resids]}")
    results["fold_tests"].append({
        "gamma": g_test,
        "residuals": [str(r) for r in resids],
        "verdict": verdict
    })
    # Update P_cur for next test if converged
    if resids[-1] < 1e-20:
        P_cur_arr = np.array([[[float(P_test[i][j][l])
                                 for l in range(P_cur.shape[2])]
                                for j in range(P_cur.shape[1])]
                               for i in range(P_cur.shape[0])], dtype=np.float64)
        P_cur = P_cur_arr
        g_cur = g_test
        log(f"  Using converged P as starting point for next gamma")

out_path = OUT / "mp_fold_test.json"
with open(out_path, "w") as fh:
    json.dump(results, fh, indent=2)
log(f"Results written to {out_path}")
log("DONE")
