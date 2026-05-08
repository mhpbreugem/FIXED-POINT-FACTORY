#!/usr/bin/env python3
"""Compute trade volume and value of information for every REE checkpoint.

Reads .npz checkpoints from projects/REZN/checkpoints/ (and an optional seed
at results/full_ree/seed_*.npz). Writes projects/REZN/figures/all_metrics.csv.

Requires the REZN paper repo at $REZN_SRC (default ~/rezn-source).
Install deps: pip install numba numpy
"""
import os, sys, math
from pathlib import Path
import numpy as np

REZN_SRC = Path(os.environ.get("REZN_SRC", Path.home() / "rezn-source"))
sys.path.insert(0, str(REZN_SRC))
from code.contour_K3_halo import _agent_evidence_K3, _bayes
from code.demand import x_crra
from numba import njit

ROOT = Path(__file__).resolve().parents[3]

@njit(cache=False)
def compute_posteriors_K3(P_full, u_full, inner_lo, inner_hi, tau_vec, gamma_vec, W_vec):
    G = P_full.shape[0]
    posteriors = np.zeros((3, G, G, G), dtype=np.float64)
    for i in range(inner_lo, inner_hi):
        acc = np.empty(2, dtype=np.float64)
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full[i, j, l]
                _agent_evidence_K3(P_full[i, :, :], p, u_full, tau_vec[1], tau_vec[2], acc)
                posteriors[0, i, j, l] = _bayes(u_full[i], tau_vec[0], acc[0], acc[1])
                _agent_evidence_K3(P_full[:, j, :], p, u_full, tau_vec[0], tau_vec[2], acc)
                posteriors[1, i, j, l] = _bayes(u_full[j], tau_vec[1], acc[0], acc[1])
                _agent_evidence_K3(P_full[:, :, l], p, u_full, tau_vec[0], tau_vec[1], acc)
                posteriors[2, i, j, l] = _bayes(u_full[l], tau_vec[2], acc[0], acc[1])
    return posteriors

def signal_density(u, v, tau):
    mean = 0.5 if v == 1 else -0.5
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-0.5 * tau * (u - mean) ** 2)

def crra_eu(mu, x, p, gamma, W):
    W1 = W + x * (1 - p); W0 = W - x * p
    if W1 <= 0 or W0 <= 0: return -1e18
    if abs(gamma - 1) < 1e-9:
        return mu * math.log(W1) + (1 - mu) * math.log(W0)
    return mu * W1 ** (1 - gamma) / (1 - gamma) + (1 - mu) * W0 ** (1 - gamma) / (1 - gamma)

def crra_ce(eu, gamma):
    if eu <= -1e17: return 0
    if abs(gamma - 1) < 1e-9: return math.exp(eu)
    val = (1 - gamma) * eu
    return val ** (1 / (1 - gamma)) if val > 0 else 0

def compute_metrics(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    P_full = d['P_full']; u_full = d['u_full']
    G_inner = int(d['G_inner']); pad = int(d['pad'])
    tau_vec = d['tau_vec']; gamma_vec = d['gamma_vec']; W_vec = d['W_vec']
    inner_lo, inner_hi = pad, pad + G_inner
    gamma = float(gamma_vec[0]); tau = float(tau_vec[0]); W = float(W_vec[0])
    posteriors = compute_posteriors_K3(P_full, u_full, inner_lo, inner_hi, tau_vec, gamma_vec, W_vec)
    u_inner = u_full[inner_lo:inner_hi]
    G = G_inner
    f0 = np.array([signal_density(u, 0, tau) for u in u_inner])
    f1 = np.array([signal_density(u, 1, tau) for u in u_inner])
    du = u_inner[1] - u_inner[0]
    trap = np.ones(G); trap[0] = 0.5; trap[-1] = 0.5; trap *= du
    eps = 1e-12

    # Volume
    vol = np.zeros(3); ws = 0
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = P_full[inner_lo+i, inner_lo+j, inner_lo+l]
                w = 0.5 * (f0[i]*f0[j]*f0[l] + f1[i]*f1[j]*f1[l]) * trap[i]*trap[j]*trap[l]
                ws += w
                if w <= 0 or p <= eps or p >= 1-eps: continue
                for k in range(3):
                    mu_k = posteriors[k, inner_lo+i, inner_lo+j, inner_lo+l]
                    if mu_k <= eps or mu_k >= 1-eps: continue
                    vol[k] += w * abs(x_crra(mu_k, p, gamma_vec[k], W_vec[k]))
    V_vol = 0.5 * vol.sum() / ws

    # Value of info
    ce_diff = 0; tot_w = 0
    for i in range(G):
        for j in range(G):
            for l in range(G):
                p = P_full[inner_lo+i, inner_lo+j, inner_lo+l]
                mu_m = posteriors[0, inner_lo+i, inner_lo+j, inner_lo+l]
                if mu_m <= eps or mu_m >= 1-eps or p <= eps or p >= 1-eps: continue
                lm = math.log(mu_m / (1 - mu_m))
                x_m = x_crra(mu_m, p, gamma, W)
                ce_m = crra_ce(crra_eu(mu_m, x_m, p, gamma, W), gamma)
                for k in range(G):
                    z = lm + tau * u_inner[k]
                    mu_p = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
                    if mu_p <= eps or mu_p >= 1-eps: continue
                    x_p = x_crra(mu_p, p, gamma, W)
                    ce_p = crra_ce(crra_eu(mu_p, x_p, p, gamma, W), gamma)
                    w = 0.5 * (f0[i]*f0[j]*f0[l]*f0[k] + f1[i]*f1[j]*f1[l]*f1[k]) * trap[i]*trap[j]*trap[l]*trap[k]
                    ce_diff += w * (ce_p - ce_m); tot_w += w
    V_info = ce_diff / tot_w if tot_w > 0 else 0
    return gamma, tau, V_vol, V_info

if __name__ == "__main__":
    ckpts = sorted((ROOT / "projects/REZN/checkpoints").glob("*.npz"))
    seed = ROOT / "results/full_ree/seed_g050_t0200_3d.npz"
    if seed.exists():
        ckpts.append(seed)
    out = ROOT / "projects/REZN/figures/all_metrics.csv"
    out.parent.mkdir(exist_ok=True, parents=True)
    with open(out, "w") as f:
        f.write("gamma,tau,volume,V_info,task_id\n")
        for ckpt in ckpts:
            try:
                g, t, vo, vi = compute_metrics(ckpt)
                f.write(f"{g},{t},{vo:.6f},{vi:.8f},{ckpt.stem}\n")
                print(f"  {ckpt.stem:<28} γ={g} τ={t} V_vol={vo:.4f} V_info={vi:.5f}")
            except Exception as e:
                print(f"  {ckpt.stem}: ERROR {e}")
    print(f"\nWrote {out}")
