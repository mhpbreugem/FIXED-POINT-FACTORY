"""Visualize the cusp / critical-point story: lookup mu(p, u_k) vs p,
|grad P| on slice with critical points, and Cheby coef decay.
"""
import os, sys, json, glob
sys.path.insert(0, "/tmp"); sys.path.insert(0, "/tmp/cheby_h0")
import numpy as np
from numpy.polynomial import chebyshev as cheb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dd_k3_cheby_lobatto import (lobatto_grid, cheby_fit_lobatto_3d,
                                          lookup_analytic_lobatto)

OUT = "/tmp/dd_k3_cheby_cusp_figs"
os.makedirs(OUT, exist_ok=True)


def load_r4_fp(key):
    f = f"/tmp/dd_k3_sweep_fps/{key}.npz"
    if not os.path.exists(f): return None, None, None
    d = np.load(f)
    return ((d["P"] if "P" in d.files else (d["mu_hi"]+d["mu_lo"])).astype(np.float64),
              float(d["gamma"]), float(d["tau"]))


# Pick gamma=100, tau=1 (kinky regime) and gamma=1, tau=0.2 (smooth)
cells = [("g100_t1.0000", "kinky"), ("g1_t0.2000", "smooth")]

# --- Fig 1: lookup mu(p, u_k) vs p for selected u_k, showing cusps ---
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for col, (key, label) in enumerate(cells):
    P, gamma, tau = load_r4_fp(key)
    if P is None: continue
    G = P.shape[0]
    from lin_cdf_strict import make_cdf_uniform_grid
    u_grid = make_cdf_uniform_grid(G)
    U_MAX = max(abs(u_grid[0]), abs(u_grid[-1]))
    # Re-solve at Lobatto for a clean Cheby fit
    from dd_k3_cheby_lobatto import solve_fp_lobatto
    print(f"solving {key} at Lobatto G={G}...", flush=True)
    P_lob, F, u_lob, U_MAX = solve_fp_lobatto(gamma, tau, G)
    print(f"  done F={F:.2e}", flush=True)
    coefs = cheby_fit_lobatto_3d(P_lob, U_MAX)
    ax = axs[col]
    p_eval = np.linspace(0.005, 0.995, 400)
    for u_k in [-1.5, 0.0, 1.5]:
        mu_vals = [lookup_analytic_lobatto(coefs, U_MAX, p, u_k, tau, NQK=24)
                      for p in p_eval]
        ax.plot(p_eval, mu_vals, "-", lw=1.0, alpha=0.8, label=f"$u_k={u_k}$")
    ax.set_xlabel("$p$"); ax.set_ylabel("$\\mu(p, u_k)$")
    ax.set_title(f"{key} ({label}): $\\gamma$={gamma:g}, $\\tau$={tau}")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
plt.suptitle("Lookup $\\mu(p, u_k)$: smooth regime vs kinky regime", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_mu_cusps.png", dpi=140); plt.close()


# --- Fig 2: |grad P|^2 on slice with critical points marked ---
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for col, (key, label) in enumerate(cells):
    P, gamma, tau = load_r4_fp(key)
    if P is None: continue
    G = P.shape[0]
    from dd_k3_cheby_lobatto import solve_fp_lobatto
    P_lob, F, u_lob, U_MAX = solve_fp_lobatto(gamma, tau, G)
    coefs = cheby_fit_lobatto_3d(P_lob, U_MAX)
    # Slice at u_1 = 0
    n_fine = 80
    u_fine = np.linspace(-U_MAX*0.98, U_MAX*0.98, n_fine)
    xi_fine = u_fine / U_MAX
    grad2 = np.zeros((n_fine, n_fine))
    # Derivatives along axis 1 and 2 (u_2, u_3)
    deg = coefs.shape[0] - 1
    # Slice at xi_1 = 0 -> sum_i coefs[i, :, :] T_i(0)
    T_at_0 = np.zeros(deg+1); T_at_0[0] = 1.0
    if deg >= 1: T_at_0[1] = 0.0
    for i in range(1, deg):
        T_at_0[i+1] = 2*0.0*T_at_0[i] - T_at_0[i-1]
    coefs_slice = np.einsum("i,ijk->jk", T_at_0, coefs)
    # Derivative w.r.t. u_2: chebder along axis 0
    dcoefs_u2 = np.zeros((coefs_slice.shape[0]-1, coefs_slice.shape[1]))
    for k in range(coefs_slice.shape[1]):
        dcoefs_u2[:, k] = cheb.chebder(coefs_slice[:, k], 1) / U_MAX
    # Derivative w.r.t. u_3: chebder along axis 1
    dcoefs_u3 = np.zeros((coefs_slice.shape[0], coefs_slice.shape[1]-1))
    for i in range(coefs_slice.shape[0]):
        dcoefs_u3[i, :] = cheb.chebder(coefs_slice[i, :], 1) / U_MAX
    for j, xi_2 in enumerate(xi_fine):
        for k, xi_3 in enumerate(xi_fine):
            dPdu2 = cheb.chebval2d(xi_2, xi_3, dcoefs_u2)
            dPdu3 = cheb.chebval2d(xi_2, xi_3, dcoefs_u3)
            grad2[j, k] = dPdu2*dPdu2 + dPdu3*dPdu3
    ax = axs[col]
    im = ax.contourf(u_fine, u_fine, np.log10(grad2 + 1e-12),
                          levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, label="$\\log_{10}|\\nabla P|^2$")
    # Find critical points (minima of |grad|^2)
    from scipy.ndimage import minimum_filter
    local_min = (grad2 == minimum_filter(grad2, size=5))
    crit_mask = local_min & (grad2 < 1e-2)
    crit_j, crit_k = np.where(crit_mask)
    ax.plot(u_fine[crit_j], u_fine[crit_k], "rx", ms=8, mew=2,
              label=f"{len(crit_j)} critical points")
    ax.set_xlabel("$u_2$"); ax.set_ylabel("$u_3$")
    ax.set_title(f"{key} ({label}): $|\\nabla P|^2$ on slice $u_1{{=}}0$")
    ax.legend(fontsize=8)
plt.suptitle("Critical points cause cusps in $\\mu(p, u_k)$", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_grad_critical.png", dpi=140); plt.close()


# --- Fig 3: Cheby coef decay log-log: sanity, Lin-CDF R4 FP, Cheby-native iter ---
fig, ax = plt.subplots(figsize=(10, 6))
# Sanity functions
data = {
    "poly deg=4":           [(7, 9e-17), (11, 7e-17), (15, 1e-16), (21, 2e-16), (31, 1e-16)],
    "sigmoid(0.5T)":         [(7, 2.4e-5), (11, 1.8e-8), (15, 1.4e-11), (21, 4e-16), (31, 1e-17)],
    "sigmoid(2T)":           [(7, 3.6e-3), (11, 2.6e-4), (15, 1.6e-5), (21, 2.9e-7), (31, 3.9e-10)],
    "sigmoid(10T) near-step":[(7, 2e-2), (11, 9.6e-3), (15, 3.9e-3), (21, 1.3e-3), (31, 2.9e-4)],
}
for name, pts in data.items():
    Gs, ecs = zip(*pts)
    ax.semilogy(Gs, ecs, "--", lw=1.0, ms=4, marker="o", alpha=0.5, label=name)
# Lin-CDF R4 FP at gamma=100, tau=1
ax.semilogy([11, 15, 21], [0.049, 0.020, 0.011], "-", lw=2, ms=8, marker="s",
                color="C3", label="Lin-CDF R4 FP @ (100,1.0)")
# Cheby-native at G=11, iterations
ax.axhline(3e-4, color="C0", ls="-", lw=2, alpha=0.5,
              label="Cheby-native FP edge floor @ (100,1.0), G=11")
ax.set_xlabel("$G$ (Lobatto grid)"); ax.set_ylabel("max edge Cheby coef")
ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)
ax.axhline(1e-16, color="k", lw=0.5, alpha=0.3)
ax.text(31, 2e-16, "machine eps", fontsize=8, ha="right")
ax.set_title("Cheby decay: smooth functions (dashed) vs FPs (solid)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_decay.png", dpi=140); plt.close()


# --- Fig 4: regime map across (gamma, tau) ---
try:
    d = json.load(open("/tmp/dd_k3_cheby_map.json"))
    GAMMAS = sorted(set(v["gamma"] for v in d.values()))
    TAUS = sorted(set(v["tau"] for v in d.values()))
    nG = len(GAMMAS); nT = len(TAUS)
    gi = {g: i for i, g in enumerate(GAMMAS)}
    ti = {t: i for i, t in enumerate(TAUS)}
    # Edge at largest G
    M = np.full((nT, nG), np.nan)
    for k, v in d.items():
        gammaval = v["gamma"]; tauval = v["tau"]
        # find largest G that has edge_coef
        Gs_ok = [(int(G_s), val.get("edge_coef")) for G_s, val in v["by_G"].items()
                    if val.get("edge_coef") is not None]
        if not Gs_ok: continue
        Gs_ok.sort()
        M[ti[tauval], gi[gammaval]] = Gs_ok[-1][1]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.log10(M), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(nG)); ax.set_xticklabels([f"{g:g}" for g in GAMMAS])
    ax.set_yticks(range(nT)); ax.set_yticklabels([f"{t:g}" for t in TAUS])
    ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\tau$")
    plt.colorbar(im, ax=ax, label="$\\log_{10}$ edge coef (at largest $G$)")
    for i in range(nT):
        for j in range(nG):
            if np.isfinite(M[i,j]):
                ax.text(j, i, f"{np.log10(M[i,j]):.1f}", ha="center", va="center",
                          fontsize=8, color="white" if M[i,j]<1e-6 else "black")
    ax.set_title("$(\\gamma, \\tau)$ regime: smooth ($\\log<-6$) vs kinky")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig4_regime.png", dpi=140); plt.close()
except Exception as e: print("fig4 fail:", e)


print(f"figs in {OUT}: {sorted(os.listdir(OUT))}")
