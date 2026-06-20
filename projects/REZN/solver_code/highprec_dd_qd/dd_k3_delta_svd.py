"""Quick: what's the SVD rank of delta? If rank ~ 1-2, a low-rank
multiplicative correction L = a(u_k) + b(p) + sum_k c_k(p)*d_k(u_k) would
close the gap. If rank ~ G, no low-rank fix exists and we need
piecewise/multi-element."""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

DATA = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight"
OUT = f"{DATA}/lpub_lpriv_v2/figs"


def make_cdf_uniform_grid(G, eps_q=0.01):
    return norm.ppf(np.linspace(eps_q, 1 - eps_q, G))


def make_p_grid(Gp, eps=1e-3):
    lo, hi = np.log(eps/(1-eps)), np.log((1-eps)/eps)
    return 1.0 / (1.0 + np.exp(-np.linspace(lo, hi, Gp)))


cells = [(100, 0.2), (100, 1.0), (10, 1.0), (1000, 1.0)]
fig, axes = plt.subplots(1, len(cells), figsize=(5*len(cells), 4))

for ax, (g, t) in zip(axes, cells):
    f = f"{DATA}/dd_k3_strict_fp_g{g}_t{t:.4f}.npz"
    if not os.path.exists(f): continue
    d = np.load(f)
    mu = np.clip(d["mu_strict"].astype(np.float64), 1e-9, 1-1e-9)
    L = np.log(mu/(1-mu))
    Gp, G = L.shape
    # ANOVA
    mu_L = L.mean()
    a = L.mean(axis=1)  # row (p) effect
    b = L.mean(axis=0)  # col (u_k) effect
    delta = L - mu_L - (a - mu_L)[:, None] - (b - mu_L)[None, :]
    # Add back the mean (centering)
    delta = L - a[:, None] - b[None, :] + mu_L

    # SVD
    U, s, Vt = np.linalg.svd(delta, full_matrices=False)
    s_L = np.linalg.svd(L, compute_uv=False)
    # Normalize by Frobenius norm
    s_n = s / s.max()
    s_L_n = s_L / s_L.max()
    ax.semilogy(s_n, 'r-s', label=r"$\sigma_k(\delta)/\sigma_1$", ms=5)
    ax.semilogy(s_L_n, 'b-o', label=r"$\sigma_k(L)/\sigma_1$", ms=5, alpha=0.5)
    ax.set_xlabel("rank k")
    ax.set_ylabel(r"singular value (normalized)")
    ax.set_title(f"$\\gamma={g}, \\tau={t}$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    # print rank-k truncation error
    for k in [1, 2, 3, 5]:
        if k <= len(s):
            err = np.linalg.norm(s[k:])
            tot = np.linalg.norm(s)
            print(f"  g={g} t={t}: rank-{k} truncation captures "
                  f"{1 - err/tot:.4f} of delta Frobenius norm")

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_svd.png", dpi=120)
plt.close()
print(f"saved fig4_svd.png")
