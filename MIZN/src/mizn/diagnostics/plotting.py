"""Per-run plot regeneration: convergence trajectory + price slices."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_plots(P: np.ndarray, hist: np.ndarray, params, out: Path):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # 1. convergence
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.semilogy(np.arange(1, len(hist) + 1), np.maximum(hist, 1e-20),
                  'o-', color='#27ae60', linewidth=1.5)
    ax.set_xlabel('iteration')
    ax.set_ylabel(r'$\|F\|_\infty$')
    ax.set_title(rf'convergence — {params.model}, $G={params.G}$')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'convergence.png', dpi=130)
    plt.close()
    # 2. price slices at u = u_bar and theta = theta_bar
    G = params.G
    u = np.linspace(-params.umax, params.umax, G)
    mid = G // 2
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(u, P[:, mid], 'o-', label=r'$P(\theta, u=u_{\bar{}})$')
    ax.plot(u, P[mid, :], 's--', label=r'$P(\theta=\bar\theta, u)$')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(r'$\theta$ or $u$')
    ax.set_ylabel(r'$P$')
    ax.set_title(rf'price slices — {params.model}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'P_slices.png', dpi=130)
    plt.close()
