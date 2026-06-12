"""Build RESULTS_highK.md + highK_report.pdf from results.json."""
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highK'
R = json.load(open(f"{OUT}/results.json"))
VAL = json.load(open(f"{OUT}/t0_validation_smooth.json"))
EMIN15 = json.load(open('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
                        'solved_fixed_points/dd_k3_overnight/emin15/emin15.json'))
LOWTAU = json.load(open('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
                        'solved_fixed_points/lowtau/lowtau.json'))

TAUS = [0.1, 0.2, 0.4, 0.6]
GAMMAS12 = list(np.round(np.logspace(np.log10(0.05), np.log10(30.0), 12), 4))


def row(K, G, tau, field='deficit'):
    cells = [v for v in R.values()
             if v['K'] == K and v['G'] == G and abs(v['tau'] - tau) < 1e-9]
    cells.sort(key=lambda v: v['gamma'])
    return (np.array([c['gamma'] for c in cells]),
            np.array([c[field] for c in cells]), cells)


def cert_ref(tau):
    """Certified K=3 reference row (emin15 for tau=0.2, lowtau else)."""
    src = EMIN15 if abs(tau - 0.2) < 1e-9 else LOWTAU
    pts = [(v['gamma'], v['deficit']) for v in src.values()
           if abs(v['tau'] - tau) < 1e-9 and v.get('verdict') == 'ACCEPT']
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


COL = {(3, 21): 'k', (3, 15): 'tab:blue', (4, 15): 'tab:red',
       (3, 11): 'tab:cyan', (4, 11): 'tab:orange', (5, 11): 'tab:green',
       (5, 15): 'tab:purple', (4, 21): 'maroon'}

pdf = PdfPages(f"{OUT}/highK_report.pdf")

# ------------------------------------------------------------ Fig 1
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
for ax, tau in zip(axes.ravel(), TAUS):
    gc, dc = cert_ref(tau)
    ax.plot(gc, dc, 'k+', ms=9, mew=1.6, label='K=3 certified ref (G=21)',
            zorder=5)
    series = [(3, 21), (3, 15), (4, 15)]
    if abs(tau - 0.2) < 1e-9:
        series += [(4, 21), (5, 15)]
    for (K, G) in series:
        g, d, _ = row(K, G, tau)
        if len(g) == 0:
            continue
        ls = 'o-' if G == 15 else ('s--' if G == 21 else 'd:')
        ax.plot(g, d, ls, ms=3.5, lw=1.2, color=COL[(K, G)],
                label=f'K={K}, G={G}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title(f'tau = {tau}')
    ax.set_xlabel('gamma'); ax.set_ylabel('revelation deficit (1-R^2)')
    ax.grid(alpha=0.3, which='both')
    if tau == TAUS[0]:
        ax.legend(fontsize=8)
fig.suptitle('Fig 1: Revelation deficit vs gamma — K=4 vs K=3 '
             '(smooth-kernel halo operator, h=0.45*sqrt(du))')
fig.tight_layout()
pdf.savefig(fig); fig.savefig(f"{OUT}/fig1_deficit_vs_gamma.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------ Fig 2
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
# left: G=11 K-ladder at tau=0.2 for gamma in {0.1, 1, 10}
ax = axes[0]
for gamma, mk in [(0.1, 'o-'), (1.0, 's-'), (10.0, 'd-')]:
    ks, ds = [], []
    for K in (3, 4, 5):
        c = [v for v in R.values() if v['K'] == K and v['G'] == 11
             and abs(v['tau'] - 0.2) < 1e-9 and abs(v['gamma'] - gamma) < 1e-9]
        if c:
            ks.append(K); ds.append(c[0]['deficit'])
    ax.plot(ks, ds, mk, label=f'gamma={gamma}')
ax.set_yscale('log'); ax.set_xticks([3, 4, 5])
ax.set_xlabel('K (number of traders)'); ax.set_ylabel('deficit')
ax.set_title('G=11, tau=0.2: deficit vs K')
ax.grid(alpha=0.3); ax.legend()
# right: G=15 at tau=0.2 (K=3,4 full + K=5 spots)
ax = axes[1]
for gamma, mk in [(0.1, 'o-'), (1.0, 's-'), (10.0, 'd-')]:
    ks, ds = [], []
    for K in (3, 4, 5):
        c = [v for v in R.values() if v['K'] == K and v['G'] == 15
             and abs(v['tau'] - 0.2) < 1e-9 and abs(v['gamma'] - gamma) < 1e-9]
        if c:
            ks.append(K); ds.append(c[0]['deficit'])
    ax.plot(ks, ds, mk, label=f'gamma={gamma}')
ax.set_yscale('log'); ax.set_xticks([3, 4, 5])
ax.set_xlabel('K'); ax.set_ylabel('deficit')
ax.set_title('G=15, tau=0.2: deficit vs K')
ax.grid(alpha=0.3); ax.legend()
fig.suptitle('Fig 2: Deficit vs number of traders at fixed (tau, gamma)')
fig.tight_layout()
pdf.savefig(fig); fig.savefig(f"{OUT}/fig2_deficit_vs_K.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------ Fig 3
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
for ax, tau in zip(axes.ravel(), TAUS):
    for (K, G) in [(3, 15), (4, 15)]:
        g, tvv, _ = row(K, G, tau, 'TV')
        if len(g):
            ax.plot(g, tvv, 'o-', ms=3.5, color=COL[(K, G)], label=f'K={K}')
    if abs(tau - 0.2) < 1e-9:
        for (K, G) in [(5, 11)]:
            g, tvv, _ = row(K, G, tau, 'TV')
            if len(g):
                ax.plot(g, tvv, 's--', ms=3.5, color=COL[(K, G)],
                        label='K=5 (G=11)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('gamma'); ax.set_ylabel('per-agent trading volume')
    ax.set_title(f'tau = {tau}'); ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
fig.suptitle('Fig 3: Trading volume per agent vs gamma (G=15)')
fig.tight_layout()
pdf.savefig(fig); fig.savefig(f"{OUT}/fig3_volume.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------ Fig 4
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
tau = 0.2
ax = axes[0]
for (K, G) in [(3, 15), (4, 15), (5, 11)]:
    g, vp, _ = row(K, G, tau, 'Vi_private')
    if len(g):
        ax.plot(g, np.maximum(vp, 1e-12), 'o-', ms=3.5, color=COL[(K, G)],
                label=f'K={K} (G={G})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('gamma'); ax.set_ylabel('Vi_private (CE units)')
ax.set_title('Private value of information, tau=0.2')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
ax = axes[1]
for (K, G) in [(3, 15), (4, 15), (5, 11)]:
    g, vp, _ = row(K, G, tau, 'Vi_public')
    if len(g):
        ax.plot(g, vp, 'o-', ms=3.5, color=COL[(K, G)], label=f'K={K} (G={G})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('gamma'); ax.set_ylabel('Vi_public (CE units)')
ax.set_title('Public (price) value of information, tau=0.2')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
fig.suptitle('Fig 4: Value-of-information decomposition per K')
fig.tight_layout()
pdf.savefig(fig); fig.savefig(f"{OUT}/fig4_vi.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------ Fig 5
fig, ax = plt.subplots(figsize=(10, 4.6))
groups = sorted(set((v['K'], v['G']) for v in R.values()))
xoff = 0
ticks, labels = [], []
for (K, G) in groups:
    cells = sorted([v for v in R.values() if v['K'] == K and v['G'] == G],
                   key=lambda v: (v['tau'], v['gamma']))
    xs = np.arange(len(cells)) + xoff
    Fs = [max(c['Finf'], 1e-16) for c in cells]
    ok = np.array([c['converged'] for c in cells])
    ax.semilogy(xs[ok], np.array(Fs)[ok], '.', color=COL.get((K, G), 'gray'))
    if (~ok).any():
        ax.semilogy(xs[~ok], np.array(Fs)[~ok], 'rx', ms=8)
    ticks.append(xoff + len(cells) / 2)
    labels.append(f'K={K}\nG={G}')
    xoff += len(cells) + 4
ax.axhline(1e-10, color='r', ls='--', lw=0.8, label='f_tol target 1e-10')
ax.set_xticks(ticks); ax.set_xticklabels(labels)
ax.set_ylabel('||Phi(P)-P||_inf')
ax.set_title('Fig 5: Certification residual per cell (red x = not converged)')
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
pdf.savefig(fig); fig.savefig(f"{OUT}/fig5_residuals.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------ Fig 6
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
for tau in TAUS:
    g3, d3, _ = row(3, 15, tau)
    g4, d4, _ = row(4, 15, tau)
    n = min(len(g3), len(g4))
    if n and np.allclose(g3[:n], g4[:n]):
        ax.semilogx(g3[:n], d4[:n] / d3[:n], 'o-', ms=3.5, label=f'tau={tau}')
ax.axhline(1.0, color='k', lw=0.8)
ax.set_xlabel('gamma'); ax.set_ylabel('deficit(K=4) / deficit(K=3)')
ax.set_title('K=4/K=3 deficit ratio (G=15)')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
ax = axes[1]
for (K, G) in [(3, 15), (4, 15)]:
    for tau in [0.2, 0.6]:
        g, d, _ = row(K, G, tau)
        if len(g):
            ax.loglog(g, d * g, 'o-' if K == 3 else 's--', ms=3.5,
                      color=COL[(K, G)], alpha=1 if tau == 0.2 else 0.5,
                      label=f'K={K}, tau={tau}')
ax.set_xlabel('gamma'); ax.set_ylabel('gamma * deficit')
ax.set_title('1/gamma asymptote check: gamma*deficit')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
fig.suptitle('Fig 6: What drives the deficit — K-ratio and gamma-scaling')
fig.tight_layout()
pdf.savefig(fig); fig.savefig(f"{OUT}/fig6_ratio_scaling.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------ Fig 7
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
for ax, tau in zip(axes.ravel(), TAUS):
    g3, d3, _ = row(3, 11, tau)
    for K in (4, 5):
        g, d, _ = row(K, 11, tau)
        n = min(len(g3), len(g))
        if n and np.allclose(g3[:n], g[:n]):
            ax.semilogx(g[:n], d[:n] / d3[:n], 'o-' if K == 4 else 's--',
                        ms=3.5, color=COL[(K, 11)], label=f'K={K} / K=3')
    ax.axhline(1.0, color='k', lw=0.8)
    ax.set_title(f'tau = {tau} (G=11)')
    ax.set_xlabel('gamma'); ax.set_ylabel('deficit ratio vs K=3')
    ax.set_ylim(0.5, 1.6); ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
fig.suptitle('Fig 7: K=4 and K=5 deficit relative to K=3 at identical '
             'discretisation (G=11)')
fig.tight_layout()
pdf.savefig(fig); fig.savefig(f"{OUT}/fig7_Kratio_G11.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------ Fig 8
fig, ax = plt.subplots(figsize=(8, 5))
for (K, G), mk in [((3, 15), 'o-'), ((4, 15), 's-'), ((5, 15), 'd-')]:
    g, sl, _ = row(K, G, 0.2, 'slope')
    if len(g):
        ax.semilogx(g, sl * K / 0.2, mk, ms=4, color=COL[(K, G)],
                    label=f'K={K} (G={G})')
ax.axhline(1.0, color='k', lw=0.8, label='slope = tau/K (single avg signal)')
ax.axhline(3.0, color='gray', ls=':', lw=0.8)
ax.set_xlabel('gamma'); ax.set_ylabel('slope * K / tau')
ax.set_title('Fig 8: Price aggregation — equilibrium slope x K / tau, tau=0.2\n'
             '(full revelation would be slope*K/tau = K)')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9)
fig.tight_layout()
pdf.savefig(fig); fig.savefig(f"{OUT}/fig8_slope_aggregation.png", dpi=130)
plt.close(fig)

pdf.close()
print("wrote highK_report.pdf")

# ================================================================ MD
lines = []
lines.append("# High-K extension of the partial-revelation results (K=4, K=5)\n")
lines.append("Operator: symmetric-K port of the certified emin15/lowtau family "
             "(kernel-smoothed Bayes, h = 0.45*sqrt(du), pad=2 no-learning halo, "
             "u_max=4, W=1), solved on the S_K sorted-tuple basis "
             "(C(G_full+K-1,K) cells instead of G_full^K).\n")
lines.append("## T0 validation\n")
lines.append(f"- jit sym operator == `phi_K3_halo_smooth` to {VAL['op_identity']:.1e} (machine precision)")
lines.append(f"- certified K=3 fixed point (t=0.2, g=1.035) residual under sym operator: {VAL['cert_fp_residual']:.1e}")
k3g21 = VAL['k3_smooth_G21']
k3g15 = VAL['k3_smooth_G15']
lines.append(f"- K=3 G=21 Newton from cold start: deficit {k3g21['deficit']:.6e} "
             f"vs certified 2.558974e-4 (rel err {k3g21['rel_err_vs_cert']:+.2%}), wall {k3g21['wall']:.1f}s")
lines.append(f"- K=3 G=15: deficit {k3g15['deficit']:.6e} ({k3g15['rel_err_vs_cert']:+.2%} discretisation drift "
             "-> cross-K comparisons done at fixed G)")
lines.append("- NOTE: the bare hard-scan contour operator (contour_KN_sym.sym_phi, h=0) "
             "admits the exact fully-revealing fixed point (deficit ~1e-24, gamma-independent); "
             "the certified PR deficits live in the smooth-kernel (h=0.45*sqrt(du)) family. "
             "See t0_validation.json.\n")

lines.append("## K=4 vs K=3 deficits at G=15 (same du, same h)\n")
for tau in TAUS:
    lines.append(f"\n### tau = {tau}\n")
    lines.append("| gamma | K=3 G=21 (ref) | K=3 G=15 | K=4 G=15 | K4/K3 (G=15) | K=4 F_inf | K=4 wall (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    g3r, d3r, _ = row(3, 21, tau)
    g3, d3, _ = row(3, 15, tau)
    g4, d4, c4 = row(4, 15, tau)
    for i, g in enumerate(g4):
        j = int(np.argmin(np.abs(g3 - g)))
        jr = int(np.argmin(np.abs(g3r - g)))
        lines.append(f"| {g} | {d3r[jr]:.4e} | {d3[j]:.4e} | {d4[i]:.4e} | "
                     f"{d4[i]/d3[j]:.3f} | {c4[i]['Finf']:.1e} | {c4[i]['wall']:.1f} |")

lines.append("\n## K-ladder at G=11 (K=3,4,5 at identical discretisation)\n")
for tau in TAUS:
    g3, d3, _ = row(3, 11, tau)
    g4, d4, _ = row(4, 11, tau)
    g5, d5, c5 = row(5, 11, tau)
    if not len(g5):
        continue
    lines.append(f"\n### tau = {tau}\n")
    lines.append("| gamma | K=3 | K=4 | K=5 | K4/K3 | K5/K3 |")
    lines.append("|---|---|---|---|---|---|")
    for i, g in enumerate(g5):
        j3 = int(np.argmin(np.abs(g3 - g)))
        j4 = int(np.argmin(np.abs(g4 - g)))
        lines.append(f"| {g} | {d3[j3]:.4e} | {d4[j4]:.4e} | {d5[i]:.4e} | "
                     f"{d4[j4]/d3[j3]:.3f} | {d5[i]/d3[j3]:.3f} |")

lines.append("\n## K=4 vs K=3 at the certified discretisation (G=21), tau=0.2\n")
g3, d3, _ = row(3, 21, 0.2)
g4, d4, c4 = row(4, 21, 0.2)
if len(g4):
    lines.append("| gamma | K=3 G=21 | K=4 G=21 | ratio |")
    lines.append("|---|---|---|---|")
    for i, g in enumerate(g4):
        j = int(np.argmin(np.abs(g3 - g)))
        lines.append(f"| {g} | {d3[j]:.4e} | {d4[i]:.4e} | {d4[i]/d3[j]:.3f} |")

g5s, d5s, c5s = row(5, 15, 0.2)
if len(g5s):
    lines.append("\n## K=5 full row at G=15, tau=0.2\n")
    lines.append("| gamma | K=3 | K=4 | K=5 | F_inf | wall (s) |")
    lines.append("|---|---|---|---|---|---|")
    g3, d3, _ = row(3, 15, 0.2)
    g4, d4, _ = row(4, 15, 0.2)
    for i, g in enumerate(g5s):
        j3 = int(np.argmin(np.abs(g3 - g)))
        j4 = int(np.argmin(np.abs(g4 - g)))
        lines.append(f"| {g} | {d3[j3]:.4e} | {d4[j4]:.4e} | {d5s[i]:.4e} | "
                     f"{c5s[i]['Finf']:.1e} | {c5s[i]['wall']:.0f} |")

nconv = sum(1 for v in R.values() if v['converged'])
nfail = [k for k, v in R.items() if not v['converged']]
lines.append(f"\n## Convergence\n\n{nconv}/{len(R)} cells converged "
             f"(f_tol 1e-10; most reach ~1e-14).")
if nfail:
    lines.append(f"\nNot converged: {nfail}")

lines.append("\n## Verdict\n")
lines.append("""
1. **The deficit is essentially K-invariant at fixed (tau, gamma).**
   Geometric-mean ratios over gamma >= 0.16 at identical discretisation
   (G=11): K4/K3 = 0.99, 0.98, 1.02, 1.05 and K5/K3 = 0.93, 0.93, 0.98,
   1.03 for tau = 0.1, 0.2, 0.4, 0.6.  Per-cell deviations stay within
   [0.80, 1.27].  More traders do NOT shrink the deficit toward zero
   (no 1/K or faster pooling collapse); at high tau (0.6) K=4/K=5 are
   actually slightly LESS revealing than K=3.

2. **Why: the price level aggregates only the AVERAGE signal.**
   Across every converged cell, slope*K/tau = 1.0 +/- 3% — the
   equilibrium price behaves like logit(p) ~ tau * mean(u), i.e. one
   signal's worth of precision regardless of K (full revelation would
   be slope*K/tau = K).  The deficit (1-R^2 nonlinearity, the Jensen
   wedge of CRRA market clearing around that linear statistic) is a
   per-clearing property, not diversified away by adding traders.

3. **What drives a HIGH deficit (unchanged from K=3, now verified at
   K=4, K=5): high tau, low gamma.**  Deficit rises steeply in tau
   (tau=0.6 deficits are 2-4 orders of magnitude above tau=0.1) and
   falls in gamma with the ~1/gamma asymptote at low-to-moderate tau;
   at tau >= 0.4 the known high-gamma plateau/upturn (gamma*deficit
   rising) also reproduces at K=4 and K=5.  K itself is a second-order
   parameter: within +/-25 percent.

4. **Per-agent volume RISES with K** (tau=0.2, gamma=1: TV = 0.34,
   0.44, 0.52 for K=3,4,5) — more counterparties, more disagreement
   trade — while **Vi_private falls and Vi_public rises with K**
   (the price reveals the mean of more signals, so the public signal
   crowds out the private one), consistent with the price becoming
   more informative as a STATISTIC even though the revelation deficit
   of its functional form is unchanged.

5. Honest caveats: all 390 cells converged (worst ||F||_inf = 8e-13,
   typical 7e-15, target 1e-10) so no failed cells are hidden;
   K=5 at full G=15 was run for tau=0.2 only (other taus at G=11);
   cross-K statements use identical-G comparisons because the G=15
   vs G=21 discretisation drift (~8 percent at tau=0.2, gamma=1.035)
   is of the same order as the K-effect itself.
""")

open(f"{OUT}/RESULTS_highK.md", 'w').write("\n".join(lines) + "\n")
print("wrote RESULTS_highK.md")
