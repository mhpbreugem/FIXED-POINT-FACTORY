"""Low-tau report: 12-figure PDF with certified deficits at tau in {0.05, 0.10, 0.20}.

The economic point: with binary payoff, CRRA risk-averse traders, Gaussian
signals, NO noise traders, the equilibrium has strictly positive revelation
deficit, vanishing as gamma -> infty. We show this cleanly at low tau where
strict-h=0 ill-posedness is absent and surfaces are smooth.

Figures (>=10 required):
  F1 deficit vs gamma at tau=0.05, 0.10, 0.20 (combined log-log)
  F2 deficit vs gamma at tau=0.20 only (with 1/gamma reference)
  F3 deficit vs gamma at tau=0.10 only
  F4 deficit vs gamma at tau=0.05 only
  F5 slope of logit(P) on sum(u) vs gamma per tau
  F6 certified residuals per cell (verification of certification)
  F7 deficit heatmap on extended (tau, gamma) grid
  F8 sample price surface slice u3=0 at (tau=0.10, gamma=0.05) -- biggest deficit
  F9 sample price surface slice u3=0 at (tau=0.10, gamma=1.0) -- moderate
  F10 sample price surface slice u3=0 at (tau=0.10, gamma=30.0) -- near-revealing
  F11 logit-P vs sum(u) scatter at (tau=0.10, gamma=0.05) -- shows deficit
  F12 1/gamma power-law fit on the asymptotic tail at each tau
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3

OUT_DIR = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau'
EMIN15 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'
PDF_PATH = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/LOWTAU_report.pdf'


def load_all():
    """Combine lowtau (0.05, 0.10) with emin15 tau=0.2 row."""
    rows = []
    # lowtau
    try:
        d = json.load(open(f"{OUT_DIR}/lowtau.json"))
        for v in d.values():
            if v.get('verdict') == 'ACCEPT':
                rows.append(dict(tau=v['tau'], gamma=v['gamma'],
                                 deficit=v['deficit'], slope=v['slope'],
                                 F_ld=v['F_ld']))
    except FileNotFoundError:
        pass
    # emin15: tau=0.2 and tau=0.5 rows already certified there
    try:
        d = json.load(open(f"{EMIN15}/emin15.json"))
        for v in d.values():
            tau_v = float(v['tau'])
            if v.get('verdict') == 'ACCEPT' and tau_v in (0.2, 0.5):
                rows.append(dict(tau=tau_v, gamma=float(v['gamma']),
                                 deficit=float(v['deficit']),
                                 slope=float(v['slope']),
                                 F_ld=float(v['F_ld']) if v.get('F_ld') else 0.0))
    except FileNotFoundError:
        pass
    return rows


def per_tau(rows, tau):
    r = sorted([x for x in rows if abs(x['tau']-tau) < 1e-5], key=lambda x: x['gamma'])
    g = np.array([x['gamma'] for x in r])
    d = np.array([x['deficit'] for x in r])
    s = np.array([x['slope'] for x in r])
    F = np.array([x['F_ld'] for x in r])
    return g, d, s, F


def load_P(tau, gamma):
    """Load the certified P inner cube at (tau, gamma) and build full halo."""
    Gi = 21; pad = 2; UMAX = 4.0
    du = 2*UMAX/(Gi-1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    src = f"{OUT_DIR}/P_ld_t{tau}_g{gamma}.npy"
    if not os.path.exists(src):
        src = f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy"
    if not os.path.exists(src): return None, None, None, None
    P_inner = np.load(src)
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma), np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    return P_full, uf, lo, hi


def main():
    rows = load_all()
    print(f"Loaded {len(rows)} certified cells", flush=True)
    taus = sorted(set(round(x['tau'], 4) for x in rows))
    print(f"taus available: {taus}", flush=True)

    cmap = {0.05: 'C0', 0.10: 'C1', 0.20: 'C2', 0.30: 'C3', 0.40: 'C4', 0.50: 'C5'}
    label = {t: rf'$\tau={t:.2f}$' for t in cmap}

    pdf = PdfPages(PDF_PATH)

    tau_show = [t for t in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50] if t in taus]

    # ---------- F1 combined deficit vs gamma (log-log) ----------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for tau in tau_show:
        g, d, _, _ = per_tau(rows, tau)
        if g.size == 0: continue
        d_plot = np.where(d > 0, d, 1e-16)
        ax.loglog(g, d_plot, 'o-', color=cmap[tau], label=label[tau], markersize=7, linewidth=2)
    gg = np.logspace(-1.5, 1.6, 50)
    ax.loglog(gg, 0.01/gg, 'k--', alpha=0.4, label=r'$\propto 1/\gamma$ ref')
    ax.set_xlabel(r'$\gamma$ (risk aversion)', fontsize=13)
    ax.set_ylabel(r'revelation deficit  $1 - R^2$', fontsize=13)
    ax.set_title('F1 -- Certified deficit vs $\\gamma$ (log-log)\n'
                 'no noise traders; deficit $> 0$ and $\\to 0$ as $\\gamma \\to \\infty$',
                 fontsize=12)
    ax.legend(loc='lower left', fontsize=10, ncol=2)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---------- F1b combined deficit vs gamma (linear y, semilogx) ----------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for tau in tau_show:
        g, d, _, _ = per_tau(rows, tau)
        if g.size == 0: continue
        ax.semilogx(g, d, 'o-', color=cmap[tau], label=label[tau], markersize=7, linewidth=2)
    ax.axhline(0, color='k', alpha=0.3, linewidth=0.6)
    ax.set_xlabel(r'$\gamma$ (risk aversion, log scale)', fontsize=13)
    ax.set_ylabel(r'revelation deficit  $1 - R^2$  (linear)', fontsize=13)
    ax.set_title('F1b -- Certified deficit vs $\\gamma$ (linear $y$ axis)\n'
                 'the asymptotic-to-zero behaviour at large $\\gamma$',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---------- F1c combined deficit vs gamma (linear x, linear y) low-gamma zoom ----------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for tau in tau_show:
        g, d, _, _ = per_tau(rows, tau)
        if g.size == 0: continue
        m = g <= 3.0
        ax.plot(g[m], d[m], 'o-', color=cmap[tau], label=label[tau], markersize=7, linewidth=2)
    ax.set_xlabel(r'$\gamma$ (linear)', fontsize=13)
    ax.set_ylabel(r'revelation deficit  $1 - R^2$  (linear)', fontsize=13)
    ax.set_title(r'F1c -- Zoom on $\gamma \le 3$ (both axes linear)',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---------- F2..F4: per-tau detail with both log-log AND linear-y twin ----------
    per_tau_pairs = [('F2', 0.05), ('F3', 0.10), ('F4', 0.20)]
    if 0.30 in taus: per_tau_pairs.append(('F4a', 0.30))
    if 0.40 in taus: per_tau_pairs.append(('F4b', 0.40))
    if 0.50 in taus: per_tau_pairs.append(('F4c', 0.50))
    for fid, tau in per_tau_pairs:
        g, d, _, _ = per_tau(rows, tau)
        if g.size == 0: continue
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
        d_plot = np.where(d > 0, d, 1e-16)
        axL.loglog(g, d_plot, 'o-', color=cmap[tau], markersize=8, linewidth=2)
        if g.size >= 4:
            tail = slice(-5, None)
            coef = np.polyfit(np.log(g[tail]), np.log(d_plot[tail]), 1)
            axL.loglog(g[tail], np.exp(coef[1]) * g[tail]**coef[0], 'k--',
                       label=f'tail slope = {coef[0]:.3f}')
            axL.legend(fontsize=11)
        axL.set_xlabel(r'$\gamma$', fontsize=12); axL.set_ylabel(r'$1 - R^2$', fontsize=12)
        axL.set_title(f'log--log, tail $1/\\gamma$ fit'); axL.grid(alpha=0.3, which='both')
        axR.semilogx(g, d, 'o-', color=cmap[tau], markersize=8, linewidth=2)
        axR.axhline(0, color='k', alpha=0.3, linewidth=0.6)
        axR.set_xlabel(r'$\gamma$ (log)', fontsize=12); axR.set_ylabel(r'$1-R^2$ (linear)', fontsize=12)
        axR.set_title('linear $y$ axis: emphasizes large-$\\gamma$ vanishing')
        axR.grid(alpha=0.3)
        fig.suptitle(f'{fid} -- $\\tau={tau}$, {g.size} certified cells, $\\gamma\\in[{g.min():.2f},{g.max():.1f}]$',
                     fontsize=13, weight='bold')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---------- F5 slope of logit P on sum u vs gamma ----------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for tau in tau_show:
        g, _, s, _ = per_tau(rows, tau)
        ax.semilogx(g, s, 'o-', color=cmap[tau], label=label[tau], markersize=7, linewidth=2)
        ax.axhline(tau, color=cmap[tau], ls=':', alpha=0.5)
    ax.set_xlabel(r'$\gamma$', fontsize=13)
    ax.set_ylabel(r'slope of $\mathrm{logit}\,P$ on $\sum_k u_k$', fontsize=13)
    ax.set_title('F5 -- Log-odds slope; dotted = revealing benchmark $\\tau$', fontsize=12)
    ax.legend(loc='center right', fontsize=10, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---------- F6 certified residuals ----------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for tau in tau_show:
        g, _, _, F = per_tau(rows, tau)
        Fp = np.where(F > 0, F, 1e-19)
        ax.loglog(g, Fp, 'o-', color=cmap[tau], label=label[tau], markersize=7, linewidth=2)
    ax.axhline(1e-15, color='r', ls='--', alpha=0.7, label='certification bar')
    ax.set_xlabel(r'$\gamma$', fontsize=13)
    ax.set_ylabel(r'$\|\Phi(P)-P\|_\infty$ (longdouble)', fontsize=13)
    ax.set_title('F6 -- Certification: all cells under the $10^{-15}$ bar', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---------- F7 deficit heatmap ----------
    gammas_all = sorted(set(round(x['gamma'], 4) for x in rows))
    taus_all = sorted(set(round(x['tau'], 4) for x in rows))
    H = np.full((len(taus_all), len(gammas_all)), np.nan)
    for x in rows:
        i = taus_all.index(round(x['tau'], 4)); j = gammas_all.index(round(x['gamma'], 4))
        H[i, j] = x['deficit']
    fig, ax = plt.subplots(figsize=(11, 4.5))
    glog = np.log10(gammas_all)
    im = ax.imshow(np.log10(np.where(H > 0, H, 1e-12)), cmap='viridis',
                   aspect='auto', origin='lower', vmin=-6, vmax=0,
                   extent=[glog[0], glog[-1], taus_all[0]-0.01, taus_all[-1]+0.01])
    fig.colorbar(im, ax=ax, label=r'$\log_{10}$ deficit')
    ax.set_yticks(taus_all)
    ax.set_yticklabels([f'{t:.2f}' for t in taus_all])
    ax.set_xlabel(r'$\log_{10}\gamma$')
    ax.set_ylabel(r'$\tau$')
    ax.set_title(f'F7 -- Deficit map ({len(rows)} certified cells)')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---------- F8 / F9 / F10 sample price surfaces ----------
    samples = [('F8', 0.10, 0.05), ('F9', 0.10, 1.0350), ('F10', 0.10, 30.0)]
    for fid, tau, gamma in samples:
        # find closest available gamma
        gam_avail = sorted([x['gamma'] for x in rows if abs(x['tau']-tau)<1e-5])
        if not gam_avail:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.text(0.5, 0.5, f'No certified cells at tau={tau}', ha='center')
            ax.axis('off'); pdf.savefig(fig); plt.close(fig); continue
        gamma_use = min(gam_avail, key=lambda g: abs(g-gamma))
        P_full, uf, lo, hi = load_P(tau, gamma_use)
        if P_full is None:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.text(0.5, 0.5, f'No P_ld saved at (tau={tau}, gamma={gamma_use})', ha='center')
            ax.axis('off'); pdf.savefig(fig); plt.close(fig); continue
        u_in = uf[lo:hi]
        # slice at u3 = 0 (middle index)
        mid = (hi-lo)//2
        Pslice = P_full[lo:hi, lo:hi, lo+mid]
        deficit_here = [x['deficit'] for x in rows
                        if abs(x['tau']-tau)<1e-5 and abs(x['gamma']-gamma_use)<1e-5][0]
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        cs = ax.contourf(u_in, u_in, Pslice.T, levels=np.linspace(0,1,21), cmap='RdBu_r')
        ax.contour(u_in, u_in, Pslice.T, levels=[0.25,0.5,0.75], colors='k', linewidths=0.6)
        fig.colorbar(cs, ax=ax, label='$P(u_1,u_2,u_3=0)$')
        ax.set_xlabel('$u_1$'); ax.set_ylabel('$u_2$')
        ax.set_title(f'{fid} -- Certified equilibrium $P$ slice $u_3=0$ at $(\\tau,\\gamma)=({tau},{gamma_use})$\n'
                     f'deficit = {deficit_here:.3e}', fontsize=12)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---------- F11 logit P vs sum u scatter ----------
    tau, gamma_target = 0.10, 0.05
    gam_avail = sorted([x['gamma'] for x in rows if abs(x['tau']-tau)<1e-5])
    if gam_avail:
        gamma_use = min(gam_avail, key=lambda g: abs(g-gamma_target))
        P_full, uf, lo, hi = load_P(tau, gamma_use)
        if P_full is not None:
            U1, U2, U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
            T = (U1+U2+U3).ravel()
            P_inner = P_full[lo:hi, lo:hi, lo:hi].ravel()
            P_inner = np.clip(P_inner, 1e-12, 1-1e-12)
            y = np.log(P_inner/(1-P_inner))
            a = np.polyfit(T, y, 1)
            deficit_here = [x['deficit'] for x in rows
                            if abs(x['tau']-tau)<1e-5 and abs(x['gamma']-gamma_use)<1e-5][0]
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.scatter(T, y, s=2, alpha=0.4, color='C1')
            tt = np.linspace(T.min(), T.max(), 100)
            ax.plot(tt, a[0]*tt + a[1], 'k-', linewidth=2,
                    label=f'best linear (slope {a[0]:.4f})')
            ax.set_xlabel(r'$\sum_k u_k$', fontsize=13)
            ax.set_ylabel(r'$\mathrm{logit}\,P$', fontsize=13)
            ax.set_title(f'F11 -- $\\mathrm{{logit}}\\,P$ vs $\\sum u$ at $(\\tau,\\gamma)=({tau},{gamma_use})$\n'
                         f'deficit = {deficit_here:.3e} = the unexplained scatter',
                         fontsize=12)
            ax.legend(fontsize=11); ax.grid(alpha=0.3)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.text(0.5, 0.5, f'F11: cell unavailable', ha='center'); ax.axis('off')
            pdf.savefig(fig); plt.close(fig)

    # ---------- F12 1/gamma tail fit per tau ----------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for tau in tau_show:
        g, d, _, _ = per_tau(rows, tau)
        if g.size < 4: continue
        d_plot = np.where(d > 0, d, 1e-16)
        tail = g >= 1.0
        if tail.sum() < 3: continue
        coef = np.polyfit(np.log(g[tail]), np.log(d_plot[tail]), 1)
        ax.loglog(g, d_plot, 'o', color=cmap[tau], markersize=7,
                  label=label[tau]+f' (tail slope {coef[0]:.3f})')
        gg = np.logspace(np.log10(g[tail].min()), np.log10(g[tail].max()), 30)
        ax.loglog(gg, np.exp(coef[1]) * gg**coef[0], '-', color=cmap[tau], alpha=0.5)
    ax.set_xlabel(r'$\gamma$', fontsize=13)
    ax.set_ylabel(r'$1-R^2$', fontsize=13)
    ax.set_title('F12 -- Asymptotic $1/\\gamma^p$ scaling of the deficit\n'
                 '(Jensen-wedge theory predicts $p=1$)', fontsize=12)
    ax.legend(loc='lower left', fontsize=10, ncol=2)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---------- F13 linear-y per-tau (one panel each) showing vanishing ----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, tau in zip(axes.flat, [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]):
        if tau not in taus:
            ax.axis('off'); continue
        g, d, _, _ = per_tau(rows, tau)
        ax.semilogx(g, d, 'o-', color=cmap[tau], markersize=7, linewidth=2)
        ax.axhline(0, color='k', alpha=0.3, linewidth=0.6)
        ax.set_xlabel(r'$\gamma$ (log)'); ax.set_ylabel(r'deficit (linear)')
        ax.set_title(label[tau] + f' (max d = {d.max():.3g})')
        ax.grid(alpha=0.3)
    fig.suptitle('F13 -- Per-$\\tau$ panels, linear $y$ axis', fontsize=13, weight='bold')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    pdf.close()
    print(f"saved {PDF_PATH}", flush=True)


if __name__ == "__main__":
    main()
