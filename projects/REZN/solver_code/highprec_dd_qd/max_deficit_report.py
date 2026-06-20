"""Max-deficit report: aggregate lowtau + emin15 + hightau into a single
PDF showing the certified deficit map over (tau, gamma).

Figures:
  F1 deficit vs gamma overlaying ALL tau values certified
  F2 deficit heatmap on (tau, gamma) grid (log10 deficit)
  F3 max-deficit-per-tau envelope (does it saturate as tau grows?)
  F4 high-tau certification residuals (success vs stall)
  F5 low-gamma zoom showing the deficit "ceiling" at gamma in [0.01, 0.1]
  F6 value-of-information decomposition at the high-tau / low-gamma cell
  F7 logit-P scatter at the maximum-deficit cell
  F8 price surface slice at the maximum-deficit cell
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3

LOWTAU = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau'
EMIN15 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'
PDF_PATH = f'{LOWTAU}/MAX_DEFICIT_report.pdf'


def load_all():
    """Combine lowtau, emin15, hightau into a single list of certified cells.
    Returns rows: list of dicts with tau, gamma, deficit, slope, F_ld, src."""
    rows = []
    seen = set()  # dedupe (tau, gamma) preferring lowtau then hightau then emin15
    sources = [
        (f"{LOWTAU}/hightau.json", 'hightau'),
        (f"{LOWTAU}/lowtau.json", 'lowtau'),
        (f"{EMIN15}/emin15.json", 'emin15'),
    ]
    for path, name in sources:
        if not os.path.exists(path): continue
        d = json.load(open(path))
        for v in d.values():
            if v.get('verdict') != 'ACCEPT': continue
            tau = float(v['tau']); gamma = float(v['gamma'])
            key = (round(tau, 4), round(gamma, 4))
            if key in seen: continue
            seen.add(key)
            rows.append(dict(tau=tau, gamma=gamma,
                             deficit=float(v['deficit']),
                             slope=float(v['slope']),
                             F_ld=float(v['F_ld']) if v.get('F_ld') is not None else 0.0,
                             src=name))
    return rows


def load_stalls():
    """All cells that did NOT accept, for the stall map."""
    rows = []
    for path, name in [(f"{LOWTAU}/hightau.json", 'hightau'),
                       (f"{LOWTAU}/lowtau.json", 'lowtau'),
                       (f"{EMIN15}/emin15.json", 'emin15')]:
        if not os.path.exists(path): continue
        d = json.load(open(path))
        for v in d.values():
            if v.get('verdict') == 'ACCEPT': continue
            tau = float(v['tau']); gamma = float(v['gamma'])
            F64 = v.get('F64')
            try: F64 = float(F64) if F64 is not None else None
            except Exception: F64 = None
            rows.append(dict(tau=tau, gamma=gamma, F64=F64,
                             verdict=v.get('verdict'), src=name))
    return rows


def per_tau(rows, tau):
    r = sorted([x for x in rows if abs(x['tau']-tau) < 1e-5], key=lambda x: x['gamma'])
    return (np.array([x['gamma'] for x in r]),
            np.array([x['deficit'] for x in r]),
            np.array([x['slope'] for x in r]),
            np.array([x['F_ld'] for x in r]))


def load_P(tau, gamma):
    Gi = 21; pad = 2; UMAX = 4.0
    du = 2*UMAX/(Gi-1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    for d in (LOWTAU, EMIN15):
        src = f"{d}/P_ld_t{tau}_g{gamma}.npy"
        if os.path.exists(src):
            P_inner = np.load(src)
            P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma), np.full(3, 1.0))
            P_full[lo:hi, lo:hi, lo:hi] = P_inner
            return P_full, uf, lo, hi
    return None, None, None, None


def color_for_tau(tau, all_taus):
    """Continuous colormap over tau range."""
    if len(all_taus) <= 1: return 'C0'
    cmap = plt.cm.viridis
    i = all_taus.index(tau)
    return cmap(i / (len(all_taus)-1))


def main():
    rows = load_all()
    stalls = load_stalls()
    print(f"Loaded {len(rows)} certified cells, {len(stalls)} stalled records", flush=True)
    all_taus = sorted(set(round(x['tau'], 4) for x in rows))
    print(f"Certified taus: {all_taus}", flush=True)
    all_gammas = sorted(set(round(x['gamma'], 4) for x in rows))

    metrics = {}
    if os.path.exists(f"{LOWTAU}/metrics.json"):
        m = json.load(open(f"{LOWTAU}/metrics.json"))
        for k, v in m.items():
            metrics[(round(v['tau'], 4), round(v['gamma'], 4))] = v

    # find the maximum-deficit cell
    max_row = max(rows, key=lambda x: x['deficit'])
    print(f"MAX deficit = {max_row['deficit']:.4f} at "
          f"(tau={max_row['tau']}, gamma={max_row['gamma']}), src={max_row['src']}",
          flush=True)

    pdf = PdfPages(PDF_PATH)

    # ---- F1: deficit vs gamma overlay all taus ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for tau in all_taus:
        g, d, _, _ = per_tau(rows, tau)
        if g.size == 0: continue
        c = color_for_tau(tau, all_taus)
        ax.loglog(g, np.where(d > 0, d, 1e-16), 'o-', color=c,
                  label=rf'$\tau={tau}$', markersize=5, linewidth=1.5)
    gg = np.logspace(-2.1, 1.6, 50)
    ax.loglog(gg, 0.02/gg, 'k--', alpha=0.4, label=r'$\propto 1/\gamma$ ref')
    ax.set_xlabel(r'$\gamma$ (risk aversion)', fontsize=13)
    ax.set_ylabel(r'revelation deficit  $1 - R^2$', fontsize=13)
    ax.set_title(f'F1 -- Certified deficit vs $\\gamma$  ({len(rows)} cells, '
                 f'{len(all_taus)} $\\tau$ values)', fontsize=12)
    ax.legend(loc='lower left', fontsize=9, ncol=3)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- F1b: same, linear y ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for tau in all_taus:
        g, d, _, _ = per_tau(rows, tau)
        if g.size == 0: continue
        c = color_for_tau(tau, all_taus)
        ax.semilogx(g, d, 'o-', color=c, label=rf'$\tau={tau}$',
                    markersize=5, linewidth=1.5)
    ax.set_xlabel(r'$\gamma$ (log)', fontsize=13)
    ax.set_ylabel(r'revelation deficit  $1 - R^2$  (linear)', fontsize=13)
    ax.set_title('F1b -- Deficit vs $\\gamma$ (linear $y$): the saturating ceiling',
                 fontsize=12)
    ax.axhline(max_row['deficit'], color='r', ls=':', alpha=0.5,
               label=f"max d = {max_row['deficit']:.3f}")
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- F2: deficit heatmap ----
    H = np.full((len(all_taus), len(all_gammas)), np.nan)
    for x in rows:
        i = all_taus.index(round(x['tau'], 4))
        j = all_gammas.index(round(x['gamma'], 4))
        H[i, j] = x['deficit']
    fig, ax = plt.subplots(figsize=(13, 6))
    glog = np.log10(all_gammas)
    # use pcolormesh so cells line up cleanly when grids are irregular
    g_edges = np.concatenate([[2*glog[0]-glog[1]],
                              0.5*(glog[1:]+glog[:-1]),
                              [2*glog[-1]-glog[-2]]])
    t_edges = np.concatenate([[2*all_taus[0]-all_taus[1]],
                              0.5*(np.array(all_taus[1:])+np.array(all_taus[:-1])),
                              [2*all_taus[-1]-all_taus[-2]]])
    pm = ax.pcolormesh(g_edges, t_edges, np.log10(np.where(H > 0, H, 1e-12)),
                       cmap='viridis', vmin=-6, vmax=0, shading='auto')
    fig.colorbar(pm, ax=ax, label=r'$\log_{10}$ deficit')
    # Mark the max
    ax.plot(np.log10(max_row['gamma']), max_row['tau'], '*', color='red',
            markersize=20, markeredgecolor='white',
            label=f"max @ ($\\tau$={max_row['tau']}, $\\gamma$={max_row['gamma']:.3g})")
    ax.set_xlabel(r'$\log_{10}\gamma$', fontsize=13)
    ax.set_ylabel(r'$\tau$', fontsize=13)
    ax.set_yticks(all_taus)
    ax.set_yticklabels([f'{t:.2f}' for t in all_taus])
    ax.set_title(f'F2 -- Certified deficit map: {len(rows)} cells', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- F3: max deficit per tau (the envelope) ----
    fig, ax = plt.subplots(figsize=(9, 6))
    max_per_tau = []
    min_per_tau = []
    for tau in all_taus:
        g, d, _, _ = per_tau(rows, tau)
        if d.size == 0: continue
        max_per_tau.append((tau, d.max(), g[d.argmax()]))
        min_per_tau.append((tau, d.min(), g[d.argmin()]))
    tau_vals = [x[0] for x in max_per_tau]
    max_d = [x[1] for x in max_per_tau]
    arg_g = [x[2] for x in max_per_tau]
    ax.plot(tau_vals, max_d, 'o-', color='C3', linewidth=2.5, markersize=9,
            label='max deficit per $\\tau$')
    for tv, md, ag in zip(tau_vals, max_d, arg_g):
        ax.annotate(f'$\\gamma$={ag:.3g}', xy=(tv, md), xytext=(5, 4),
                    textcoords='offset points', fontsize=8)
    ax.axhline(1.0, color='k', ls=':', alpha=0.5, label='theoretical upper bound 1')
    ax.set_xlabel(r'$\tau$', fontsize=13)
    ax.set_ylabel(r'max revelation deficit $1-R^2$', fontsize=13)
    ax.set_title('F3 -- Maximum certified deficit per $\\tau$ (envelope)\n'
                 'each marker labelled with the $\\gamma$ at which the max occurs',
                 fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- F4: certification residuals + stalls ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for tau in all_taus:
        g, _, _, F = per_tau(rows, tau)
        c = color_for_tau(tau, all_taus)
        ax.loglog(g, np.where(F > 0, F, 1e-19), 'o', color=c,
                  label=rf'$\tau={tau}$', markersize=5)
    # stall cells
    sg = [s['gamma'] for s in stalls]
    st = [s['tau'] for s in stalls]
    if sg:
        ax.scatter(sg, np.full(len(sg), 1e-7), marker='x', color='red',
                   s=40, label=f'stalled cells ({len(sg)})', zorder=5)
    ax.axhline(1e-15, color='r', ls='--', alpha=0.7, label='certification bar $10^{-15}$')
    ax.set_xlabel(r'$\gamma$', fontsize=13)
    ax.set_ylabel(r'$\|\Phi(P)-P\|_\infty$  (longdouble)', fontsize=13)
    ax.set_title('F4 -- Certification residuals: ACCEPT below bar; X = stalled',
                 fontsize=12)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- F5: low-gamma zoom showing the deficit ceiling ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for tau in all_taus:
        g, d, _, _ = per_tau(rows, tau)
        if g.size == 0: continue
        m = g <= 0.15
        if m.sum() == 0: continue
        c = color_for_tau(tau, all_taus)
        ax.plot(g[m], d[m], 'o-', color=c, label=rf'$\tau={tau}$',
                markersize=7, linewidth=1.8)
    ax.set_xlabel(r'$\gamma$ (linear)', fontsize=13)
    ax.set_ylabel(r'revelation deficit $1-R^2$', fontsize=13)
    ax.set_title('F5 -- Low-$\\gamma$ corner $\\gamma\\in[0.01,0.15]$: the deficit ceiling\n'
                 'deficit asymptotes to a finite limit < 1 even as $\\gamma\\to 0$',
                 fontsize=12)
    ax.axhline(1.0, color='k', ls=':', alpha=0.4, label='upper bound 1')
    ax.legend(loc='center right', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- F6: VoI decomposition at MAX-deficit tau row (if metrics present) ----
    fig, ax = plt.subplots(figsize=(9, 6))
    if metrics:
        # pick a high-tau slice (the largest tau for which we have metrics)
        avail_taus = sorted(set(t for (t, g) in metrics))
        tau_pick = avail_taus[-1] if avail_taus else None
        if tau_pick is not None:
            cells = sorted([(g, metrics[(tau_pick, g)]) for (t, g) in metrics
                            if abs(t-tau_pick) < 1e-5])
            g_arr = np.array([c[0] for c in cells])
            vpub = np.array([max(c[1]['Vi_public'], 0) for c in cells])
            vpriv = np.array([max(c[1]['Vi_private'], 0) for c in cells])
            vfrgap = np.array([max(c[1]['Vi_FR_gap'], 0) for c in cells])
            ax.semilogx(g_arr, vpub+vpriv+vfrgap, '-', color='gray', alpha=0.4,
                        label='total $CE_{FR}-CE_{prior}$')
            ax.fill_between(g_arr, 0, vpub, color='C2', alpha=0.6,
                            label=r'$V^{\rm public}$ (price info)')
            ax.fill_between(g_arr, vpub, vpub+vpriv, color='C0', alpha=0.6,
                            label=r'$V^{\rm private}$ (own signal)')
            ax.fill_between(g_arr, vpub+vpriv, vpub+vpriv+vfrgap, color='C3', alpha=0.5,
                            label=r'$V^{\rm FRgap}$ (welfare loss vs FR)')
            ax.set_xlabel(r'$\gamma$ (log)', fontsize=12)
            ax.set_ylabel('CE units', fontsize=12)
            ax.set_title(f'F6 -- Value-of-information decomposition at $\\tau$={tau_pick}\n'
                         'shows where in $\\gamma$ the welfare gap is largest',
                         fontsize=12)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No metrics available', ha='center'); ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'metrics.json absent -- run lowtau_metrics.py',
                ha='center'); ax.axis('off')
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # ---- F7: logit-P scatter at max-deficit cell ----
    tau_m, gamma_m = max_row['tau'], max_row['gamma']
    P_full, uf, lo, hi = load_P(tau_m, gamma_m)
    if P_full is not None:
        U1,U2,U3 = np.meshgrid(uf[lo:hi], uf[lo:hi], uf[lo:hi], indexing='ij')
        T = (U1+U2+U3).ravel()
        P_inner = P_full[lo:hi, lo:hi, lo:hi].ravel()
        Pc = np.clip(P_inner, 1e-12, 1-1e-12)
        y = np.log(Pc/(1-Pc))
        a = np.polyfit(T, y, 1)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(T, y, s=2, alpha=0.3, color='C3')
        tt = np.linspace(T.min(), T.max(), 100)
        ax.plot(tt, a[0]*tt + a[1], 'k-', linewidth=2,
                label=f'best linear (slope {a[0]:.4f})')
        ax.set_xlabel(r'$\sum_k u_k$', fontsize=13)
        ax.set_ylabel(r'$\mathrm{logit}\,P$', fontsize=13)
        ax.set_title(f'F7 -- logit$P$ vs $\\sum u$ at the MAX-deficit cell '
                     f'($\\tau$={tau_m}, $\\gamma$={gamma_m})\n'
                     f'deficit = {max_row["deficit"]:.4f} = the unexplained scatter',
                     fontsize=12)
        ax.legend(fontsize=11); ax.grid(alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ---- F8: price surface slice at max cell ----
        u_in = uf[lo:hi]
        mid = (hi-lo)//2
        Pslice = P_full[lo:hi, lo:hi, lo+mid]
        fig, ax = plt.subplots(figsize=(9, 7))
        cs = ax.contourf(u_in, u_in, Pslice.T, levels=np.linspace(0,1,21), cmap='RdBu_r')
        ax.contour(u_in, u_in, Pslice.T, levels=[0.25,0.5,0.75], colors='k', linewidths=0.6)
        fig.colorbar(cs, ax=ax, label='$P(u_1, u_2, u_3=0)$')
        ax.set_xlabel('$u_1$'); ax.set_ylabel('$u_2$')
        ax.set_title(f'F8 -- Equilibrium price slice $u_3=0$ at MAX-deficit cell '
                     f'$(\\tau,\\gamma)=({tau_m},{gamma_m})$\n'
                     f'price hugs 0.5 with weak signal aggregation', fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    else:
        for fig_label in ['F7', 'F8']:
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.text(0.5, 0.5, f'{fig_label}: max-cell P file unavailable', ha='center')
            ax.axis('off'); pdf.savefig(fig); plt.close(fig)

    # ---- F9: per-fixed-CPU frontier ----
    # For each tau, the cell where deficit per second of wall time is largest is the
    # economically relevant frontier. Approximation: assume wall ~ const, so the
    # frontier is just the max-deficit cell per tau (which is F3). Add it explicitly:
    fig, ax = plt.subplots(figsize=(9, 6))
    for tv, md, ag in zip(tau_vals, max_d, arg_g):
        ax.semilogx(ag, md, 'o', color=color_for_tau(tv, all_taus),
                    markersize=10)
        ax.annotate(f'$\\tau$={tv}', xy=(ag, md), xytext=(7, 4),
                    textcoords='offset points', fontsize=9)
    ax.set_xlabel(r'$\gamma$ (log)', fontsize=13)
    ax.set_ylabel('max deficit attained', fontsize=13)
    ax.set_title('F9 -- Locus of max-deficit cells in $(\\gamma, \\tau)$ space\n'
                 'the "economically relevant" worst-case-per-tau frontier', fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    pdf.close()
    print(f"saved {PDF_PATH}", flush=True)


if __name__ == "__main__":
    main()
