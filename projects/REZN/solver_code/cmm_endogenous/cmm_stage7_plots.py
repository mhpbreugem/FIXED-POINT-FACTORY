"""Stage 7 deliverable plots: tilt heatmap (T1) + tau-ladder deficit plot (T2/T3)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/cmm_endogenous'
SQRT2 = float(np.sqrt(2.0))


def heatmap():
    scr = json.load(open(f"{OUT}/stage7_tilt_screen.json"))['cells']
    taus = sorted(set(v['tau'] for v in scr.values()))
    gams = sorted(set(v['gamma'] for v in scr.values()))
    Z = np.full((len(taus), len(gams)), np.nan)
    C = np.empty((len(taus), len(gams)), dtype=object)
    for v in scr.values():
        i = taus.index(v['tau']); j = gams.index(v['gamma'])
        Z[i, j] = v['max_tilt']; C[i, j] = v['cls']
    fig, ax = plt.subplots(figsize=(13, 4.2))
    norm = TwoSlopeNorm(vmin=0.5, vcenter=SQRT2, vmax=np.nanmax(Z))
    im = ax.imshow(Z, aspect='auto', origin='lower', cmap='RdYlGn_r',
                   norm=norm)
    for i in range(len(taus)):
        for j in range(len(gams)):
            if np.isnan(Z[i, j]):
                ax.text(j, i, 'rej', ha='center', va='center', fontsize=7,
                        color='gray')
                continue
            cls = C[i, j]
            mark = {'WELL-POSED': 'W', 'MARGINAL': 'M', 'ILL-POSED': ''}[cls]
            ax.text(j, i, f"{Z[i,j]:.2f}\n{mark}", ha='center', va='center',
                    fontsize=7,
                    fontweight='bold' if cls != 'ILL-POSED' else 'normal')
    ax.set_xticks(range(len(gams)))
    ax.set_xticklabels([f"{g:g}" for g in gams], rotation=45, fontsize=8)
    ax.set_yticks(range(len(taus)))
    ax.set_yticklabels([f"{t:g}" for t in taus])
    ax.set_xlabel('gamma'); ax.set_ylabel('tau')
    ax.set_title('Stage 7 T1: max transverse tilt |grad H| of kernel-FP level surfaces\n'
                 '(strict h=0 graph premise needs tilt < sqrt(2)=1.414; '
                 'W = well-posed (<1), M = marginal)')
    cb = fig.colorbar(im, ax=ax); cb.set_label('max |grad H|')
    cb.ax.axhline(SQRT2, color='k', lw=1.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage7_tilt_heatmap.png", dpi=150)
    print(f"-> {OUT}/stage7_tilt_heatmap.png")


def ladder_plot():
    lad = json.load(open(f"{OUT}/stage7_tau_ladder.json"))
    em = json.load(open('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
                        'solved_fixed_points/dd_k3_overnight/emin15/emin15.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = {0.098: 'tab:red', 1.035: 'tab:blue'}
    taustar = {}
    for key in [k for k in lad if k.startswith('g')]:
        gamma = float(key[1:])
        col = lad[key]
        recs = sorted((v for v in col.values() if v.get('done')),
                      key=lambda r: r['tau'])
        t = [r['tau'] for r in recs]
        d = [r['deficit_strict_h0'] for r in recs]
        mx = [r['final']['max_r'] for r in recs]
        md = [r['final']['med_r'] for r in recs]
        conv = [r['final']['max_r'] < 1e-6 for r in recs]
        c = colors.get(gamma, 'k')
        # kernel deficits
        tk = sorted(v['tau'] for v in em.values()
                    if v['gamma'] == gamma and v['verdict'] == 'ACCEPT')
        dk = [em[f"t{x}_g{gamma}"]['deficit'] for x in tk]
        ax = axes[0]
        ax.plot(tk, dk, 'o--', color=c, alpha=0.45, ms=4,
                label=f'kernel G=21, gamma={gamma}')
        okt = [x for x, cv in zip(t, conv) if cv]
        okd = [y for y, cv in zip(d, conv) if cv]
        bat = [x for x, cv in zip(t, conv) if not cv]
        bad = [y for y, cv in zip(d, conv) if not cv]
        ax.plot(t, d, '-', color=c, lw=1, alpha=0.6)
        ax.plot(okt, okd, 's', color=c, ms=8,
                label=f'strict h=0 (converged), gamma={gamma}')
        ax.plot(bat, bad, 'x', color=c, ms=8, mew=2,
                label=f'strict h=0 (PLATEAU, not eq.), gamma={gamma}')
        ts = next((x for x, cv in zip(t, conv) if not cv), None)
        # tau*: first tau (walking up) where convergence fails
        if ts is not None:
            taustar[gamma] = ts
            ax.axvline(ts, color=c, ls=':', lw=1.5)
            ax.text(ts, ax.get_ylim()[1]*0.05, f' tau*<= {ts:g}', color=c,
                    fontsize=8, rotation=90, va='bottom')
        axes[1].semilogy(t, mx, 'o-', color=c, label=f'max|r|, gamma={gamma}')
        axes[1].semilogy(t, md, 's--', color=c, alpha=0.5,
                         label=f'med|r|, gamma={gamma}')
    axes[0].errorbar([2.0], [0.268], yerr=[0.010], fmt='*', color='k', ms=14,
                     capsize=4, label='d_inf anchor (t2,g0.098)')
    axes[0].set_xlabel('tau'); axes[0].set_ylabel('revelation deficit (unweighted 1-R^2)')
    axes[0].set_title('strict h=0 deficit vs tau (squares = converged equilibria;\n'
                      'x = LM plateau, no strict-h=0 equilibrium reached)')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].axhline(1e-6, color='gray', ls='--', lw=1)
    axes[1].text(0.1, 1.4e-6, 'convergence tol', fontsize=7, color='gray')
    axes[1].set_xlabel('tau'); axes[1].set_ylabel('final residual')
    axes[1].set_title('LM final residuals per tau')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage7_ladder_plot.png", dpi=150)
    print(f"-> {OUT}/stage7_ladder_plot.png  tau* = {taustar}")


if __name__ == '__main__':
    if 'heatmap' in sys.argv or len(sys.argv) == 1:
        heatmap()
    if 'ladder' in sys.argv or len(sys.argv) == 1:
        ladder_plot()
