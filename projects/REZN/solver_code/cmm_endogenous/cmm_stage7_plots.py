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
    rev = json.load(open(f"{OUT}/stage7_revealing_check.json"))
    em = json.load(open('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
                        'solved_fixed_points/dd_k3_overnight/emin15/emin15.json'))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {0.098: 'tab:red', 1.035: 'tab:blue'}
    for key in [k for k in lad if k.startswith('g')]:
        gamma = float(key[1:])
        col = lad[key]
        recs = sorted((v for v in col.values()
                       if isinstance(v, dict) and v.get('done')),
                      key=lambda r: r['tau'])
        t = [r['tau'] for r in recs]
        mx = [r['final']['max_r'] for r in recs]
        md = [r['final']['med_r'] for r in recs]
        rv = [rev[f"{key}_t{r['tau']}"]['revealing_max_r'] for r in recs]
        c = colors.get(gamma, 'k')
        # kernel deficits
        tk = sorted(v['tau'] for v in em.values()
                    if v['gamma'] == gamma and v['verdict'] == 'ACCEPT')
        dk = [em[f"t{x}_g{gamma}"]['deficit'] for x in tk]
        ax = axes[0]
        ax.plot(tk, dk, 'o--', color=c, alpha=0.6, ms=5,
                label=f'kernel G=21 FP deficit, gamma={gamma}')
        # strict h=0 equilibrium deficit: exactly 0 (revealing solution,
        # certified max|r| ~ 3e-15 at each cell)
        ax.plot(t, [0.0]*len(t), '-', color=c, lw=2.5, alpha=0.35)
        ax.plot(t, [0.0]*len(t), 's', color=c, ms=7, mfc='white',
                label=f'strict h=0 equilibrium (revealing, exact), gamma={gamma}')
        axes[1].semilogy(t, mx, 'o-', color=c,
                         label=f'LM-from-kernel plateau max|r|, g={gamma}')
        axes[1].semilogy(t, md, 's--', color=c, alpha=0.5,
                         label=f'... med|r|, g={gamma}')
        axes[1].semilogy(t, rv, '^:', color=c, alpha=0.8, ms=5,
                         label=f'revealing-solution max|r|, g={gamma}')
    # tau* annotation (gamma=1.035): kernel branch well-posed at 0.2,
    # ill-posed at 0.5 (screen + LM jam)
    axes[1].axvspan(0.2, 0.5, color='tab:blue', alpha=0.08)
    axes[1].text(0.3, 3e-9, 'tau*(1.035)\nin (0.2, 0.5]', fontsize=8,
                 color='tab:blue', ha='center')
    axes[1].text(0.07, 3e-12, 'gamma=0.098: ill-posed at ALL tau\n'
                 '(no tau*; plateaus everywhere)', fontsize=8,
                 color='tab:red')
    axes[0].errorbar([2.0], [0.268], yerr=[0.010], fmt='*', color='k',
                     ms=14, capsize=4,
                     label='d_inf (deep-ladder h->0 anchor, t2 g0.098)')
    axes[0].set_xlabel('tau')
    axes[0].set_ylabel('revelation deficit (unweighted 1-R^2)')
    axes[0].set_title('Deficit vs tau: the strict h=0 EQUILIBRIUM is fully revealing\n'
                      '(deficit identically 0); kernel-FP deficits do NOT converge to it')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].axhline(1e-6, color='gray', ls='--', lw=1)
    axes[1].text(1.5, 1.6e-6, 'convergence tol', fontsize=7, color='gray')
    axes[1].set_xlabel('tau'); axes[1].set_ylabel('max residual')
    axes[1].set_title('Strict-h=0 residuals: LM from kernel warm start plateaus\n'
                      'at O(0.1) (no nontrivial equilibrium); revealing solution ~1e-15')
    axes[1].legend(fontsize=7, loc='center right'); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stage7_ladder_plot.png", dpi=150)
    print(f"-> {OUT}/stage7_ladder_plot.png")


if __name__ == '__main__':
    if 'heatmap' in sys.argv or len(sys.argv) == 1:
        heatmap()
    if 'ladder' in sys.argv or len(sys.argv) == 1:
        ladder_plot()
