"""Figures for the stall diagnosis: eigenvalue track + step scaling +
fold curve."""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/stall_diagnosis'


def main():
    ph1 = json.load(open(f"{OUT}/phase1_tau2.0.json"))
    pts = [p for p in ph1['points'] if p.get('ok')]
    g = np.array([p['gamma'] for p in pts])
    rho = np.array([p['rho'] for p in pts])
    sig = np.array([p['sigma_min_ImJ'] for p in pts])
    d1 = np.array([p['eig_closest_1']['dist'] for p in pts])
    symB = np.array([p['lead_eigs'][0]['sym_break'] for p in pts])
    gfail = [p['gamma'] for p in ph1['points'] if not p.get('ok')]

    try:
        ph2 = json.load(open(f"{OUT}/phase2_stepscaling.json"))
    except FileNotFoundError:
        ph2 = None
    try:
        ph3 = json.load(open(f"{OUT}/phase3_foldcurve.json"))
    except FileNotFoundError:
        ph3 = None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(g, rho, 'o-', color='tab:red', label=r'$\rho(J)$ (leading $|\lambda|$)')
    ax.plot(g, sig, 's-', color='tab:blue', label=r'$\sigma_{\min}(I-J)$')
    ax.plot(g, d1, '^-', color='tab:green', label=r'$|\lambda_{c}-1|$ (closest to 1)')
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    if gfail:
        ax.axvline(gfail[0], color='gray', ls=':', label=f'first NK stall g={gfail[0]}')
    if ph2:
        gs = ph2.get('deepest_gamma')
        if gs:
            ax.axvline(gs, color='purple', ls='-.', lw=1,
                       label=f'deepest reachable g={gs:.4f}')
    ax.set_xlabel(r'$\gamma$'); ax.set_yscale('log')
    ax.set_title(r'Spectrum of $J=D\Phi$ along $\tau=2.0$, $G=13$')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(g, rho, 'o-', color='tab:red')
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_ylabel(r'$\rho(J)$', color='tab:red')
    ax2.plot(g, symB, 'd--', color='tab:orange')
    ax2.set_ylabel('sym-breaking fraction of leading eigvec', color='tab:orange')
    ax2.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$\gamma$')
    ax.set_title('Leading eigenvalue and its symmetry character')
    ax.grid(alpha=0.3)

    ax = axes[2]
    if ph2:
        steps = [r['step'] for r in ph2['runs']]
        gmax = [r['gamma_max'] for r in ph2['runs']]
        ax.semilogx(steps, gmax, 'o-')
        ax.set_xlabel(r'$\gamma$ continuation step')
        ax.set_ylabel(r'$\gamma_{\max}$ reachable')
        ttl = 'Step-size scaling (saturation = fold)'
        if ph3:
            fp = {p['tau']: p.get('gamma_star') for p in ph3['fold_points']}
            ttl += '\nfold: ' + ', '.join(
                f"$\\gamma^*({t})={v:.4g}$" for t, v in fp.items() if v)
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(f"{OUT}/figs/eigtrack_stepscaling.png", dpi=130)
    print(f"saved {OUT}/figs/eigtrack_stepscaling.png")


if __name__ == '__main__':
    main()
