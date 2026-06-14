"""Publishable-quality PDF report for the (tau, gamma) phase sweep.

Produces a multi-page PDF combining a single high-resolution headline
figure with interpolated boundary contours, plus tables and notes.
"""
import os, json, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.interpolate import RegularGridInterpolator

ROOT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_monotone_phase_pub'
JSON_PATH = f'{ROOT}/phase.json'
PDF_PATH  = f'{ROOT}/phase_pub_report.pdf'

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})


def text_page(pdf, lines, title=None, fontsize=10):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    y = 0.96
    if title is not None:
        ax.text(0.5, y, title, ha='center', va='top',
                 fontsize=16, weight='bold')
        y -= 0.05
    for ln in lines:
        ax.text(0.06, y, ln, ha='left', va='top',
                 fontsize=fontsize, family='monospace')
        y -= 0.022
    pdf.savefig(fig); plt.close(fig)


def main():
    data = json.load(open(JSON_PATH))
    rows = data['rows']
    gammas = data['gammas']; taus = data['taus']
    Gi = data['G']; M = data['M']
    code = {'monotone_converged': 0, 'fixed_but_nonmonotone': 1,
            'cycle': 2, 'not_converged': 3, 'solver_failed': 4}
    status_mat = np.full((len(gammas), len(taus)), -1, dtype=int)
    deficit_mat = np.full((len(gammas), len(taus)), np.nan)
    viol_mat = np.full((len(gammas), len(taus)), np.nan)
    wall_mat = np.full((len(gammas), len(taus)), np.nan)
    for r in rows:
        ig = gammas.index(r['gamma']); it = taus.index(r['tau'])
        status_mat[ig, it] = code.get(r['status'], -1)
        if 'deficit_tail' in r:
            deficit_mat[ig, it] = r['deficit_tail']
            viol_mat[ig, it]    = r['max_viol_raw_tail']
        wall_mat[ig, it] = r.get('wall', np.nan)

    log_gammas = np.log10(gammas); log_taus = np.log10(taus)

    with PdfPages(PDF_PATH) as pdf:
        # Title page
        text_page(pdf, [
            f"Compiled: {datetime.date.today().isoformat()}",
            "",
            "Model & solver",
            "--------------",
            "  * REZN economy with K=3 CRRA-like informed traders,",
            "    Gaussian signals, no noise traders.",
            "  * Endogenous-price (CMM) discrete-bin solver:",
            f"      coverage grid G = {Gi}",
            f"      price bins   M = {M}",
            "  * Iterate: soft K-means assignment of agents to price bins,",
            "    isotonic (PAV) projection of the price-on-state map onto the",
            "    monotone cone, max 40 outer iterations.",
            "",
            "Sweep grid",
            "----------",
            f"  gamma in {gammas}    ({len(gammas)} values, log-spaced)",
            f"  tau   in {taus}    ({len(taus)} values, log-spaced)",
            f"  total {len(gammas)*len(taus)} cells",
            "",
            "Status definitions",
            "------------------",
            "  monotone_converged    : RAW violation below Delta p / 2 ",
            "                          (a strict discrete-price equilibrium).",
            "  fixed_but_nonmonotone : fixed point of the projected map, but the",
            "                          un-projected price-on-state field has",
            "                          true non-monotone steps above the grid floor.",
            "  cycle / not_converged : iterate did not stabilise.",
            "  solver_failed         : numerical exception during the run.",
            "",
            "Headline",
            "--------",
            f"   * {(status_mat==0).sum()} of {status_mat.size} cells are monotone-converged.",
            f"   * {(status_mat==1).sum()} cells are fixed but non-monotone.",
            f"   * {(status_mat==2).sum()} cells in cycle, {(status_mat==3).sum()} no-conv, {(status_mat==4).sum()} failed.",
            "",
            "The 'good zone' grows monotonically with gamma and shrinks",
            "monotonically with tau, with the boundary roughly tracing a",
            "tau ~ gamma^(1/2)-shaped curve on the log-log plane.",
        ], title='Monotone discrete-price equilibria in (tau, gamma):\n'
                 'publishable-quality phase diagram')

        # Single big publication-quality 1x3 figure
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
        cmap_status = ListedColormap(['#27ae60', '#e67e22', '#c0392b',
                                       '#8e44ad', '#7f8c8d'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap_status.N)

        # Panel A: status with log-log axes
        im0 = axes[0].pcolormesh(
            np.r_[log_taus - (log_taus[1]-log_taus[0])/2,
                   log_taus[-1] + (log_taus[-1]-log_taus[-2])/2],
            np.r_[log_gammas - (log_gammas[1]-log_gammas[0])/2,
                   log_gammas[-1] + (log_gammas[-1]-log_gammas[-2])/2],
            status_mat, cmap=cmap_status, norm=norm, shading='auto')
        axes[0].set_xticks(log_taus)
        axes[0].set_xticklabels([f'{t:g}' for t in taus], rotation=45)
        axes[0].set_yticks(log_gammas)
        axes[0].set_yticklabels([f'{g:g}' for g in gammas])
        axes[0].set_xlabel(r'signal noise $\tau$ (log scale)')
        axes[0].set_ylabel(r'risk aversion $\gamma$ (log scale)')
        axes[0].set_title('(a) equilibrium status')
        cbar = fig.colorbar(im0, ax=axes[0], ticks=[0,1,2,3,4], fraction=0.05)
        cbar.ax.set_yticklabels(['mono', 'non-mono', 'cycle', 'no-conv', 'fail'],
                                  fontsize=8)
        # boundary contour from the binary status mask via interpolation
        try:
            grid_t = np.linspace(log_taus[0], log_taus[-1], 200)
            grid_g = np.linspace(log_gammas[0], log_gammas[-1], 200)
            mono_mask = (status_mat == 0).astype(float)
            interp = RegularGridInterpolator(
                (log_gammas, log_taus), mono_mask,
                bounds_error=False, fill_value=0.0)
            GG, TT = np.meshgrid(grid_g, grid_t, indexing='ij')
            ZZ = interp((GG, TT))
            axes[0].contour(TT, GG, ZZ, levels=[0.5],
                              colors='white', linewidths=2.0, linestyles='--')
        except Exception:
            pass

        # Panel B: deficit heatmap
        im1 = axes[1].pcolormesh(
            np.r_[log_taus - (log_taus[1]-log_taus[0])/2,
                   log_taus[-1] + (log_taus[-1]-log_taus[-2])/2],
            np.r_[log_gammas - (log_gammas[1]-log_gammas[0])/2,
                   log_gammas[-1] + (log_gammas[-1]-log_gammas[-2])/2],
            deficit_mat, cmap='viridis', shading='auto')
        axes[1].set_xticks(log_taus)
        axes[1].set_xticklabels([f'{t:g}' for t in taus], rotation=45)
        axes[1].set_yticks(log_gammas)
        axes[1].set_yticklabels([f'{g:g}' for g in gammas])
        axes[1].set_xlabel(r'$\tau$ (log scale)')
        axes[1].set_ylabel(r'$\gamma$ (log scale)')
        axes[1].set_title(r'(b) revelation deficit $1-R^2$')
        fig.colorbar(im1, ax=axes[1], fraction=0.05)

        # Panel C: log10 violation
        with np.errstate(divide='ignore'):
            log_viol = np.log10(np.maximum(viol_mat, 1e-16))
        im2 = axes[2].pcolormesh(
            np.r_[log_taus - (log_taus[1]-log_taus[0])/2,
                   log_taus[-1] + (log_taus[-1]-log_taus[-2])/2],
            np.r_[log_gammas - (log_gammas[1]-log_gammas[0])/2,
                   log_gammas[-1] + (log_gammas[-1]-log_gammas[-2])/2],
            log_viol, cmap='magma', shading='auto')
        axes[2].set_xticks(log_taus)
        axes[2].set_xticklabels([f'{t:g}' for t in taus], rotation=45)
        axes[2].set_yticks(log_gammas)
        axes[2].set_yticklabels([f'{g:g}' for g in gammas])
        axes[2].set_xlabel(r'$\tau$ (log scale)')
        axes[2].set_ylabel(r'$\gamma$ (log scale)')
        axes[2].set_title(r'(c) $\log_{10}$ raw monotonicity violation')
        fig.colorbar(im2, ax=axes[2], fraction=0.05)
        dp = 0.9 / (M-1); thresh = 0.5 * dp
        try:
            cs = axes[2].contour(log_taus, log_gammas, viol_mat,
                                  levels=[thresh],
                                  colors='white', linewidths=1.6,
                                  linestyles='--')
            axes[2].clabel(cs, fmt=lambda v: f' Δp/2 ', fontsize=8)
        except Exception:
            pass

        fig.suptitle(rf'(τ, γ) monotone phase diagram — $G={Gi}$, $M={M}$, '
                      f'{len(gammas)}×{len(taus)} grid', y=1.02, fontsize=13)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300); plt.close(fig)

        # Page 3 — per-gamma monotone share + per-tau deficit profile
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        frac_mono = (status_mat == 0).mean(axis=1)
        axes[0].plot(gammas, frac_mono, 'o-', color='#27ae60', linewidth=2)
        axes[0].set_xscale('log')
        axes[0].set_xlabel(r'$\gamma$ (log scale)')
        axes[0].set_ylabel(r'fraction of $\tau$-grid monotone')
        axes[0].set_title(r'monotone share vs $\gamma$')
        axes[0].grid(True, which='both', alpha=0.3); axes[0].set_ylim(-0.05, 1.05)
        # Per-tau deficit at several gammas
        sel_g = [gammas[0], gammas[len(gammas)//2], gammas[-1]]
        markers = ['o', 's', '^']
        for g, mk in zip(sel_g, markers):
            ig = gammas.index(g)
            axes[1].plot(taus, deficit_mat[ig, :], mk + '-',
                          label=fr'$\gamma={g:g}$', linewidth=1.7)
        axes[1].set_xscale('log')
        axes[1].set_xlabel(r'$\tau$ (log scale)')
        axes[1].set_ylabel(r'deficit $1-R^2$')
        axes[1].set_title(r'deficit profile across $\tau$')
        axes[1].legend(); axes[1].grid(True, which='both', alpha=0.3)
        plt.tight_layout(); pdf.savefig(fig, dpi=300); plt.close(fig)

        # Page 4 — numerical table
        hdr = '  gamma \\ tau  |  ' + '  '.join(f'{t:>5g}' for t in taus)
        sep = '  ' + '-' * len(hdr)
        lines = ['Status matrix (0=mono, 1=non-mono, 2=cycle, 3=no-conv, 4=fail)',
                 '', hdr, sep]
        for ig, g in enumerate(gammas):
            row = f'  gamma={g:>4g}    |  ' + '  '.join(
                f'{int(status_mat[ig,it]):>5d}' for it in range(len(taus)))
            lines.append(row)
        lines += ['', '', 'Tail-mean deficit (1 - R^2)', '', hdr, sep]
        for ig, g in enumerate(gammas):
            row = f'  gamma={g:>4g}    |  ' + '  '.join(
                f'{deficit_mat[ig,it]:>5.2f}' if np.isfinite(deficit_mat[ig,it]) else '   nan'
                for it in range(len(taus)))
            lines.append(row)
        text_page(pdf, lines, title='Phase tables', fontsize=8)

        # Page 5 — methodology / disclosure
        text_page(pdf, [
            "Reading the figures",
            "-------------------",
            "  Panel (a) Status: green = strictly monotone discrete-price",
            "    equilibrium (RAW violation < Δp/2). Orange = fixed point of",
            "    the projected map but underlying RAW field has genuine",
            "    non-monotone steps. Dashed white line is the smoothed",
            "    boundary between green and non-green at the 0.5-mass contour.",
            "",
            "  Panel (b) Deficit: tail mean of (1 - R^2) of the logit-price",
            "    regression on the sufficient statistic.  Even monotone",
            "    equilibria may exhibit a sizeable revelation deficit.",
            "",
            "  Panel (c) RAW max non-monotone step on log10 scale, with the",
            "    Δp/2 contour superimposed.",
            "",
            "Limitations and follow-ups",
            "--------------------------",
            "  * Grid still fixed at G = 21 for the underlying coverage map.",
            "  * Some borderline cells (gamma ~ 1, tau >= 2.5) flip status",
            "    between cycle and fixed; sensitivity to h_p_factor was not",
            "    fully explored here.",
            "  * The dashed boundary curve in panel (a) is an interpolated",
            "    eyeball aid only; reading the boundary from the cells is",
            "    the rigorous interpretation.",
        ], title='Notes & follow-ups')

    print(f"Saved {PDF_PATH}")


if __name__ == '__main__':
    main()
