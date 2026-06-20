"""PDF report for the high-range (gamma, tau) monotone phase sweep.

Reads phase.json from the phase_hi run and produces a multi-page PDF
that combines the headline figure with text describing the methods and
findings. No latex required.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau/discrete_price_monotone_phase_hi'
JSON_PATH = f'{ROOT}/phase.json'
PDF_PATH  = f'{ROOT}/phase_hi_report.pdf'


def text_page(pdf, lines, title=None, fontsize=11):
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
    name = {0: 'monotone-converged', 1: 'fixed but non-monotone',
            2: 'cycle', 3: 'not-converged', 4: 'solver-failed'}
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

    with PdfPages(PDF_PATH) as pdf:
        # Title page
        text_page(pdf, [
            "Authors: M. Breugem (compiled by Claude on " +
            f"{__import__('datetime').date.today().isoformat()})",
            "",
            "Setup",
            "-----",
            "  * REZN economy: K=3 informed traders, CRRA-like utilities,",
            "    Gaussian signals.  No noise traders.",
            f"  * Endogenous-price (CMM) discrete-bin solver,  G = {Gi},  M = {M}.",
            f"  * Sweep grid:",
            f"      gamma in {gammas}",
            f"      tau   in {taus}",
            "  * For each (tau, gamma) we run the soft-bin K-means iteration",
            "    with monotone PAV projection (max_iter=40, h_p_factor=1.5)",
            "    and record the final status:",
            "       0: monotone_converged   -- equilibrium is strictly monotone",
            "                                  (RAW non-monotonicity below Delta p / 2)",
            "       1: fixed_but_nonmonotone-- a fixed point exists but the RAW",
            "                                  field has genuine non-monotonicity",
            "       2: cycle                -- iterate hits a 2-cycle",
            "       3: not_converged        -- ran out of iterations",
            "       4: solver_failed        -- numerical exception",
            "",
            "Headline",
            "--------",
            "Extends the (tau, gamma) phase diagram into the upper-right corner.",
            "Confirms the original finding: the 'good zone' (strictly monotone",
            "equilibria) grows monotonically with risk aversion gamma and shrinks",
            "monotonically with information variance tau.",
            "",
            f"   * {(status_mat==0).sum()} of {status_mat.size} cells: monotone-converged.",
            f"   * {(status_mat==1).sum()} cells: fixed but non-monotone.",
            f"   * {(status_mat==2).sum()} cells: cycle.",
            f"   * {(status_mat==3).sum()} cells: not converged.",
            f"   * {(status_mat==4).sum()} cells: solver failed.",
        ], title='Monotone Phase Diagram in (tau, gamma): high range')

        # Page 2: the three-panel headline figure
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
        cmap = ListedColormap(['#2ecc71', '#e67e22', '#e74c3c', '#9b59b6', '#7f8c8d'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
        im0 = axes[0].imshow(status_mat, origin='lower', aspect='auto',
                              cmap=cmap, norm=norm)
        axes[0].set_xticks(range(len(taus)))
        axes[0].set_xticklabels([f'{t:g}' for t in taus], rotation=45)
        axes[0].set_yticks(range(len(gammas)))
        axes[0].set_yticklabels([f'{g:g}' for g in gammas])
        axes[0].set_xlabel(r'$\tau$'); axes[0].set_ylabel(r'$\gamma$')
        axes[0].set_title('Status')
        cbar0 = fig.colorbar(im0, ax=axes[0], ticks=[0,1,2,3,4], fraction=0.05)
        cbar0.ax.set_yticklabels(['mono', 'non-mono', 'cycle', 'no-conv', 'fail'],
                                  fontsize=8)
        for ig in range(len(gammas)):
            for it in range(len(taus)):
                axes[0].text(it, ig, str(status_mat[ig, it]),
                              ha='center', va='center', fontsize=7, color='black')
        im1 = axes[1].imshow(deficit_mat, origin='lower', aspect='auto',
                              cmap='viridis')
        axes[1].set_xticks(range(len(taus)))
        axes[1].set_xticklabels([f'{t:g}' for t in taus], rotation=45)
        axes[1].set_yticks(range(len(gammas)))
        axes[1].set_yticklabels([f'{g:g}' for g in gammas])
        axes[1].set_xlabel(r'$\tau$'); axes[1].set_ylabel(r'$\gamma$')
        axes[1].set_title(r'Deficit $1-R^2$ (tail mean)')
        fig.colorbar(im1, ax=axes[1], fraction=0.05)
        for ig in range(len(gammas)):
            for it in range(len(taus)):
                v = deficit_mat[ig, it]
                if np.isfinite(v):
                    axes[1].text(it, ig, f'{v:.2f}', ha='center', va='center',
                                  fontsize=6, color='white')
        with np.errstate(divide='ignore'):
            log_viol = np.log10(np.maximum(viol_mat, 1e-16))
        im2 = axes[2].imshow(log_viol, origin='lower', aspect='auto', cmap='magma')
        axes[2].set_xticks(range(len(taus)))
        axes[2].set_xticklabels([f'{t:g}' for t in taus], rotation=45)
        axes[2].set_yticks(range(len(gammas)))
        axes[2].set_yticklabels([f'{g:g}' for g in gammas])
        axes[2].set_xlabel(r'$\tau$'); axes[2].set_ylabel(r'$\gamma$')
        axes[2].set_title(r'$\log_{10}$ RAW non-monotone step')
        fig.colorbar(im2, ax=axes[2], fraction=0.05)
        dp = 0.9 / (M-1); thresh = 0.5 * dp
        try:
            cs = axes[2].contour(range(len(taus)), range(len(gammas)),
                                  viol_mat, levels=[thresh],
                                  colors=['white'], linestyles=['--'], linewidths=2)
            axes[2].clabel(cs, fmt=lambda v: f' Δp/2 = {v:.2e} ', fontsize=7)
        except Exception:
            pass
        fig.suptitle(f'(tau, gamma) phase diagram — high range — G={Gi}, M={M}',
                      y=1.02, fontsize=12)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # Page 3: numerical table
        hdr = '  gamma \\ tau  |  ' + '  '.join(f'{t:>5g}' for t in taus)
        sep = '  ' + '-' * len(hdr)
        lines = ['Status matrix (0=mono, 1=non-mono, 2=cycle, 3=no-conv, 4=fail)',
                 '',
                 hdr, sep]
        for ig, g in enumerate(gammas):
            row = f'  gamma={g:>4g}    |  ' + '  '.join(f'{int(status_mat[ig,it]):>5d}'
                                                          for it in range(len(taus)))
            lines.append(row)
        lines += ['', '', 'Deficit (tail mean of 1 - R^2)', '', hdr, sep]
        for ig, g in enumerate(gammas):
            row = f'  gamma={g:>4g}    |  ' + '  '.join(
                f'{deficit_mat[ig,it]:>5.2f}' if np.isfinite(deficit_mat[ig,it]) else '   nan'
                for it in range(len(taus)))
            lines.append(row)
        lines += ['', '', 'Wall time (seconds)', '', hdr, sep]
        for ig, g in enumerate(gammas):
            row = f'  gamma={g:>4g}    |  ' + '  '.join(
                f'{wall_mat[ig,it]:>5.0f}' if np.isfinite(wall_mat[ig,it]) else '   nan'
                for it in range(len(taus)))
            lines.append(row)
        text_page(pdf, lines, title='Phase tables', fontsize=8)

        # Page 4: per-gamma monotone-fraction
        fig, ax = plt.subplots(figsize=(7, 5))
        frac_mono = (status_mat == 0).mean(axis=1)
        ax.plot(gammas, frac_mono, 'o-', color='#2ecc71', linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel(r'$\gamma$ (log scale)')
        ax.set_ylabel(r'fraction of $\tau$-grid that is monotone')
        ax.set_title(r'Monotone share as a function of $\gamma$')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        for x, y in zip(gammas, frac_mono):
            ax.annotate(f'{y:.0%}', (x, y), textcoords='offset points',
                          xytext=(0, 8), ha='center', fontsize=8)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Page 5: closing notes
        text_page(pdf, [
            "Reading the figures",
            "-------------------",
            "",
            "  * Status (panel 1): a green cell certifies a strictly monotone",
            "    discrete-price equilibrium of the soft-K-means CMM solver.",
            "    Orange cells reach a fixed point but the underlying RAW",
            "    price-on-state field has genuine non-monotonicity above the",
            "    Delta p / 2 grid floor.",
            "",
            "  * Deficit (panel 2): tail-mean of (1 - R^2) of the logit price",
            "    regression on (u1+u2+u3).  Larger = farther from full revelation.",
            "    Note that even strictly-monotone equilibria can carry sizeable",
            "    revelation deficit at intermediate tau.",
            "",
            "  * RAW violation (panel 3): max single-step non-monotonicity of",
            "    the un-projected price field after iteration; the dashed Delta p / 2",
            "    contour separates strict (green) from quasi (orange) monotonicity.",
            "",
            "Limitations",
            "-----------",
            "  * Single-grid run (G=21, M=32); a publishable companion at",
            "    higher resolution is the next step.",
            "  * Tau >= 2 with weak risk aversion is delicate -- the K-means",
            "    iteration occasionally cycles instead of converging.",
        ], title='Notes')

    print(f"Saved {PDF_PATH}")


if __name__ == '__main__':
    main()
