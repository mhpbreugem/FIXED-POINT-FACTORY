"""Build the full 20-page gamma-sweep PDF."""
import json
data = json.load(open('/tmp/cheby_h0/gamma_sweep_data.json'))
gammas = sorted([d['gamma'] for d in data.values()])

# Contour pages
contour_lines = []
for g in gammas:
    r = data[f'gamma={g}']
    contour_lines.append(r'\begin{center}')
    contour_lines.append(rf'\includegraphics[width=\linewidth]{{04_contours_gamma{g}.png}}')
    contour_lines.append(r'\end{center}')
    contour_lines.append(rf'$\gamma={g}$, NQ={r["NQ"]}, $\|F\|_\infty={r["F_final"]:.2e}$, '
                          rf'slope $\alpha^*={r["slope"]:.4f}$, '
                          rf'deficit $1{{-}}R^2={r["deficit"]:.4f}$.')
    contour_lines.append(r'\newpage')
open('/tmp/cheby_h0/contour_pages.tex', 'w').write('\n'.join(contour_lines))

# mu pages
mu_lines = []
for g in gammas:
    r = data[f'gamma={g}']
    mu_lines.append(r'\begin{center}')
    mu_lines.append(rf'\includegraphics[width=\linewidth]{{05_mu_gamma{g}.png}}')
    mu_lines.append(r'\end{center}')
    mu_lines.append(rf'$\gamma={g}$. Posterior $\mu(p, u_k)$ table from the converged FP. '
                     rf'Slope $\alpha^*={r["slope"]:.4f}$, deficit $={r["deficit"]:.4f}$.')
    mu_lines.append(r'\newpage')
open('/tmp/cheby_h0/mu_pages.tex', 'w').write('\n'.join(mu_lines))

# Slice pages
slice_lines = []
for g in gammas:
    r = data[f'gamma={g}']
    slice_lines.append(r'\begin{center}')
    slice_lines.append(rf'\includegraphics[width=\linewidth]{{07_slices_gamma{g}.png}}')
    slice_lines.append(r'\end{center}')
    slice_lines.append(rf'$\gamma={g}$: $P(u_1, u_2, u_3)$ at three slices.')
    slice_lines.append(r'\newpage')
open('/tmp/cheby_h0/slice_pages.tex', 'w').write('\n'.join(slice_lines))

# Summary table
tab_lines = [
    r'\begin{center}',
    r'\begin{tabular}{cccccc}',
    r'\toprule',
    r'$\gamma$ & best $N_Q$ & $\|F\|_\infty$ & slope $\alpha^*$ & deficit $1-R^2$ & solve time \\',
    r'\midrule',
]
for g in gammas:
    r = data[f'gamma={g}']
    star = r' $\star$' if r['F_final'] < 1e-12 else ''
    tab_lines.append(
        rf'{g} & {r["NQ"]} & ${r["F_final"]:.2e}${star} & {r["slope"]:.4f} & '
        rf'{r["deficit"]:.4f} & {r["t_solve"]:.1f}\,s \\')
tab_lines.append(r'\bottomrule')
tab_lines.append(r'\end{tabular}')
tab_lines.append(r'\end{center}')
tab_lines.append(r'\vspace{0.5em}')
tab_lines.append(r'$\star$ = converged to machine $\varepsilon$ (6 of 9). '
                  r'The 3 that did not (~$10^{-3}$ floor) would need wider $N_Q$ search.')
open('/tmp/cheby_h0/summary_table.tex', 'w').write('\n'.join(tab_lines))

print('all helper .tex files written')
