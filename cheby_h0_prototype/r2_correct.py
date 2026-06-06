"""Compute the CORRECT R² for the gamma sweep: nonparametric f(T)-R².

For each FP:
  R²_linear = 1 - SS_res / SS_tot where pred = α*T + β   (linear fit)
  R²_polyfit_k = 1 - SS_res / SS_tot where pred = poly_k(T)  (degree-k poly fit in T)
  R²_grouped = ANOVA: 1 - within-group-variance / total-variance,
               grouping cells by unique T value (tolerance ~ machine eps)
               (best possible f(T) fit; this is the relevant deficit)

The relevant economic statistic for the paper is:
  deficit_oneToOne = 1 - R²_grouped
This is the fraction of logit(P) variance NOT explainable by ANY f(T).
If 0: P = f(T) exactly (one-to-one). If positive: same T gives multiple
P values; the price is NOT a sufficient statistic for T.
"""
import sys, json, os
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cheby_numba import U_NODES, TAU, N_GRID

G = N_GRID
U1, U2, U3 = np.meshgrid(U_NODES, U_NODES, U_NODES, indexing='ij')
T = TAU*(U1+U2+U3)

def compute_R2_metrics(P, T):
    Pc = np.clip(P, 1e-15, 1-1e-15)
    L = np.log(Pc/(1-Pc)).ravel()
    Tf = T.ravel()
    ss_tot = float(np.sum((L - L.mean())**2))

    # 1. Linear R² (existing)
    slope = float(np.sum(L*Tf) / np.sum(Tf**2))
    intercept = float(np.mean(L - slope*Tf))
    pred_lin = slope*Tf + intercept
    R2_lin = 1 - float(np.sum((L - pred_lin)**2) / ss_tot)

    # 2. Polynomial-in-T R² at various degrees
    R2_poly = {}
    for deg in [1, 3, 5, 7, 9]:
        coefs = np.polyfit(Tf, L, deg)
        pred = np.polyval(coefs, Tf)
        R2_poly[deg] = 1 - float(np.sum((L - pred)**2) / ss_tot)

    # 3. Nonparametric ANOVA: best possible f(T) fit
    # Group cells by T value (with tolerance for numerical noise)
    unique_T, inverse = np.unique(np.round(Tf, decimals=10), return_inverse=True)
    n_groups = len(unique_T)
    # Within-group variance
    within_ss = 0.0
    group_means = np.empty(n_groups)
    group_sizes = np.empty(n_groups)
    for g in range(n_groups):
        mask = (inverse == g)
        n_g = int(mask.sum())
        L_g = L[mask]
        group_means[g] = L_g.mean()
        group_sizes[g] = n_g
        within_ss += float(np.sum((L_g - L_g.mean())**2))
    R2_nonparam = 1 - within_ss / ss_tot
    deficit_oneToOne = 1 - R2_nonparam   # = within_ss/total_ss

    # Singletons vs multi-cell T groups
    n_singleton = int(np.sum(group_sizes == 1))
    n_multi = n_groups - n_singleton

    # Max within-group spread (worst one-to-one breakdown)
    max_within_spread = 0.0
    for g in range(n_groups):
        mask = (inverse == g)
        if mask.sum() > 1:
            L_g = L[mask]
            spread = float(L_g.max() - L_g.min())
            if spread > max_within_spread:
                max_within_spread = spread

    return dict(
        R2_lin=R2_lin,
        R2_poly=R2_poly,
        R2_nonparam=R2_nonparam,
        deficit_lin=1 - R2_lin,
        deficit_poly={d: 1-v for d, v in R2_poly.items()},
        deficit_oneToOne=deficit_oneToOne,
        n_unique_T=n_groups,
        n_singleton_T=n_singleton,
        n_multi_T=n_multi,
        max_within_T_spread=max_within_spread,
        within_ss=within_ss,
        total_ss=ss_tot,
    )


# Load all FPs and recompute
data = json.load(open('/tmp/cheby_h0/gamma_sweep_data.json'))
gammas = sorted([d['gamma'] for d in data.values()])
out = {}
print(f'{"gamma":>6} {"R2_lin":>9} {"R2_poly9":>10} {"R2_nonparam":>13} {"def_oneToOne":>13} {"unique_T":>9} {"max spread":>10}')
for g in gammas:
    P = np.load(f'/tmp/cheby_h0/fps_gamma/P_FP_gamma{g}.npy')
    m = compute_R2_metrics(P, T)
    out[f'gamma={g}'] = m
    print(f'{g:>6.2f} {m["R2_lin"]:>9.5f} {m["R2_poly"][9]:>10.5f} '
          f'{m["R2_nonparam"]:>13.5f} {m["deficit_oneToOne"]:>13.5f} '
          f'{m["n_unique_T"]:>9d} {m["max_within_T_spread"]:>10.3f}')

json.dump(out, open('/tmp/cheby_h0/r2_correct.json', 'w'),
            indent=2, default=str)
print('\nsaved r2_correct.json')

print(f'\n=== Interpretation ===')
print(f'For each gamma, three "deficit" numbers:')
print(f'  deficit_lin    = 1 - R²_linear      (vs slope-of-T fit only)')
print(f'  deficit_poly9  = 1 - R²_poly_deg9   (vs degree-9 polynomial in T)')
print(f'  deficit_oneToOne = 1 - R²_nonparam  (vs best possible f(T))')
print(f'')
print(f'The third one is the ECONOMICALLY MEANINGFUL one:')
print(f'  if = 0: P = f(T) exactly -> price is one-to-one with T (T-revealing)')
print(f'  if > 0: P depends on individual u_k, not just T -> price is NOT')
print(f'          a sufficient statistic for T (one-to-one mapping breaks)')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
ax.semilogx(gammas, [out[f'gamma={g}']['deficit_lin'] for g in gammas],
              'o-', markersize=8, label=r'1 - $R^2_{\rm linear}$ (vs $\alpha T$)',
              color='tab:blue')
ax.semilogx(gammas, [out[f'gamma={g}']['deficit_poly'][3] for g in gammas],
              's-', markersize=8, label=r'1 - $R^2_{\rm poly3}$',
              color='tab:purple')
ax.semilogx(gammas, [out[f'gamma={g}']['deficit_poly'][9] for g in gammas],
              '^-', markersize=8, label=r'1 - $R^2_{\rm poly9}$',
              color='tab:orange')
ax.semilogx(gammas, [out[f'gamma={g}']['deficit_oneToOne'] for g in gammas],
              'D-', markersize=10, lw=2,
              label=r'1 - $R^2_{\rm nonparam}$ (best $f(T)$ fit)',
              color='tab:red')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel('deficit')
ax.set_title('Three deficit measures vs $\\gamma$')
ax.legend(fontsize=10); ax.grid(alpha=0.3, which='both')

ax = axes[1]
# Gap between linear and nonparam deficit
gap = [out[f'gamma={g}']['deficit_lin'] - out[f'gamma={g}']['deficit_oneToOne']
       for g in gammas]
ax.semilogx(gammas, gap, 'o-', markersize=8, color='tab:green')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$1-R^2_{\rm lin}$ $-$ $1-R^2_{\rm nonparam}$')
ax.set_title('Linear vs nonparametric deficit gap\n'
              '(how much of "linear deficit" is just nonlinearity in $f(T)$)')
ax.grid(alpha=0.3, which='both')

plt.suptitle('Deficits: linear vs nonparametric $f(T)$\n'
              r'The nonparam deficit ($1-R^2_{\rm nonparam}$) is the true measure of'
              r' one-to-one $P \leftrightarrow T$ failure',
              fontsize=12)
plt.tight_layout()
import os
os.makedirs('/tmp/cheby_h0/figs/gamma_sweep', exist_ok=True)
plt.savefig('/tmp/cheby_h0/figs/gamma_sweep/08_correct_R2.png',
              dpi=140, bbox_inches='tight')
plt.close()
print('saved 08_correct_R2.png')
