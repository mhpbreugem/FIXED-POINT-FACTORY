"""TEST A: validate the CRRA market-clearing Jensen expansion directly.
For random belief triples mu in (0.05,0.95), equal weights, gamma in a grid,
compute exact cleared price p* = clear_crra, and compare logit(p*) against the
prediction mbar + (1/2-p*)*Var(m)/gamma. No contour/inference involved."""
import os, sys, json, time
sys.path.insert(0, '/tmp/rezn-source')
import numpy as np
from code.demand import clear_crra
from code.signals import logit as _logit

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_jensen_gap'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + '/run.log', 'a')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

logit = np.vectorize(lambda p: _logit(float(p)))
rng = np.random.default_rng(20260529)
N = 40000
GAMMAS = [0.1, 0.3, 1.0, 3.0]
W = np.array([1.0, 1.0, 1.0])

log('=== TEST A : CRRA clearing Jensen expansion ===', time.ctime())
mu = rng.uniform(0.05, 0.95, size=(N, 3))
m = logit(mu)                       # log-odds beliefs
mbar = m.mean(axis=1)
Varm = m.var(axis=1)                # population variance over 3 agents (ddof=0)

results = {}
allrows = {'gap': [], 'pred_self': [], 'gamma': [], 'Varm': []}
for g in GAMMAS:
    gv = np.full(3, g)
    pstar = np.array([clear_crra(np.ascontiguousarray(mu[k]), gv, W) for k in range(N)])
    lp = logit(pstar)
    gap = lp - mbar                         # actual Jensen gap
    pred_self = (0.5 - pstar) * Varm / g    # implicit form (p* on RHS)
    # explicit form: use the price implied by mbar (the avg-belief price logit)
    p_mbar = 1.0 / (1.0 + np.exp(-mbar))
    pred_expl = (0.5 - p_mbar) * Varm / g

    err_self = lp - (mbar + pred_self)
    err_expl = lp - (mbar + pred_expl)
    corr = float(np.corrcoef(gap, pred_self)[0, 1])
    corr_expl = float(np.corrcoef(gap, pred_expl)[0, 1])
    # sign-change check: sign of gap should match sign of (1/2 - p*)
    sgn_match = float(np.mean(np.sign(gap) == np.sign(0.5 - pstar)))
    # restrict to non-tiny dispersion for the sign test to be meaningful
    mask = Varm > 0.05
    sgn_match_disp = float(np.mean(np.sign(gap[mask]) == np.sign(0.5 - pstar[mask])))

    results[f'gamma={g}'] = dict(
        corr_self=corr, corr_explicit=corr_expl,
        rms_err_self=float(np.sqrt(np.mean(err_self**2))),
        max_err_self=float(np.max(np.abs(err_self))),
        rms_err_explicit=float(np.sqrt(np.mean(err_expl**2))),
        max_err_explicit=float(np.max(np.abs(err_expl))),
        mean_abs_gap=float(np.mean(np.abs(gap))),
        sign_match_frac=sgn_match, sign_match_frac_disp=sgn_match_disp,
    )
    allrows['gap'].append(gap); allrows['pred_self'].append(pred_self)
    allrows['gamma'].append(np.full(N, g)); allrows['Varm'].append(Varm)
    log(f'  gamma={g:<4}: corr(self)={corr:.5f} rms_self={results[f"gamma={g}"]["rms_err_self"]:.2e} '
        f'max_self={results[f"gamma={g}"]["max_err_self"]:.2e} mean|gap|={results[f"gamma={g}"]["mean_abs_gap"]:.4f} '
        f'sign_match(disp>.05)={sgn_match_disp:.4f}')

# P3: gap ~ 1/gamma. For fixed beliefs, mean|gap| vs gamma. Use same mu across gammas already.
mg = np.array([results[f'gamma={g}']['mean_abs_gap'] for g in GAMMAS])
slope_1g = float(np.polyfit(np.log(GAMMAS), np.log(mg), 1)[0])
results['P3_loglog_slope_meanabsgap_vs_gamma'] = slope_1g
log(f'  P3: log-log slope mean|gap| vs gamma = {slope_1g:.4f} (expect ~ -1)')

# P4: gap grows with Var(m). Within gamma=1, bin by Varm.
g1 = 1.0; gv = np.full(3, g1)
idx = np.argsort(Varm)
# correlation of |gap| with Varm at gamma=1
i1 = GAMMAS.index(1.0)
gap1 = allrows['gap'][i1]
corr_gap_var = float(np.corrcoef(np.abs(gap1), Varm)[0, 1])
results['P4_corr_absgap_vs_Varm_g1'] = corr_gap_var
log(f'  P4: corr(|gap|, Var(m)) at gamma=1 = {corr_gap_var:.4f} (expect >0)')

json.dump(results, open(OUT + '/testA_report.json', 'w'), indent=2)

# scatter plot gap vs (1/2-p*)Var/gamma colored by gamma
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
colors = plt.cm.viridis(np.linspace(0, 1, len(GAMMAS)))
gap_all = np.concatenate(allrows['gap']); pred_all = np.concatenate(allrows['pred_self'])
for c, g in zip(colors, GAMMAS):
    i = GAMMAS.index(g)
    sub = slice(0, 4000)
    ax.scatter(allrows['pred_self'][i][sub], allrows['gap'][i][sub], s=2, alpha=0.3, color=c, label=f'γ={g}')
lim = np.percentile(np.abs(np.concatenate([gap_all, pred_all])), 99.5)
ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1, label='y=x')
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('predicted gap  (1/2 - p*)·Var(m)/γ'); ax.set_ylabel('actual gap  logit(p*) - mbar')
ax.set_title('TEST A: CRRA-clearing Jensen gap vs demand-curvature prediction')
ax.legend(); ax.grid(ls=':'); plt.tight_layout()
plt.savefig(OUT + '/testA_scatter.png', dpi=130, bbox_inches='tight'); plt.close()
log('  wrote testA_scatter.png and testA_report.json')
log('=== TEST A done ===')
