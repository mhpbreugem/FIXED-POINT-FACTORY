"""TEST B: measure the Jensen gap on the genuine nailed noisy-PR equilibrium.
Nail the smoothed K=3 PR fixed point at kernel_h=0.30 via Newton-Krylov for
several gamma. At each nailed P*, extract per-state agent beliefs mu_i by
replicating the operator's smoothed Bayes step, then compute the actual gap
logit(P*)-mbar vs predicted (1/2-P*)Var(m)/gamma."""
import os, sys, json, time, warnings
sys.path.insert(0, '/tmp/rezn-source')
import numpy as np
from numba import njit, prange
from code.contour_K3_halo import (init_no_learning_K3, phi_K3_halo_smooth,
                                   _agent_evidence_K3_smooth, _bayes)
from code.signals import logit as _logit
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence
warnings.filterwarnings('ignore')

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_jensen_gap'
os.makedirs(OUT, exist_ok=True)
LOG = open(OUT + '/run.log', 'a')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

K = 3; Gi = 9; pad = 2; UMAX = 4.0; Gf = Gi + 2 * pad
du = 2 * UMAX / (Gi - 1)
uf = np.array([-UMAX + (q - pad) * du for q in range(Gf)])
lo, hi = pad, pad + Gi
slc = (slice(lo, hi),) * K
TAU = 2.0; KH = 0.30
tv = np.full(K, TAU); wv = np.full(K, 1.0)
ui = uf[lo:hi]
U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
T = TAU * (U1 + U2 + U3)

logit_v = np.vectorize(lambda p: _logit(float(p)))

# Extract per-state agent beliefs by replicating the operator's Bayes step.
@njit(cache=True, parallel=True)
def extract_mu(P_full, u_full, inner_lo, inner_hi, tau_vec, kernel_h):
    Gi = inner_hi - inner_lo
    MU = np.empty((Gi, Gi, Gi, 3), dtype=np.float64)
    for i in prange(inner_lo, inner_hi):
        acc = np.empty(2, dtype=np.float64)
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full[i, j, l]
                _agent_evidence_K3_smooth(P_full[i, :, :], p, u_full,
                                          tau_vec[1], tau_vec[2], kernel_h, acc)
                m0 = _bayes(u_full[i], tau_vec[0], acc[0], acc[1])
                _agent_evidence_K3_smooth(P_full[:, j, :], p, u_full,
                                          tau_vec[0], tau_vec[2], kernel_h, acc)
                m1 = _bayes(u_full[j], tau_vec[1], acc[0], acc[1])
                _agent_evidence_K3_smooth(P_full[:, :, l], p, u_full,
                                          tau_vec[0], tau_vec[1], kernel_h, acc)
                m2 = _bayes(u_full[l], tau_vec[2], acc[0], acc[1])
                ii, jj, ll = i - inner_lo, j - inner_lo, l - inner_lo
                MU[ii, jj, ll, 0] = m0
                MU[ii, jj, ll, 1] = m1
                MU[ii, jj, ll, 2] = m2
    return MU

GAMMAS = [0.1, 0.2, 0.4, 0.8, 1.6]
log('=== TEST B : Jensen gap on nailed noisy-PR equilibrium (kernel_h=%.2f, tau=%d) ===' % (KH, TAU), time.ctime())

rows = []
P0_cache = None
for g in GAMMAS:
    gv = np.full(K, g)
    halo = init_no_learning_K3(uf, tv, gv, wv)
    def resid(xflat):
        Pf = halo.copy(); Pf[slc] = xflat.reshape((Gi,) * K)
        return (phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, KH) - Pf)[slc].ravel()
    P0 = halo[slc].ravel().copy() if P0_cache is None else P0_cache.copy()
    conv = True
    try:
        sol = newton_krylov(resid, P0, f_tol=1e-10, maxiter=300, method='lgmres')
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(resid(sol))))
    P0_cache = sol.copy()
    Pstar = halo.copy(); Pstar[slc] = sol.reshape((Gi,) * K)

    MU = extract_mu(Pstar, uf, lo, hi, tv, g)        # (Gi,Gi,Gi,3)
    m = logit_v(MU)                                   # log-odds beliefs
    mbar = m.mean(axis=3)
    Varm = m.var(axis=3)                              # equal weights -> plain var
    Pin = np.clip(Pstar[slc], 1e-12, 1 - 1e-12)
    lp = logit_v(Pin)
    gap = lp - mbar
    pred = (0.5 - Pin) * Varm / g

    gf = gap.ravel(); pf = pred.ravel(); pp = Pin.ravel(); vf = Varm.ravel()
    corr = float(np.corrcoef(gf, pf)[0, 1])
    # revelation deficit: 1 - R^2 of actual gap explained by predicted gap (regression through scaling)
    A = np.vstack([pf, np.ones_like(pf)]).T
    coef, _, _, _ = np.linalg.lstsq(A, gf, rcond=None)
    resid_fit = gf - A @ coef
    R2 = 1.0 - np.sum(resid_fit**2) / max(np.sum((gf - gf.mean())**2), 1e-30)
    deficit = 1.0 - R2
    slope = float(coef[0])
    sign_match = float(np.mean(np.sign(gf) == np.sign(0.5 - pp)))
    mask = vf > 0.02
    sign_match_disp = float(np.mean(np.sign(gf[mask]) == np.sign(0.5 - pp[mask]))) if mask.any() else float('nan')
    row = dict(gamma=g, Finf=Finf, converged=conv,
               corr=corr, R2=float(R2), deficit=float(deficit), slope=slope,
               mean_abs_gap=float(np.mean(np.abs(gf))),
               mean_Varm=float(np.mean(vf)),
               sign_match_frac=sign_match, sign_match_frac_disp=sign_match_disp)
    rows.append(row)
    log(f'  gamma={g:<4} ||F||={Finf:.2e} conv={conv} | corr={corr:.4f} R2={R2:.4f} '
        f'deficit={deficit:.4f} slope={slope:.3f} mean|gap|={row["mean_abs_gap"]:.4f} '
        f'signmatch(disp)={sign_match_disp:.4f}')
    np.save(OUT + f'/testB_gap_g{g}.npy', np.column_stack([gf, pf, pp, vf]))

# P3 fit: deficit ~ gamma^slope (log-log). Also mean|gap| ~ 1/gamma.
gs = np.array([r['gamma'] for r in rows])
defs = np.array([r['deficit'] for r in rows])
mags = np.array([r['mean_abs_gap'] for r in rows])
sl_def = float(np.polyfit(np.log(gs), np.log(np.maximum(defs, 1e-12)), 1)[0])
sl_mag = float(np.polyfit(np.log(gs), np.log(mags), 1)[0])
report = dict(kernel_h=KH, tau=TAU, G_inner=Gi, rows=rows,
              loglog_slope_deficit_vs_gamma=sl_def,
              loglog_slope_meanabsgap_vs_gamma=sl_mag)
log(f'  P3: log-log slope deficit vs gamma = {sl_def:.3f}; mean|gap| vs gamma = {sl_mag:.3f} (expect ~ -1 for gap)')
json.dump(report, open(OUT + '/testB_report.json', 'w'), indent=2)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), dpi=130)
colors = plt.cm.plasma(np.linspace(0, 0.9, len(GAMMAS)))
# panel 0: gap vs (1/2 - P*) sign structure
for c, r in zip(colors, rows):
    d = np.load(OUT + f'/testB_gap_g{r["gamma"]}.npy')
    ax[0].scatter(0.5 - d[:, 2], d[:, 0], s=8, alpha=0.5, color=c, label=f'γ={r["gamma"]}')
ax[0].axhline(0, c='k', lw=.5); ax[0].axvline(0, c='k', lw=.5)
ax[0].set_xlabel('1/2 - P*'); ax[0].set_ylabel('actual gap  logit(P*) - mbar')
ax[0].set_title('TEST B: gap sign structure (P2: sign flips at P*=1/2)'); ax[0].legend(fontsize=8); ax[0].grid(ls=':')
# panel 1: actual vs predicted gap
for c, r in zip(colors, rows):
    d = np.load(OUT + f'/testB_gap_g{r["gamma"]}.npy')
    ax[1].scatter(d[:, 1], d[:, 0], s=8, alpha=0.5, color=c, label=f'γ={r["gamma"]}')
lim = max(abs(ax[1].get_xlim()[0]), abs(ax[1].get_xlim()[1]))
ax[1].plot([-lim, lim], [-lim, lim], 'r--', lw=1, label='y=x')
ax[1].set_xlabel('predicted (1/2-P*)Var(m)/γ'); ax[1].set_ylabel('actual gap')
ax[1].set_title('actual vs predicted gap'); ax[1].legend(fontsize=8); ax[1].grid(ls=':')
# panel 2: deficit vs gamma log-log
ax[2].loglog(gs, defs, 'o-', label=f'deficit (slope {sl_def:.2f})')
ax[2].loglog(gs, mags, 's-', label=f'mean|gap| (slope {sl_mag:.2f})')
ax[2].loglog(gs, defs[0] * (gs / gs[0])**(-1.0), 'k:', label='slope -1 ref')
ax[2].set_xlabel('gamma'); ax[2].set_title('P3: deficit / |gap| vs gamma (expect ~1/γ)')
ax[2].legend(fontsize=8); ax[2].grid(ls=':', which='both')
plt.suptitle('TEST B: Jensen gap on the genuine nailed noisy-PR K=3 equilibrium (kernel_h=0.30)', weight='bold')
plt.tight_layout(); plt.savefig(OUT + '/testB_figures.png', dpi=130, bbox_inches='tight'); plt.close()
log('  wrote testB_figures.png and testB_report.json')
log('=== TEST B done ===')
