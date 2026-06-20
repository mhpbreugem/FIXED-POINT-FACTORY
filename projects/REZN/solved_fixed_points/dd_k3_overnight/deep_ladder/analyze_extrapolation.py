"""Continuum (h->0) extrapolation of deficit and slope from the deep G-ladders.

Per cell, with h = 0.45*sqrt(8/(G-1)):
  - q-free nonlinear fit:  m(h) = m_inf + a*h^q   (curve_fit)
  - q=1 linear fit:        m(h) = m_inf + a*h
  - q=2 linear fit:        m(h) = m_inf + a*h^2
Honest error bar: the interval spanning all three extrapolated m_inf values
(centered on the q-free value). Robustness: q-free fit on the deepest 5
rungs only is also reported.
"""
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/deep_ladder'
results = json.load(open(f"{OUT}/results.json"))


def fits(h, m):
    """Return dict of extrapolations for metric m(h)."""
    out = {}
    # q-free nonlinear
    def model(h, m_inf, a, q):
        return m_inf + a * h**q
    span = m[0] - m[-1]
    p0 = [max(m[-1] - 0.5*abs(span), -1.0) if span >= 0 else m[-1] + 0.5*abs(span),
          span / (h[0]**1.5 - h[-1]**1.5 + 1e-30), 1.5]
    try:
        popt, pcov = curve_fit(model, h, m, p0=[m[-1], p0[1], 1.5],
                               bounds=([-np.inf, -np.inf, 0.2],
                                       [np.inf, np.inf, 4.0]), maxfev=20000)
        resid = m - model(h, *popt)
        out['qfree'] = dict(m_inf=float(popt[0]), a=float(popt[1]),
                            q=float(popt[2]),
                            rms=float(np.sqrt(np.mean(resid**2))))
    except Exception as e:
        out['qfree'] = dict(error=str(e))
    # fixed-q linear fits
    for q in (1.0, 2.0):
        A = np.vstack([np.ones_like(h), h**q]).T
        coef, *_ = np.linalg.lstsq(A, m, rcond=None)
        resid = m - A @ coef
        out[f'q{int(q)}'] = dict(m_inf=float(coef[0]), a=float(coef[1]), q=q,
                                 rms=float(np.sqrt(np.mean(resid**2))))
    # robustness: q-free on deepest 5 rungs
    if len(h) >= 5:
        try:
            popt, _ = curve_fit(model, h[-5:], m[-5:],
                                p0=[m[-1], out['qfree'].get('a', 1.0),
                                    out['qfree'].get('q', 1.5)],
                                bounds=([-np.inf, -np.inf, 0.2],
                                        [np.inf, np.inf, 4.0]), maxfev=20000)
            out['qfree_deep5'] = dict(m_inf=float(popt[0]), a=float(popt[1]),
                                      q=float(popt[2]))
        except Exception as e:
            out['qfree_deep5'] = dict(error=str(e))
    # central value + honest error bar
    vals = [out[k]['m_inf'] for k in ('qfree', 'q1', 'q2') if 'm_inf' in out[k]]
    c = out['qfree'].get('m_inf', np.mean(vals))
    out['central'] = float(c)
    out['err'] = float(max(abs(v - c) for v in vals))
    return out


analysis = {}
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()
for i, (key, cell) in enumerate(results.items()):
    lad = sorted(cell['ladder'], key=lambda r: -r['h'])
    h = np.array([r['h'] for r in lad])
    dfc = np.array([r['deficit'] for r in lad])
    slp = np.array([r['slope'] for r in lad])
    G = [r['G'] for r in lad]
    fd = fits(h, dfc)
    fs = fits(h, slp)
    g21 = next(r for r in lad if r['G'] == 21)
    inside = abs(g21['deficit'] - fd['central']) <= fd['err']
    analysis[key] = dict(tau=cell['tau'], gamma=cell['gamma'], G=G,
                         h=list(h), deficit=list(dfc), slope=list(slp),
                         deficit_fits=fd, slope_fits=fs,
                         g21_deficit=g21['deficit'], g21_inside_bar=bool(inside))

    ax = axes[i]
    ax.plot(h, dfc, 'ko', ms=6, label='ladder rungs', zorder=5)
    hh = np.linspace(0, h.max()*1.05, 200)
    if 'm_inf' in fd['qfree']:
        p = fd['qfree']
        ax.plot(hh, p['m_inf'] + p['a']*hh**p['q'], 'b-',
                label=f"q-free (q={p['q']:.2f})")
        ax.plot(0, p['m_inf'], 'b*', ms=12)
    p = fd['q1']
    ax.plot(hh, p['m_inf'] + p['a']*hh, 'r--', label='q=1')
    ax.plot(0, p['m_inf'], 'r*', ms=10)
    p = fd['q2']
    ax.plot(hh, p['m_inf'] + p['a']*hh**2, 'g-.', label='q=2')
    ax.plot(0, p['m_inf'], 'g*', ms=10)
    ax.errorbar([0], [fd['central']], yerr=[fd['err']], fmt='none',
                ecolor='k', elinewidth=2, capsize=5, zorder=6)
    ax.set_title(f"tau={cell['tau']}, gamma={cell['gamma']}\n"
                 f"d_inf={fd['central']:.4f} +/- {fd['err']:.4f}")
    ax.set_xlabel('h = 0.45*sqrt(du)')
    ax.set_ylabel('deficit = 1 - R^2')
    ax.legend(fontsize=8)
for j in range(len(results), 6):
    axes[j].axis('off')
fig.suptitle('K=3 CRRA REE joint-limit: deficit vs h, deep G-ladders with '
             'continuum extrapolation', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{OUT}/extrapolation.png", dpi=150)
json.dump(analysis, open(f"{OUT}/analysis.json", 'w'), indent=2)

# console table
print(f"{'cell':>16} {'rungs':>6} {'d(G21)':>8} {'d_inf':>9} {'+/-':>8} "
      f"{'q_free':>7} {'q_deep5':>8} {'in bar':>7} {'s_inf':>8} {'s_err':>8}")
for key, a in analysis.items():
    fd = a['deficit_fits']; fs = a['slope_fits']
    qd5 = fd.get('qfree_deep5', {}).get('q', float('nan'))
    print(f"{key:>16} {len(a['G']):>6} {a['g21_deficit']:>8.4f} "
          f"{fd['central']:>9.4f} {fd['err']:>8.4f} "
          f"{fd['qfree'].get('q', float('nan')):>7.3f} {qd5:>8.3f} "
          f"{str(a['g21_inside_bar']):>7} {fs['central']:>8.4f} {fs['err']:>8.4f}")
