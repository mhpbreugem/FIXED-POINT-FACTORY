"""Line plots of 1-R^2 (revelation deficit) from the consistent-operator 2D sweep --
the Jensen-gap law made visible: log-log deficit vs gamma (slope -> -1 at high gamma);
deficit vs tau (rising with signal precision). Reads sweep2d.json, no recompute."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.cm as cm
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, 'sweep2d.json')))['rows']
G = sorted(set(x['gamma'] for x in r)); T = sorted(set(x['tau'] for x in r))

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), dpi=140)

# (a) deficit vs gamma at each tau (log-log; visualize 1/gamma scaling)
cols = cm.viridis(np.linspace(0.15, 0.9, len(T)))
for tau, col in zip(T, cols):
    g = np.array([x['gamma'] for x in r if x['tau'] == tau])
    d = np.array([x['deficit'] for x in r if x['tau'] == tau])
    idx = np.argsort(g); g = g[idx]; d = d[idx]
    ax[0].loglog(g, d, 'o-', color=col, lw=1.8, ms=6, label=f'τ={tau}')
# guideline 1/gamma
g_ref = np.array([G[0], G[-1]]); ax[0].loglog(g_ref, 0.3 / g_ref, 'k--', alpha=0.4, label='∝ 1/γ guide')
ax[0].set_xlabel('risk aversion γ'); ax[0].set_ylabel('revelation deficit  1−R²')
ax[0].set_title('1−R² vs γ (Jensen-gap law: deficit ∝ Var(m)/γ)')
ax[0].legend(fontsize=9); ax[0].grid(ls=':', which='both', alpha=0.5)

# (b) deficit vs tau at each gamma (log-x; rises with signal precision)
cols2 = cm.plasma(np.linspace(0.1, 0.9, len(G)))
for gam, col in zip(G, cols2):
    t = np.array([x['tau'] for x in r if x['gamma'] == gam])
    d = np.array([x['deficit'] for x in r if x['gamma'] == gam])
    idx = np.argsort(t); t = t[idx]; d = d[idx]
    ax[1].semilogx(t, d, 's-', color=col, lw=1.6, ms=6, label=f'γ={gam}')
ax[1].set_xlabel('signal precision τ'); ax[1].set_ylabel('revelation deficit  1−R²')
ax[1].set_title('1−R² vs τ (gap grows with belief dispersion)')
ax[1].legend(fontsize=8, ncol=2); ax[1].grid(ls=':', which='both', alpha=0.5)

plt.suptitle('Genuine K=3 CRRA PR equilibrium (consistent co-area operator, G=17, all nailed):  1−R² across (γ,τ)', weight='bold')
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'deficit_lineplots.png'), dpi=140, bbox_inches='tight'); plt.close()
print('wrote deficit_lineplots.png')
