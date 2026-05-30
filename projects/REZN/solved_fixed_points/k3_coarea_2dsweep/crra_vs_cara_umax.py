"""Side-by-side UMAX-test: CARA collapses, CRRA persists.

This is the decisive figure that vindicates the paper's headline:
  - CARA without noise traders -> FR (deficit -> 0) when the box is open enough
    to escape the FR clip-saturation. Same operator, same precision range:
    deficit drops 6 orders of magnitude as UMAX 4 -> 12.
  - CRRA at gamma=0.1, tau=2 retains a deficit ~0.3-0.4 across ALL UMAX,
    even growing slightly. The gap is the wealth-curvature (Jensen) effect,
    intrinsic to CRRA and not a numerical artifact.

Reads cara_umax_diag.json (G=9) and crra_umax_diag.json (G=11) at tau=2.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
KCARA = os.path.join(os.path.dirname(HERE), 'k3_cara', 'cara_umax_diag.json')

cara = json.load(open(KCARA))['A_fixedG']
crra = json.load(open(os.path.join(HERE, 'crra_umax_diag.json')))['rows']
try:
    crra17 = json.load(open(os.path.join(HERE, 'crra_umax_diag_G17.json')))['rows']
except FileNotFoundError:
    crra17 = []

uc = np.array([r['umax'] for r in cara], float); dc = np.array([r['deficit'] for r in cara], float)
ur = np.array([r['umax'] for r in crra], float); dr = np.array([r['deficit'] for r in crra], float)
u17 = np.array([r['umax'] for r in crra17], float) if crra17 else np.array([])
d17 = np.array([r['deficit'] for r in crra17], float) if crra17 else np.array([])

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), dpi=140)

# (left) shared linear scale: CRRA stays high, CARA hugs zero
if u17.size:
    ax[0].plot(u17, d17, 'D-', color='C0', lw=2.5, ms=10, label='CRRA  γ=0.1, τ=2, G=17 (headline)')
ax[0].plot(ur, dr, 's-', color='C2', lw=2.5, ms=10, label='CRRA  γ=0.1, τ=2, G=11')
ax[0].plot(uc, dc, 'o-', color='C3', lw=2.5, ms=10, label='CARA  τ=2, G=9  (Hellwig: FR, deficit→0)')
ax[0].axhline(0, color='k', ls=':', alpha=0.5)
ax[0].set_xlabel('box half-width UMAX'); ax[0].set_ylabel('revelation deficit 1−R²')
ax[0].set_title('CRRA gap is intrinsic; CARA "gap" was a box-clip artifact')
ax[0].legend(fontsize=10, loc='center right'); ax[0].grid(ls=':')
for u, d in zip(uc, dc):
    ax[0].annotate(f'{d:.1e}', (u, d), textcoords='offset points', xytext=(6, -12), fontsize=8, color='C3')
for u, d in zip(ur, dr):
    ax[0].annotate(f'{d:.3f}', (u, d), textcoords='offset points', xytext=(6, 8), fontsize=8, color='C2')

# (right) log y axis so the CARA collapse is visible
if u17.size:
    ax[1].semilogy(u17, d17, 'D-', color='C0', lw=2.5, ms=10, label='CRRA  γ=0.1, τ=2, G=17')
ax[1].semilogy(ur, dr, 's-', color='C2', lw=2.5, ms=10, label='CRRA  γ=0.1, τ=2, G=11')
ax[1].semilogy(uc, np.maximum(dc, 1e-15), 'o-', color='C3', lw=2.5, ms=10, label='CARA  τ=2, G=9')
ax[1].set_xlabel('UMAX'); ax[1].set_ylabel('deficit  (log)')
ax[1].set_title('Log scale: CARA spans 6 orders; CRRA flat')
ax[1].legend(fontsize=10); ax[1].grid(ls=':', which='both', alpha=0.5)

plt.suptitle('Same UMAX-collapse test, opposite outcomes:  CRRA has a genuine PR equilibrium (the deficit survives); CARA does not.', weight='bold', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'crra_vs_cara_umax.png'), dpi=140, bbox_inches='tight'); plt.close()
print('wrote crra_vs_cara_umax.png')
