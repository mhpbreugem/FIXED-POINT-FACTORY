"""Plot the UMAX-collapse of the CARA "deficit": at fixed G=9, varying the box
half-width UMAX from 4 -> 12, the residual deficit drops from 3.8e-3 to 7e-9
(six orders of magnitude). This is the numerical confirmation of Hellwig
(the analytic CARA equilibrium is FR; the residual was 100% the box-clip on FR)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, 'cara_umax_diag.json')))
A = J['A_fixedG']

umax = np.array([r['umax'] for r in A], float)
defi = np.array([r['deficit'] for r in A], float)

fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=140)

# (left) deficit vs UMAX with the clip-saturation threshold marker
ax[0].semilogy(umax, defi, 'o-', color='C0', lw=2, ms=10, label='CARA deficit at G=9 (FR halo)')
thresh = np.log(1e9) / (3 * 2.0)  # ~3.45, where 3*tau*UMAX = log(1/clip)
ax[0].axvline(thresh, color='C3', ls='--', lw=1.5, label=f'clip threshold UMAX = log(1/eps)/3τ ≈ {thresh:.2f}')
ax[0].axhline(0, color='k', ls=':', alpha=0.5)
ax[0].set_xlabel('box half-width UMAX'); ax[0].set_ylabel('CARA deficit (log)')
ax[0].set_title('Hellwig vindicated: CARA deficit collapses past the clip threshold')
ax[0].legend(fontsize=9, loc='upper right'); ax[0].grid(ls=':', which='both', alpha=0.5)
for u, d in zip(umax, defi):
    ax[0].annotate(f'{d:.1e}', (u, d), textcoords='offset points', xytext=(8, 8), fontsize=9)

# (right) physical interpretation: at the box edges, |τΣu| = 3·τ·UMAX. When this exceeds
# log(1/clip)=log(1e9)≈20.7, the analytic FR price σ(τΣu) is hard-clipped, contaminating
# the Picard attractor. Plot the maximum |argument| of σ at the box corner.
arg = 3 * 2.0 * umax
ax[1].plot(umax, arg, 'o-', color='C2', lw=2, ms=10, label='|3τΣu| at box corner')
ax[1].axhline(np.log(1e9), color='C3', ls='--', lw=1.5, label='clip activates at |arg| = log(1/ε)')
ax[1].set_xlabel('UMAX'); ax[1].set_ylabel('max |arg of σ| at box corner')
ax[1].set_title('At UMAX ≥ 4, the analytic FR price is clip-saturated at the corners\n→ the discrete operator sees a wrong p there → spurious "deficit"')
ax[1].legend(fontsize=9, loc='upper left'); ax[1].grid(ls=':')

plt.suptitle('CARA without noise traders → fully revealing (Hellwig). The numerical "deficit" was the box-clip on σ(τΣu).', weight='bold', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'cara_umax_collapse.png'), dpi=140, bbox_inches='tight'); plt.close()
print('wrote cara_umax_collapse.png')
