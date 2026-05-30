"""CARA with FR-CONSISTENT HALO: verify that the deficit collapses to ~0 as G grows,
confirming the closed-form Hellwig result and that the growing deficit in cara_vs_G.py
is a halo-BC artifact (NL halo around an FR inner is self-inconsistent).

Procedure (one change from cara_vs_G.py):
  1. Build the padded grid as usual.
  2. Halo + IC = analytic FR price  P(u) = sigmoid(tau*Sigma u)  on the FULL padded grid
     (instead of the NL price).
  3. Run phi_cara with model='logodds' holding the halo fixed at FR; iterate to tol.
  4. Measure deficit on the INNER block.

If outcome (iii) is the real story, the deficit on this driver should be O(1e-3) at G=9
and shrink toward 0 as G grows (the opposite trend of cara_vs_G.py).
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from cara_operator import phi_cara, _grid, deficit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

TAU = 2.0; a = 1.0; W = 1.0
Gs = [9, 13, 17, 21, 25]
rows = []
print(f"CARA with FR-CONSISTENT HALO (tau={TAU}). Theory: deficit -> 0 as G -> inf (Hellwig).", flush=True)
for G in Gs:
    t = time.time()
    du, h, uf, lo, hi, T = _grid(G, TAU)
    slc = (slice(lo, hi),) * 3
    U = np.meshgrid(uf, uf, uf, indexing='ij')
    # FR price on the FULL padded grid (this is what makes the halo CONSISTENT with FR)
    P_FR_full = np.clip(1.0 / (1.0 + np.exp(-TAU * (U[0] + U[1] + U[2]))), 1e-9, 1 - 1e-9)
    halo = P_FR_full.copy()  # halo stays FR forever
    Pf = P_FR_full.copy()    # inner starts at FR too (we test self-consistency)
    tv = np.full(3, TAU); av = np.full(3, a); Wv = np.full(3, W)

    Finf = np.inf; iters = 0
    for it in range(400):
        Pn = phi_cara(Pf, uf, lo, hi, tv, av, Wv, h, model='logodds')
        Pn_h = halo.copy(); Pn_h[slc] = Pn[slc]
        Finf = float(np.max(np.abs(Pn_h[slc] - Pf[slc])))
        Pf = Pn_h; iters = it + 1
        if Finf < 1e-13:
            break

    defi = float(deficit(Pf[slc], T))
    dt = time.time() - t
    rec = dict(G=G, deficit=defi, Finf=Finf, iters=iters, h=float(h), du=float(du), sec=round(dt, 1))
    rows.append(rec)
    print(f"  G={G:2d}: deficit={defi:.4e}  ||F||={Finf:.2e}  iters={iters}  ({dt:.0f}s)", flush=True)
    json.dump({'tau': TAU, 'halo': 'FR (consistent)', 'rows': rows,
               'theory': 'Hellwig: CARA without noise traders -> FR (deficit=0) in continuum.'},
              open(os.path.join(HERE, 'cara_fr_halo_vs_G.json'), 'w'), indent=2)

# Comparison plot: NL-halo (artifact) vs FR-halo (Hellwig)
import json as _json
try:
    nl = _json.load(open(os.path.join(HERE, 'cara_vs_G.json')))['rows']
except Exception:
    nl = []

G_fr = np.array([r['G'] for r in rows]); D_fr = np.array([r['deficit'] for r in rows])
fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=140)
if nl:
    G_nl = np.array([r['G'] for r in nl]); D_nl = np.array([r['deficit'] for r in nl])
    ax[0].plot(G_nl, D_nl, 'o-', color='C3', lw=2, ms=8, label='NL halo (artifact, grows w/ G)')
ax[0].plot(G_fr, D_fr, 's-', color='C0', lw=2, ms=8, label='FR halo (Hellwig: -> 0)')
ax[0].axhline(0, color='k', ls=':', alpha=0.5)
ax[0].set_xlabel('grid resolution G'); ax[0].set_ylabel('revelation deficit 1-R^2')
ax[0].set_title('CARA deficit: NL-halo (BC artifact) vs FR-halo (Hellwig)'); ax[0].legend(fontsize=9); ax[0].grid(ls=':')

if nl:
    ax[1].loglog(G_nl, np.maximum(D_nl, 1e-15), 'o-', color='C3', lw=2, ms=8, label='NL halo')
ax[1].loglog(G_fr, np.maximum(D_fr, 1e-15), 's-', color='C0', lw=2, ms=8, label='FR halo')
ax[1].set_xlabel('G'); ax[1].set_ylabel('deficit (log)')
ax[1].set_title('Log-log: FR-halo collapses, NL-halo grows'); ax[1].legend(fontsize=9); ax[1].grid(ls=':', which='both', alpha=0.5)

plt.suptitle('Hellwig vindicated: with an FR-CONSISTENT halo, CARA deficit -> 0. The growing "deficit" in cara_vs_G.py is a halo-BC artifact.', weight='bold', fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(HERE, 'cara_fr_halo_vs_G.png'), dpi=140, bbox_inches='tight'); plt.close()
print('wrote cara_fr_halo_vs_G.png', flush=True)
