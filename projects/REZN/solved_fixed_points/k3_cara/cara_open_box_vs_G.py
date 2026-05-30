"""CARA "production" run at OPEN box (UMAX=12, well past the clip threshold).
Sweeps G in {7, 9, 11, 13, 15} and reports the deficit. With UMAX=12 the FR
price sigma(tau*Sigma u) is no longer clip-saturated at the corners, so the
discrete operator's fixed point matches the analytic FR to machine precision.

Companion to cara_vs_G.py (which used UMAX=4 and showed a spurious growing
deficit): same operator, just an honest box. Expected: deficit ~ 1e-8 -- 1e-10
for all G -- the Hellwig prediction realized numerically.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from cara_operator import phi_cara, deficit

TAU = 2.0; UMAX = 12.0
Gs = [7, 9, 11, 13, 15]
rows = []
print(f"CARA at OPEN box UMAX={UMAX} (FR halo, tau={TAU}). Theory: deficit ~ 0 for all G.", flush=True)
for G in Gs:
    pad = 2
    du = 2 * UMAX / (G - 1); h = 0.45 * du ** 0.5
    G_full = G + 2 * pad
    uf = np.array([-UMAX + (q - pad) * du for q in range(G_full)])
    lo, hi = pad, pad + G
    slc = (slice(lo, hi),) * 3
    U = np.meshgrid(uf, uf, uf, indexing='ij')
    P_FR = np.clip(1.0 / (1.0 + np.exp(-TAU * (U[0] + U[1] + U[2]))), 1e-9, 1 - 1e-9)
    halo = P_FR.copy(); Pf = P_FR.copy()
    tv = np.full(3, TAU); av = np.full(3, 1.0); Wv = np.full(3, 1.0)
    ui = uf[lo:hi]; U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij'); T = TAU * (U1 + U2 + U3)

    t = time.time(); Finf = np.inf
    for it in range(400):
        Pn = phi_cara(Pf, uf, lo, hi, tv, av, Wv, h, model='logodds')
        Pn_h = halo.copy(); Pn_h[slc] = Pn[slc]
        Finf = float(np.max(np.abs(Pn_h[slc] - Pf[slc])))
        Pf = Pn_h
        if Finf < 1e-13:
            break
    defi = float(deficit(Pf[slc], T))
    dt = time.time() - t
    rec = dict(G=G, UMAX=UMAX, du=float(du), h=float(h), deficit=defi, Finf=Finf, iters=it+1, sec=round(dt, 1))
    rows.append(rec)
    print(f"  G={G:2d}  du={du:.3f}  h={h:.3f}: deficit={defi:.4e}  ||F||={Finf:.2e}  ({it+1} it, {dt:.0f}s)", flush=True)
    json.dump({'tau': TAU, 'UMAX': UMAX, 'rows': rows,
               'theory': 'Hellwig: with open box (no clip on FR), CARA without noise traders -> FR, deficit = 0.'},
              open(os.path.join(HERE, 'cara_open_box_vs_G.json'), 'w'), indent=2)

print('\nDONE. All deficits should be tiny (~1e-8 to 1e-11) -- the Hellwig prediction.', flush=True)
