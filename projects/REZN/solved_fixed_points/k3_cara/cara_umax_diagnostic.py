"""CARA deficit vs UMAX (box half-width) at FIXED G: if the residual is driven by
the box-edge clip on p (analytic FR = sigmoid(tau*Sigma u) saturates clip when
|tau*Sigma u| >> log(1/eps_clip) = log(1e9) = 20.7, i.e. when 3*tau*UMAX > 20.7 ->
UMAX > 3.45 at tau=2), pushing UMAX out should kill the deficit.

We do this at fixed G_inner = 9 and vary UMAX in {4, 6, 8, 12}. Constant du to keep
the kernel band sensible, so G grows with UMAX (we use G_inner = 2*UMAX/du + 1 with
du = 4/8 = 0.5 -> G=9 at UMAX=4, etc.). Two protocols:

  (A) FIXED G=9 (du grows with UMAX): tests pure UMAX effect with fixed cell count.
  (B) FIXED du=0.5 (G grows with UMAX): tests UMAX with constant resolution.

If (A) collapses the deficit and (B) does too, the artifact is the box-edge clip,
not the resolution. If only (B) does, it's the resolution. If neither, it's deeper.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from cara_operator import phi_cara, deficit

TAU = 2.0; a = 1.0; W = 1.0

def run(G_inner, umax, model='logodds', halo='FR'):
    pad = 2
    du = 2 * umax / (G_inner - 1)
    h = 0.45 * du ** 0.5
    G_full = G_inner + 2 * pad
    uf = np.array([-umax + (q - pad) * du for q in range(G_full)])
    lo, hi = pad, pad + G_inner
    slc = (slice(lo, hi),) * 3
    U = np.meshgrid(uf, uf, uf, indexing='ij')
    if halo == 'FR':
        P_FR = np.clip(1.0 / (1.0 + np.exp(-TAU * (U[0] + U[1] + U[2]))), 1e-9, 1 - 1e-9)
        halo_full = P_FR.copy(); Pf = P_FR.copy()
    else:  # NL
        from cara_operator import init_no_learning_cara
        tv = np.full(3, TAU); av = np.full(3, a); Wv = np.full(3, W)
        halo_full = init_no_learning_cara(uf, tv, av, Wv, model='logodds')
        Pf = halo_full.copy()
    tv = np.full(3, TAU); av = np.full(3, a); Wv = np.full(3, W)
    ui = uf[lo:hi]; U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij'); T = TAU * (U1 + U2 + U3)
    for it in range(400):
        Pn = phi_cara(Pf, uf, lo, hi, tv, av, Wv, h, model=model)
        Pn_h = halo_full.copy(); Pn_h[slc] = Pn[slc]
        Finf = float(np.max(np.abs(Pn_h[slc] - Pf[slc])))
        Pf = Pn_h
        if Finf < 1e-13:
            break
    return float(deficit(Pf[slc], T)), Finf, it + 1, du, h

# (A) FIXED G=9, varying UMAX
print("(A) FIXED G_inner=9, varying UMAX (FR halo)", flush=True)
A_rows = []
for umax in [4, 6, 8, 12]:
    t = time.time()
    d, F, it, du, h = run(9, umax, halo='FR')
    dt = time.time() - t
    rec = dict(G=9, umax=umax, du=float(du), h=float(h), deficit=d, Finf=F, iters=it, sec=round(dt, 1))
    A_rows.append(rec)
    print(f"  G=9 UMAX={umax:2d} du={du:.3f} h={h:.3f}: deficit={d:.4e}  ||F||={F:.1e}  ({it} it, {dt:.0f}s)", flush=True)

# (B) FIXED du = 0.5, varying UMAX -> G grows
print("\n(B) FIXED du=0.5, G_inner = 2*UMAX/du + 1 (FR halo)", flush=True)
B_rows = []
for umax in [4, 6, 8]:
    G = int(round(2*umax/0.5)) + 1
    t = time.time()
    d, F, it, du, h = run(G, umax, halo='FR')
    dt = time.time() - t
    rec = dict(G=G, umax=umax, du=float(du), h=float(h), deficit=d, Finf=F, iters=it, sec=round(dt, 1))
    B_rows.append(rec)
    print(f"  G={G:2d} UMAX={umax:2d} du={du:.3f} h={h:.3f}: deficit={d:.4e}  ||F||={F:.1e}  ({it} it, {dt:.0f}s)", flush=True)

json.dump({'tau': TAU, 'A_fixedG': A_rows, 'B_fixed_du': B_rows,
           'hypothesis': 'box-edge clip on p saturates analytic FR when 3*tau*UMAX > log(1e9) ~ 20.7; '
                          'pushing UMAX out kills the discrete-FP deficit -> confirms the discretization artifact is the box-clip.'},
          open(os.path.join(HERE, 'cara_umax_diag.json'), 'w'), indent=2)
print('\nwrote cara_umax_diag.json', flush=True)
