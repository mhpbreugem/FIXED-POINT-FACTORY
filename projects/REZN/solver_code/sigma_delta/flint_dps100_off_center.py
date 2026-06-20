"""Test flint dps=100 Phi at OFF-CENTER cells where float64 plateau lives.
Specifically: corner (8,8,8), edge (8,4,4), midface (4,8,4), and a generic
asymmetric cell (1,3,7). Compare flint vs float64 |Phi - FR|.
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
flint.ctx.prec = 333
print(f'flint prec: {flint.ctx.prec} bits (~{int(flint.ctx.prec*0.301)} dec digits)')

# Reuse most of flint_dps100_one_cell — re-import its definitions
from flint_dps100_one_cell import (arb, UMAX, TAU, GAMMA, G, NQ, SUB, ui, h,
    f_signal, natural_spline_M, spline_eval, spline_roots,
    gnodes, gweights, slice_evidence, bayes, crra_clear_3, P_FR)

# Also load float64 PhP from existing reference
sys.path.insert(0, '/tmp')
import hfree_operator as H
ui_f = np.linspace(-4, 4, G)
U1, U2, U3 = np.meshgrid(ui_f, ui_f, ui_f, indexing='ij')
T_f = 2.0*(U1+U2+U3)
P_FR_f = 1.0/(1.0+np.exp(-T_f))
gnodes_f, gweights_f = H.gauss_legendre(NQ, -4.0, 4.0)
print('Computing float64 Phi(FR) once...', flush=True)
PhP_f64 = H.phi_hfree(P_FR_f, ui_f, gnodes_f, gweights_f, np.full(3,2.0), np.full(3,10.0), np.full(3,1.0), SUB)
F_f64 = PhP_f64 - P_FR_f
print(f'float64 ||F||_∞ = {float(np.max(np.abs(F_f64))):.4e}')
idx_max = np.unravel_index(np.argmax(np.abs(F_f64)), F_f64.shape)
print(f'  worst cell: {idx_max} u=({ui_f[idx_max[0]]:+.2f},{ui_f[idx_max[1]]:+.2f},{ui_f[idx_max[2]]:+.2f})')
print(f'  Phi[worst]-FR[worst] = {F_f64[idx_max]:+.4e}', flush=True)

# Now test flint at the worst cell + a few others
test_cells = [tuple(idx_max), (0,0,0), (8,8,8), (8,4,4), (4,8,4), (1,3,7)]
results = []
for (i,j,k) in test_cells:
    print(f'\nCell ({i},{j},{k}) u=({float(ui[i]):+.2f},{float(ui[j]):+.2f},{float(ui[k]):+.2f})...', flush=True)
    ts = time.time()
    p_cell = P_FR[i][j][k]
    S_a = [[P_FR[i][a][b] for b in range(G)] for a in range(G)]
    A0a, A1a = slice_evidence(S_a, p_cell, TAU, TAU)
    mu0 = bayes(ui[i], TAU, A0a, A1a)
    S_b = [[P_FR[a][j][b] for b in range(G)] for a in range(G)]
    A0b, A1b = slice_evidence(S_b, p_cell, TAU, TAU)
    mu1 = bayes(ui[j], TAU, A0b, A1b)
    S_c = [[P_FR[a][b][k] for b in range(G)] for a in range(G)]
    A0c, A1c = slice_evidence(S_c, p_cell, TAU, TAU)
    mu2 = bayes(ui[k], TAU, A0c, A1c)
    p_new = crra_clear_3(mu0, mu1, mu2, GAMMA)
    diff_arb = p_new - p_cell
    diff_f64 = F_f64[i,j,k]
    print(f'  arb Phi-FR = {float(diff_arb):+.4e} (radius {float(p_new.rad()):.1e})  '
          f'float64 = {diff_f64:+.4e}  ({time.time()-ts:.0f}s)')
    results.append(dict(i=i, j=j, k=k, u=[float(ui[i]), float(ui[j]), float(ui[k])],
                          p_FR=float(p_cell), arb_diff=float(diff_arb),
                          arb_rad=float(p_new.rad()), float64_diff=float(diff_f64)))
    json.dump(results, open(os.path.join(HERE,'flint_dps100_off_center.json'),'w'), indent=2, default=str)

# Verdict
print('\n=== VERDICT ===')
max_abs_arb = max(abs(r['arb_diff']) for r in results)
max_abs_f64 = max(abs(r['float64_diff']) for r in results)
print(f'  max |Phi-FR|_arb = {max_abs_arb:.3e}')
print(f'  max |Phi-FR|_f64 = {max_abs_f64:.3e}')
if max_abs_arb > 1e-30 and abs(max_abs_arb - max_abs_f64) / max_abs_f64 < 0.01:
    print('  arithmetic agrees with float64: discretization, not precision, is the floor')
elif max_abs_arb < 1e-30:
    print('  flint gives essentially zero — float64 floor IS arithmetic, flint cracks it')
else:
    print('  partial improvement; arithmetic matters but discretization also caps')
print('saved')
