"""Cross-check: take each known u-grid PR fixed point, interpolate onto the
CDF-cube ζ-grid (G=13, NQ=64), apply phi_cdf_numba once, and report ferr.

If a u-grid FP is also a CDF-cube FP, ferr should be ~0 (within discretization
error). If it's NOT, ferr ~ O(1) showing the FPs are different operators in
different coordinate frames.

Tests:
  (a) k3_coarea_sweep G=17 kernel h≈0.32  → PR slope=0.18 deficit=0.28
  (b) k3_hfree_smooth G=9 h=0             → PR slope=0.36 deficit=0.17
  (c) k3_hfree_smooth G=9 h=0 NK reverify → same as (b), tighter ||F||
  (d) k3_hfree_smooth G=13 h=0 BEST       → PR slope=0.47 deficit=0.11
  (e) ugrid_h0 (linear scan, h=0)         → oscillating, near-FR
  (f) ugrid_h0_cubic (Hermite, h=0)       → oscillating, near-FR
  (g) cdf_cube_NB_g0.1 at G=9              → CDF-cube own FP (sanity check)
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cdf_cube_numba import (phi_cdf_numba, build_zeta_grid, gauss_legendre,
                              metrics, TAU, GAMMA, TAB_Z, TAB_U, TAB_DUDZ)

# Setup CDF-cube grid (match sweep config)
G = 13; NQ = 64
zeta_arr, u_arr_cdf, h, z0 = build_zeta_grid(G)
gl_z_nodes, gl_z_weights = gauss_legendre(NQ, zeta_arr[0], zeta_arr[-1])
print(f'CDF-cube target: G={G}, NQ={NQ}, ζ∈[{zeta_arr[0]:.4f},{zeta_arr[-1]:.4f}]')
print(f'  u_arr_cdf range: [{u_arr_cdf.min():.3f}, {u_arr_cdf.max():.3f}]')

# JIT warmup
print('JIT warmup...', flush=True)
import time; t = time.time()
P_warm = np.full((G,G,G), 0.5)
_ = phi_cdf_numba(P_warm, zeta_arr, h, z0, gl_z_nodes, gl_z_weights,
                    TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU)
print(f'  done {time.time()-t:.1f}s')

def interp_to_cdf_cube(P_ugrid, u_source_grid):
    """Trilinear-interp u-grid FP onto CDF-cube u_arr_cdf."""
    interp = RegularGridInterpolator((u_source_grid, u_source_grid, u_source_grid),
                                        P_ugrid, bounds_error=False, fill_value=None,
                                        method='linear')
    U1, U2, U3 = np.meshgrid(u_arr_cdf, u_arr_cdf, u_arr_cdf, indexing='ij')
    # Clamp into source range
    lo, hi = u_source_grid[0], u_source_grid[-1]
    U1c = np.clip(U1, lo, hi); U2c = np.clip(U2, lo, hi); U3c = np.clip(U3, lo, hi)
    pts = np.stack([U1c.ravel(), U2c.ravel(), U3c.ravel()], axis=1)
    P_interp = interp(pts).reshape((G, G, G))
    return P_interp

# Load all u-grid FPs at γ=0.1
def load_fp(path, G_src, label):
    P = np.load(path)
    return dict(P=P, G_src=G_src, u_src=np.linspace(-4.0, 4.0, G_src), label=label)

fps = [
    load_fp('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_sweep/P_g0.1_t2.0.npy', 17, '(a) kernel co-area G=17 h≈0.32 (PR slope=0.18)'),
    load_fp('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth/P_nailed_G9.npy', 9, '(b) hfree_smooth G=9 h=0 (PR slope=0.36)'),
    load_fp('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth/P_nailed_G9_NK.npy', 9, '(c) hfree_smooth G=9 h=0 NK reverify (||F||=3e-11)'),
    load_fp('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth/P_nailed_G13_BEST.npy', 13, '(d) hfree_smooth G=13 h=0 BEST (PR slope=0.47 ||F||=0.076)'),
    load_fp('/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta/ugrid_h0_FP.npy', 17, '(e) u-grid h=0 linear scan G=17 (oscillating, near-FR)'),
    load_fp('/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta/ugrid_h0_cubic_FP.npy', 17, '(f) u-grid h=0 Hermite cubic G=17 (oscillating, near-FR)'),
]
# Self-consistency check: CDF-cube own FP
try:
    P_cdf_own = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/sigma_delta/cdf_cube_NB_g0.1.npy')
    G_self = P_cdf_own.shape[0]
    # this was at G=9, not G=13. Skip if shape doesn't match
    if P_cdf_own.shape == (G, G, G):
        fps.append(dict(P=P_cdf_own, G_src=G, u_src=u_arr_cdf, label=f'(g) cdf_cube γ=0.1 own FP G={G} (sanity check)'))
    else:
        fps.append(dict(P=P_cdf_own, G_src=G_self, u_src=u_arr_cdf if G_self==G else build_zeta_grid(G_self)[1], label=f'(g) cdf_cube γ=0.1 own FP G={G_self}'))
except Exception as e:
    print(f'cdf_own load failed: {e}')

# For each FP: interpolate, apply phi, compute ferr
results = []
print('\n=== CROSS-CHECK: u-grid FPs as CDF-cube FPs ===\n')
for fp in fps:
    print(f'\n{fp["label"]}')
    print(f'  source: G_src={fp["G_src"]}, P range [{fp["P"].min():.4f}, {fp["P"].max():.4f}]')
    # Interpolate
    P_interp = interp_to_cdf_cube(fp['P'], fp['u_src'])
    print(f'  interp to CDF-cube G={G}: P range [{P_interp.min():.4f}, {P_interp.max():.4f}]')
    m_src = metrics(P_interp, u_arr_cdf)
    print(f'  interp metrics (CDF cells): slope={m_src["slope_T"]:.4f} deficit={m_src["deficit"]:.4f} d_FR={m_src["d_FR"]:.4f}')

    # Apply Phi
    ts = time.time()
    P_after = phi_cdf_numba(P_interp.copy(), zeta_arr, h, z0,
                              gl_z_nodes, gl_z_weights,
                              TAB_Z, TAB_U, TAB_DUDZ, GAMMA, TAU)
    ferr = float(np.max(np.abs(P_after - P_interp)))
    rms = float(np.sqrt(np.mean((P_after - P_interp)**2)))
    m_after = metrics(P_after, u_arr_cdf)
    print(f'  Phi applied ({time.time()-ts:.1f}s):')
    print(f'    after metrics: slope={m_after["slope_T"]:.4f} deficit={m_after["deficit"]:.4f} d_FR={m_after["d_FR"]:.4f}')
    print(f'    >>> ferr = max|Phi(P) - P| = {ferr:.4e}, rms = {rms:.4e} <<<')
    results.append(dict(label=fp['label'], G_src=fp['G_src'],
                          interp_slope=m_src['slope_T'], interp_deficit=m_src['deficit'],
                          interp_d_FR=m_src['d_FR'],
                          after_slope=m_after['slope_T'], after_deficit=m_after['deficit'],
                          after_d_FR=m_after['d_FR'],
                          ferr=ferr, rms=rms))

# Summary
print('\n\n=== SUMMARY ===')
print(f'{"label":85} {"ferr":>10} {"rms":>10}')
for r in results:
    verdict = 'FP-AGREEMENT' if r['ferr'] < 0.05 else ('NEAR-AGREE' if r['ferr'] < 0.15 else 'DIFFERENT')
    print(f'{r["label"]:85} {r["ferr"]:>10.4e} {r["rms"]:>10.4e}  [{verdict}]')

json.dump({'G_cdf':G, 'NQ_cdf':NQ, 'results':results,
            'verdict':'A u-grid FP is also a CDF-cube FP iff ferr~0; otherwise the two operators have different FPs.'},
          open(os.path.join(HERE,'check_ugrid_FPs_on_cdf_cube.json'),'w'), indent=2, default=str)
print('\nsaved check_ugrid_FPs_on_cdf_cube.json')
