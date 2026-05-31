"""Point-by-point verification of the u-grid kernel FP on the σ-δ ξ-cube.

For each σ-δ inner cell (i,j,k):
  - ξ-coords (ξ₁, ξ₂_Σ, ξ₃_δ)
  - physical coords (u₁, Σ, δ) and equivalently (u₁, u₂, u₃)
  - P_interp (= u-grid FP trilinear-interp at this cell)
  - Phi(P_interp) (one V11 σ-δ Phi application)
  - residual r = Phi - P_interp
  - |r|

Outputs: CSV with all 11³ = 1331 inner cells, plus summary tables
(worst-25, best-25, residual histogram, per-axis projection).
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import csv
from v11_strict_h0_hardwired import TOT_u, TOT_S, TOT_d, TAU

# Load the two arrays produced by fix_mapping_BC.py
P_sd = np.load(os.path.join(HERE, 'fixed_mapping_P_sd.npy'))   # u-grid interp on σ-δ cube
P_phi = np.load(os.path.join(HERE, 'fixed_mapping_P_phi.npy'))  # V11 Phi(P_sd)

G_FULL = P_sd.shape[0]
INNER_LO, INNER_HI = 1, G_FULL - 1
G_INNER = INNER_HI - INNER_LO

xi_arr = np.linspace(-1.0, 1.0, G_FULL)
safe = np.clip(xi_arr, -0.9999999, 0.9999999)
u_arr = TOT_u * np.arctanh(safe)
S_arr = TOT_S * np.arctanh(safe)
d_arr = TOT_d * np.arctanh(safe)

print(f'σ-δ cube G_FULL={G_FULL}, inner range [{INNER_LO},{INNER_HI})')
print(f'TOT_u={TOT_u}  TOT_S={TOT_S}  TOT_d={TOT_d}  τ={TAU}')

# Build per-cell records (inner only)
records = []
for i in range(INNER_LO, INNER_HI):
    for j in range(INNER_LO, INNER_HI):
        for k in range(INNER_LO, INNER_HI):
            xi1, xi2, xi3 = xi_arr[i], xi_arr[j], xi_arr[k]
            u1, S, dlt = u_arr[i], S_arr[j], d_arr[k]
            u2 = 0.5*(S + dlt); u3 = 0.5*(S - dlt)
            pi = float(P_sd[i,j,k])
            pp = float(P_phi[i,j,k])
            r  = pp - pi
            records.append(dict(i=i, j=j, k=k,
                                xi1=xi1, xi2=xi2, xi3=xi3,
                                u1=u1, Sig=S, delta=dlt, u2=u2, u3=u3,
                                P_interp=pi, Phi=pp, resid=r, abs_resid=abs(r)))

# Save full CSV
csv_path = os.path.join(HERE, 'verify_pointwise_full.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader(); w.writerows(records)
print(f'\nWrote {len(records)} cells to {csv_path}')

# Sort by |residual|
records.sort(key=lambda r: -r['abs_resid'])

def fmt(r):
    return (f"({r['i']:2d},{r['j']:2d},{r['k']:2d})  "
            f"ξ=({r['xi1']:+.3f},{r['xi2']:+.3f},{r['xi3']:+.3f})  "
            f"u=({r['u1']:+5.2f},{r['u2']:+5.2f},{r['u3']:+5.2f})  "
            f"P_int={r['P_interp']:.4f}  Φ={r['Phi']:.4f}  r={r['resid']:+.4f}")

print(f'\n=== WORST-25 cells (largest |Phi - P_interp|) ===')
print(f'{"idx":18}{"ξ-coords":26}{"u-coords":24}{"P_int":>9}{"Φ":>10}{"resid":>10}')
for r in records[:25]:
    print(fmt(r))

print(f'\n=== BEST-25 cells (smallest |Phi - P_interp|) ===')
print(f'{"idx":18}{"ξ-coords":26}{"u-coords":24}{"P_int":>9}{"Φ":>10}{"resid":>10}')
for r in records[-25:]:
    print(fmt(r))

# Stats
abs_res = np.array([r['abs_resid'] for r in records])
res = np.array([r['resid'] for r in records])
P_int = np.array([r['P_interp'] for r in records])
Phi   = np.array([r['Phi'] for r in records])
print(f'\n=== Residual stats over {len(records)} interior cells ===')
print(f'  max |r|   = {abs_res.max():.4e}')
print(f'  mean |r|  = {abs_res.mean():.4e}')
print(f'  rms r     = {np.sqrt((res**2).mean()):.4e}')
print(f'  median |r|= {np.median(abs_res):.4e}')
print(f'  signed mean r = {res.mean():+.4e}  (positive ⇒ Phi pulls UP)')
print(f'  fraction with r>0 = {(res>0).mean():.3f}')
print(f'  fraction with r<0 = {(res<0).mean():.3f}')

# Sign breakdown by P_interp regime
print(f'\n=== Residual sign by P_interp regime ===')
for lo, hi in [(0.0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.0)]:
    mask = (P_int>=lo)&(P_int<hi)
    if mask.sum()==0: continue
    print(f'  P_int∈[{lo:.1f},{hi:.1f}):  n={mask.sum():4d}  '
          f'mean(Phi-P)={res[mask].mean():+.4f}  '
          f'rms={np.sqrt((res[mask]**2).mean()):.4f}  '
          f'mean Phi={Phi[mask].mean():.4f}')

# Residual histogram
print(f'\n=== Histogram of signed residual ===')
edges = np.linspace(-1, 1, 21)
hist, _ = np.histogram(res, bins=edges)
for k in range(len(edges)-1):
    bar = '#' * min(60, hist[k])
    if hist[k] > 0:
        print(f'  [{edges[k]:+.2f},{edges[k+1]:+.2f}):  {hist[k]:5d}  {bar}')

# Projection: mean |r| along each axis
res3 = np.zeros((G_INNER,)*3)
absr3 = np.zeros((G_INNER,)*3)
for r in records:
    ii = r['i']-INNER_LO; jj = r['j']-INNER_LO; kk = r['k']-INNER_LO
    res3[ii,jj,kk] = r['resid']
    absr3[ii,jj,kk] = r['abs_resid']

print(f'\n=== Mean |r| sliced along ξ₁ (i-axis, u₁ direction) ===')
print(f'  i  ξ₁     u₁     mean|r|    max|r|')
for ii in range(G_INNER):
    sl = absr3[ii,:,:]
    print(f'  {ii+INNER_LO:2d} {xi_arr[ii+INNER_LO]:+.3f} {u_arr[ii+INNER_LO]:+6.2f}  '
          f'{sl.mean():.4f}    {sl.max():.4f}')

print(f'\n=== Mean |r| sliced along ξ₂ (j-axis, Σ direction) ===')
print(f'  j  ξ₂     Σ      mean|r|    max|r|')
for jj in range(G_INNER):
    sl = absr3[:,jj,:]
    print(f'  {jj+INNER_LO:2d} {xi_arr[jj+INNER_LO]:+.3f} {S_arr[jj+INNER_LO]:+6.2f}  '
          f'{sl.mean():.4f}    {sl.max():.4f}')

print(f'\n=== Mean |r| sliced along ξ₃ (k-axis, δ direction) ===')
print(f'  k  ξ₃     δ      mean|r|    max|r|')
for kk in range(G_INNER):
    sl = absr3[:,:,kk]
    print(f'  {kk+INNER_LO:2d} {xi_arr[kk+INNER_LO]:+.3f} {d_arr[kk+INNER_LO]:+6.2f}  '
          f'{sl.mean():.4f}    {sl.max():.4f}')

# Where are the worst cells concentrated?
print(f'\n=== Anatomy of the worst 25 cells ===')
worst = records[:25]
i_vals = [r['i'] for r in worst]; j_vals = [r['j'] for r in worst]; k_vals = [r['k'] for r in worst]
print(f'  i (ξ₁/u₁) values: {sorted(set(i_vals))}')
print(f'  j (ξ₂/Σ)  values: {sorted(set(j_vals))}')
print(f'  k (ξ₃/δ)  values: {sorted(set(k_vals))}')
print(f'  signed: positive r count = {sum(1 for r in worst if r["resid"]>0)} / negative = {sum(1 for r in worst if r["resid"]<0)}')

# Summary JSON
summary = dict(
    n_cells=len(records),
    max_abs_resid=float(abs_res.max()),
    mean_abs_resid=float(abs_res.mean()),
    rms_resid=float(np.sqrt((res**2).mean())),
    signed_mean=float(res.mean()),
    frac_positive=float((res>0).mean()),
    worst_cell_idx=[int(records[0]['i']), int(records[0]['j']), int(records[0]['k'])],
    worst_cell_u=[float(records[0]['u1']), float(records[0]['u2']), float(records[0]['u3'])],
    worst_cell_P_interp=float(records[0]['P_interp']),
    worst_cell_Phi=float(records[0]['Phi']),
    worst_cell_resid=float(records[0]['resid']),
)
json.dump(summary, open(os.path.join(HERE,'verify_pointwise_summary.json'),'w'), indent=2)
print('\nsaved summary & full CSV')
