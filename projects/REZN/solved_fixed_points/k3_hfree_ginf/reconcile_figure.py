"""Reconciliation: does the h-free (no-kernel) method converge to the same
continuum PR deficit as the kernel (co-area) method as G->inf?
Overlays both deficit-vs-G trends + Richardson extrapolations. Run after the
robust h-free limit finishes (reads ginf_robust.json + kernel report.json)."""
import json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BASE = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points'
OUT = os.path.join(BASE, 'k3_hfree_ginf')

def load_rows(path, gkey, dkey='deficit'):
    d = json.load(open(path)); rows = d.get('rows') or []
    out = [(r.get(gkey), r.get(dkey)) for r in rows if isinstance(r, dict) and r.get(gkey) and r.get(dkey) is not None]
    return sorted(out)

# h-free (robust orbit-average + self-continuation)
hf = load_rows(os.path.join(OUT, 'ginf_robust.json'), 'G')
# kernel co-area ladder
kf = load_rows(os.path.join(BASE, 'k3_coarea_limit', 'report.json'), 'G_inner')

def extrap(rows, p):
    G = np.array([r[0] for r in rows], float); D = np.array([r[1] for r in rows], float)
    A = np.column_stack([np.ones_like(G), 1.0 / G**p]); c, *_ = np.linalg.lstsq(A, D, rcond=None)
    return c[0]

print("h-free deficit vs G:", hf)
print("kernel deficit vs G:", kf)
res = {'hfree': hf, 'kernel': kf}
for lab, rows in [('hfree', hf), ('kernel', kf)]:
    if len(rows) >= 3:
        res[f'{lab}_extrap_1/G'] = float(extrap(rows, 1))
        res[f'{lab}_extrap_1/G2'] = float(extrap(rows, 2))
        print(f"{lab}: extrap(1/G)={res[f'{lab}_extrap_1/G']:.4f}  extrap(1/G^2)={res[f'{lab}_extrap_1/G2']:.4f}")
if 'hfree_extrap_1/G' in res and 'kernel_extrap_1/G' in res:
    gap = abs(res['hfree_extrap_1/G'] - res['kernel_extrap_1/G'])
    res['reconciled'] = bool(gap < 0.03)
    res['continuum_gap'] = float(gap)
    print(f"VERDICT: |hfree - kernel| continuum extrap = {gap:.4f} -> {'RECONCILED' if gap<0.03 else 'DISAGREE (one biased)'}")
json.dump(res, open(os.path.join(OUT, 'reconcile.json'), 'w'), indent=2)

fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
if hf: ax.plot([r[0] for r in hf], [r[1] for r in hf], 'o-', label='h-free (no kernel)', color='C0')
if kf: ax.plot([r[0] for r in kf], [r[1] for r in kf], 's-', label='kernel (co-area band)', color='C1')
for lab, col in [('hfree', 'C0'), ('kernel', 'C1')]:
    k = f'{lab}_extrap_1/G'
    if k in res: ax.axhline(res[k], ls=':', color=col, alpha=0.7, label=f'{lab} G->inf extrap = {res[k]:.3f}')
ax.set_xlabel('grid G_inner'); ax.set_ylabel('revelation deficit 1-R^2')
ax.set_title('Reconciliation: h-free vs kernel method -> same continuum PR deficit? (tau=2, gamma=0.1)')
ax.legend(fontsize=9); ax.grid(ls=':')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'reconcile.png'), dpi=140, bbox_inches='tight'); plt.close()
print('wrote', os.path.join(OUT, 'reconcile.png'))
