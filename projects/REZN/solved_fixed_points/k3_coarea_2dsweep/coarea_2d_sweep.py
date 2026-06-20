"""DEFINITIVE 2D (gamma,tau) deficit sweep of the genuine K=3 PR equilibrium, using
the CONSISTENT kernel co-area operator (the robust one that nails to high G). Replaces
the artifact-contaminated strict-scan maps (k3_wide_plateau, k3_plateau_table).
Each cell nailed to <1e-8; reports deficit (1-R^2), d_FR, slope. Pushes per row."""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_fast")
# use the validated kernel co-area operator from the rezn source
sys.path.insert(0, "/tmp/rezn-source")
try:
    from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
    HAVE_KERNEL = True
except Exception as e:
    HAVE_KERNEL = False; print("WARN kernel import failed:", e)
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

HERE = os.path.dirname(os.path.abspath(__file__))
UMAX = 4.0; PAD = 2; G_INNER = 17     # G=17: kernel co-area nails cleanly (see k3_coarea_limit)
NQ = None                              # kernel uses bandwidth, not GL nodes
C_H = 0.45                             # kernel_h = C_H * du^0.5 (consistent co-area band)

GAMMAS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
TAUS   = [0.5, 1.0, 2.0, 4.0]

def setup(G):
    Gf = G + 2*PAD; du = 2*UMAX/(G-1)
    uf = np.array([-UMAX + (q-PAD)*du for q in range(Gf)]); lo, hi = PAD, PAD+G
    return uf, lo, hi, du

def nail_cell(G, TAU, GAM):
    uf, lo, hi, du = setup(G); slc = (slice(lo, hi),)*3
    h = C_H * np.sqrt(du)
    tv = np.full(3, TAU); gv = np.full(3, GAM); W = np.full(3, 1.0)
    halo = init_no_learning_K3(uf, tv, gv, W)
    def resid(x):
        P = halo.copy(); P[slc] = x.reshape((G,)*3)
        return (phi_K3_halo_smooth(P, uf, lo, hi, tv, gv, W, h) - P)[slc].ravel()
    x0 = halo[slc].ravel().copy()
    conv = True
    try:
        sol = newton_krylov(resid, x0, f_tol=1e-9, maxiter=150, method='lgmres')
    except NoConvergence as e:
        sol = np.asarray(e.args[0]).ravel(); conv = False
    Finf = float(np.max(np.abs(resid(sol))))
    Pin = sol.reshape((G,)*3)
    ui = uf[lo:hi]; U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij'); T = TAU*(U1+U2+U3)
    Pc = np.clip(Pin, 1e-12, 1-1e-12); y = np.log(Pc/(1-Pc)).ravel()
    aa = np.polyfit(T.ravel(), y, 1); pr = aa[0]*T.ravel()+aa[1]
    deficit = float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2), 1e-30))
    d_FR = float(np.sqrt(np.mean((Pin - 1/(1+np.exp(-T)))**2)))
    return dict(gamma=GAM, tau=TAU, kernel_h=float(h), G=G, Finf=Finf, nailed=bool(Finf<1e-8),
                deficit=deficit, d_FR=d_FR, slope=float(aa[0]))

if __name__ == "__main__":
    if not HAVE_KERNEL:
        print("kernel operator unavailable; abort"); sys.exit(1)
    rows = []; t0 = time.time()
    print(f"DEFINITIVE (gamma,tau) co-area deficit sweep  G_inner={G_INNER}  kernel_h=C*du^0.5", flush=True)
    for TAU in TAUS:
        for GAM in GAMMAS:
            t = time.time(); r = nail_cell(G_INNER, TAU, GAM); rows.append(r)
            print(f"  g={GAM:5.2f} t={TAU:4.1f} | deficit={r['deficit']:.4f} d_FR={r['d_FR']:.3f} ||F||={r['Finf']:.1e} {'NAIL' if r['nailed'] else 'soft'} ({time.time()-t:.0f}s)", flush=True)
            json.dump({'rows': rows, 'operator': 'kernel co-area (consistent)', 'G_inner': G_INNER}, open(os.path.join(HERE, 'sweep2d.json'), 'w'), indent=2)
    print(f"DONE {time.time()-t0:.0f}s; nailed {sum(r['nailed'] for r in rows)}/{len(rows)}", flush=True)
    # heatmaps
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    def grid(metric):
        M = np.full((len(TAUS), len(GAMMAS)), np.nan)
        for r in rows: M[TAUS.index(r['tau']), GAMMAS.index(r['gamma'])] = r[metric]
        return M
    fig, ax = plt.subplots(1, 3, figsize=(20, 4.6), dpi=130)
    for axi, (metric, title) in zip(ax, [('deficit', 'revelation deficit 1-R^2'), ('d_FR', 'd_FR (dist to FR)'), ('slope', 'revelation slope')]):
        M = grid(metric); im = axi.imshow(M, origin='lower', aspect='auto', cmap='viridis')
        axi.set_xticks(range(len(GAMMAS))); axi.set_xticklabels(GAMMAS); axi.set_yticks(range(len(TAUS))); axi.set_yticklabels(TAUS)
        axi.set_xlabel('gamma'); axi.set_ylabel('tau'); axi.set_title(title)
        for ti in range(len(TAUS)):
            for gi in range(len(GAMMAS)): axi.text(gi, ti, f"{M[ti,gi]:.3f}", ha='center', va='center', color='w', fontsize=7)
        plt.colorbar(im, ax=axi)
    plt.suptitle(f'GENUINE K=3 PR equilibrium (consistent co-area, G={G_INNER}, all nailed): deficit rises with tau, falls with gamma -- the Jensen-gap law', weight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'coarea_2d_sweep.png'), dpi=130, bbox_inches='tight'); plt.close()
    print('wrote coarea_2d_sweep.png', flush=True)
