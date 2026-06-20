"""Joint co-area limit for the K=3 CRRA REE contour-Phi operator.

Tests whether the DETERMINISTIC (h->0) partially-revealing equilibrium is a
genuine SMOOTH fixed point, by treating the Gaussian kernel bandwidth h as a
numerical co-area/interior-point barrier parameter and taking the JOINT limit
(h->0, grid->0 with grid spacing Delta_u << h, via h = C*Delta_u^0.5).

Contrast with the prior fixed-grid h->0 study (k3_noisy_pr.py) which stalled
once h fell below grid spacing (ties returned). Here h/Delta_u -> inf so the
band always spans a growing number of cells -> consistent smooth quadrature.

Distinguish:
 (i)   deficit -> positive grid-independent const = genuine deterministic PR
 (ii)  deficit -> 0                                = PR collapses to FR
 (iii) cannot nail along path                      = deeper obstruction
"""
import os, sys, json, time, warnings
sys.path.insert(0, '/tmp/rezn-source')
import numpy as np
from code.contour_K3_halo import (init_no_learning_K3, phi_K3_halo_smooth,
                                   phi_K3_halo)
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    try:
        from scipy.optimize._nonlin import NoConvergence
    except ImportError:
        class NoConvergence(Exception): pass
warnings.filterwarnings('ignore')

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_limit'
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'run.log'), 'a')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

K = 3; pad = 2; UMAX = 4.0
TAU = 2.0; GAMMA = 0.1
C = 0.45              # h = C * Delta_u^0.5 ; h(G=9)=0.45 matches prior nails
G_LIST = [9, 13, 17, 21, 25, 31]
tv = np.full(K, TAU); gv = np.full(K, GAMMA); wv = np.full(K, 1.0)


def build_grid(Gi):
    du = 2 * UMAX / (Gi - 1)
    Gf = Gi + 2 * pad
    uf = np.array([-UMAX + (q - pad) * du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def metrics(Pin, T, P_FR, Xreg):
    Pc = np.clip(Pin, 1e-12, 1 - 1e-12)
    y = np.log(Pc / (1 - Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    pr = a[0] * T.ravel() + a[1]
    defi = np.sum((y - pr) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    coef, _, _, _ = np.linalg.lstsq(Xreg, y, rcond=None)
    return dict(deficit=float(defi),
                d_FR=float(np.sqrt(np.mean((Pin - P_FR) ** 2))),
                b1=float(coef[0]), b2=float(coef[1]), b3=float(coef[2]),
                slope_T=float(a[0]))


def interp_solution(P_inner_old, Gi_old, Gi_new):
    """Trilinear interpolation of an inner cube (Gi_old^3) onto Gi_new^3.
    Both grids span the same inner domain [-UMAX, UMAX]."""
    from scipy.interpolate import RegularGridInterpolator
    ax_old = np.linspace(-UMAX, UMAX, Gi_old)
    ax_new = np.linspace(-UMAX, UMAX, Gi_new)
    rgi = RegularGridInterpolator((ax_old, ax_old, ax_old),
                                  P_inner_old, bounds_error=False,
                                  fill_value=None)
    A, B, Cc = np.meshgrid(ax_new, ax_new, ax_new, indexing='ij')
    pts = np.column_stack([A.ravel(), B.ravel(), Cc.ravel()])
    return rgi(pts).reshape((Gi_new,) * 3)


def run():
    t0 = time.time()
    log(f"=== JOINT CO-AREA LIMIT  tau={TAU} gamma={GAMMA}  h=C*du^0.5, C={C} ===")
    log(f"date 2026-05-29  G_LIST={G_LIST}")
    rows = []
    P_prev_inner = None; Gi_prev = None
    for Gi in G_LIST:
        du, uf, lo, hi = build_grid(Gi)
        h = C * du ** 0.5
        slc = (slice(lo, hi),) * K
        ui = uf[lo:hi]
        U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
        T = TAU * (U1 + U2 + U3)
        P_FR = 1 / (1 + np.exp(-T))
        Xreg = np.column_stack([U1.ravel(), U2.ravel(), U3.ravel(),
                                np.ones(U1.size)])
        halo = init_no_learning_K3(uf, tv, gv, wv)

        def resid(xflat):
            Pf = halo.copy(); Pf[slc] = xflat.reshape((Gi,) * K)
            return (phi_K3_halo_smooth(Pf, uf, lo, hi, tv, gv, wv, h)
                    - Pf)[slc].ravel()

        # initial condition: warm-start from interpolated coarser nail
        if P_prev_inner is not None:
            x0 = interp_solution(P_prev_inner, Gi_prev, Gi).ravel()
        else:
            x0 = halo[slc].ravel().copy()

        cnt = {'n': 0}
        def cb(x, fx): cnt['n'] += 1
        conv = True
        t1 = time.time()
        try:
            sol = newton_krylov(resid, x0, f_tol=1e-10, maxiter=300,
                                callback=cb, method='lgmres')
        except NoConvergence as e:
            sol = np.asarray(e.args[0]).ravel(); conv = False
        Finf = float(np.max(np.abs(resid(sol))))
        sol_cube = sol.reshape((Gi,) * K)
        m = metrics(sol_cube, T, P_FR, Xreg)

        # strict-scan CONTROL: damped Picard plateau (hard edge-crossing)
        Ps = halo.copy()
        for _ in range(80):
            Pn = phi_K3_halo(Ps, uf, lo, hi, tv, gv, wv)
            Ps = 0.75 * Ps + 0.25 * Pn
        f_strict = float(np.max(np.abs(
            (phi_K3_halo(Ps, uf, lo, hi, tv, gv, wv) - Ps)[slc])))
        ms = metrics(Ps[slc], T, P_FR, Xreg)

        row = dict(G_inner=Gi, du=float(du), h=float(h),
                   h_over_du=float(h / du), Finf=Finf, iters=cnt['n'],
                   converged=conv, **m,
                   strict_Finf=f_strict, strict_deficit=ms['deficit'],
                   strict_d_FR=ms['d_FR'], strict_b1=ms['b1'],
                   walltime_s=round(time.time() - t1, 1))
        rows.append(row)
        log(f"G={Gi:3d} du={du:.4f} h={h:.4f} h/du={h/du:.2f} | "
            f"NAILED ||F||={Finf:.2e} it={cnt['n']} conv={conv} | "
            f"deficit={m['deficit']:.5f} d_FR={m['d_FR']:.4f} "
            f"beta={m['b1']:.3f} slopeT={m['slope_T']:.4f} || "
            f"STRICT ||F||={f_strict:.4f} def={ms['deficit']:.4f} "
            f"({time.time()-t0:.0f}s, this {time.time()-t1:.0f}s)")

        # persist finest nail surface
        np.save(os.path.join(OUT, f'P_inner_G{Gi}.npy'), sol_cube)
        P_prev_inner = sol_cube; Gi_prev = Gi

        # incremental report write + Richardson on what we have so far
        richardson = richardson_extrap(rows)
        json.dump(dict(tau=TAU, gamma=GAMMA, C=C, h_law='h=C*du^0.5',
                       G_LIST=G_LIST, rows=rows, richardson=richardson),
                  open(os.path.join(OUT, 'report.json'), 'w'), indent=2)
    return rows


def richardson_extrap(rows):
    """Extrapolate deficit & d_FR to h=0 along the path. h decreases as G
    grows. Fit y(h) = y0 + a*h^q via log-log of successive differences for
    the order q, then a linear-in-h^q fit for y0. Robust simple version:
    fit y = y0 + a*h + b*h^2 (least squares) if >=3 pts; report y0 and an
    h^q power-law estimate of the convergence order."""
    conv_rows = [r for r in rows if r['converged'] and r['Finf'] < 1e-7]
    if len(conv_rows) < 3:
        return dict(note='need >=3 nailed points', n=len(conv_rows))
    h = np.array([r['h'] for r in conv_rows])
    out = {}
    for key in ('deficit', 'd_FR', 'b1', 'slope_T'):
        y = np.array([r[key] for r in conv_rows])
        # polynomial in h: y = y0 + a*h + b*h^2
        A = np.column_stack([np.ones_like(h), h, h ** 2])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        y0_poly = float(coef[0])
        # power-law order from consecutive triples (Aitken-like on diffs)
        order = None
        if len(y) >= 4:
            d = np.diff(y)
            r1 = d[1:] / d[:-1]
            hr = h[1:-1] / h[:-2]
            with np.errstate(all='ignore'):
                qs = np.log(np.abs(r1)) / np.log(hr)
            order = float(np.nanmedian(qs))
        out[key] = dict(h0_extrap=y0_poly, poly_coef=[float(c) for c in coef],
                        power_order=order,
                        finest_value=float(y[-1]),
                        coarsest_value=float(y[0]))
    out['n_points'] = len(conv_rows)
    out['h_values'] = [float(x) for x in h]
    return out


def make_figs(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cr = [r for r in rows if r['converged']]
    h = np.array([r['h'] for r in cr])
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), dpi=130)

    ax[0].semilogy(h, [r['Finf'] for r in cr], 'o-', label='co-area nailed ||F||')
    ax[0].semilogy(h, [r['strict_Finf'] for r in cr], 's--', c='r',
                   label='strict-scan plateau ||F||')
    ax[0].axhline(1e-9, ls=':', c='g', label='nail bar 1e-9')
    ax[0].invert_xaxis()
    ax[0].set_xlabel('h = C*du^0.5  (joint path, h->0 as G->inf)')
    ax[0].set_ylabel('||Phi_h - P||_inf')
    ax[0].set_title('(a) Nailability along joint co-area path')
    ax[0].legend(fontsize=8); ax[0].grid(ls=':')

    rich = richardson_extrap(rows)
    ax[1].plot(h, [r['deficit'] for r in cr], 'o-', label='co-area deficit')
    ax[1].plot(h, [r['d_FR'] for r in cr], '^-', label='co-area d_FR')
    ax[1].plot(h, [r['strict_deficit'] for r in cr], 's--', c='r',
               label='strict-scan deficit (control)')
    if 'deficit' in rich:
        ax[1].axhline(rich['deficit']['h0_extrap'], ls=':', c='b',
                      label=f"Richardson deficit(h=0)={rich['deficit']['h0_extrap']:.4f}")
        ax[1].axhline(rich['d_FR']['h0_extrap'], ls=':', c='g',
                      label=f"Richardson d_FR(h=0)={rich['d_FR']['h0_extrap']:.4f}")
    ax[1].invert_xaxis()
    ax[1].set_xlabel('h = C*du^0.5'); ax[1].set_ylabel('deficit / d_FR')
    ax[1].set_title('(b) PR metrics vs h with Richardson -> h=0')
    ax[1].legend(fontsize=7.5); ax[1].grid(ls=':')

    # converged finest price surface: mid-slice
    Gi = cr[-1]['G_inner']
    P = np.load(os.path.join(OUT, f'P_inner_G{Gi}.npy'))
    mid = Gi // 2
    ax[2].imshow(P[:, :, mid], origin='lower', aspect='auto',
                 extent=[-UMAX, UMAX, -UMAX, UMAX], cmap='viridis')
    ax[2].set_xlabel('u_2'); ax[2].set_ylabel('u_1')
    ax[2].set_title(f'(c) Converged PR price surface G={Gi} (u_3=mid)')
    plt.colorbar(ax[2].images[0], ax=ax[2], fraction=0.046)

    plt.suptitle('K=3 CRRA REE joint co-area limit (h->0, du->0, du<<h): '
                 'is deterministic smooth PR genuine?  tau=2, gamma=0.1',
                 weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'k3_coarea_limit.png'), dpi=130,
                bbox_inches='tight')
    plt.close()
    log('wrote figure')


if __name__ == '__main__':
    rows = run()
    make_figs(rows)
    log(f"DONE all grids in {sum(r['walltime_s'] for r in rows):.0f}s cumulative")
