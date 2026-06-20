"""Nail the STRICT h=0 exact co-area operator's fixed point F(P)=Phi(P)-P=0.

flint Newton with flint-FD Jacobian (eps in arb) + backtracking line search.
Warm-start from the kernel-nailed co-area equilibrium (G9) interpolated to the
strict grid. Drive ||F||inf down as far as possible; TARGET < 1e-100.

Diagnoses any floor: marching-squares is only C0 (the contour-integral operator
has kinks where a cell contour passes exactly through a grid node / Morse-critical
p). If Newton floors, we perturb UMAX slightly and see if the floor MOVES (kink)
or is intrinsic.
"""
import sys, os, json, time
sys.path.insert(0, '/tmp')
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from strict_h0_operator import (arb, ctx, init_no_learning, phi_strict,
                                build_grid)

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_strict_h0_exact'
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'run.log'), 'a')
TEE = open('/tmp/strict_h0.log', 'a')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + '\n'); LOG.flush()
    TEE.write(s + '\n'); TEE.flush()

TAU = arb(2); GAMMA = arb('0.1'); W = arb(1)
G_INNER = int(os.environ.get('GINNER', '7'))
PAD = 2
UMAX = float(os.environ.get('UMAX', '4.0'))


def make_arb_full(Pin, halo, lo, hi, G):
    P = [[[halo[i][j][l] for l in range(G)] for j in range(G)] for i in range(G)]
    for ii, i in enumerate(range(lo, hi)):
        for jj, j in enumerate(range(lo, hi)):
            for ll, l in enumerate(range(lo, hi)):
                P[i][j][l] = arb(repr(float(Pin[ii, jj, ll])))
    return P


def warm_start(G_inner):
    P9 = np.load(os.path.join(
        os.path.dirname(OUT), 'k3_coarea_limit', 'P_inner_G9.npy'))
    ax9 = np.linspace(-UMAX, UMAX, 9)
    axn = np.linspace(-UMAX, UMAX, G_inner)
    rgi = RegularGridInterpolator((ax9, ax9, ax9), P9,
                                  bounds_error=False, fill_value=None)
    A, B, C = np.meshgrid(axn, axn, axn, indexing='ij')
    return rgi(np.column_stack([A.ravel(), B.ravel(), C.ravel()])
               ).reshape((G_inner,) * 3)


def main():
    t0 = time.time()
    log(f"\n=== STRICT h=0 NAIL  G_inner={G_INNER} pad={PAD} UMAX={UMAX} "
        f"tau=2 gamma=0.1  prec={ctx.prec}bits  ({time.strftime('%H:%M:%S')}) ===")
    tv = [TAU] * 3; gv = [GAMMA] * 3; wv = [W] * 3
    du, u_full, lo, hi = build_grid(G_INNER, PAD, UMAX)
    G = len(u_full)
    n = G_INNER ** 3
    idx = [(i, j, l) for i in range(lo, hi)
           for j in range(lo, hi) for l in range(lo, hi)]

    halo = init_no_learning(u_full, tv, gv, wv)
    log(f"halo built ({time.time()-t0:.0f}s)")

    Pin0 = warm_start(G_INNER)
    P = make_arb_full(Pin0, halo, lo, hi, G)

    def get_x(P):
        return [P[i][j][l] for (i, j, l) in idx]

    def set_x(x):
        Pf = [[[halo[i][j][l] for l in range(G)] for j in range(G)]
              for i in range(G)]
        for q, (i, j, l) in enumerate(idx):
            Pf[i][j][l] = x[q]
        return Pf

    def residual(x):
        Pf = set_x(x)
        Pn = phi_strict(Pf, u_full, lo, hi, tv, gv, wv, du)
        return [Pn[i][j][l] - x[q] for q, (i, j, l) in enumerate(idx)]

    def norm_inf(v):
        m = arb(0)
        for e in v:
            a = abs(e)
            if a > m:
                m = a
        return m

    x = get_x(P)
    F = residual(x)
    Finf = norm_inf(F)
    log(f"warm-start ||F||inf = {float(Finf):.6e}")

    EPSFD = arb(10) ** -50
    traj = [float(Finf)]
    best_inf = Finf
    best_x = x[:]

    MAXIT = int(os.environ.get('MAXIT', '12'))
    for it in range(MAXIT):
        tj = time.time()
        # FD Jacobian J[a][b] = dF_a/dx_b  (forward difference, eps in arb)
        # column b: perturb x_b -> recompute full residual
        J = [[arb(0)] * n for _ in range(n)]
        for b in range(n):
            xp = x[:]
            xp[b] = xp[b] + EPSFD
            Fp = residual(xp)
            for a in range(n):
                J[a][b] = (Fp[a] - F[a]) / EPSFD
        tjac = time.time() - tj

        # solve J dx = -F via Gaussian elimination with partial pivot (arb)
        dx = solve_arb(J, [-f for f in F], n)
        if dx is None:
            log(f"it {it}: singular Jacobian, stop")
            break

        # backtracking line search on ||F||inf
        lam = arb(1)
        improved = False
        for _ls in range(40):
            xt = [x[q] + lam * dx[q] for q in range(n)]
            # keep inside (0,1)
            ok = all((xt[q] > 0 and xt[q] < 1) for q in range(n))
            if ok:
                Ft = residual(xt)
                Fti = norm_inf(Ft)
                if Fti < Finf:
                    x = xt; F = Ft; Finf = Fti; improved = True
                    break
            lam = lam * arb('0.5')
        traj.append(float(Finf))
        if Finf < best_inf:
            best_inf = Finf; best_x = x[:]
        log(f"it {it:2d}: ||F||inf = {float(Finf):.6e}  lam={float(lam):.2e} "
            f"jac={tjac:.0f}s ls_ok={improved} tot={(time.time()-t0):.0f}s")
        # save progress
        save_state(best_x, idx, G_INNER, lo, traj, float(best_inf), t0, du, halo, G)
        if not improved:
            log(f"it {it}: line search failed to improve -> FLOOR at {float(Finf):.3e}")
            break
        if Finf < arb(10) ** -100:
            log(f"it {it}: REACHED < 1e-100")
            break
        if it > 0 and traj[-1] > 0.5 * traj[-2]:
            # slow progress; keep going a bit but note
            pass

    log(f"LOWEST ||F||inf = {float(best_inf):.6e}  ({'< 1e-100 NAILED' if best_inf < arb(10)**-100 else 'FLOOR'})")
    finalize(best_x, idx, G_INNER, lo, traj, best_inf, t0, du, halo, G, u_full,
             tv, gv, wv)


def solve_arb(J, b, n):
    """Gaussian elimination with partial pivoting in arb. Returns x or None."""
    # augmented
    M = [row[:] + [b[i]] for i, row in enumerate(J)]
    for col in range(n):
        # pivot
        piv = col
        pm = abs(M[col][col])
        for r in range(col + 1, n):
            v = abs(M[r][col])
            if v > pm:
                pm = v; piv = r
        if pm == 0:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        inv = arb(1) / M[col][col]
        for r in range(n):
            if r == col:
                continue
            fac = M[r][col] * inv
            if fac == 0:
                continue
            for c in range(col, n + 1):
                M[r][c] = M[r][c] - fac * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def save_state(best_x, idx, G_inner, lo, traj, best_inf, t0, du, halo, G):
    cube = np.empty((G_inner,) * 3, dtype=object)
    for q, (i, j, l) in enumerate(idx):
        cube[i - lo, j - lo, l - lo] = best_x[q].str(40, radius=False)
    np.save(os.path.join(OUT, 'P_strict_h0.npy'), cube)


def deficit_and_report(best_x, idx, G_inner, lo, du, u_full):
    UM = UMAX
    axn = np.linspace(-UM, UM, G_inner)
    P = np.empty((G_inner,) * 3)
    for q, (i, j, l) in enumerate(idx):
        P[i - lo, j - lo, l - lo] = float(best_x[q])
    U1, U2, U3 = np.meshgrid(axn, axn, axn, indexing='ij')
    T = 2.0 * (U1 + U2 + U3)
    Pc = np.clip(P, 1e-15, 1 - 1e-15)
    y = np.log(Pc / (1 - Pc)).ravel()
    t = T.ravel()
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    defi = float(ss_res / max(ss_tot, 1e-30))
    return defi, P


def finalize(best_x, idx, G_inner, lo, traj, best_inf, t0, du, halo, G, u_full,
             tv, gv, wv):
    save_state(best_x, idx, G_inner, lo, traj, float(best_inf), t0, du, halo, G)
    defi, P = deficit_and_report(best_x, idx, G_inner, lo, du, u_full)
    log(f"deficit (1-R^2) at nailed point = {defi:.5f}  (genuine PR ~0.27-0.29?)")
    rep = dict(
        config=dict(tau=2.0, gamma=0.1, W=1.0, G_inner=G_inner, pad=PAD,
                    UMAX=UMAX, prec_bits=ctx.prec, du=float(du)),
        operator='strict h=0 marching-squares co-area, NO kernel/bandwidth',
        lowest_Finf=float(best_inf),
        lowest_Finf_str=best_inf.str(20, radius=False),
        reached_1e_100=bool(best_inf < arb(10) ** -100),
        nail_trajectory=traj,
        deficit_1_minus_R2=defi,
        walltime_s=round(time.time() - t0, 1),
    )
    # merge validation
    try:
        v = json.load(open('/tmp/strict_h0_validate.json'))
        rep['validation'] = v
    except Exception:
        pass
    json.dump(rep, open(os.path.join(OUT, 'report.json'), 'w'), indent=2)
    log("wrote report.json; NAIL_DONE")


if __name__ == '__main__':
    main()
