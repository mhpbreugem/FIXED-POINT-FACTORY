"""Nail the STRICT h=0 CDF-derivative co-area fixed point for K=3 CRRA REE.

F(P) = Phi_strict_cdf(P) - P = 0 on the inner G_inner^3 block (halo fixed at
the no-learning surface).  flint-arb Newton with a localized finite-difference
Jacobian + line search.  Target ||F||_inf < 1e-100.

Localized Jacobian: perturbing inner unknown P[a][b][c] only changes residuals
at nodes that share a coordinate with (a,b,c) (the operator at node (i,j,l)
reads the three planar slices P[i,:,:], P[:,j,:], P[:,:,l]; node (a,b,c)
appears in slice0 iff i==a, slice1 iff j==b, slice2 iff l==c).  So only the
union {i==a} U {j==b} U {l==c} of output nodes is affected -> ~O(3*G^2) per
column instead of O(G^3).  All arithmetic in flint arb.
"""
import os, sys, json, time
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
sys.path.insert(0, '/tmp')
import numpy as np
from flint import ctx, arb
import cdf_h0_operator as OP
from cdf_h0_operator import (agent_evidence_cdf, bayes, clear_crra, lam)

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_strict_h0_cdf'
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, 'run.log'), 'a')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

G_INNER = 9
PAD = 2
UMAX = 4.0
TAU = arb('2.0')
GAMMA = arb('0.1')
W = arb('1.0')
NSUB = 3
TARGET = arb(10) ** -100

tv = [TAU, TAU, TAU]
gv = [GAMMA, GAMMA, GAMMA]
wv = [W, W, W]


def node_residual(P, u_full, i, j, l, du, NSUB):
    """Phi at a single inner node (i,j,l): returns new price (arb)."""
    G = len(u_full)
    p = P[i][j][l]
    sl0 = [[P[i][a][b] for b in range(G)] for a in range(G)]
    A0, A1 = agent_evidence_cdf(sl0, p, u_full, tv[1], tv[2], du, G, NSUB)
    mu0 = bayes(u_full[i], tv[0], A0, A1)
    sl1 = [[P[a][j][b] for b in range(G)] for a in range(G)]
    A0, A1 = agent_evidence_cdf(sl1, p, u_full, tv[0], tv[2], du, G, NSUB)
    mu1 = bayes(u_full[j], tv[1], A0, A1)
    sl2 = [[P[a][b][l] for b in range(G)] for a in range(G)]
    A0, A1 = agent_evidence_cdf(sl2, p, u_full, tv[0], tv[1], du, G, NSUB)
    mu2 = bayes(u_full[l], tv[2], A0, A1)
    return clear_crra([mu0, mu1, mu2], gv, wv)


def full_phi_inner(P, u_full, lo, hi, du, NSUB):
    """Phi on all inner nodes -> dict (i,j,l)->arb new price."""
    out = {}
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                out[(i, j, l)] = node_residual(P, u_full, i, j, l, du, NSUB)
    return out


def residual_vec(P, idx, u_full, lo, hi, du, NSUB):
    """F = Phi(P)-P at inner nodes, returned as list of arb in idx order."""
    phi = full_phi_inner(P, u_full, lo, hi, du, NSUB)
    return [phi[k] - P[k[0]][k[1]][k[2]] for k in idx]


def affected_nodes(a, b, c, lo, hi):
    """Inner output nodes whose residual depends on inner unknown (a,b,c)."""
    s = set()
    for j in range(lo, hi):
        for l in range(lo, hi):
            s.add((a, j, l))
    for i in range(lo, hi):
        for l in range(lo, hi):
            s.add((i, b, l))
    for i in range(lo, hi):
        for j in range(lo, hi):
            s.add((i, j, c))
    return s


def main():
    t0 = time.time()
    log("")
    log("=" * 78)
    log(f"=== STRICT h=0 CDF-DERIVATIVE NAIL  K=3  tau={float(TAU)} "
        f"gamma={float(GAMMA)} W={float(W)} ===")
    log(f"date 2026-05-29  G_inner={G_INNER} pad={PAD} UMAX={UMAX} NSUB={NSUB} "
        f"prec_bits={ctx.prec}")

    du, u_full, lo, hi = OP.build_grid(G_INNER, PAD, UMAX)
    G = len(u_full)

    # warm start from kernel-nailed co-area G9 (interpolated trivially: same Gi)
    P64 = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
                  'solved_fixed_points/k3_coarea_limit/P_inner_G9.npy')
    P = OP.init_no_learning(u_full, tv, gv, wv)   # halo + inner default
    for ii in range(G_INNER):
        for jj in range(G_INNER):
            for ll in range(G_INNER):
                P[lo + ii][lo + jj][lo + ll] = arb(repr(float(P64[ii, jj, ll])))

    idx = [(i, j, l) for i in range(lo, hi)
           for j in range(lo, hi) for l in range(lo, hi)]
    nidx = len(idx)
    pos = {k: q for q, k in enumerate(idx)}
    log(f"unknowns={nidx}  warm-start from k3_coarea_limit G9")

    def set_P(vec):
        for q, k in enumerate(idx):
            P[k[0]][k[1]][k[2]] = vec[q]

    def get_P():
        return [P[k[0]][k[1]][k[2]] for k in idx]

    eps_fd = arb(10) ** -40

    x = get_P()
    F = residual_vec(P, idx, u_full, lo, hi, du, NSUB)
    Finf = max(abs(f) for f in F)
    log(f"initial ||F||inf = {Finf}")

    traj = [float(Finf)]
    lowest = Finf
    lowest_x = list(x)

    MAX_NEWTON = 12
    for it in range(MAX_NEWTON):
        ts = time.time()
        # ---- localized FD Jacobian (dense nidx x nidx in flint) ----
        # J[r, q] = dF_r / dx_q.  Column q: perturb x_q, recompute residual
        # only at affected output nodes.
        # Build columns.
        Jcols = [[arb(0)] * nidx for _ in range(nidx)]  # Jcols[q][r]
        # baseline phi map (already have F; need phi at affected nodes on perturb)
        for q, k in enumerate(idx):
            a, b, c = k
            x0 = x[q]
            P[a][b][c] = x0 + eps_fd
            aff = affected_nodes(a, b, c, lo, hi)
            for nd in aff:
                phi_new = node_residual(P, u_full, nd[0], nd[1], nd[2], du, NSUB)
                Fnew = phi_new - P[nd[0]][nd[1]][nd[2]]
                r = pos[nd]
                Jcols[q][r] = (Fnew - F[r]) / eps_fd
            P[a][b][c] = x0  # restore
        # assemble dense J (r,q)
        # Solve J dx = -F  via flint Gaussian elimination with partial pivot.
        dx = solve_flint(Jcols, F, nidx)
        # ---- line search on ||F|| ----
        best_t = None; best_Finf = None; best_x = None
        for tstep in (arb(1), arb('0.5'), arb('0.25'), arb('0.125'),
                      arb('0.0625')):
            xt = [x[r] - tstep * dx[r] for r in range(nidx)]
            set_P(xt)
            Ft = residual_vec(P, idx, u_full, lo, hi, du, NSUB)
            Fti = max(abs(f) for f in Ft)
            if best_Finf is None or Fti < best_Finf:
                best_Finf = Fti; best_t = tstep; best_x = xt; best_F = Ft
        x = best_x; F = best_F; set_P(x)
        Finf = best_Finf
        traj.append(float(Finf))
        if Finf < lowest:
            lowest = Finf; lowest_x = list(x)
        log(f"newton it={it+1} step_t={float(best_t)} ||F||inf={Finf} "
            f"({time.time()-ts:.0f}s, tot {time.time()-t0:.0f}s)")
        if Finf < TARGET:
            log(f"REACHED TARGET <1e-100 at it={it+1}")
            break
        # stall detection
        if it >= 2 and traj[-1] > traj[-2] * arb('0.9') and float(Finf) > 1e-90:
            # not improving much; if floored, stop
            if float(traj[-1]) > float(traj[-2]) * 0.5:
                log(f"STALL: improvement stalled at ||F||={Finf}")
                # keep going a couple more then break
                if it >= 5:
                    break

    set_P(lowest_x)
    reached = lowest < TARGET
    log(f"LOWEST ||F||inf = {lowest}   reached_1e_100={bool(reached)}")

    # ---- metrics: deficit (1-R^2 of logit P vs T = tau*sum u) + kink diag ----
    Pin = np.zeros((G_INNER, G_INNER, G_INNER))
    for q, k in enumerate(idx):
        Pin[k[0] - lo, k[1] - lo, k[2] - lo] = float(lowest_x[q])
    np.save(os.path.join(OUT, 'P_strict_h0_cdf.npy'), Pin)

    ui = np.array([float(u_full[lo + q]) for q in range(G_INNER)])
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing='ij')
    T = float(TAU) * (U1 + U2 + U3)
    Pc = np.clip(Pin, 1e-300, 1 - 1e-16)
    y = np.log(Pc / (1 - Pc)).ravel()
    A = np.polyfit(T.ravel(), y, 1)
    pr = A[0] * T.ravel() + A[1]
    deficit = float(np.sum((y - pr) ** 2) /
                    max(np.sum((y - y.mean()) ** 2), 1e-30))
    P_FR = 1 / (1 + np.exp(-T))
    d_FR = float(np.sqrt(np.mean((Pin - P_FR) ** 2)))

    # ---- knot/kink diagnosis: is any nailed price exactly at a cell min/max
    # P-value (a CDF C0 knot)?  For each inner node price p, scan all slices'
    # cell corner values; report the min distance from p to any cell extreme. ----
    min_knot_dist = float('inf')
    set_P(lowest_x)
    for (i, j, l) in idx:
        p = float(P[i][j][l])
        for (sl, fa) in (([[float(P[i][a][b]) for b in range(G)]
                           for a in range(G)], None),):
            for a in range(G - 1):
                for b in range(G - 1):
                    cs = [sl[a][b], sl[a + 1][b], sl[a][b + 1], sl[a + 1][b + 1]]
                    for cv in (min(cs), max(cs)):
                        d = abs(p - cv)
                        if d < min_knot_dist:
                            min_knot_dist = d
    log(f"deficit(1-R2)={deficit}  d_FR={d_FR}  min_knot_dist={min_knot_dist:.3e}")

    rep = dict(
        config=dict(tau=float(TAU), gamma=float(GAMMA), W=float(W),
                    G_inner=G_INNER, pad=PAD, UMAX=UMAX, NSUB=NSUB,
                    prec_bits=ctx.prec, du=float(du), eps_fd=1e-40),
        operator="strict h=0 CDF-derivative (sub-level-set) co-area, "
                 "NO kernel/bandwidth, NO explicit 1/|gradP| division",
        warm_start="k3_coarea_limit/P_inner_G9.npy (kernel-nailed co-area)",
        validation=json.load(open(os.path.join(OUT, 'validate.json'))),
        nail_trajectory=traj,
        lowest_Finf=float(lowest),
        lowest_Finf_str=str(lowest),
        reached_1e_100=bool(reached),
        deficit_1_minus_R2=deficit,
        d_FR=d_FR,
        kink_diag=dict(min_price_to_cell_extreme_dist=min_knot_dist,
                       note="C0 CDF knots occur when a nailed price equals a "
                            "cell min/max P; small dist => kink-limited floor"),
        walltime_s=round(time.time() - t0, 1),
    )
    json.dump(rep, open(os.path.join(OUT, 'report.json'), 'w'), indent=2)
    log(f"DONE  lowest||F||={lowest}  reached_1e_100={reached}  "
        f"deficit={deficit:.4f}  ({time.time()-t0:.0f}s)")
    return rep


def solve_flint(Jcols, F, n):
    """Solve J dx = -F.  Jcols[q][r] = J[r,q] (column-major).  Use flint
    arb_mat.solve (fast LU).  Returns dx (list of arb)."""
    from flint import arb_mat
    J = arb_mat([[Jcols[q][r] for q in range(n)] for r in range(n)])
    rhs = arb_mat([[-F[r]] for r in range(n)])
    x = J.solve(rhs)
    return [x[r, 0] for r in range(n)]


if __name__ == '__main__':
    main()
