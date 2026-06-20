"""Stage 6b: localize residual plateau in (a,b) per surface; test margin-2
vertex exclusion (drop outermost vertex ring) on the converged-state stats."""
import sys, json
import numpy as np

sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/cmm_endogenous')
from cmm_stage6b_solver import (Problem, load_P_full, make_levels, OUT,
                                 build_pad, heval, solve_c, E_A, E_B, E_T,
                                 DAS, DBS, SQ3, FC_FLOOR)


def mu_probe(pb, H_m, p_m, iv):
    """Replicate the residual's mu computation at one vertex (python)."""
    Hp = build_pad(np.ascontiguousarray(H_m))
    av, bv = pb.va[iv], pb.vb[iv]
    tv, _, _ = heval(Hp, pb.a0, pb.da, pb.Na, pb.b0, pb.db, pb.Nb, av, bv)
    u = av*E_A + bv*E_B + tv*E_T
    tau = pb.tau
    ds = pb.s_arr[1] - pb.s_arr[0]
    mu = np.empty(3)
    for k in range(3):
        X = u[k]
        jl0 = [1, 2, 0][k]; jl1 = [2, 0, 1][k]
        c0 = 0.5*(u[jl0] + u[jl1])
        A0 = A1 = 0.0
        for i_s, s in enumerate(pb.s_arr):
            c, Ha, Hb, F, ok = solve_c(Hp, pb.a0, pb.da, pb.Na, pb.b0,
                                        pb.db, pb.Nb, X, E_A[k], E_B[k],
                                        DAS[k], DBS[k], s, c0, 40)
            Fc = max(2.0/SQ3 + Ha*E_A[k] + Hb*E_B[k], FC_FLOOR)
            cp = (Ha*DAS[k] + Hb*DBS[k])/Fc
            dsig = np.sqrt((cp+1)**2 + (cp-1)**2)
            wq = ds if (0 < i_s < pb.s_arr.size-1) else 0.5*ds
            A0 += wq*np.exp(-0.5*tau*((c+s+0.5)**2 + (c-s+0.5)**2))*dsig
            A1 += wq*np.exp(-0.5*tau*((c+s-0.5)**2 + (c-s-0.5)**2))*dsig
        f0X = np.exp(-0.5*tau*(X+0.5)**2); f1X = np.exp(-0.5*tau*(X-0.5)**2)
        mu[k] = f1X*A1/(f0X*A0 + f1X*A1)
    return mu, tv

tau, gamma = 2.0, 0.098
du, uf, lo, hi, P_inner, P_full = load_P_full(21, tau, gamma)
res = json.load(open(f"{OUT}/stage6b_results.json"))
cfg = res['config']
M, Na, margin = cfg['M'], cfg['Na'], cfg['margin']
p_levels = np.array(cfg['p_levels'])
Hs = np.load(f"{OUT}/stage6b_H.npy")
pb = Problem(M, Na, half_width=4.5, n_vert_margin=margin, tau=tau, gamma=gamma)

nv_side = Na - 2*margin
print(f"M={M} Na={Na} margin={margin} -> vertex grid {nv_side}x{nv_side}")
w = np.sqrt(pb.va**2 + pb.vb**2)
for m in range(M):
    r, _, nf = pb.residual(np.ascontiguousarray(Hs[m]), p_levels[m])
    R = np.abs(r).reshape(nv_side, nv_side)
    # ring index = distance (in rings) from the vertex-grid boundary
    idx = np.arange(nv_side)
    ring = np.minimum(np.minimum(idx[:, None], idx[None, :]),
                      np.minimum(nv_side-1-idx[:, None], nv_side-1-idx[None, :]))
    print(f"m={m} p={p_levels[m]:.3f}: max|r|={R.max():.3e} "
          f"med={np.median(R):.3e}")
    for q in range(min(4, (nv_side+1)//2)):
        sel = ring == q
        print(f"   ring {q} (outermost-{q}): max={R[sel].max():.3e} "
              f"med={np.median(R[sel]):.3e}  n={sel.sum()}")
    inner = ring >= 1
    print(f"   rings>=1: max={R[inner].max():.3e} med={np.median(R[inner]):.3e}")
    inner2 = ring >= 2
    print(f"   rings>=2: max={R[inner2].max():.3e} med={np.median(R[inner2]):.3e}")
    # top-5 worst vertices with mu probe
    order = np.argsort(np.abs(r))[::-1][:5]
    for i in order:
        mu, tv = mu_probe(pb, Hs[m], p_levels[m], int(i))
        print(f"     worst: (a,b)=({pb.va[i]:+.2f},{pb.vb[i]:+.2f}) "
              f"|w|={w[i]:.2f} t={tv:+.2f} |r|={abs(r[i]):.3e} "
              f"mu=({mu[0]:.3e},{mu[1]:.3e},{mu[2]:.3e})")
