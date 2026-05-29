"""Nail the h-FREE SMOOTH co-area K=3 CRRA PR fixed point to 1e-100 at 100-dec.

Path: python-flint (arb) operator + symmetry reduction (165 unknowns at G=9).
Warm-start from the float64 h-free nail (P_nailed_G9.npy, ||F||~1.3e-13).

Newton scheme:
  * Build the 165x165 dense FD Jacobian of the symmetry-reduced residual ONCE
    in fast numba float64 (each column = 1 float64 operator eval). At the root
    the Jacobian is essentially exact; we reuse it (chord/frozen-Jacobian
    Newton) for the arb polish.
  * Each arb Newton step: F_arb = reduce(Phi_arb(expand(x))) - x  (full
    100-dec residual). Solve J dx = -F_arb via the float64 LU factorization
    applied to the arb RHS (direction from float64 J is accurate near the
    root; magnitude carried in arb). x <- x + dx.
  Starting at ||F||~1e-13 this drives ||F||inf down quadratically to ~1e-100
  in a handful of steps.
"""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_100")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_fast")
from itertools import combinations_with_replacement
import numpy as np
import flint
from flint import arb
import hfree_operator as Hf64
import hfree_operator_arb as Ha

OUTDIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_100"
LOG = "/tmp/hfree100.log"
UMAX = 4.0
NQ = 40
SUB = 4
G = 9
TAU = 2.0
GAMMA = 0.1
W = 1.0


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- sym reducer
class SymReducer3:
    def __init__(self, G):
        self.G = G
        self.multisets = list(combinations_with_replacement(range(G), 3))
        self.n_red = len(self.multisets)
        ms_to_red = {ms: r for r, ms in enumerate(self.multisets)}
        red_of_cell = np.empty((G, G, G), dtype=np.int64)
        orbit_count = np.zeros(self.n_red, dtype=np.int64)
        for i in range(G):
            for j in range(G):
                for l in range(G):
                    r = ms_to_red[tuple(sorted((i, j, l)))]
                    red_of_cell[i, j, l] = r
                    orbit_count[r] += 1
        self.red_of_cell = red_of_cell
        self.orbit_count = orbit_count

    def expand_np(self, vec_red):
        return vec_red[self.red_of_cell]

    def reduce_np(self, full):
        acc = np.bincount(self.red_of_cell.ravel(), weights=full.ravel(),
                          minlength=self.n_red)
        return acc / self.orbit_count


# ---- arb expand/reduce (operate on python lists of arb) ----
def expand_arb(vec_red, red):
    G = red.G
    roc = red.red_of_cell
    return [[[vec_red[roc[i, j, l]] for l in range(G)]
             for j in range(G)] for i in range(G)]


def reduce_arb(full, red):
    G = red.G
    acc = [Ha.ZERO] * red.n_red
    roc = red.red_of_cell
    for i in range(G):
        for j in range(G):
            for l in range(G):
                r = roc[i, j, l]
                acc[r] = acc[r] + full[i][j][l]
    return [acc[r] / arb(int(red.orbit_count[r])) for r in range(red.n_red)]


def main():
    # Working precision: the operator loses ~48-52 digits internally (contour
    # 1/|der| weighting + spline Thomas solve), measured by the precision-scaling
    # diagnostic: 110-dec -> ~58 good digits, 160-dec -> ~109 good digits. To
    # NAIL ||F||inf < 1e-100 the residual must be accurate well below 1e-100, so
    # we run at 180-dec working precision (~130 reliable digits of headroom).
    WORK_DEC = 180
    Ha.set_precision(WORK_DEC)
    log(f"=== h-free 100-dec NAIL start  prec={flint.ctx.prec} bits "
        f"(~{flint.ctx.prec/3.3219:.0f} dec working; >100 reliable digits) ===")
    log(f"G={G} tau={TAU} gamma={GAMMA} W={W} Nq={NQ} sub={SUB} impl=python-flint(arb)")

    ui = np.linspace(-UMAX, UMAX, G)
    gn, gw = Hf64.gauss_legendre(NQ, -UMAX, UMAX)
    tau = np.full(3, TAU); gam = np.full(3, GAMMA); Wv = np.full(3, 1.0)
    red = SymReducer3(G)
    log(f"sym-reduced unknowns n_red={red.n_red}")

    # warm start
    P0 = np.load(os.path.join(
        "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points",
        "k3_hfree_fast", "P_nailed_G9.npy"))
    x0 = red.reduce_np(P0)  # symmetrize + read

    # float64 reduced residual
    def Fred_np(vec):
        P = red.expand_np(vec)
        Pn = Hf64.phi_hfree(P, ui, gn, gw, tau, gam, Wv, SUB)
        return red.reduce_np(Pn) - vec

    f0 = Fred_np(x0)
    log(f"warm-start float64 reduced ||F||inf = {np.max(np.abs(f0)):.3e}")

    # ---- build dense FD Jacobian of reduced residual in float64 (once) ----
    n = red.n_red
    teval = time.time()
    F_base = Fred_np(x0)
    dt1 = time.time() - teval
    log(f"single float64 reduced eval = {dt1:.2f}s; building {n}x{n} FD Jacobian"
        f" (~{n*dt1/60:.1f} min)...")
    eps = 1e-7
    J = np.empty((n, n))
    tJ = time.time()
    for k in range(n):
        xp = x0.copy(); xp[k] += eps
        J[:, k] = (Fred_np(xp) - F_base) / eps
        if k % 25 == 0:
            log(f"  Jacobian col {k}/{n}  ({time.time()-tJ:.0f}s)")
    log(f"Jacobian built in {time.time()-tJ:.0f}s; cond={np.linalg.cond(J):.3e}")
    import scipy.linalg as sla
    lu, piv = sla.lu_factor(J)

    # ---- arb setup ----
    ui_a = [arb(repr(float(x))) for x in ui]
    gn_a, gw_a = Ha.gauss_legendre(NQ, -UMAX, UMAX)
    tau_a = [arb(int(TAU))] * 3
    gam_a = [arb(repr(GAMMA))] * 3
    W_a = [arb(1)] * 3
    twopi = Ha._twopi()

    # arb residual: returns list of arb (length n)
    def Fred_arb(vec_arb):
        P = expand_arb(vec_arb, red)
        Pn = Ha.phi_hfree(P, ui_a, gn_a, gw_a, tau_a, gam_a, W_a, SUB)
        red_v = reduce_arb(Pn, red)
        return [red_v[k] - vec_arb[k] for k in range(n)]

    def norm_inf_arb(vec):
        m = Ha.ZERO
        for v in vec:
            av = abs(v)
            if av > m:
                m = av
        return m

    # arb state
    x = [arb(repr(float(x0[k]))) for k in range(n)]
    teval = time.time()
    F = Fred_arb(x)
    dt_arb = time.time() - teval
    Finf = norm_inf_arb(F)
    log(f"single arb (100-dec) reduced eval = {dt_arb:.2f}s")
    log(f"arb warm-start ||F||inf = {float(Finf):.6e}")

    traj = [float(Finf)]
    best_x = list(x)
    best_Finf = Finf
    MAXSTEP = 40
    for step in range(1, MAXSTEP + 1):
        # solve J dx = -F  using float64 LU on the arb RHS.
        # Convert F to a high-precision-safe representation: we need dx in arb.
        # Apply J^{-1} columnwise: dx_i = sum_j (Jinv)_ij * (-F_j). We compute
        # Jinv once (float64) and form the arb matvec so dx carries arb digits.
        if step == 1:
            Jinv = sla.lu_solve((lu, piv), np.eye(n))
            Jinv_arb = [[arb(repr(float(Jinv[i, j]))) for j in range(n)]
                        for i in range(n)]
        negF = [(-F[j]) for j in range(n)]
        dx = []
        for i in range(n):
            s = Ha.ZERO
            row = Jinv_arb[i]
            for j in range(n):
                s = s + row[j] * negF[j]
            dx.append(s)
        xnew = [x[k] + dx[k] for k in range(n)]
        Fnew = Fred_arb(xnew)
        Finf_new = norm_inf_arb(Fnew)
        log(f"Newton step {step}: ||F||inf = {float(Finf_new):.6e}")
        traj.append(float(Finf_new))
        x = xnew
        F = Fnew
        if Finf_new < best_Finf:
            best_Finf = Finf_new
            best_x = list(x)
        # stop if at/below target or stagnating (no longer improving by >5%)
        if float(Finf_new) < 1e-100:
            log("reached < 1e-100")
            break
        if step >= 4 and float(Finf_new) > 0.95 * float(traj[-2]):
            log("stagnated (residual floor reached)")
            break

    x = best_x
    Finf = best_Finf
    lowest = float(Finf)
    reached = lowest < 1e-100
    log(f"LOWEST ||F||inf = {lowest:.6e}  reached_1e_100={reached}")

    # deficit at nailed point (float64 metrics on the arb solution)
    P_full_arb = expand_arb(x, red)
    P_full = np.array([[[float(P_full_arb[i][j][l]) for l in range(G)]
                        for j in range(G)] for i in range(G)])
    m = metrics(P_full, ui, TAU)
    log(f"deficit at nailed h-free PR = {m['deficit']:.6f}  slope_T={m['slope_T']:.5f}")

    # save P_nailed (arb strings)
    P_str = [[[x_.str(115, radius=False) if hasattr(x_, 'str') else str(x_)
               for x_ in row] for row in plane] for plane in P_full_arb]
    with open(os.path.join(OUTDIR, "P_nailed_100dec.json"), "w") as f:
        json.dump(P_str, f)
    # reduced strings too
    xred_str = [v.str(115, radius=False) for v in x]
    with open(os.path.join(OUTDIR, "P_nailed_100dec_reduced.json"), "w") as f:
        json.dump(xred_str, f)

    report = dict(
        title="K=3 CRRA REE h-FREE SMOOTH co-area -- strict h=0 NO-kernel "
              "100-decimal nail",
        date="2026-05-29",
        impl_path="python-flint (arb) operator + float64-frozen-Jacobian "
                  "arb Newton + K=3 symmetry reduction (165 unknowns)",
        why_not_cython="python-flint chosen per task fallback: correctness + "
                       "the 1e-100 result prioritized over raw speed; "
                       "warm-start keeps the flint nail cheap regardless.",
        G=G, tau=TAU, gamma=GAMMA, W=W, Nq=NQ, sub=SUB, UMAX=UMAX,
        prec_bits=flint.ctx.prec, prec_dec=round(flint.ctx.prec / 3.3219, 1),
        n_red=red.n_red,
        NO_kernel=True, NO_bandwidth=True, NO_smoothing_parameter=True,
        strict_h_zero=True,
        s_per_arb_eval_100dec=round(dt_arb, 2),
        s_per_float64_eval=round(dt1, 2),
        warmstart_float64_Finf=float(np.max(np.abs(f0))),
        arb_warmstart_Finf=traj[0],
        nail_trajectory_Finf=traj,
        lowest_Finf=lowest,
        reached_1e_100=bool(reached),
        deficit=m["deficit"], slope_T=m["slope_T"], d_FR=m["d_FR"],
    )
    with open(os.path.join(OUTDIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log("report.json + P_nailed_100dec.json written")
    log(f"VERDICT strict-h=0 no-kernel PR nailed to 1e-100 at 100-dec: "
        f"{'YES' if reached else 'NO'} (lowest ||F||inf={lowest:.3e})")
    return report


def metrics(P, ui, TAU):
    G = ui.size
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    T = TAU * (U1 + U2 + U3)
    Pc = np.clip(P, 1e-12, 1 - 1e-12)
    y = np.log(Pc / (1 - Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    pr = a[0] * T.ravel() + a[1]
    deficit = float(np.sum((y - pr) ** 2) /
                    max(np.sum((y - y.mean()) ** 2), 1e-30))
    P_FR = 1.0 / (1.0 + np.exp(-T))
    d_FR = float(np.sqrt(np.mean((P - P_FR) ** 2)))
    return dict(deficit=deficit, d_FR=d_FR, slope_T=float(a[0]))


if __name__ == "__main__":
    main()
