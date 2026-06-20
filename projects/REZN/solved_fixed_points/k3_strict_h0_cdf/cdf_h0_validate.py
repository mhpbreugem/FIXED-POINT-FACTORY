"""Validation of the CDF-derivative co-area density A_v(p)=dG_v/dp.

(a) On a known analytic surface P=sigmoid(k(u_b^2 - u_c)) -- which has a
    CRITICAL LINE u_b=0 where dP/du_b=0 (|grad P| collapses to |dP/du_c|, and
    along the bottom of the parabola the level set is tangent -> the place
    where the marching-squares 1/|grad| operator BLEW UP) -- check that the
    CDF-derivative A(p) matches the EXACT co-area integral, INCLUDING near the
    critical price p* = sigmoid(0) where the level set is tangent to u_b=0.

    Exact reference: A_v(p) = d/dp INT_{P<p} f du.  We build an INDEPENDENT
    reference two ways:
      (R1) Analytic.  With P=sigmoid(k(u_b^2-u_c)), {P<p} = {u_c > u_b^2 - z/k}
           where z=logit(p).  So
             G(p) = INT_{u_b} f_b(u_b) [ INT_{u_c > u_b^2 - z/k} f_c(u_c) du_c ] du_b
                  = INT_{u_b} f_b(u_b) * Q(u_b^2 - z/k) du_b ,  Q = upper-CDF.
           A(p)=dG/dp = (dz/dp) INT_{u_b} f_b(u_b) f_c(u_b^2 - z/k) (1/k) du_b.
           Computed by fine 1D Gauss-Legendre in u_b (no 2D grid, no co-area
           singularity) -> ground truth, finite everywhere incl critical p.
    Compare CDF-derivative grid operator to R1 over a sweep of p incl p*.

(b) Continuity of A(p) in p: finite differences of the operator A(p) are
    bounded (no jumps) across the sweep, esp at cell-knot prices.
"""
import os, sys, json, time
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
sys.path.insert(0, '/tmp')
from flint import ctx, arb
import cdf_h0_operator as OP
from cdf_h0_operator import f_signal, agent_evidence_cdf, VM0, VM1, HALF, ONE, TWO, PI

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_strict_h0_cdf'
os.makedirs(OUT, exist_ok=True)

# Use mean-0 signals for the analytic test so f is symmetric Gaussian.
def fG(u, tau, mean):
    d = u - mean
    return (tau / (TWO * PI)).sqrt() * (-HALF * tau * d * d).exp()

def sigmoid(x):
    if x >= 0:
        e = (-x).exp(); return ONE / (ONE + e)
    e = x.exp(); return e / (ONE + e)

def logit(p):
    return p.log() - (ONE - p).log()

# High-order Gauss-Legendre on [-1,1] (build via Newton on Legendre)
def gauss_legendre(N):
    nodes = []; weights = []
    for i in range(1, N + 1):
        x = (PI * (arb(i) - arb('0.25')) / (arb(N) + HALF)).cos()  # init
        for _ in range(100):
            p0 = ONE; p1 = x
            for k in range(2, N + 1):
                p2 = ((TWO * arb(k) - ONE) * x * p1 - (arb(k) - ONE) * p0) / arb(k)
                p0 = p1; p1 = p2
            dp = arb(N) * (x * p1 - p0) / (x * x - ONE)
            dx = p1 / dp
            x = x - dx
            if abs(float(dx)) < 1e-110:
                break
        nodes.append(x)
        weights.append(TWO / ((ONE - x * x) * dp * dp))
    return nodes, weights


def main():
    t0 = time.time()
    rep = {}
    # ---- test surface params ----
    k = arb('1.5')
    TAU_B = arb('1.0')   # f_b density precision (mean 0)
    TAU_C = arb('1.0')
    MEAN = arb('0')
    UMAX = 6.0

    # critical price: at u_b=0 the level set u_c = -z/k is tangent; the
    # "critical" structure is around z=0 -> p* = 0.5 (the parabola vertex).
    pstar = arb('0.5')

    # ---- R1 analytic reference A(p) via fine 1D GL in u_b ----
    Nb = 60
    nb, wb = gauss_legendre(Nb)
    # map [-1,1] -> [-UMAX,UMAX]
    ub_nodes = [arb(UMAX) * x for x in nb]
    ub_w = [arb(UMAX) * w for w in wb]

    def A_exact(p):
        z = logit(p)
        dzdp = ONE / (p * (ONE - p))
        s = arb(0)
        for n in range(Nb):
            ub = ub_nodes[n]
            uc_star = ub * ub - z / k
            fb = fG(ub, TAU_B, MEAN)
            fc = fG(uc_star, TAU_C, MEAN)
            s = s + ub_w[n] * fb * fc / k
        return dzdp * s

    # ---- grid operator A(p) on a fine 2D slice ----
    # slice axis a = u_b (with tau_a), b = u_c (tau_b). P[i][j]=sigmoid(k(ub_i^2-uc_j))
    G = 81
    NSUB = 6
    du = arb(2 * UMAX) / arb(G - 1)
    u_axis = [arb(-UMAX) + arb(q) * du for q in range(G)]
    P_slice = [[sigmoid(k * (u_axis[i] * u_axis[i] - u_axis[j]))
                for j in range(G)] for i in range(G)]

    def A_grid(p):
        # agent over this slice: A0 uses mean VM0 etc; but for the pure
        # geometric test we want f with MEAN=0 on BOTH axes -> reuse the
        # operator by setting taus and reading A0 with means built in.
        # The operator hardcodes means VM0/VM1. To match the analytic test
        # (mean 0 on both), we call it and combine: it returns A for mean
        # -0.5 (A0) and +0.5 (A1). We instead want mean-0; so compute a
        # mean-0 variant inline mirroring agent_evidence_cdf.
        return _A_grid_mean0(P_slice, p, u_axis, TAU_B, TAU_C, du, G, NSUB)

    p_list = [arb('0.10'), arb('0.25'), arb('0.40'), arb('0.48'),
              arb('0.50'), arb('0.52'), arb('0.60'), arb('0.75'), arb('0.90')]
    errs = []
    rows = []
    for p in p_list:
        ae = A_exact(p)
        ag = A_grid(p)
        rel = abs(float((ag - ae) / ae)) if float(ae) != 0 else float('nan')
        errs.append(rel)
        near = abs(float(p - pstar)) < 1e-9 or abs(float(p) - 0.5) < 0.03
        rows.append(dict(p=float(p), A_exact=float(ae), A_grid=float(ag),
                         rel_err=rel, near_critical=bool(near)))
        print(f"p={float(p):.3f} A_exact={float(ae):.6e} A_grid={float(ag):.6e} "
              f"rel={rel:.3e} {'<-critical' if near else ''}", flush=True)

    max_rel = max(errs)
    crit_rel = [r['rel_err'] for r in rows if r['near_critical']]
    rep['known_surface_max_relerr'] = max_rel
    rep['near_critical_max_relerr'] = max(crit_rel) if crit_rel else None
    rep['rows'] = rows
    rep['finite_near_critical'] = all(
        (r['A_grid'] == r['A_grid'] and abs(r['A_grid']) < 1e6)
        for r in rows if r['near_critical'])

    # ---- (b) continuity: dense sweep, max jump ----
    dp = arb('0.0025')
    pgrid = [arb('0.05') + arb(q) * dp for q in range(int(0.90 / 0.0025))]
    vals = [float(A_grid(p)) for p in pgrid]
    jumps = [abs(vals[q + 1] - vals[q]) for q in range(len(vals) - 1)]
    # normalize jump by local scale
    scale = max(vals) if max(vals) > 0 else 1.0
    rep['continuity_max_abs_jump'] = max(jumps)
    rep['continuity_max_rel_jump'] = max(jumps) / scale
    rep['continuity_dp'] = float(dp)
    rep['continuity_npts'] = len(pgrid)
    print(f"\ncontinuity: max abs jump (dp={float(dp)}) = {max(jumps):.3e}  "
          f"rel = {max(jumps)/scale:.3e}", flush=True)

    rep['walltime_s'] = round(time.time() - t0, 1)
    rep['config'] = dict(k=float(k), TAU_B=float(TAU_B), TAU_C=float(TAU_C),
                         UMAX=UMAX, G=G, NSUB=NSUB, Nb_GL=Nb,
                         prec_bits=ctx.prec)
    json.dump(rep, open(os.path.join(OUT, 'validate.json'), 'w'), indent=2)
    print(f"\nVALIDATION: max_rel={max_rel:.3e}  near_crit_max_rel="
          f"{rep['near_critical_max_relerr']:.3e}  finite_near_crit="
          f"{rep['finite_near_critical']}  cont_rel_jump="
          f"{rep['continuity_max_rel_jump']:.3e}  ({rep['walltime_s']}s)",
          flush=True)
    return rep


def _A_grid_mean0(P_slice, p_target, u_axis, tau_a, tau_b, du, n, NSUB):
    """Mean-0 variant of agent_evidence_cdf returning a single A (both signal
    means 0) for the geometric validation."""
    from cdf_h0_operator import dA_dp_affine, _bilin
    A = arb(0)
    hsub = ONE / arb(NSUB)
    cell_area = du * du
    sub_area = cell_area * hsub * hsub
    MEAN = arb('0')
    for i in range(n - 1):
        ua_i = u_axis[i]
        for j in range(n - 1):
            ub_j = u_axis[j]
            P00 = P_slice[i][j]; P10 = P_slice[i + 1][j]
            P01 = P_slice[i][j + 1]; P11 = P_slice[i + 1][j + 1]
            cmin = P00; cmax = P00
            for c in (P10, P01, P11):
                if c < cmin: cmin = c
                if c > cmax: cmax = c
            if p_target <= cmin or p_target >= cmax:
                continue
            for a in range(NSUB):
                s0 = arb(a) * hsub; s1 = arb(a + 1) * hsub
                sc = (s0 + s1) * HALF
                ua = ua_i + sc * du
                fa = fG(ua, tau_a, MEAN)
                for b in range(NSUB):
                    t0 = arb(b) * hsub; t1 = arb(b + 1) * hsub
                    tc = (t0 + t1) * HALF
                    q00 = _bilin(P00, P10, P01, P11, s0, t0)
                    q10 = _bilin(P00, P10, P01, P11, s1, t0)
                    q01 = _bilin(P00, P10, P01, P11, s0, t1)
                    q11 = _bilin(P00, P10, P01, P11, s1, t1)
                    smin = q00; smax = q00
                    for c in (q10, q01, q11):
                        if c < smin: smin = c
                        if c > smax: smax = c
                    if p_target <= smin or p_target >= smax:
                        continue
                    bx = q10 - q00; by = q01 - q00
                    dadp = dA_dp_affine(q00, bx, by, p_target)
                    if dadp <= 0:
                        continue
                    ub = ub_j + tc * du
                    fb = fG(ub, tau_b, MEAN)
                    A = A + dadp * sub_area * fa * fb
    return A


def fG(u, tau, mean):
    d = u - mean
    return (tau / (TWO * PI)).sqrt() * (-HALF * tau * d * d).exp()


if __name__ == '__main__':
    main()
