"""Validation of the strict h=0 co-area integrator.

(a) Known analytic surface: P depends only on ua (a monotone ramp), so the level
    set {P=p} is the vertical line ua=ua*(p), and
       A_v(p) = int f_v(ua*) f_v(ub) / |dP/dua| dub
              = [f_v(ua*) / |dP/dua|] * int_{ub} f_v(ub) dub
    With P = sigmoid(k*ua), dP/dua = k P(1-P), ua* = logit(p)/k, and ub over the
    full real line int f_v(ub) dub = 1 (a Gaussian density). On a finite grid the
    ub-integral is the truncated Gaussian mass over [ulo,uhi]. We compare the
    marching-squares result to this closed form.

(b) Continuity of A_v(p) in p: sample many p, confirm no O(0.1) jumps (the naive
    edge-crossing scan would jump; the exact contour integral must be smooth).

Also a consistency check vs the Gaussian-band (kernel) operator as h->0.
"""
import sys, json, time
sys.path.insert(0, '/tmp')
from strict_h0_operator import (arb, ctx, ONE, TWO, PI, HALF, VM0, VM1,
                                f_signal, agent_evidence_strict, build_grid)

t0 = time.time()
LOG = open('/tmp/strict_h0_validate.out', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()

log("=== STRICT h=0 co-area integrator validation ===")
log("prec bits =", ctx.prec)

# ---- build a FINE grid for the analytic test (isolate integrator accuracy from
#      the bilinear discretization error of representing a smooth surface) ----
G_inner, pad, UMAX = 81, 0, 6.0
du, u_full, lo, hi = build_grid(G_inner, pad, UMAX)
G = len(u_full)
ulo, uhi = u_full[0], u_full[-1]
log(f"grid G={G} du={float(du):.4f} u in [{float(ulo):.3f},{float(uhi):.3f}]")

# ---- analytic test surface P = sigmoid(k*ua) (depends only on axis a) ----
TAU_A = arb(2); TAU_B = arb(2)
K = arb('1.3')

def sig(x):
    return ONE / (ONE + (-x).exp())

# slice[a][b] = sigmoid(k * u_full[a])
slc = [[sig(K * u_full[a]) for b in range(G)] for a in range(G)]

def gauss_cdf_mass(tau, vm):
    """int_{ulo}^{uhi} f_v(ub) dub for f_v Gaussian(mean vm, var 1/tau)."""
    # = Phi((uhi-vm)sqrt(tau)) - Phi((ulo-vm)sqrt(tau)); use erf
    s = tau.sqrt()
    def Phi(x):
        return HALF * (ONE + (x / TWO.sqrt()).erf())
    return Phi((uhi - vm) * s) - Phi((ulo - vm) * s)

def analytic_A(p, vm, tau_a, tau_b):
    """closed form for the ramp surface."""
    # ua* = logit(p)/k ; dP/dua = k*p*(1-p)
    ua_star = (p.log() - (ONE - p).log()) / K
    grad = K * p * (ONE - p)
    fa = f_signal(ua_star, vm, tau_a)
    mass_b = gauss_cdf_mass(tau_b, vm)
    return fa / grad * mass_b

log("\n(a) KNOWN-SURFACE check: P=sigmoid(k*ua), exact co-area vs marching-squares")
errs = []
for ps in ['0.3', '0.45', '0.5', '0.6', '0.72']:
    p = arb(ps)
    A0, A1 = agent_evidence_strict(slc, p, u_full, TAU_A, TAU_B, du, G)
    ex0 = analytic_A(p, VM0, TAU_A, TAU_B)
    ex1 = analytic_A(p, VM1, TAU_A, TAU_B)
    e0 = abs(float((A0 - ex0) / ex0))
    e1 = abs(float((A1 - ex1) / ex1))
    errs.append(max(e0, e1))
    log(f"  p={ps}: A0={float(A0):.10e} exact={float(ex0):.10e} relerr={e0:.3e} | "
        f"A1 relerr={e1:.3e}")
maxerr_a = max(errs)
log(f"  MAX rel error (known surface) = {maxerr_a:.3e}")

# ---- consistency vs Gaussian-band operator as h->0 on the SAME ramp slice ----
log("\n(consistency) strict h=0 vs Gaussian-band kernel as h->0 (same ramp slice)")
def band_A(p, h, vm, tau_a, tau_b):
    inv2h2 = HALF / (h * h)
    s = arb(0)
    for a in range(G):
        fa = f_signal(u_full[a], vm, tau_a)
        for b in range(G):
            fb = f_signal(u_full[b], vm, tau_b)
            d = slc[a][b] - p
            s = s + (-d * d * inv2h2).exp() * fa * fb
    return s
# band sum approximates int K_h(P-p) f f du_a du_b. As h->0 with grid fixed the
# Riemann band-sum -> du^2 * (1/(h sqrt(2pi)))... compare NORMALISED shapes.
# Simpler/cleaner consistency: integral of A over p should match; instead show
# the band operator converges to strict as h shrinks (ratio -> const indep of p).
p_test = arb('0.5')
prev_ratio = None
for hs in ['0.6', '0.4', '0.25', '0.15']:
    h = arb(hs)
    bA0 = band_A(p_test, h, VM0, TAU_A, TAU_B)
    sA0, _ = agent_evidence_strict(slc, p_test, u_full, TAU_A, TAU_B, du, G)
    # band sum has units of [density * 1] summed over grid pts; convert to
    # integral estimate: * du^2, then it approximates h*sqrt(2pi)*A_strict
    band_int = bA0 * du * du
    pred = h * (TWO * PI).sqrt() * sA0
    ratio = float(band_int / pred)
    log(f"  h={hs}: band_int/(h sqrt2pi A_strict) = {ratio:.6f}  (->1 as h->0, grid permitting)")

# ---- (b) continuity of A_v(p) in p ----
log("\n(b) CONTINUITY of A_v(p): interior p-sweep, compare to NAIVE edge-scan")
# interior range 0.15..0.85 where the level set is well inside the grid; here a
# C0-continuous A0 must have jumps that vanish as dp->0 (proportional to dp).
def naive_scan_A(p, vm_a, vm_b, tau_a, tau_b):
    """The biased 'count edge crossings' scan: for each cell that the level p
    crosses, add f_a*f_b at the cell-center (NO 1/|gradP|). This is the method
    the task warns has O(0.1) tie jumps."""
    s = arb(0)
    for a in range(G - 1):
        for b in range(G - 1):
            corners = [slc[a][b], slc[a+1][b], slc[a][b+1], slc[a+1][b+1]]
            above = sum(1 for c in corners if c >= p)
            if above == 0 or above == 4:
                continue
            uam = (u_full[a] + u_full[a+1]) * HALF
            ubm = (u_full[b] + u_full[b+1]) * HALF
            s = s + f_signal(uam, vm_a, tau_a) * f_signal(ubm, vm_b, tau_b)
    return s

def dp_jump_stats(dp, use_naive):
    npts = int(0.7 / dp) + 1
    prev = None
    mx = arb(0)
    for k in range(npts):
        pv = 0.15 + k * float(dp)
        p = arb(str(round(pv, 8)))
        if use_naive:
            v = naive_scan_A(p, VM0, VM0, TAU_A, TAU_B)
        else:
            v, _ = agent_evidence_strict(slc, p, u_full, TAU_A, TAU_B, du, G)
        if prev is not None and prev > 0:
            j = abs((v - prev) / prev)
            if j > mx:
                mx = j
        prev = v
    return float(mx)

# strict: jump should roughly HALVE when dp halves (C0/C1 continuous)
j_strict_coarse = dp_jump_stats(0.02, False)
j_strict_fine = dp_jump_stats(0.01, False)
j_naive_coarse = dp_jump_stats(0.02, True)
j_naive_fine = dp_jump_stats(0.01, True)
log(f"  STRICT contour : max interior rel jump  dp=0.02 -> {j_strict_coarse:.3e},"
    f"  dp=0.01 -> {j_strict_fine:.3e}  (ratio {j_strict_coarse/max(j_strict_fine,1e-300):.2f} ~ continuous)")
log(f"  NAIVE edge-scan: max interior rel jump  dp=0.02 -> {j_naive_coarse:.3e},"
    f"  dp=0.01 -> {j_naive_fine:.3e}  (ratio ~1 => DISCONTINUOUS tie-jumps)")
maxjump = j_strict_fine

report = dict(
    known_surface_max_relerr=maxerr_a,
    continuity_strict_jump_dp02=j_strict_coarse,
    continuity_strict_jump_dp01=j_strict_fine,
    continuity_naive_jump_dp02=j_naive_coarse,
    continuity_naive_jump_dp01=j_naive_fine,
    prec_bits=ctx.prec,
    grid=dict(G_inner=G_inner, pad=pad, UMAX=UMAX, du=float(du)),
    walltime_s=round(time.time() - t0, 1),
)
json.dump(report, open('/tmp/strict_h0_validate.json', 'w'), indent=2)
log("\nvalidation walltime", report['walltime_s'], "s")
log("VALIDATION_DONE")
