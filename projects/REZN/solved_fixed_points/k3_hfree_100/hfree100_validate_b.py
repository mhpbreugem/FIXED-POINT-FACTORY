"""Validation (b): precision self-consistency. The SAME arb operator run at
~110-dec vs ~210-dec must agree to ~1e-100 on a test P -- confirms the port
scales to full precision with NO hidden float / fixed-low-precision floor.

Test P is an analytic SMOOTH surface given EXACTLY in arb (sigmoid(0.4*sum u))
so the input itself carries no float64 floor that would mask the comparison.
"""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_100")
import flint
from flint import arb
import hfree_operator_arb as Ha

UMAX = 4.0
NQ = 40
SUB = 4
G = 9
TAU = 2.0
GAMMA = 0.1


def run_analytic(dec):
    Ha.set_precision(dec)
    ui_a = [arb(-4) + arb(i) * arb(1) for i in range(G)]   # exact integer grid
    P_a = [[[Ha.lam(arb("0.4") * (ui_a[i] + ui_a[j] + ui_a[l]))
             for l in range(G)] for j in range(G)] for i in range(G)]
    gn_a, gw_a = Ha.gauss_legendre(NQ, -UMAX, UMAX)
    tau_a = [arb(int(TAU))] * 3
    gam_a = [arb("0.1")] * 3      # gamma=0.1 exact-ish; same value both precisions
    W_a = [arb(1)] * 3
    t0 = time.time()
    out = Ha.phi_hfree(P_a, ui_a, gn_a, gw_a, tau_a, gam_a, W_a, SUB)
    return out, time.time() - t0, flint.ctx.prec


out100, dt100, pb100 = run_analytic(110)
out200, dt200, pb200 = run_analytic(210)

maxdiff = arb(0)
for i in range(G):
    for j in range(G):
        for l in range(G):
            d = abs(out100[i][j][l] - out200[i][j][l])
            if d > maxdiff:
                maxdiff = d
md = float(maxdiff)
print(f"prec100={pb100} bits ({dt100:.1f}s), prec200={pb200} bits ({dt200:.1f}s)")
print(f"max |out_100dec - out_200dec| = {md:.3e}")
res = dict(prec100_bits=pb100, prec200_bits=pb200, t_100=dt100, t_200=dt200,
           max_abs_diff_100_vs_200=md, max_abs_diff_str=maxdiff.str(8),
           agree_1e_100=bool(md < 1e-95),
           testP="sigmoid(0.4*(u1+u2+u3)) exact in arb")
with open("/tmp/hfree100_valid_b.json", "w") as f:
    json.dump(res, f, indent=2)
print("VALID_B", json.dumps(res))
