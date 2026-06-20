"""Validation (a): 100-dec arb operator vs float64 hfree_operator.phi_hfree.
On the nailed test P at G=9, tau=2, gamma=0.1. Expect match ~1e-12 (float64).
"""
import os, sys, time, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
sys.path.insert(0, "/tmp")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_100")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_fast")
import numpy as np
import flint
from flint import arb
import hfree_operator as Hf64        # float64 numba (from k3_hfree_fast)
import hfree_operator_arb as Ha      # arb

UMAX = 4.0
NQ = 40
SUB = 4
G = 9
TAU = 2.0
GAMMA = 0.1

P = np.load("/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_fast/P_nailed_G9.npy")

ui = np.linspace(-UMAX, UMAX, G)
gn, gw = Hf64.gauss_legendre(NQ, -UMAX, UMAX)
tau = np.full(3, TAU); gam = np.full(3, GAMMA); W = np.full(3, 1.0)

t0 = time.time()
out_f64 = Hf64.phi_hfree(P, ui, gn, gw, tau, gam, W, SUB)
t_f64 = time.time() - t0
print(f"float64 phi eval: {t_f64:.3f}s", flush=True)

# arb inputs
def to_arb_grid(ui_np):
    return [arb(repr(float(x))) for x in ui_np]
def to_arb_P(P_np):
    return [[[arb(repr(float(P_np[i][j][l]))) for l in range(G)]
             for j in range(G)] for i in range(G)]

ui_a = to_arb_grid(ui)
P_a = to_arb_P(P)
gn_a, gw_a = Ha.gauss_legendre(NQ, -UMAX, UMAX)
tau_a = [arb(TAU)] * 3
gam_a = [arb(repr(GAMMA))] * 3
W_a = [arb(1)] * 3

t0 = time.time()
out_a = Ha.phi_hfree(P_a, ui_a, gn_a, gw_a, tau_a, gam_a, W_a, SUB)
t_a = time.time() - t0
print(f"arb phi eval (100-dec): {t_a:.3f}s", flush=True)

# compare
maxdiff = 0.0
for i in range(G):
    for j in range(G):
        for l in range(G):
            af = float(out_a[i][j][l])
            d = abs(af - float(out_f64[i][j][l]))
            if d > maxdiff:
                maxdiff = d
print(f"max |arb - float64| = {maxdiff:.3e}")

res = dict(t_f64=t_f64, t_arb=t_a, max_abs_diff_vs_float64=maxdiff,
           NQ=NQ, SUB=SUB, G=G, prec_bits=flint.ctx.prec)
with open("/tmp/hfree100_valid_a.json", "w") as f:
    json.dump(res, f, indent=2)
print("VALID_A", json.dumps(res))
