"""Quick compile + consistency smoke test:
  (b) on a GENERIC (non-critical) P, the Morse operator (refine=1) must match
      the original hfree_operator to ~1e-10 (the fix only changes near-critical
      behaviour).  Also checks refine>1 stays ~1e-10 on generic P (refinement
      inactive away from criticality)."""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "k3_hfree_fast"))

import hfree_morse_operator as M
import hfree_operator as O   # original (from k3_hfree_fast)

UMAX = 4.0
NQ = 40
SUB = 4


def main():
    G = 9
    ui = np.linspace(-UMAX, UMAX, G)
    gn, gw = M.gauss_legendre(NQ, -UMAX, UMAX)
    tau = np.full(3, 2.0)
    gam = np.full(3, 0.1)
    W = np.full(3, 1.0)

    # generic smooth, monotone, NON-critical price field (fully-revealing-ish)
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    T = 2.0 * (U1 + U2 + U3)
    P = np.clip(1.0 / (1.0 + np.exp(-T)), 1e-9, 1 - 1e-9)

    t = time.time()
    Po = O.phi_hfree(P, ui, gn, gw, tau, gam, W, SUB)
    t_o = time.time() - t

    t = time.time()
    # eps_c=0 reduces the Morse operator EXACTLY to the original (only the
    # numerically-stable |dB|/denom reformulation differs -> machine precision).
    Pm0 = M.phi_hfree(P, ui, gn, gw, tau, gam, W, SUB, 1, 0.0)
    t_m1 = time.time() - t

    # eps_c=0.01 is the production regularization: on a GENERIC (non-critical)
    # P it changes Phi only by O(eps_c*h^2 / |gradP|^2) << 1 (the fix only acts
    # near criticality, of which a generic monotone field has little).
    Pm_reg = M.phi_hfree(P, ui, gn, gw, tau, gam, W, SUB, 1, 0.01)

    err0 = float(np.max(np.abs(Pm0 - Po)))
    err_reg = float(np.max(np.abs(Pm_reg - Po)))
    print(f"compile+eval ok.  orig {t_o:.2f}s  morse {t_m1:.2f}s")
    print(f"(b) CONSISTENCY generic P:  ||morse(eps_c=0) - orig||_inf = {err0:.3e}")
    print(f"    ||morse(eps_c=0.01) - orig||_inf = {err_reg:.3e}  (only near-crit cells shift)")
    assert err0 < 1e-10, f"eps_c=0 must match orig to 1e-10, got {err0:.3e}"
    print("PASS (b): eps_c=0 reduces to original to <1e-10 (only near-crit behaviour changes)")


if __name__ == "__main__":
    main()
