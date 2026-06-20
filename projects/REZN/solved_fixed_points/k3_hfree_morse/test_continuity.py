"""Validation (a): CONTINUITY across a Morse-critical price.

Pick a slice of a real nailed price field, locate a critical value p* of the
slice surface (where |grad P| has a minimum on a level set -> a saddle/extremum
where the contour topology changes), then sweep p across p* at decreasing step
and confirm A_v(p) and mu(p) are CONTINUOUS for the Morse operator (jump -> 0 as
step -> 0), vs the ORIGINAL operator which kinks / jumps there.

Quantifies max |Delta A_v| / step and max |Delta mu| / step on a fine sweep
(a finite, bounded difference quotient = continuous; a blow-up = jump).
"""
import os, sys, json
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "k3_hfree_fast"))
import hfree_morse_operator as M
import hfree_operator as O

UMAX = 4.0
NQ = 40
SUB = 4
BASE = os.path.dirname(HERE)


def find_critical_price(S, h, u0, gn):
    """Scan the slice spline gradient magnitude on a fine grid; return a price
    p* near where min|grad P| over a level set is smallest = a Morse-critical
    value (saddle/extremum), and the global price range."""
    G = S.shape[0]
    colM = np.empty((G, G))
    for ib in range(G):
        colM[ib, :] = M.natural_spline_M(S[:, ib], h)
    rowM = np.empty((G, G))
    for ia in range(G):
        rowM[ia, :] = M.natural_spline_M(S[ia, :], h)
    # sample grad magnitude on a fine mesh
    nf = 121
    us = np.linspace(u0, u0 + (G - 1) * h, nf)
    Pv = np.empty((nf, nf)); Gm = np.empty((nf, nf))
    for ia, ua in enumerate(us):
        # value/dB along row ua via column splines then row spline
        rowvals = np.empty(G); rowdA = np.empty(G)
        for ib in range(G):
            v, d = M.spline_eval(S[:, ib], colM[ib, :], h, u0, ua)
            rowvals[ib] = v; rowdA[ib] = d
        Mr = M.natural_spline_M(rowvals, h)
        MrdA = M.natural_spline_M(rowdA, h)
        for ib, ub in enumerate(us):
            val, dB = M.spline_eval(rowvals, Mr, h, u0, ub)
            dA, _ = M.spline_eval(rowdA, MrdA, h, u0, ub)
            Pv[ia, ib] = val
            Gm[ia, ib] = np.hypot(dA, dB)
    # interior critical point: smallest grad-magnitude not on the boundary
    interior = Gm[5:-5, 5:-5]
    idx = np.unravel_index(np.argmin(interior), interior.shape)
    p_star = float(Pv[5:-5, 5:-5][idx])
    return p_star, float(Pv.min()), float(Pv.max())


EPS_C = 0.01


def sweep(opfn, S, h, u0, ps, gn, gw, tauA, tauB, morse=False):
    if not morse:
        A0, A1 = opfn(S, h, u0, ps, gn, gw, tauA, tauB, SUB)
    else:
        A0, A1 = opfn(S, h, u0, ps, gn, gw, tauA, tauB, SUB, 1, EPS_C)
    return A0, A1


def main():
    G = 17
    P = np.load(os.path.join(BASE, "k3_coarea_limit", f"P_inner_G{G}.npy"))
    # symmetric, so any agent slice works; take agent-0 slice at a mid own-index
    i_own = G // 2
    ui = np.linspace(-UMAX, UMAX, G)
    h = ui[1] - ui[0]; u0 = ui[0]
    gn, gw = M.gauss_legendre(NQ, -UMAX, UMAX)
    S = P[i_own, :, :]
    tauA = 2.0; tauB = 2.0; tau_own = 2.0
    u_own = ui[i_own]

    p_star, pmin, pmax = find_critical_price(S, h, u0, gn)
    print(f"slice critical price p* ~ {p_star:.6f}  (range {pmin:.4f}..{pmax:.4f})")

    # fine sweep across p* at two step sizes -> check jump shrinks with step
    res = {"G": G, "p_star": p_star, "p_range": [pmin, pmax]}
    half = max(0.02, 0.05 * (pmax - pmin))
    for npts, key in [(81, "coarse"), (321, "fine"), (1281, "finer")]:
        ps = np.clip(np.linspace(p_star - half, p_star + half, npts),
                     pmin + 1e-4, pmax - 1e-4)
        step = ps[1] - ps[0]
        # original
        A0o, A1o = sweep(O.slice_Av_sweep, S, h, u0, ps, gn, gw, tauA, tauB)
        muo = np.array([O.bayes(u_own, tau_own, A0o[k], A1o[k])
                        for k in range(len(ps))])
        # morse (regularized)
        A0m, A1m = sweep(M.slice_Av_sweep, S, h, u0, ps, gn, gw, tauA, tauB,
                         morse=True)
        mum = np.array([M.bayes_from_ratio(u_own, tau_own, A0m[k], A1m[k])
                        for k in range(len(ps))])
        # difference-quotient: max |dA/dp|*step = max consecutive jump
        rA = (A0o + A1o)
        rM = (A0m + A1m)
        jump_o_A = float(np.max(np.abs(np.diff(rA))))
        jump_m_A = float(np.max(np.abs(np.diff(rM))))
        jump_o_mu = float(np.max(np.abs(np.diff(muo))))
        jump_m_mu = float(np.max(np.abs(np.diff(mum))))
        res[key] = dict(step=float(step),
                        orig_maxjump_A=jump_o_A, morse_maxjump_A=jump_m_A,
                        orig_maxjump_mu=jump_o_mu, morse_maxjump_mu=jump_m_mu)
        print(f"  [{key}] step={step:.2e}  maxjump A: orig={jump_o_A:.3e} "
              f"morse={jump_m_A:.3e} | maxjump mu: orig={jump_o_mu:.3e} "
              f"morse={jump_m_mu:.3e}")

    res["eps_c"] = EPS_C
    # continuity verdict: as the sweep step shrinks (4x then 4x more pts), a
    # CONTINUOUS function's max consecutive mu-jump scales ~ step (-> 0); a true
    # discontinuity (jump) does NOT shrink.  We measure the shrink factor across
    # two successive 4x refinements for both operators.
    c = res["coarse"]; f = res["fine"]; ff = res["finer"]
    res["step_ratio"] = c["step"] / f["step"]
    res["morse_mu_jump_shrink_1"] = c["morse_maxjump_mu"] / max(f["morse_maxjump_mu"], 1e-30)
    res["morse_mu_jump_shrink_2"] = f["morse_maxjump_mu"] / max(ff["morse_maxjump_mu"], 1e-30)
    res["orig_mu_jump_shrink_1"] = c["orig_maxjump_mu"] / max(f["orig_maxjump_mu"], 1e-30)
    res["orig_mu_jump_shrink_2"] = f["orig_maxjump_mu"] / max(ff["orig_maxjump_mu"], 1e-30)
    # CONTINUOUS if the mu-jump keeps shrinking as step shrinks (both refinements)
    res["morse_continuous"] = bool(
        f["morse_maxjump_mu"] < c["morse_maxjump_mu"] * 0.7 and
        ff["morse_maxjump_mu"] < f["morse_maxjump_mu"] * 0.7)
    res["orig_jumps"] = bool(
        not (ff["orig_maxjump_mu"] < f["orig_maxjump_mu"] * 0.7))
    print(f"  step shrinks 4x then 4x:")
    print(f"    MORSE mu-jump shrink: {res['morse_mu_jump_shrink_1']:.2f}x then "
          f"{res['morse_mu_jump_shrink_2']:.2f}x  (continuous -> keeps shrinking ~4x)")
    print(f"    ORIG  mu-jump shrink: {res['orig_mu_jump_shrink_1']:.2f}x then "
          f"{res['orig_mu_jump_shrink_2']:.2f}x  (jump -> stalls ~1x)")
    print(f"  MORSE CONTINUOUS across p*: {res['morse_continuous']}  | "
          f"ORIG jumps (non-shrinking): {res['orig_jumps']}")
    res["orig_peakA"] = float(c["orig_maxjump_A"])
    res["morse_peakA"] = float(c["morse_maxjump_A"])
    json.dump(res, open(os.path.join(HERE, "continuity.json"), "w"), indent=2)
    print("wrote continuity.json")
    return res


if __name__ == "__main__":
    main()
