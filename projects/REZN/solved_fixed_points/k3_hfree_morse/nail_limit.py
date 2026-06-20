"""THE NAIL + CONTINUUM driver for the Morse-robust no-kernel operator.

(c) Self-continuation warm-start (G=9 solution interpolated up to 13 then 17),
    orbit-average residual + Newton-Krylov.  Reports ||F||_inf at G=9,13,17.
    TARGET < 1e-9 at all three (the OLD operator floored ~1e-3 at G>=13).
(d) Richardson-extrapolate deficit(G) from G=9,13,17 to G->inf; compare to the
    KERNEL method continuum (~0.26, from k3_coarea_limit/report.json).
    RECONCILED if |extrap_hfree - kernel_continuum| < ~0.02.

Also writes the deficit-vs-G figure (Morse-robust h-free vs kernel, both
extrapolated) and report.json with validation (a)+(b) merged in.
"""
import os, sys, json, time
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import morse_solver as S
from morse_solver import UMAX, SymReducer3

BASE = os.path.dirname(HERE)
LOG = open(os.path.join(HERE, "run.log"), "a")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


def interp_up(Pprev, Gprev, Gnew):
    axp = np.linspace(-UMAX, UMAX, Gprev)
    axn = np.linspace(-UMAX, UMAX, Gnew)
    rgi = RegularGridInterpolator((axp, axp, axp), Pprev,
                                  bounds_error=False, fill_value=None)
    A, B, C = np.meshgrid(axn, axn, axn, indexing="ij")
    pts = np.column_stack([A.ravel(), B.ravel(), C.ravel()])
    return np.clip(rgi(pts).reshape((Gnew,) * 3), 1e-9, 1 - 1e-9)


def richardson(Gs, D):
    """Fit D(G) = c0 + c1/G^p + c2/G^(2p) by least squares over a few p, pick
    the p with smallest residual; return c0 = G->inf extrapolation."""
    Gs = np.asarray(Gs, float); D = np.asarray(D, float)
    best = None
    for p in [1.0, 1.3, 1.5, 2.0]:
        if len(Gs) >= 3:
            Amat = np.column_stack([np.ones_like(Gs), Gs**(-p), Gs**(-2 * p)])
        else:
            Amat = np.column_stack([np.ones_like(Gs), Gs**(-p)])
        c, res, *_ = np.linalg.lstsq(Amat, D, rcond=None)
        pred = Amat @ c
        r = float(np.sum((pred - D) ** 2))
        if best is None or r < best[0]:
            best = (r, p, float(c[0]), c.tolist())
    return dict(h0_extrap=best[2], power=best[1], resid=best[0],
                coef=best[3], finest=float(D[-1]), coarsest=float(D[0]))


def load_kernel_continuum():
    """Kernel co-area method's continuum deficit + per-G deficits."""
    rep = json.load(open(os.path.join(BASE, "k3_coarea_limit", "report.json")))
    krows = [(r["G_inner"], r["deficit"]) for r in rep["rows"]
             if r.get("deficit") is not None]
    kcont = rep["richardson"]["deficit"]["h0_extrap"]
    return krows, float(kcont)


def main():
    TAU, GAMMA = 2.0, 0.1
    G_LIST = [9, 13, 17]
    log("=" * 70)
    log("MORSE-ROBUST no-kernel h-free PR nail + continuum   tau=2 gamma=0.1")
    log(f"date 2026-05-29  UMAX={UMAX} NQ={S.NQ} SUB={S.SUB} eps_c={S.EPS_C} "
        f"(geo softened by eps_c*h^2)")
    rows = []
    Pprev, Gprev = None, None
    for G in G_LIST:
        t = time.time()
        x0 = None
        if Pprev is not None:
            red = SymReducer3(G)
            P0 = interp_up(Pprev, Gprev, G)
            x0 = red.reduce(P0)
        P, sol, Finf, info = S.solve_morse_pr(
            G, TAU, GAMMA, x0_red=x0, f_tol=1e-9,
            verbose=lambda n, f: log(f"    G={G} nk it {n} ||F||={f:.3e}"))
        info["walltime_s"] = round(time.time() - t, 1)
        Pprev, Gprev = P, G
        np.save(os.path.join(HERE, f"P_morse_G{G}.npy"), P)
        rows.append(info)
        log(f"  G={G:2d}: ||F||={Finf:.3e} {'NAILED' if Finf<1e-9 else 'FLOOR'} "
            f"deficit={info['deficit']:.5f} d_FR={info['d_FR']:.4f} "
            f"iters={info['iters']} picard={info['used_picard']} "
            f"({info['walltime_s']}s)")
        # checkpoint after each G
        json.dump({"rows": rows}, open(os.path.join(HERE, "_nail_ckpt.json"),
                                        "w"), indent=2)

    # ---- (d) continuum extrapolation + kernel reconciliation ----
    Gs = [r["G"] for r in rows]
    D = [r["deficit"] for r in rows]
    rich = richardson(Gs, D)
    krows, kcont = load_kernel_continuum()
    gap = abs(rich["h0_extrap"] - kcont)
    reconciled = bool(gap < 0.02)
    log("-" * 70)
    log(f"(c) NAIL ||F||_inf:  " +
        "  ".join(f"G={r['G']}:{r['Finf']:.2e}" for r in rows))
    nailed_all = all(r["Finf"] < 1e-9 for r in rows)
    nailed_hi = all(r["Finf"] < 1e-9 for r in rows if r["G"] >= 13)
    log(f"    all <1e-9 ? {nailed_all}   |   G>=13 (the key) <1e-9 ? {nailed_hi}")
    log(f"    deficits: " +
        "  ".join(f"G={r['G']}:{r['deficit']:.4f}" for r in rows))
    log(f"(d) Morse h-free continuum extrap (G->inf) = {rich['h0_extrap']:.4f} "
        f"(power {rich['power']}, finest {rich['finest']:.4f})")
    log(f"    kernel (co-area) continuum             = {kcont:.4f}")
    log(f"    |gap| = {gap:.4f}  ->  {'RECONCILED' if reconciled else 'DISAGREE'} "
        f"(threshold 0.02)")

    # ---- merge validation (a)/(b) if present ----
    cont = None
    cpath = os.path.join(HERE, "continuity.json")
    if os.path.exists(cpath):
        cont = json.load(open(cpath))

    report = {
        "operator": "hfree_morse_operator (no kernel; geo softened by eps_c*h^2)",
        "params": {"tau": TAU, "gamma": GAMMA, "UMAX": UMAX, "NQ": S.NQ,
                   "SUB": S.SUB, "REFINE": S.REFINE, "eps_c": S.EPS_C},
        "validation_a_continuity": cont,
        "validation_b_consistency": {
            "eps_c0_vs_original_inf": "see test_consistency.py: 8.9e-16 "
            "(eps_c=0 reduces EXACTLY to original; only near-crit behaviour "
            "changes at eps_c>0)"},
        "validation_c_nail": {
            "rows": rows,
            "nailed_all_G": nailed_all,
            "nailed_G_ge_13": nailed_hi,
            "key_criterion": "||F||_inf < 1e-9 at G=13 AND G=17"},
        "validation_d_continuum": {
            "Gs": Gs, "deficits": D,
            "morse_hfree_continuum": rich["h0_extrap"],
            "richardson": rich,
            "kernel_continuum": kcont,
            "kernel_rows": krows,
            "gap": gap, "reconciled": reconciled,
            "threshold": 0.02},
    }
    json.dump(report, open(os.path.join(HERE, "report.json"), "w"), indent=2)
    log("wrote report.json")

    # ---- figure: deficit vs G (Morse h-free vs kernel), both extrapolated ----
    try:
        make_figure(rows, rich, krows, kcont)
        log("wrote deficit_vs_G.png")
    except Exception as e:
        log(f"figure error: {type(e).__name__}: {e}")
    log("DONE")
    return report


def make_figure(rows, rich, krows, kcont):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=140)
    Gs = [r["G"] for r in rows]; D = [r["deficit"] for r in rows]
    ax.plot(Gs, D, "o-", color="C0", lw=2, ms=8,
            label="Morse-robust h-free (NO kernel)")
    kG = [g for g, _ in krows]; kD = [d for _, d in krows]
    ax.plot(kG, kD, "s--", color="C1", lw=1.5, ms=6,
            label="kernel (co-area band)")
    ax.axhline(rich["h0_extrap"], ls=":", color="C0", alpha=0.8,
               label=f"h-free G->inf = {rich['h0_extrap']:.3f}")
    ax.axhline(kcont, ls=":", color="C1", alpha=0.8,
               label=f"kernel G->inf = {kcont:.3f}")
    gap = abs(rich["h0_extrap"] - kcont)
    verdict = "RECONCILED" if gap < 0.02 else "DISAGREE"
    ax.set_xlabel("grid resolution G")
    ax.set_ylabel("revelation deficit  1 - R^2")
    ax.set_title(f"No-kernel (Morse-robust) vs kernel co-area: continuum PR "
                 f"deficit\ntau=2 gamma=0.1   |gap|={gap:.3f} -> {verdict}")
    ax.legend(fontsize=9); ax.grid(ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "deficit_vs_G.png"), dpi=140,
                bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
