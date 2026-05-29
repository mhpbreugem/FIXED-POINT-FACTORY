"""Finalize: read the nailed Morse solutions (P_morse_G{9,13,17}.npy +
_nail_ckpt.json), compute (c) nail summary, (d) Richardson continuum +
reconciliation with the kernel co-area continuum, merge (a)/(b), write
report.json, run.log lines, and the deficit-vs-G figure.

Decoupled from the (expensive) solve so it can be re-run instantly."""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)

UMAX = 4.0
TAU = 2.0


def metrics(P, G):
    ui = np.linspace(-UMAX, UMAX, G)
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    T = TAU * (U1 + U2 + U3)
    Pc = np.clip(P, 1e-12, 1 - 1e-12)
    y = np.log(Pc / (1 - Pc)).ravel()
    a = np.polyfit(T.ravel(), y, 1)
    pr = a[0] * T.ravel() + a[1]
    deficit = float(np.sum((y - pr) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30))
    P_FR = 1.0 / (1.0 + np.exp(-T))
    d_FR = float(np.sqrt(np.mean((P - P_FR) ** 2)))
    return deficit, d_FR


def richardson(Gs, D):
    Gs = np.asarray(Gs, float); D = np.asarray(D, float)
    best = None
    for p in [1.0, 1.3, 1.5, 2.0]:
        if len(Gs) >= 3:
            Amat = np.column_stack([np.ones_like(Gs), Gs**(-p), Gs**(-2 * p)])
        else:
            Amat = np.column_stack([np.ones_like(Gs), Gs**(-p)])
        c, *_ = np.linalg.lstsq(Amat, D, rcond=None)
        r = float(np.sum((Amat @ c - D) ** 2))
        if best is None or r < best[0]:
            best = (r, p, float(c[0]), c.tolist())
    # also a plain 1/G two-point-style linear fit for robustness
    Alin = np.column_stack([np.ones_like(Gs), 1.0 / Gs])
    clin, *_ = np.linalg.lstsq(Alin, D, rcond=None)
    return dict(h0_extrap=best[2], power=best[1], resid=best[0], coef=best[3],
                linear_1overG_extrap=float(clin[0]),
                finest=float(D[-1]), coarsest=float(D[0]))


def load_kernel():
    rep = json.load(open(os.path.join(BASE, "k3_coarea_limit", "report.json")))
    krows = [(r["G_inner"], r["deficit"]) for r in rep["rows"] if r.get("deficit") is not None]
    kcont = float(rep["richardson"]["deficit"]["h0_extrap"])
    return krows, kcont


def main():
    ckpt = json.load(open(os.path.join(HERE, "_nail_ckpt.json")))
    rows = sorted(ckpt["rows"], key=lambda r: r["G"])
    # recompute deficit straight from the saved P (authoritative)
    for r in rows:
        p = os.path.join(HERE, f"P_morse_G{r['G']}.npy")
        if os.path.exists(p):
            d, dfr = metrics(np.load(p), r["G"])
            r["deficit_recomputed"] = d
            r["d_FR_recomputed"] = dfr
    Gs = [r["G"] for r in rows]
    D = [r.get("deficit_recomputed", r["deficit"]) for r in rows]
    rich = richardson(Gs, D)
    krows, kcont = load_kernel()
    gap = abs(rich["h0_extrap"] - kcont)
    reconciled = bool(gap < 0.02)
    nailed_all = all(r["Finf"] < 1e-9 for r in rows)
    nailed_hi = all(r["Finf"] < 1e-9 for r in rows if r["G"] >= 13)

    cont = None
    cpath = os.path.join(HERE, "continuity.json")
    if os.path.exists(cpath):
        cont = json.load(open(cpath))

    report = {
        "operator": "hfree_morse_operator (NO kernel; v-independent geometric "
                    "weight 1/|gradP| softened by eps_c*h^2 at Morse-critical prices)",
        "params": {"tau": TAU, "gamma": 0.1, "UMAX": UMAX, "NQ": 40, "SUB": 4,
                   "eps_c": 1e-3, "note": "eps2=eps_c*h^2 -> 0 as G->inf"},
        "validation_a_continuity": cont,
        "validation_b_consistency": {
            "eps_c0_vs_original_inf": 8.882e-16,
            "note": "eps_c=0 reduces the Morse operator EXACTLY to the original "
                    "hfree_operator (machine precision on a generic non-critical "
                    "P); eps_c>0 changes only near-critical behaviour."},
        "validation_c_nail": {
            "rows": rows,
            "Finf_by_G": {r["G"]: r["Finf"] for r in rows},
            "deficit_by_G": {r["G"]: r.get("deficit_recomputed", r["deficit"]) for r in rows},
            "nailed_all_G": nailed_all,
            "nailed_G_ge_13": nailed_hi,
            "key_criterion": "||F||_inf < 1e-9 at G=13 AND G=17",
            "old_operator_floor": "~1e-3 at G>=13 (k3_hfree_ginf/ginf_robust.json)"},
        "validation_d_continuum": {
            "Gs": Gs, "deficits": D,
            "morse_hfree_continuum": rich["h0_extrap"],
            "richardson": rich,
            "kernel_continuum": kcont,
            "kernel_rows": krows,
            "gap": gap, "reconciled": reconciled, "threshold": 0.02},
    }
    json.dump(report, open(os.path.join(HERE, "report.json"), "w"), indent=2)

    L = open(os.path.join(HERE, "run.log"), "a")
    def log(*a):
        m = " ".join(str(x) for x in a); print(m, flush=True); L.write(m + "\n")
    log("-" * 70)
    log("(c) NAIL ||F||_inf: " + "  ".join(f"G={r['G']}:{r['Finf']:.2e}" for r in rows))
    log(f"    nailed all G ? {nailed_all}  |  G>=13 (key) <1e-9 ? {nailed_hi}")
    log("    deficits:      " + "  ".join(f"G={g}:{d:.4f}" for g, d in zip(Gs, D)))
    log(f"(d) Morse h-free continuum (G->inf) = {rich['h0_extrap']:.4f} "
        f"(power {rich['power']}; linear-1/G {rich['linear_1overG_extrap']:.4f})")
    log(f"    kernel (co-area) continuum     = {kcont:.4f}")
    log(f"    |gap| = {gap:.4f}  ->  {'RECONCILED' if reconciled else 'DISAGREE'} (thr 0.02)")
    log("wrote report.json")

    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=140)
        ax.plot(Gs, D, "o-", color="C0", lw=2, ms=8, label="Morse-robust h-free (NO kernel)")
        ax.plot([g for g, _ in krows], [d for _, d in krows], "s--", color="C1",
                lw=1.5, ms=6, label="kernel (co-area band)")
        ax.axhline(rich["h0_extrap"], ls=":", color="C0", alpha=0.85,
                   label=f"h-free G->inf = {rich['h0_extrap']:.3f}")
        ax.axhline(kcont, ls=":", color="C1", alpha=0.85,
                   label=f"kernel G->inf = {kcont:.3f}")
        verdict = "RECONCILED" if gap < 0.02 else "DISAGREE"
        ax.set_xlabel("grid resolution G"); ax.set_ylabel("revelation deficit  1 - R^2")
        ax.set_title(f"No-kernel (Morse-robust) vs kernel co-area: continuum PR deficit\n"
                     f"tau=2 gamma=0.1   |gap|={gap:.3f} -> {verdict}")
        ax.legend(fontsize=9); ax.grid(ls=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, "deficit_vs_G.png"), dpi=140, bbox_inches="tight")
        plt.close()
        log("wrote deficit_vs_G.png")
    except Exception as e:
        log(f"figure error: {type(e).__name__}: {e}")
    L.close()
    return report


if __name__ == "__main__":
    main()
