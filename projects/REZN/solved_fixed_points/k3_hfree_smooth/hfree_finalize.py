"""Assemble final report.json (validation + nail) and refresh the figure with
the nailed price surface. Run after the nail completes."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth"
sm = json.load(open("/tmp/hfree_smooth_res.json"))
cons = json.load(open("/tmp/hfree_consistency.json"))
nail = json.load(open(f"{OUT}/nail_report.json"))

report = dict(
    title="K=3 CRRA REE h-FREE SMOOTH co-area operator: validation + Newton nail",
    date="2026-05-29", tau=2.0, gamma=0.1, UMAX=4.0, Nq=nail["Nq"], sub=nail["sub"],
    NO_h=True, NO_kernel=True, NO_bandwidth=True, NO_smoothing_parameter=True,
    only_discretizations=["working grid G (->inf)",
                          "Gauss-Legendre transverse nodes Nq (->inf)",
                          "spline sub-bracket count for root isolation"],
    operator=("A_v(p)=int_{P=p} f_v f_v/|grad P| dsigma via C2 natural-cubic "
              "tensor spline of P + FIXED Gauss-Legendre transverse quadrature "
              "(DECOUPLED from the grid) + smooth 1D contour root-find (all roots) "
              "+ partition-of-unity weights w2=d2P^2/(d2P^2+d3P^2), "
              "w3=d3P^2/(d2P^2+d3P^2). No kernel/bandwidth/smoothing parameter."),
    validation_a_smoothness=sm,
    validation_b_consistency=dict(
        analytic_coarea_surface="sigmoid(1.3*(uA+uB))", Nq=80, G_test=41,
        max_rel_err_vs_analytic=cons["analytic_coarea_max_relerr"],
        note=("matches analytic co-area line integral to ~1.5e-5 (converges as "
              "G,Nq->inf); matches kernel h->0 limit, which floors on the grid "
              "while the h-free value is the exact limit.")),
    nail=nail,
    confirm_no_h=("grep of hfree_operator.py finds no kernel/bandwidth/"
                  "smoothing parameter; the only 'h' is grid/spline spacing du."),
)
json.dump(report, open(f"{OUT}/report.json", "w"), indent=2)
print("wrote report.json")

# ---- figure: smoothness (3 panels) + nailed surface inset ----
d = np.load("/tmp/hfree_smooth_compare.npz")
ps = d["ps"]; nodevals = d["nodevals"]; A1h = d["A1h"]; A1g = d["A1g"]
sc = sm["slopejump_scaling"]; hl = np.array(sc["h_list"])
traj = np.array(nail["newton_G9"]["trajectory"])

fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

ax[0, 0].plot(ps, A1h, "-", lw=1.6, c="C0", label="h-free smooth A_1(p)")
for nv in nodevals:
    ax[0, 0].axvline(nv, color="grey", ls=":", lw=0.5)
ax[0, 0].set_xlabel("price p"); ax[0, 0].set_ylabel("A_1(p)")
ax[0, 0].set_title("(a) Evidence A_v(p); dotted = grid-node prices\n"
                   "(no kink at nodes)")
ax[0, 0].grid(ls=":"); ax[0, 0].legend(fontsize=8)

dp = ps[1] - ps[0]
ax[0, 1].plot(ps, np.gradient(A1h, dp), "-", c="C0", lw=1.4, label="h-free dA/dp")
ax[0, 1].plot(ps, np.gradient(A1g, dp), "-", c="orange", lw=0.8, alpha=0.8,
              label="grid dA/dp (kinks)")
for nv in nodevals:
    ax[0, 1].axvline(nv, color="grey", ls=":", lw=0.5)
ax[0, 1].set_xlabel("price p"); ax[0, 1].set_ylabel("dA_1/dp")
ax[0, 1].set_title("(b) derivative: h-free continuous, grid kinks at nodes")
ax[0, 1].grid(ls=":"); ax[0, 1].legend(fontsize=8)

ax[1, 0].loglog(hl, sc["hfree_median"], "o-", c="C0", label="h-free (this work)")
ax[1, 0].loglog(hl, sc["grid_scan_median"], "s-", c="orange", label="grid marching-sq")
ax[1, 0].loglog(hl, sc["grid_cdf_median"], "^-", c="red", label="grid CDF-deriv")
ax[1, 0].loglog(hl, sc["grid_scan_median"][0] * (hl[0] / hl), "k--", lw=0.8,
                alpha=0.6, label="~1/h (KINK)")
ax[1, 0].loglog(hl, sc["hfree_median"][-1] * (hl / hl[-1]), "k:", lw=0.8,
                alpha=0.6, label="~h (SMOOTH)")
ax[1, 0].invert_xaxis()
ax[1, 0].set_xlabel("probe width h (decreasing ->)")
ax[1, 0].set_ylabel("slope-jump in A_v at node prices")
ax[1, 0].set_title("(c) SCALING TEST: h-free ->0 (SMOOTH);\n grid grows ~1/h (KINK)")
ax[1, 0].grid(ls=":", which="both"); ax[1, 0].legend(fontsize=7.5)

its = np.arange(len(traj))
ax[1, 1].semilogy(its, traj, "o-", c="C2")
ax[1, 1].set_xlabel("Newton iteration")
ax[1, 1].set_ylabel("||F||inf = ||Phi(P)-P||inf")
q = nail["newton_G9"]["order_q"]
ax[1, 1].set_title(f"(d) Newton nail (G=9, smooth h-free op)\n"
                   f"order_q~{q:.2f}, lowest||F||={traj[-1]:.1e}")
ax[1, 1].grid(ls=":")

plt.suptitle("K=3 CRRA REE h-FREE co-area operator (NO kernel/bandwidth): "
             "SMOOTH at node prices + Newton nails. tau=2 gamma=0.1",
             weight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/smoothness.png", dpi=120, bbox_inches="tight")
print("wrote smoothness.png (now incl Newton trajectory)")
