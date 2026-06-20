"""1-R² vs τ at BOTH precision levels: float64 vs mpmath high-precision.
float64 1-R² parsed from the cached metrics log; high-precision computed fresh."""
import sys, os, glob, re
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_src = open("/tmp/present_metrics.py").read().split("# Load all FPs")[0]
exec(_src, globals())

REPO = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points"

# ---- float64 1-R²(τ): parse cached metrics log (keep highest G per τ) ----
f64 = {}
for line in open("/tmp/present_metrics.log"):
    m = re.search(r"τ=([\d.]+) G=\s*(\d+): 1-R²=([\d.eE+-]+)", line)
    if m:
        t = float(m.group(1)); G = int(m.group(2)); r2 = float(m.group(3))
        if t not in f64 or G > f64[t][0]:
            f64[t] = (G, r2)
f64_tau = sorted(f64)
f64_r2 = [f64[t][1] for t in f64_tau]
print(f"float64 1-R² points parsed: {len(f64_tau)}")

# ---- high-precision: compute 1-R² fresh + Δμ vs original float64 ----
hp_tau, hp_r2, dmu_tau, dmu_max = [], [], [], []
for f in sorted(glob.glob(f"{REPO}/highprec/g100.0_*.npz")):
    d = np.load(f, allow_pickle=False)
    mu = d["mu_vals"]; xi = d["xi_grid"]; p = d["p_grid"]; t = float(d["tau"]); G = mu.shape[0]
    r = measure_all(mu, xi, p, t)
    hp_tau.append(t); hp_r2.append(r["m1r2"])
    # original float64 FP of same τ,G
    f64f = f"{REPO}/g100.0_t{t:.4f}_A5.50_G{G}_Gp17.npz"
    if os.path.exists(f64f):
        mu0 = np.load(f64f, allow_pickle=False)["mu_vals"]
        if mu0.shape == mu.shape:
            dmu_tau.append(t); dmu_max.append(float(np.max(np.abs(mu - mu0))))
    print(f"  HP τ={t:.4f} G={G}: 1-R²={r['m1r2']:.6e}", flush=True)
print("Δμ:", [(round(t,4), f'{d:.2e}') for t,d in zip(dmu_tau,dmu_max)])

order = np.argsort(hp_tau)
hp_tau = list(np.array(hp_tau)[order]); hp_r2 = list(np.array(hp_r2)[order])

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(f64_tau, f64_r2, "-o", ms=3, color="#1f77b4", label="float64  (TOL 1e-10 / 1e-12)")
ax[0].plot(hp_tau, hp_r2, "s", ms=11, mfc="none", mec="#d62728", mew=2.2,
           label="mpmath dps=60  (TOL 1e-40)")
ax[0].axvline(1.018, ls="--", color="gray", alpha=0.6)
ax[0].axvline(0.929, ls=":", color="gray", alpha=0.6)
ax[0].set_yscale("log"); ax[0].set_xlabel("τ"); ax[0].set_ylabel("1 − R²")
ax[0].text(1.02, max(f64_r2)*0.9, "wall₁ τ≈1.018", fontsize=8, color="gray", rotation=90, va="top")
ax[0].text(0.93, max(f64_r2)*0.9, "wall₂ τ≈0.929", fontsize=8, color="gray", rotation=90, va="top")
ax[0].set_title("Departure from linear REE (1 − R²) vs τ — both precision levels")
ax[0].legend(); ax[0].grid(alpha=0.3, which="both")

pos = [(t, d) for t, d in zip(dmu_tau, dmu_max) if d > 0]
if pos:
    ax[1].semilogy([t for t,_ in pos], [d for _,d in pos], "D", ms=10, color="#2ca02c")
ax[1].axhline(1e-12, ls="--", color="gray", alpha=0.7)
ax[1].text(min(dmu_tau) if dmu_tau else 1.5, 1.4e-12, "float64 wall floor ~1e-12", fontsize=8, color="gray")
ax[1].set_xlabel("τ"); ax[1].set_ylabel("max |Δμ|  (mpmath − float64 FP)")
ax[1].set_title("Fixed-point shift when refined float64→1e-40\n(=size of the float64 error the wall hid)")
ax[1].grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("/tmp/r2_both_levels.png", dpi=130)
print("saved /tmp/r2_both_levels.png")
