# Welfare metrics extraction

Computes trade volume V_vol(γ, τ) and per-agent value of information
V(γ, τ) from converged REE checkpoints. Produces Fig 7 and Fig 8 in
two orientations (τ on x-axis, γ on x-axis).

## Usage

```bash
# 1. Ensure REZN paper repo is available (for contour_K3_halo functions)
git clone --depth 1 https://github.com/mhpbreugem/REZN.git ~/rezn-source

# 2. Install dependencies
pip install numba numpy matplotlib

# 3. Compute metrics for every checkpoint
python3 projects/REZN/extract/extract_metrics.py

# 4. Render PNG and pgfplots files
python3 projects/REZN/extract/make_plots.py
```

Outputs land in `projects/REZN/figures/`:
- `all_metrics.csv`              — raw data for every (γ, τ) checkpoint
- `fig7_volume_vs_tau.png`       — volume curves, τ on x-axis
- `fig7_volume_vs_gamma.png`     — volume curves, γ on x-axis
- `fig8_value_info_vs_tau.png`   — V_info curves, τ on x-axis
- `fig8_value_info_vs_gamma.png` — V_info curves, γ on x-axis
- `fig7_*.tex`, `fig8_*.tex`     — pgfplots blocks for the paper

Re-run after new checkpoints land. Fully deterministic.

## Math

See `projects/REZN/WELFARE_METRICS.md` for the integral formulas.
Both V_vol and V_info use the same ex-ante joint signal weight
w(u_1,…,u_K) = ½(∏ f_0(u_k) + ∏ f_1(u_k)).
