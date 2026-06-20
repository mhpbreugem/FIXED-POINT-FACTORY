"""Build the tau-sweep figure from the 2D continuation log."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from /tmp/cheby_h0/2dcont.log Stage A at γ=1
tau_pts = [1.0, 1.2, 1.5, 1.75, 2.0]
slope_pts = [0.3439, 0.3009, 0.2700, 0.0991, 0.0859]
deficit_pts = [0.1307, 0.1329, 0.1553, 0.5612, 0.5673]
dFR_pts = [0.0818, 0.0585, 0.0897, 0.2003, 0.2003]  # last point partial Newton
residual_pts = [1.737e-01, 1.916e-01, 1.553e+00, 9.758e-01, 1.391e+00]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: slope vs tau
ax = axes[0]
ax.plot(tau_pts[:3], slope_pts[:3], 'o-', color='green', lw=2.5, markersize=12, label='PR branch (continuation)')
ax.plot(tau_pts[3:], slope_pts[3:], 's--', color='red', lw=2, markersize=10, label='NL (no-info) basin')
ax.axvline(1.62, color='gray', linestyle=':', alpha=0.6, label='approx. fold τ_fold ≈ 1.6')
# Annotate
for ti, si in zip(tau_pts, slope_pts):
    ax.annotate(f'{si:.3f}', (ti, si), textcoords='offset points', xytext=(8, 5), fontsize=9)
# Prior session point
ax.plot([2.0], [0.3641], '*', color='blue', markersize=22, label='Prior session τ=2,γ=0.1: 0.3641')
ax.set_xlabel('τ (signal precision, γ=1 fixed)')
ax.set_ylabel('slope (logit P vs T)')
ax.set_title('τ-sweep at γ=1 from converged τ=1,γ=1 PR FP\nPR branch ends in a fold around τ≈1.6')
ax.grid(alpha=0.3); ax.legend(fontsize=10)

# Right: deficit and residual
ax = axes[1]
ax.semilogy(tau_pts, residual_pts, 'D-', color='black', lw=2, markersize=10, label='lifted ||F||_∞')
ax2 = ax.twinx()
ax2.plot(tau_pts, deficit_pts, 'o--', color='purple', lw=2, markersize=10, label='deficit (1−R²)')
ax.set_xlabel('τ')
ax.set_ylabel('lifted ||F||_∞ (log scale)', color='black')
ax2.set_ylabel('deficit', color='purple')
ax.tick_params(axis='y', labelcolor='black')
ax2.tick_params(axis='y', labelcolor='purple')
ax.set_title('Residual and deficit along the τ-sweep')
ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=10)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/G_tau_sweep_fold.png', dpi=140, bbox_inches='tight')
plt.close()
print('Saved /tmp/cheby_h0/figs/G_tau_sweep_fold.png')
