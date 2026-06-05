"""γ-sweep figure for analysis v2."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# τ=1 γ-sweep results from gamma_sweep.log
gamma = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1]
slope = [0.3439, 0.3274, 0.3239, 0.3238, 0.3119, 0.3006, 0.3003]
deficit = [0.1307, 0.1427, 0.1443, 0.1445, 0.1531, 0.1677, 0.1679]
d_FR = [0.0818, 0.0889, 0.0916, 0.0917, 0.0983, 0.1015, 0.1017]
residual = [1.737e-01, 2.963e-01, 2.885e-01, 2.762e-01, 1.680e-01, 1.297e-01, 2.230e-01]

# Prior session at τ=2 (different operator: hfree spline strict h=0)
g_prior = [0.001, 0.005, 0.01, 0.03, 0.07, 0.1, 0.15, 0.225, 0.259]
slope_prior = [0.3599, 0.3601, 0.3603, 0.3611, 0.3626, 0.3641, 0.3668, 0.3708, 0.3728]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: slope_T vs γ — main story
ax = axes[0]
ax.semilogx(g_prior, slope_prior, 'go-', lw=1.5, alpha=0.5, label='Prior: hfree strict h=0 spline, τ=2', markersize=8)
ax.semilogx(gamma, slope, 'rs-', lw=2.5, markersize=12, label='THIS: Chebyshev h=0 lift, τ=1 (γ-continuation)')
ax.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='FR (slope=1)')
ax.axhline(0.0, color='gray', linestyle=':', alpha=0.3)
# Annotate the continuation points
for gi, si in zip(gamma, slope):
    ax.annotate(f'{si:.3f}', (gi, si), textcoords='offset points', xytext=(6, -14), fontsize=8.5)
ax.set_xlabel('γ (CRRA risk aversion)')
ax.set_ylabel('slope$_T$ (logit P vs T)')
ax.set_title('Deep PR branch in two operators:\nτ=1 Chebyshev (this work) vs τ=2 spline (prior session)')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)

# Right: deficit & d_FR
ax = axes[1]
ax.semilogx(gamma, deficit, 'o-', color='purple', lw=2, markersize=10, label='deficit (1−R²)')
ax.semilogx(gamma, d_FR, 's-', color='brown', lw=2, markersize=10, label='d_FR (RMS dist from FR)')
ax.set_xlabel('γ')
ax.set_ylabel('value')
ax.set_title('Equilibrium structure metrics along τ=1 γ-sweep')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/tmp/cheby_h0/figs/H_gamma_sweep_PR.png', dpi=140, bbox_inches='tight')
plt.close()
print('Saved /tmp/cheby_h0/figs/H_gamma_sweep_PR.png')
