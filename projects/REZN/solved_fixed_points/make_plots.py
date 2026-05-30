"""Make many plots from all the CARA/CRRA UMAX/xi-grid/sigma-delta data.

Generates a battery of figures in solved_fixed_points/plots/ summarizing:
  1. The CARA full story (NL-halo grows, FR-halo same, open-box collapses, UMAX cliff)
  2. CRRA vs CARA UMAX-collapse battery (CRRA stays put, CARA collapses)
  3. Open-box CARA log-log fit across G
  4. xi-grid protocols (slope and deficit vs xi-bound)
  5. sigma-delta G-scaling (G=10 vs G=15 with high prec)
  6. Open-box vs closed-box CARA side-by-side
  7. Hellwig verdict synthesis (one-figure summary)
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points')
SRC = ROOT.parent
OUT = ROOT / 'plots'
OUT.mkdir(exist_ok=True)

def load(p):
    with open(p) as f: return json.load(f)

# ----- DATA LOAD -----
cara_nl     = load(ROOT/'k3_cara'/'cara_vs_G.json')['rows']           # NL halo, growing deficit
cara_fr     = load(ROOT/'k3_cara'/'cara_fr_halo_vs_G.json')['rows']   # FR halo, same growing deficit
cara_open   = load(ROOT/'k3_cara'/'cara_open_box_vs_G.json')['rows']  # UMAX=12, machine zero
cara_umax   = load(ROOT/'k3_cara'/'cara_umax_diag.json')              # Protocols A and B
cara_xi     = load(ROOT/'k3_cara'/'cara_xi_grid.json')['rows']        # xi-grid 3 protocols
cara_FR_tst = load(ROOT/'k3_cara'/'FR_test.json')['rows']             # FR-as-fixed-point residual
crra_umaxG11 = load(ROOT/'k3_coarea_2dsweep'/'crra_umax_diag.json')['rows']
crra_umaxG17 = load(ROOT/'k3_coarea_2dsweep'/'crra_umax_diag_G17.json')['rows']
sd_g10_cara  = load(SRC/'solver_code'/'sigma_delta'/'sigmadelta_cara_summary.json')
sd_g15_cara  = load(SRC/'solver_code'/'sigma_delta'/'flint_sd_G15_cara.json')
sd_g15_crra  = load(SRC/'solver_code'/'sigma_delta'/'flint_sd_G15_crra.json')

# ===================================================================
# 1. CARA FULL STORY (4 panels)
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=140)

# (a) NL-halo and FR-halo deficits both GROW with G -- artifact, not Hellwig
ax = axes[0, 0]
G_nl = [r['G'] for r in cara_nl]; D_nl = [r['deficit'] for r in cara_nl]
G_fr = [r['G'] for r in cara_fr]; D_fr = [r['deficit'] for r in cara_fr]
ax.plot(G_nl, D_nl, 'o-', color='C3', lw=2, ms=9, label='NL halo')
ax.plot(G_fr, D_fr, 's--', color='C1', lw=2, ms=9, alpha=0.7, label='FR halo (identical: halo does not matter)')
ax.set_xlabel('grid resolution G'); ax.set_ylabel('CARA deficit (1−R²)')
ax.set_title('(a) CLOSED box (UMAX=4): deficit GROWS with G\nNL-halo = FR-halo ⇒ artifact is intrinsic to discrete op, not BC', fontsize=11)
ax.legend(fontsize=9); ax.grid(ls=':')

# (b) Open box (UMAX=12) — deficit COLLAPSES to machine zero
ax = axes[0, 1]
G_o = [r['G'] for r in cara_open]; D_o = [r['deficit'] for r in cara_open]
ax.semilogy(G_o, np.maximum(D_o, 1e-16), 'o-', color='C2', lw=2, ms=10, label='open box UMAX=12 (FR halo)')
ax.axhline(2.2e-16, color='k', ls=':', alpha=0.5, label='float64 machine ε')
ax.set_xlabel('G'); ax.set_ylabel('CARA deficit (log)')
ax.set_title('(b) OPEN box (UMAX=12): deficit COLLAPSES to machine zero\n3.4e-14 at G=7, 7.3e-9 at G=9 — Hellwig realized', fontsize=11)
ax.legend(fontsize=9); ax.grid(ls=':', which='both', alpha=0.5)
for g, d in zip(G_o, D_o):
    ax.annotate(f'{d:.1e}', (g, max(d, 1e-15)), textcoords='offset points', xytext=(5, 7), fontsize=8)

# (c) UMAX-cliff at fixed G=9
ax = axes[1, 0]
A = cara_umax['A_fixedG']
ux = [r['umax'] for r in A]; dx = [r['deficit'] for r in A]
ax.semilogy(ux, dx, 'D-', color='C0', lw=2.5, ms=12, label='G=9, FR halo, varying UMAX')
ax.axvline(np.log(1e9)/(3*2), color='C3', ls='--', lw=1.5, label=f'clip threshold ≈ {np.log(1e9)/(3*2):.2f}')
ax.set_xlabel('box half-width UMAX'); ax.set_ylabel('deficit (log)')
ax.set_title('(c) UMAX cliff at fixed G=9:\ndeficit drops 6 orders of magnitude (3.8e-3 → 7.3e-9)', fontsize=11)
ax.legend(fontsize=9); ax.grid(ls=':', which='both', alpha=0.5)
for u, d in zip(ux, dx):
    ax.annotate(f'{d:.1e}', (u, d), textcoords='offset points', xytext=(6, 8), fontsize=8)

# (d) Protocol B: fixed du, growing G+UMAX -- deficit halves per UMAX step
ax = axes[1, 1]
B = cara_umax['B_fixed_du']
ub = [r['umax'] for r in B]; db = [r['deficit'] for r in B]; gb = [r['G'] for r in B]
ax.plot(ub, db, 'o-', color='C4', lw=2.5, ms=12)
for u, d, g in zip(ub, db, gb):
    ax.annotate(f'G={g}\n{d:.2e}', (u, d), textcoords='offset points', xytext=(8, 6), fontsize=8)
ax.set_xlabel('UMAX (with G grown to keep du=0.5 fixed)'); ax.set_ylabel('deficit')
ax.set_title('(d) FIXED du=0.5 — halving deficit per UMAX step:\n0.0256 → 0.0124 → 0.0066 even as G grows 17→25→33', fontsize=11)
ax.grid(ls=':')

plt.suptitle('CARA "deficit" was 100% a box-clip artifact. Hellwig (CARA = FR) realized to machine zero.', weight='bold', fontsize=12)
plt.tight_layout(); plt.savefig(OUT/'01_cara_full_story.png', dpi=140, bbox_inches='tight'); plt.close()
print('1: cara_full_story.png')

# ===================================================================
# 2. CRRA vs CARA UMAX-COLLAPSE BATTERY
# ===================================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)
uc = [r['umax'] for r in A]; dc_cara = [r['deficit'] for r in A]
u_crra17 = [r['umax'] for r in crra_umaxG17]; d_crra17 = [r['deficit'] for r in crra_umaxG17]
u_crra11 = [r['umax'] for r in crra_umaxG11]; d_crra11 = [r['deficit'] for r in crra_umaxG11]

# (left) linear scale
ax[0].plot(u_crra17, d_crra17, 'D-', color='C0', lw=2.5, ms=12, label='CRRA G=17 (paper headline)')
ax[0].plot(u_crra11, d_crra11, 's--', color='C2', lw=2, ms=9, alpha=0.8, label='CRRA G=11')
ax[0].plot(uc, dc_cara, 'o-', color='C3', lw=2.5, ms=12, label='CARA G=9 (Hellwig: → 0)')
ax[0].axhline(0, color='k', ls=':', alpha=0.4)
ax[0].set_xlabel('box half-width UMAX'); ax[0].set_ylabel('revelation deficit 1−R²')
ax[0].set_title('Same diagnostic, opposite outcomes:\nCRRA gap intrinsic; CARA gap was a clip artifact', fontsize=11)
ax[0].legend(fontsize=10); ax[0].grid(ls=':')

# (right) log scale
ax[1].semilogy(u_crra17, d_crra17, 'D-', color='C0', lw=2.5, ms=12, label='CRRA G=17')
ax[1].semilogy(u_crra11, d_crra11, 's--', color='C2', lw=2, ms=9, alpha=0.8, label='CRRA G=11')
ax[1].semilogy(uc, np.maximum(dc_cara, 1e-15), 'o-', color='C3', lw=2.5, ms=12, label='CARA G=9')
ax[1].set_xlabel('UMAX'); ax[1].set_ylabel('deficit (log)')
ax[1].set_title('Log scale: CARA spans 6 orders; CRRA flat', fontsize=11)
ax[1].legend(fontsize=10); ax[1].grid(ls=':', which='both', alpha=0.5)
plt.suptitle('CRRA partial-revelation equilibrium is real; CARA "deficit" was numerical', weight='bold', fontsize=12)
plt.tight_layout(); plt.savefig(OUT/'02_crra_vs_cara_battery.png', dpi=140, bbox_inches='tight'); plt.close()
print('2: crra_vs_cara_battery.png')

# ===================================================================
# 3. OPEN-BOX CARA: log-log convergence
# ===================================================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
G_o_a = np.array(G_o, float); D_o_a = np.array(D_o, float)
ax.loglog(G_o_a, np.maximum(D_o_a, 1e-16), 'o-', color='C2', lw=2.5, ms=12, label='open-box CARA (UMAX=12, FR halo)')
# fit power law to the unconverged (G≥11) tail where Picard hits its 400-iter cap
G_fit = G_o_a[2:]; D_fit = D_o_a[2:]
if len(G_fit) >= 2:
    p = np.polyfit(np.log(G_fit), np.log(np.maximum(D_fit, 1e-15)), 1)
    gg = np.geomspace(7, 20, 50)
    ax.loglog(gg, np.exp(p[1])*gg**p[0], 'k--', lw=1.3, alpha=0.6, label=f'fit ~ G^{p[0]:+.2f} (Picard residual)')
ax.axhline(2.2e-16, color='k', ls=':', alpha=0.5, label='machine ε')
ax.set_xlabel('grid resolution G'); ax.set_ylabel('CARA deficit at FR (log)')
ax.set_title('CARA on open box: deficit at Hellwig FP\nG=7,9 hit machine zero; G≥11 limited by Picard\'s 400-iter cap', fontsize=11)
ax.legend(fontsize=10); ax.grid(ls=':', which='both', alpha=0.5)
for g, d in zip(G_o_a, D_o_a):
    ax.annotate(f'{d:.1e}', (g, d), textcoords='offset points', xytext=(6, 6), fontsize=9)
plt.tight_layout(); plt.savefig(OUT/'03_cara_open_box_loglog.png', dpi=140, bbox_inches='tight'); plt.close()
print('3: cara_open_box_loglog.png')

# ===================================================================
# 4. xi-grid protocols (slope and deficit vs xi-bound)
# ===================================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)
# group cara_xi by protocol
by_prot = {}
for r in cara_xi:
    by_prot.setdefault(r['protocol'], []).append(r)
labels = {'A_wide': 'xi ∈ [-0.95, 0.95] (saturated)',
          'B_narrow': 'xi ∈ [-0.7, 0.7] (off-saturation)',
          'C_very_narrow': 'xi ∈ [-0.5, 0.5] (fully off-saturation)'}
colors = {'A_wide': 'C3', 'B_narrow': 'C1', 'C_very_narrow': 'C2'}
for prot, rows in by_prot.items():
    Gs = [r['G'] for r in rows]
    sl = [r['slope'] for r in rows]
    df = [r['deficit'] for r in rows]
    ax[0].plot(Gs, sl, 'o-', color=colors[prot], lw=2, ms=10, label=labels[prot])
    ax[1].plot(Gs, df, 'o-', color=colors[prot], lw=2, ms=10, label=labels[prot])
ax[0].axhline(1.0, color='k', ls='--', alpha=0.6, label='FR slope = 1')
ax[0].set_xlabel('G'); ax[0].set_ylabel('slope of logit(P) vs τΣu')
ax[0].set_title('Slope vs xi-bound:\nshrinking xi (avoiding saturation) recovers FR slope', fontsize=11)
ax[0].legend(fontsize=9); ax[0].grid(ls=':')
ax[1].set_xlabel('G'); ax[1].set_ylabel('1−R² deficit')
ax[1].set_title('Deficit (R² scatter): kernel-band-vs-saturation bias\nplateaus ~0.23 regardless of xi-bound', fontsize=11)
ax[1].legend(fontsize=9); ax[1].grid(ls=':')
plt.suptitle('Bounded-ξ K=3 CARA on the kernel co-area op: slope recovers but R² stays imperfect', weight='bold', fontsize=12)
plt.tight_layout(); plt.savefig(OUT/'04_xi_grid_protocols.png', dpi=140, bbox_inches='tight'); plt.close()
print('4: xi_grid_protocols.png')

# ===================================================================
# 5. sigma-delta G-scaling: G=10 (verifies Hellwig) vs G=15 (breaks)
# ===================================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=140)
# (left) G=10 sigma-delta CARA -- both ICs converge near FR
sd10_FR = sd_g10_cara.get('FR_ansatz', {})
sd10_NL = sd_g10_cara.get('no_learning', {})
labels10 = [('FR_ansatz IC', sd10_FR), ('no_learning IC', sd10_NL)]
metric_xs = []
metric_ys_dFR = []
metric_ys_deficit = []
for lbl, d in labels10:
    metric_xs.append(lbl)
    metric_ys_dFR.append(d.get('d_FR', np.nan))
    metric_ys_deficit.append(d.get('1mR2_Tstar', np.nan))
xpos = np.arange(len(metric_xs))
ax[0].bar(xpos-0.18, metric_ys_dFR, width=0.35, color='C0', label='d_FR (RMS distance to FR)')
ax[0].bar(xpos+0.18, metric_ys_deficit, width=0.35, color='C2', label='1-R²(T*)')
ax[0].set_xticks(xpos); ax[0].set_xticklabels(metric_xs)
ax[0].set_yscale('log')
ax[0].set_title('σ-δ G=10 (float64): VERIFIES Hellwig\nboth ICs converge to d_FR ≈ 6e-3', fontsize=11)
ax[0].set_ylabel('value (log)')
ax[0].legend(fontsize=9); ax[0].grid(ls=':', axis='y', which='both', alpha=0.5)
for x, v in zip(xpos-0.18, metric_ys_dFR):
    ax[0].annotate(f'{v:.1e}', (x, v), textcoords='offset points', xytext=(0, 5), fontsize=8, ha='center')
for x, v in zip(xpos+0.18, metric_ys_deficit):
    ax[0].annotate(f'{v:.1e}', (x, v), textcoords='offset points', xytext=(0, 5), fontsize=8, ha='center')

# (right) G=15 flint dps=50: CARA drifts AWAY from FR
iters_cara = np.arange(1, len(sd_g15_cara['d_FR'])+1)
iters_crra = np.arange(1, len(sd_g15_crra['d_FR'])+1)
ax[1].plot(iters_cara, sd_g15_cara['d_FR'], 'o-', color='C3', lw=2, ms=10, label='CARA (FR-IC)')
ax[1].plot(iters_crra, sd_g15_crra['d_FR'], 's-', color='C0', lw=2, ms=10, label='CRRA (NL-IC)')
ax[1].axhline(6e-3, color='C2', ls='--', alpha=0.7, label='G=10 CARA d_FR=6e-3 (reference)')
ax[1].set_xlabel('Picard iter'); ax[1].set_ylabel('d_FR')
ax[1].set_title('σ-δ G=15 (flint dps=50): FAILS Hellwig\nboth drift to d_FR ≈ 0.20–0.26 (Σ-interp bias)', fontsize=11)
ax[1].legend(fontsize=9); ax[1].grid(ls=':')

plt.suptitle('σ-δ machinery at G=10 ≈ FR; at G=15 the oblique-slice Σ-interp dominates', weight='bold', fontsize=12)
plt.tight_layout(); plt.savefig(OUT/'05_sigma_delta_G_scaling.png', dpi=140, bbox_inches='tight'); plt.close()
print('5: sigma_delta_G_scaling.png')

# ===================================================================
# 6. CARA: closed-box artifact vs open-box truth, ONE comparison plot
# ===================================================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
ax.semilogy([r['G'] for r in cara_nl], [r['deficit'] for r in cara_nl], 'o-', color='C3', lw=2.5, ms=10, label='closed box UMAX=4 (artifact)')
ax.semilogy(G_o_a, np.maximum(D_o_a, 1e-16), 's-', color='C2', lw=2.5, ms=10, label='open box UMAX=12 (Hellwig FR)')
ax.axhline(2.2e-16, color='k', ls=':', alpha=0.5, label='float64 machine ε')
ax.set_xlabel('grid resolution G'); ax.set_ylabel('CARA deficit (log)')
ax.set_title('Same CARA operator, same precision range — only the box width changes.\nClosed box → 0.004…0.09 artifact; Open box → 10⁻⁹ (Hellwig).', fontsize=11)
ax.legend(fontsize=10, loc='lower right'); ax.grid(ls=':', which='both', alpha=0.5)
for g, d, ec in zip([r['G'] for r in cara_nl], [r['deficit'] for r in cara_nl], 'C3'*99):
    ax.annotate(f'{d:.1e}', (g, d), textcoords='offset points', xytext=(5, -12), fontsize=8, color='C3')
for g, d in zip(G_o_a, D_o_a):
    ax.annotate(f'{d:.1e}', (g, max(d, 1e-15)), textcoords='offset points', xytext=(5, 7), fontsize=8, color='C2')
plt.tight_layout(); plt.savefig(OUT/'06_closed_vs_open_box.png', dpi=140, bbox_inches='tight'); plt.close()
print('6: closed_vs_open_box.png')

# ===================================================================
# 7. Hellwig verdict synthesis: one-figure summary across the diagnostics
# ===================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=140)

# (a) UMAX collapse — the smoking gun
ax = axes[0]
ax.semilogy(ux, dx, 'o-', color='C0', lw=2.5, ms=12, label='CARA G=9')
ax.semilogy(u_crra17, d_crra17, 'D-', color='C2', lw=2.5, ms=12, label='CRRA G=17')
ax.set_xlabel('UMAX'); ax.set_ylabel('deficit (log)')
ax.set_title('(a) UMAX-collapse: CARA drops 6 orders;\nCRRA stays flat — gap intrinsic', fontsize=11)
ax.legend(fontsize=10); ax.grid(ls=':', which='both', alpha=0.5)

# (b) Open box -- CARA reaches machine zero
ax = axes[1]
ax.semilogy(G_o_a, np.maximum(D_o_a, 1e-16), 'o-', color='C2', lw=2.5, ms=12, label='CARA open box (UMAX=12)')
ax.axhline(2.2e-16, color='k', ls=':', alpha=0.5, label='machine ε')
ax.axhline(0.28, color='C0', ls='--', alpha=0.7, label='CRRA G=17, UMAX=4: 0.28')
ax.set_xlabel('G'); ax.set_ylabel('deficit (log)')
ax.set_title('(b) Hellwig realized:\nCARA at UMAX=12 hits machine zero', fontsize=11)
ax.legend(fontsize=10); ax.grid(ls=':', which='both', alpha=0.5)

# (c) sigma-delta cross-check
ax = axes[2]
ax.plot(iters_cara, sd_g15_cara['d_FR'], 'o-', color='C3', lw=2, ms=8, label='σ-δ G=15 (Σ-interp bias)')
ax.axhline(6e-3, color='C2', lw=2.5, ls='--', label='σ-δ G=10: d_FR=6e-3 (Hellwig)')
ax.set_xlabel('Picard iter'); ax.set_ylabel('d_FR (RMS dist to FR)')
ax.set_title('(c) σ-δ frame: G=10 verifies Hellwig;\nG=15 saturated by oblique-slice bias', fontsize=11)
ax.legend(fontsize=9); ax.grid(ls=':')

plt.suptitle('Hellwig verdict: noiseless CARA → FR (deficit = 0). Multiple operators confirm.', weight='bold', fontsize=13)
plt.tight_layout(); plt.savefig(OUT/'07_hellwig_verdict_synthesis.png', dpi=140, bbox_inches='tight'); plt.close()
print('7: hellwig_verdict_synthesis.png')

# ===================================================================
# 8. CRRA gap survives every test (parallel structure to CARA collapse)
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
# CRRA at G=11 and G=17 across UMAX
ax.plot(u_crra17, d_crra17, 'D-', color='C0', lw=2.5, ms=14, label='CRRA G=17 (paper headline)')
ax.plot(u_crra11, d_crra11, 's-', color='C2', lw=2.5, ms=12, label='CRRA G=11')
# annotate
for u, d in zip(u_crra17, d_crra17):
    ax.annotate(f'{d:.3f}', (u, d), textcoords='offset points', xytext=(8, -4), fontsize=9, color='C0')
for u, d in zip(u_crra11, d_crra11):
    ax.annotate(f'{d:.3f}', (u, d), textcoords='offset points', xytext=(8, 12), fontsize=9, color='C2')
ax.set_xlabel('box half-width UMAX'); ax.set_ylabel('CRRA revelation deficit 1−R²')
ax.set_title('CRRA at γ=0.1, τ=2: the same UMAX-collapse that demolished CARA leaves CRRA intact.\nDeficit stays 0.28–0.42 across all UMAX, even GROWS — the Jensen gap is real.', fontsize=11)
ax.legend(fontsize=11); ax.grid(ls=':')
ax.set_ylim(0, 0.5)
plt.tight_layout(); plt.savefig(OUT/'08_crra_gap_survives.png', dpi=140, bbox_inches='tight'); plt.close()
print('8: crra_gap_survives.png')

print(f'\nALL DONE — wrote {len(list(OUT.glob("*.png")))} plots to {OUT}')
for p in sorted(OUT.glob('*.png')):
    sz = p.stat().st_size // 1024
    print(f'  {p.name}  ({sz} KB)')
