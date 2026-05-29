"""Figures + final report.json for k3_verify_coarea."""
import os, sys, json
sys.path.insert(0, '/tmp/rezn-source')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_verify_coarea'
t1 = json.load(open(os.path.join(OUT, 'test1.json')))
rows = t1['rows']
t2 = json.load(open(os.path.join(OUT, 'test2.json'))) if \
    os.path.exists(os.path.join(OUT, 'test2.json')) else {}

G = [r['G_inner'] for r in rows]
a_def = [r['a_deficit'] for r in rows]
b_def = [r['b_deficit'] for r in rows]
c_def = [r['c_deficit'] for r in rows]
diff_ac = [r['diff_ac'] for r in rows]
diff_ab = [r['diff_ab'] for r in rows]

fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), dpi=130)

ax[0].plot(G, a_def, 'o-', label='(a) h->0 kernel ladder (co-area)')
ax[0].plot(G, c_def, 's-', c='g', label='(c) co-area-WEIGHTED contour')
ax[0].plot(G, b_def, '^--', c='r', label='(b) naive UNweighted contour')
ax[0].set_xlabel('G_inner'); ax[0].set_ylabel('deficit = 1 - R^2')
ax[0].set_title('(1) Deficit vs G: three methods')
ax[0].legend(fontsize=8); ax[0].grid(ls=':')

ax[1].semilogy(G, [max(d, 1e-6) for d in diff_ac], 's-', c='g',
               label='|P_a - P_c|  (ladder vs weighted)')
ax[1].semilogy(G, [max(d, 1e-6) for d in diff_ab], '^--', c='r',
               label='|P_a - P_b|  (ladder vs naive)')
ax[1].set_xlabel('G_inner'); ax[1].set_ylabel('max |P diff|')
ax[1].set_title('(2) Price-surface agreement with ladder')
ax[1].legend(fontsize=8); ax[1].grid(ls=':')

# bias vs gap
bias_b = [r['bias_b'] for r in rows]
gap_c = [r['gap_c'] for r in rows]
ax[2].plot(G, bias_b, '^--', c='r', label='|def_b - def_a| (naive bias)')
ax[2].plot(G, gap_c, 's-', c='g', label='|def_c - def_a| (weighted gap)')
ax[2].set_xlabel('G_inner'); ax[2].set_ylabel('deficit gap to ladder')
ax[2].set_title('(3) Naive bias vs weighted gap')
ax[2].legend(fontsize=8); ax[2].grid(ls=':')

plt.suptitle('K=3 CRRA REE: h->0 ladder vs co-area-weighted vs naive contour '
             '(tau=2, gamma=0.1)', weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'test1_three_methods.png'),
            dpi=130, bbox_inches='tight')
plt.close()

# TEST2 audit plots
if t2:
    fig, ax = plt.subplots(1, 3, figsize=(19, 5), dpi=130)
    # issue 3: bandwidth exponent
    i3 = t2.get('issue3_bandwidth_exponent', {}).get('per_alpha', {})
    for al, v in i3.items():
        ax[0].plot(v['h'], v['deficit'], 'o-', label=f'alpha={al} '
                   f'(h=0:{v["h0_extrap"]:.3f})')
    ax[0].set_xlabel('h'); ax[0].set_ylabel('deficit')
    ax[0].invert_xaxis()
    ax[0].set_title('(2.3) Bandwidth-exponent sensitivity')
    ax[0].legend(fontsize=8); ax[0].grid(ls=':')
    # issue 2: coordinate dependence
    i2 = t2.get('issue2_coordinate', {})
    if i2:
        labels = ['uniform\nplain', 'signal\nweighted', 'stretched\nnodes']
        vals = [i2['deficit_uniform_plain'], i2['deficit_signal_weighted'],
                i2['deficit_stretched_nodes']]
        ax[1].bar(labels, vals, color=['C0', 'C1', 'C2'])
        ax[1].set_ylabel('deficit')
        ax[1].set_title(f'(2.2) Coordinate dependence (spread={i2["spread"]:.3f})')
        ax[1].grid(ls=':', axis='y')
    # issue 5: conditioning
    i5 = t2.get('issue5_conditioning', {})
    if i5:
        cr = i5['rows']
        ax[2].plot([r['G'] for r in cr], [r['gmres_outer_iters'] for r in cr],
                   'o-')
        ax[2].set_xlabel('G_inner'); ax[2].set_ylabel('Newton-Krylov iters')
        ax[2].set_title('(2.5) Conditioning along ladder')
        ax[2].grid(ls=':')
    plt.suptitle('K=3 co-area equilibrium: numerical-issues audit',
                 weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'test2_audit.png'), dpi=130,
                bbox_inches='tight')
    plt.close()

# ---- consolidated report.json ----
finest = rows[-1]
verdict_ac = ('AGREE' if finest['gap_c'] < 0.03 and
              finest['gap_c'] < finest['bias_b'] else 'PARTIAL')
report = dict(
    task='Verify h->0 kernel/co-area ladder == strict h=0 co-area-weighted '
         'contour, and that naive unweighted contour is biased. + numerical audit.',
    params=dict(tau=2.0, gamma=0.1, W=1.0, UMAX=4.0, pad=2,
                G_LIST=[r['G_inner'] for r in rows], h_law='h=0.45*du^0.5'),
    TEST1=dict(
        methods=dict(
            a='h->0 kernel ladder (phi_K3_halo_smooth, newton_krylov)',
            b='naive strict contour (phi_K3_halo, damped Picard plateau)',
            c='co-area-weighted strict contour (phi_K3_halo_weighted, nailed)'),
        rows=rows,
        finest=dict(G=finest['G_inner'],
                    a_deficit=finest['a_deficit'],
                    b_deficit=finest['b_deficit'],
                    c_deficit=finest['c_deficit'],
                    naive_bias_abs=finest['bias_b'],
                    weighted_gap_abs=finest['gap_c'],
                    surf_diff_ac=finest['diff_ac'],
                    surf_diff_ab=finest['diff_ab']),
        verdict=f'(a) vs (c): {verdict_ac}; naive (b) is biased '
                f'(bias {finest["bias_b"]:.3f} vs weighted gap '
                f'{finest["gap_c"]:.3f}).'),
    TEST2=t2)
json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=2)
print('wrote report.json + figures; verdict (a)vs(c):', verdict_ac)
