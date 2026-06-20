"""Generate plots from the overnight battery results.

Reads /tmp/overnight_out/{summary.json, P_*.npy} and /tmp/g21_max5000_out/.
Produces:
  1. ferr / d_FR / 1-R² scaling heatmap across (G, γ) for each IC
  2. P at û_1=0 slice for each (G, γ, IC) — grid layout
  3. Deviation from FR for each (G, γ, IC)
  4. Comparison FR-ansatz vs no-learn at each γ for each G
  5. 5000-iter trajectories for G=21 (if available)
"""
import os, sys, math, json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR_MAIN = '/tmp/overnight_out'
OUT_DIR_5K = '/tmp/g21_max5000_out'
FIG = '/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
os.makedirs(FIG, exist_ok=True)

TAU = 2.0
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def load_main():
    s_path = f'{OUT_DIR_MAIN}/summary.json'
    if not os.path.exists(s_path): return {}
    with open(s_path) as f:
        return json.load(f)

def load_5k():
    s_path = f'{OUT_DIR_5K}/summary.json'
    if not os.path.exists(s_path): return {}
    with open(s_path) as f:
        return json.load(f)

# ============ PLOT 1: heatmap of final ferr / d_FR / 1-R² over (G, γ) ============
def heatmap_summary(results, ic_name, save_path):
    Gs = sorted({r['G'] for r in results.values() if r['ic'] == ic_name})
    gammas = sorted({r['gamma'] for r in results.values() if r['ic'] == ic_name})
    if not Gs or not gammas:
        return
    metrics = ['final_ferr', 'final_d_FR', 'final_1mR2']
    titles = ['ferr', 'd_FR', '1−R²(T*)']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=140)
    for ax, m, t in zip(axes, metrics, titles):
        Z = np.full((len(Gs), len(gammas)), np.nan)
        for r in results.values():
            if r['ic'] != ic_name: continue
            i = Gs.index(r['G']); j = gammas.index(r['gamma'])
            Z[i, j] = r[m]
        im = ax.imshow(np.log10(np.where(Z > 0, Z, 1e-30)), origin='lower',
                       cmap='viridis', aspect='auto',
                       extent=[-0.5, len(gammas)-0.5, -0.5, len(Gs)-0.5])
        ax.set_xticks(range(len(gammas))); ax.set_xticklabels([f'{g}' for g in gammas])
        ax.set_yticks(range(len(Gs))); ax.set_yticklabels([f'{g}' for g in Gs])
        ax.set_xlabel('γ'); ax.set_ylabel('G')
        ax.set_title(f'{t}, IC={ic_name}')
        plt.colorbar(im, ax=ax, label=f'log10({t})')
        # annotations
        for i in range(len(Gs)):
            for j in range(len(gammas)):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, f'{Z[i,j]:.2e}', ha='center', va='center',
                             color='white', fontsize=7.5)
    plt.suptitle(f'Battery summary: G vs γ, IC = {ic_name}', fontsize=13, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'wrote {save_path}')

# ============ PLOT 2: P at û_1=0 slice grid ============
def grid_P_slices(results, ic_name, out_dir, save_path):
    Gs = sorted({r['G'] for r in results.values() if r['ic'] == ic_name})
    gammas = sorted({r['gamma'] for r in results.values() if r['ic'] == ic_name})
    if not Gs or not gammas:
        return
    fig, axes = plt.subplots(len(Gs), len(gammas), figsize=(3*len(gammas), 3*len(Gs)), dpi=140,
                              squeeze=False)
    for i, G in enumerate(Gs):
        mid = G // 2
        for j, GAMMA in enumerate(gammas):
            label = f'G{G}_g{GAMMA}_{ic_name}'
            P_path = f'{out_dir}/P_{label}.npy'
            ax = axes[i, j]
            if not os.path.exists(P_path):
                ax.set_facecolor('#eeeeee')
                ax.text(0.5, 0.5, 'pending', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            P = np.load(P_path)  # full G^3
            sl = P[mid, :, :]
            im = ax.imshow(sl.T, origin='lower', cmap='RdBu_r', vmin=0, vmax=1,
                            extent=[-1, 1, -1, 1], aspect='auto')
            ax.set_title(f'G={G}, γ={GAMMA}', fontsize=9)
            if j == 0: ax.set_ylabel(r'$\hat\delta$')
            if i == len(Gs)-1: ax.set_xlabel(r'$\hat\Sigma$')
    plt.suptitle(f'P at $\\hat u_1=0$ slice, IC = {ic_name}',
                  fontsize=13, weight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'wrote {save_path}')

# ============ PLOT 3: deviation from FR ============
def grid_deviation_slices(results, ic_name, out_dir, save_path):
    Gs = sorted({r['G'] for r in results.values() if r['ic'] == ic_name})
    gammas = sorted({r['gamma'] for r in results.values() if r['ic'] == ic_name})
    if not Gs or not gammas:
        return
    fig, axes = plt.subplots(len(Gs), len(gammas), figsize=(3*len(gammas), 3*len(Gs)), dpi=140,
                              squeeze=False)
    for i, G in enumerate(Gs):
        mid = G // 2
        xi_full = np.linspace(-1, 1, G)
        xi_inner = xi_full[1:-1]
        u_p = TOT_u * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
        S_p = TOT_S * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
        d_p = TOT_d * np.arctanh(np.clip(xi_inner, -0.999999, 0.999999))
        U1m, SIm, DEm = np.meshgrid(u_p, S_p, d_p, indexing='ij')
        S_phys = U1m + SIm
        P_FR_in = sigmoid(TAU * S_phys)
        # Embed P_FR_in into a full G x G x G by zero-padding boundary (just for visualization)
        for j, GAMMA in enumerate(gammas):
            label = f'G{G}_g{GAMMA}_{ic_name}'
            P_path = f'{out_dir}/P_{label}.npy'
            ax = axes[i, j]
            if not os.path.exists(P_path):
                ax.set_facecolor('#eeeeee')
                ax.text(0.5, 0.5, 'pending', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            P = np.load(P_path)
            P_inner = P[1:-1, 1:-1, 1:-1]
            dev = P_inner - P_FR_in
            sl = dev[mid-1, :, :]  # mid is G//2 of full; inner is (G-2,)*3, so mid-1
            vmax = max(abs(sl).max(), 1e-10)
            im = ax.imshow(sl.T, origin='lower', cmap='PiYG', vmin=-vmax, vmax=vmax,
                            extent=[-1, 1, -1, 1], aspect='auto')
            ax.set_title(f'G={G},γ={GAMMA}\nmax|dev|={vmax:.1e}', fontsize=8)
            if j == 0: ax.set_ylabel(r'$\hat\delta$')
            if i == len(Gs)-1: ax.set_xlabel(r'$\hat\Sigma$')
    plt.suptitle(f'P − P^FR at $\\hat u_1=0$, IC = {ic_name}',
                  fontsize=13, weight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'wrote {save_path}')

# ============ PLOT 4: bar chart summary ============
def bar_summary(results, save_path):
    """Final ferr per scenario (bar chart)."""
    keys = sorted(results.keys(), key=lambda k: (results[k]['G'], results[k]['gamma'], results[k]['ic']))
    Gs = [results[k]['G'] for k in keys]
    gammas = [results[k]['gamma'] for k in keys]
    ics = [results[k]['ic'] for k in keys]
    ferrs = [results[k]['final_ferr'] for k in keys]
    d_FRs = [results[k]['final_d_FR'] for k in keys]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), dpi=140)
    x = np.arange(len(keys))
    colors_fr = ['steelblue' if ic == 'FR_ansatz' else 'orange' for ic in ics]
    ax1.bar(x, ferrs, color=colors_fr)
    ax1.set_yscale('log')
    ax1.set_ylabel('final ferr')
    ax1.set_title('Final ferr per scenario')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"G{G}γ{g}\n{ic[:5]}" for G,g,ic in zip(Gs,gammas,ics)],
                         rotation=45, ha='right', fontsize=7)
    ax2.bar(x, d_FRs, color=colors_fr)
    ax2.set_yscale('log')
    ax2.set_ylabel('final d_FR')
    ax2.set_title('Final d_FR per scenario  (blue=FR_ansatz, orange=no_learn)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"G{G}γ{g}\n{ic[:5]}" for G,g,ic in zip(Gs,gammas,ics)],
                         rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'wrote {save_path}')


# ============ MAIN ============
results_main = load_main()
results_5k = load_5k()
print(f'main results loaded: {len(results_main)} scenarios')
print(f'5K results loaded: {len(results_5k)} scenarios')

if results_main:
    heatmap_summary(results_main, 'FR_ansatz', f'{FIG}/battery_heatmap_FR.png')
    heatmap_summary(results_main, 'no_learn', f'{FIG}/battery_heatmap_NL.png')
    grid_P_slices(results_main, 'FR_ansatz', OUT_DIR_MAIN, f'{FIG}/battery_P_FR.png')
    grid_P_slices(results_main, 'no_learn', OUT_DIR_MAIN, f'{FIG}/battery_P_NL.png')
    grid_deviation_slices(results_main, 'FR_ansatz', OUT_DIR_MAIN, f'{FIG}/battery_dev_FR.png')
    grid_deviation_slices(results_main, 'no_learn', OUT_DIR_MAIN, f'{FIG}/battery_dev_NL.png')
    bar_summary(results_main, f'{FIG}/battery_bars.png')

if results_5k:
    # results_5k uses different keys (no G field), set G=21
    for k, r in results_5k.items():
        r['G'] = 21
    bar_summary(results_5k, f'{FIG}/g21_5K_bars.png')

print('done')
