"""Solve u-grid kernel co-area at STRICT h=0 (phi_K3_halo, linear scan) and
compare to σ-δ V11's strict h=0 fixed point.

User hypothesis: the discrepancy between u-grid (smooth, h≈0.32) and σ-δ V11
(strict h=0) is purely the kernel smoothing, not the (u₁,u₂,u₃) vs (u₁,Σ,δ)
discretization choice. If u-grid AT h=0 converges to the SAME basin as σ-δ
V11, that's confirmed.

Anderson(m=8) Picard, γ=0.1, τ=2, G_inner=17 (same as smooth sweep).
"""
import os, sys, time, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
import numpy as np
from reznsrc.contour_K3_halo import (phi_K3_halo, phi_K3_halo_smooth,
                                       phi_K3_halo_cubic, init_no_learning_K3)

# parameters matching k3_coarea_sweep
TAU = 2.0; GAMMA = 0.1
UMAX = 4.0; PAD = 2; G_INNER = 17
Gf = G_INNER + 2*PAD; du = 2*UMAX/(G_INNER-1)
u_full = np.array([-UMAX + (q-PAD)*du for q in range(Gf)])
INNER_LO, INNER_HI = PAD, PAD + G_INNER
slc = (slice(INNER_LO, INNER_HI),) * 3

tau_vec = np.full(3, TAU); gamma_vec = np.full(3, GAMMA); W_vec = np.full(3, 1.0)
print(f'u-grid G_inner={G_INNER}, padded {Gf}, du={du}, UMAX={UMAX}')
print(f'TAU={TAU}, γ={GAMMA}')

# Initialize from no-learning ansatz
halo = init_no_learning_K3(u_full, tau_vec, gamma_vec, W_vec)

# Reference: load the SMOOTH-kernel FP at h≈0.32 (k3_coarea_sweep result)
P_ref_smooth = np.load('/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_sweep/P_g0.1_t2.0.npy')
print(f'\nLoaded smooth-kernel FP (h={0.45*math.sqrt(du):.4f}): shape={P_ref_smooth.shape}, range [{P_ref_smooth.min():.4f},{P_ref_smooth.max():.4f}]')

# Metrics setup
u_in = u_full[INNER_LO:INNER_HI]
U1, U2, U3 = np.meshgrid(u_in, u_in, u_in, indexing='ij')
T = TAU * (U1 + U2 + U3)
def metrics(P):
    eps=1e-30; Pc=np.clip(P,eps,1-eps); lp=np.log(Pc/(1-Pc))
    fl=lp.ravel(); aa = np.polyfit(T.ravel(), fl, 1)
    pr=aa[0]*T.ravel()+aa[1]; m=fl.mean()
    vt=((fl-m)**2).mean(); vr=((fl-pr)**2).mean()
    P_FR = 1/(1+np.exp(-T))
    return float(aa[0]), float(vr/vt if vt>0 else float('nan')), float(np.sqrt(np.mean((P-P_FR)**2)))

s_ref, omr_ref, dfr_ref = metrics(P_ref_smooth)
print(f'  smooth (h≈0.32) FP metrics: slope={s_ref:.4f}, 1-R²={omr_ref:.4f}, d_FR={dfr_ref:.4f}')

# JIT warmup linear-scan operator
print('\nJIT warmup phi_K3_halo (linear scan, h=0)...', flush=True); t=time.time()
P_warm = halo.copy()
_ = phi_K3_halo(P_warm, u_full, INNER_LO, INNER_HI, tau_vec, gamma_vec, W_vec)
print(f'  warmup {time.time()-t:.1f}s', flush=True)

# === Anderson(m=8) Picard for phi_K3_halo (strict h=0 linear scan) ===
def phi_x(x, phi_fn):
    P = halo.copy()
    P[slc] = x.reshape((G_INNER,)*3)
    return phi_fn(P, u_full, INNER_LO, INNER_HI, tau_vec, gamma_vec, W_vec)[slc].ravel()

def anderson(x0, phi_fn, m_max=8, max_it=80, tol=1e-7, label=''):
    x = x0.copy()
    G_h=[]; F_h=[]; ferr_hist=[]
    print(f'\n=== Anderson(m={m_max}) for {label} ===', flush=True)
    for it in range(1, max_it+1):
        ts = time.time()
        g = phi_x(x, phi_fn); f = g - x
        ferr = float(np.max(np.abs(f)))
        ferr_hist.append(ferr)
        if ferr < tol:
            print(f'  it {it:3d} ferr={ferr:.3e} CONVERGED ({time.time()-ts:.1f}s)')
            return x, True, ferr_hist
        G_h.append(g.copy()); F_h.append(f.copy())
        if len(F_h) > m_max+1: G_h.pop(0); F_h.pop(0)
        mk = len(F_h) - 1
        if mk == 0:
            x_new = g
        else:
            dF = np.column_stack([F_h[k+1]-F_h[k] for k in range(mk)])
            dG = np.column_stack([G_h[k+1]-G_h[k] for k in range(mk)])
            try:
                gc, *_ = np.linalg.lstsq(dF, f, rcond=None)
                x_new = g - dG @ gc
            except np.linalg.LinAlgError:
                x_new = g
        x_new = np.clip(x_new, 1e-12, 1-1e-12)
        x = x_new
        if it % 5 == 0 or it == 1 or it < 5:
            inner = x.reshape((G_INNER,)*3)
            s, om, df = metrics(inner)
            print(f'  it {it:3d} ferr={ferr:.3e} slope={s:.4f} 1-R²={om:.4f} d_FR={df:.4f} ({time.time()-ts:.1f}s)')
    return x, False, ferr_hist

# IC: no-learning ansatz
x_ic = halo[slc].ravel().copy()
s_ic, om_ic, df_ic = metrics(x_ic.reshape((G_INNER,)*3))
print(f'\nIC (no-learning ansatz): slope={s_ic:.4f} 1-R²={om_ic:.4f} d_FR={df_ic:.4f}')

x_final, conv, hist = anderson(x_ic, phi_K3_halo, m_max=8, max_it=80, tol=1e-7,
                                 label='phi_K3_halo (linear scan, h=0)')

inner = x_final.reshape((G_INNER,)*3)
s_h0, om_h0, df_h0 = metrics(inner)
print(f'\n=== u-grid STRICT h=0 (linear scan) FP ===')
print(f'  conv={conv}, final ferr={hist[-1]:.3e}')
print(f'  slope={s_h0:.4f} 1-R²={om_h0:.4f} d_FR={df_h0:.4f}')

# Now also CUBIC variant (closest analog of σ-δ V11)
print('\n\nJIT warmup phi_K3_halo_cubic...', flush=True); t=time.time()
P_warm = halo.copy()
_ = phi_K3_halo_cubic(P_warm, u_full, INNER_LO, INNER_HI, tau_vec, gamma_vec, W_vec)
print(f'  warmup {time.time()-t:.1f}s', flush=True)

x_cub, conv_cub, hist_cub = anderson(x_ic.copy(), phi_K3_halo_cubic, m_max=8, max_it=80,
                                       tol=1e-7, label='phi_K3_halo_cubic (Hermite, h=0)')
inner_cub = x_cub.reshape((G_INNER,)*3)
s_cub, om_cub, df_cub = metrics(inner_cub)
print(f'\n=== u-grid STRICT h=0 (Hermite cubic) FP ===')
print(f'  conv={conv_cub}, final ferr={hist_cub[-1]:.3e}')
print(f'  slope={s_cub:.4f} 1-R²={om_cub:.4f} d_FR={df_cub:.4f}')

np.save(os.path.join(HERE, 'ugrid_h0_cubic_FP.npy'), inner_cub)
print(f'\n=== Comparison ===')
print(f'  smooth-kernel (h=0.32) FP:  slope={s_ref:.4f} d_FR={dfr_ref:.4f}')
print(f'  strict-h=0 (linear scan):   slope={s_h0:.4f} d_FR={df_h0:.4f}')
print(f'  σ-δ V11 reported (from earlier tests): slope≈0.86-0.90, d_FR≈0.05-0.09')

# How close is u-grid h=0 to σ-δ V11?
# σ-δ V11 has its own basin: from iter_ugrid_bc_thick.py the deep core
# converged to slope=0.86 — should match u-grid h=0 closely
diff_to_smooth = float(np.max(np.abs(inner - P_ref_smooth)))
print(f'  max|h=0 FP - smooth FP|: {diff_to_smooth:.4f}')

np.save(os.path.join(HERE, 'ugrid_h0_FP.npy'), inner)
json.dump({'gamma':GAMMA,'tau':TAU,'G_inner':G_INNER,
           'smooth_slope':s_ref,'smooth_d_FR':dfr_ref,
           'h0_linear_converged':conv,'h0_linear_final_ferr':hist[-1] if hist else None,
           'h0_linear_slope':s_h0,'h0_linear_d_FR':df_h0,
           'h0_cubic_converged':conv_cub,'h0_cubic_final_ferr':hist_cub[-1] if hist_cub else None,
           'h0_cubic_slope':s_cub,'h0_cubic_d_FR':df_cub,
           'max_diff_h0_vs_smooth':diff_to_smooth,
           'ferr_history_linear':hist,'ferr_history_cubic':hist_cub},
          open(os.path.join(HERE, 'ugrid_h0_result.json'),'w'), indent=2)
print('\nsaved')
