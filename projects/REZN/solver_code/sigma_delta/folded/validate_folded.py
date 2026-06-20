"""Validate folded Φ against unfolded by running 3 iters on G=11."""
import os, sys, time, math
sys.path.insert(0, '/tmp')
import numpy as np
from dd_phi_sigma_delta import (phi_sigmadelta, set_boundary, crra_clear_sym)
from phi_sigma_delta_folded import (phi_folded, set_boundary_folded, unfold_full,
                                      finf_interior_folded)

G = 11
mid = G // 2
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU = 2.0; GAMMA = 0.1; W = 1.0
xi_full = np.linspace(-1.0, 1.0, G)
xi_u1 = xi_full.copy(); xi_S = xi_full.copy(); xi_d = xi_full.copy()

# Build initial P_full
def sigmoid(x): return 1.0/(1.0+np.exp(-x))
INNER_LO, INNER_HI = 1, G - 1
G_INNER = G - 2

u1_phys   = TOT_u * np.arctanh(np.clip(xi_full[INNER_LO:INNER_HI], -0.999999, 0.999999))
Sigma_phys = TOT_S * np.arctanh(np.clip(xi_full[INNER_LO:INNER_HI], -0.999999, 0.999999))
delta_phys = TOT_d * np.arctanh(np.clip(xi_full[INNER_LO:INNER_HI], -0.999999, 0.999999))
U1_m, SI_m, DE_m = np.meshgrid(u1_phys, Sigma_phys, delta_phys, indexing='ij')
U2_m = 0.5*(SI_m + DE_m); U3_m = 0.5*(SI_m - DE_m)
S_phys = U1_m + U2_m + U3_m
P_FR_in = sigmoid(TAU * S_phys)

# Unfolded
P_full = np.zeros((G,)*3)
P_full[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_FR_in
P_full = set_boundary(P_full, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)

# Folded init: copy from P_full
P_stored = np.zeros((G, mid+1, mid+1))
for i in range(G):
    for j_s in range(mid+1):
        for k_s in range(mid+1):
            P_stored[i, j_s, k_s] = P_full[i, j_s + mid, k_s + mid]
P_stored = set_boundary_folded(P_stored, G)

# Run 3 iters of both
print('warmup JIT...')
_ = phi_sigmadelta(P_full, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                    1, G-1, 1, G-1, 1, G-1)
_ = phi_folded(P_stored, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W, G)
print('done')

for it in range(1, 4):
    P_full = phi_sigmadelta(P_full, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W,
                              1, G-1, 1, G-1, 1, G-1)
    P_full = set_boundary(P_full, TOT_u, TOT_S, TOT_d, xi_u1, xi_S, xi_d)
    P_stored = phi_folded(P_stored, xi_u1, xi_S, xi_d, TOT_u, TOT_S, TOT_d, TAU, GAMMA, W, G)
    P_stored = set_boundary_folded(P_stored, G)
    P_unfolded = unfold_full(P_stored, G)
    diff = np.max(np.abs(P_unfolded - P_full))
    print(f'iter {it}: max|folded_unfolded - direct| = {diff:.3e}')

# Final cross-check: is the unfolded P symmetric (as the folded one is by construction)?
P_check = P_full
sym_err_delta = np.max(np.abs(P_check - P_check[:, :, ::-1]))
sym_err_combo = np.max(np.abs(P_check - (1.0 - P_check[::-1, ::-1, :])))
print(f'unfolded P: δ-sym error = {sym_err_delta:.3e}')
print(f'unfolded P: combined sym error = {sym_err_combo:.3e}')
