"""flint (Σ̂, δ̂) cube STRICT-contour with ADAPTIVE DAMPED Picard.

K=3 symmetric, h=0 hardwired. No-learn IC. flint at 200-dec precision.
Damping: ω auto-tunes (×0.7 on stall, ×1.05 on good contraction; floor 0.001).
"""
import time, math, json
import numpy as np
import flint
from flint import arb

DPS = 200
flint.ctx.prec = int(DPS * 3.33) + 20

G_FULL = 11
INNER_LO, INNER_HI = 1, G_FULL - 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU_F = 2.0; GAMMA_F = 0.1; W_F = 1.0
MAX_ITER = 50

LOG = '/tmp/flint_sigdelta_damp.log'
open(LOG, 'w').close()
def lg(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

def mp(x):
    if isinstance(x, str): return arb(x)
    return arb(float(x))

# --- arb primitives ---
def fsig(u, vm, tau, coef):
    d = u - vm
    return coef * (mp('-0.5') * tau * d * d).exp()

def crra_demand(mu, p, gamma, W):
    one = mp(1)
    lm = (mu / (one - mu)).log()
    lp = (p / (one - p)).log()
    R = ((lm - lp) / gamma).exp()
    return W * (R - one) / ((one - p) + R * p)

def crra_clear_sym(mus, gamma, W, steps=300):
    eps = mp('1e-150'); one = mp(1); two = mp(2); zero = mp(0)
    a, b = eps, one - eps
    for _ in range(steps):
        m = (a + b) / two
        ex = zero
        for mu in mus: ex = ex + crra_demand(mu, m, gamma, W)
        if ex > 0: a = m
        else: b = m
    return (a + b) / two

# --- Σ-interp helper ---
def interp_along_Sigma(P, i_u, k_d, Sigma_target, xi_S, TOT_S_mp):
    """P[i_u, j, k_d] interpolated along axis j at the ξ corresponding to Sigma_target."""
    G = len(xi_S)
    one = mp(1); zero = mp(0)
    # convert Sigma_target → xi
    # xi = tanh(Sigma / TOT)
    if Sigma_target > mp('1e10'):
        return P[i_u][G-1][k_d]
    if Sigma_target < mp('-1e10'):
        return P[i_u][0][k_d]
    # safer: clip
    arg = Sigma_target / TOT_S_mp
    # tanh: arb has it
    try:
        xi_t = arg.tanh()
    except Exception:
        return P[i_u][G//2][k_d]
    # linear search in xi_S
    if xi_t <= xi_S[0]: return P[i_u][0][k_d]
    if xi_t >= xi_S[-1]: return P[i_u][G-1][k_d]
    for j in range(G - 1):
        if xi_S[j] <= xi_t <= xi_S[j+1]:
            denom = xi_S[j+1] - xi_S[j]
            if denom == 0:
                return P[i_u][j][k_d]
            frac = (xi_t - xi_S[j]) / denom
            return (one - frac) * P[i_u][j][k_d] + frac * P[i_u][j+1][k_d]
    return P[i_u][G-1][k_d]

# --- agent 1 evidence (axis-aligned in (Σ, δ)) ---
def evidence_agent1(P, p_target, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm, i_u):
    G_S = len(xi_S); G_d = len(xi_d)
    zero = mp(0); one = mp(1); two = mp(2)
    PI2 = arb.pi() * two
    coef = (tau / PI2).sqrt()
    A = zero
    # scan Σ
    for k in range(G_d):
        if abs(xi_d[k]) >= one - mp('1e-15'): continue
        d_k = TOT_d_mp * xi_d[k].atanh()
        prev = P[i_u][0][k]
        for j in range(G_S - 1):
            nxt = P[i_u][j+1][k]
            dp = prev - p_target; dn = nxt - p_target
            same_zero = (dp == 0 and dn == 0)
            if not same_zero and (dp * dn) <= 0:
                den = nxt - prev
                if den != 0:
                    frac = -dp / den
                    if frac < 0: frac = zero
                    if frac > 1: frac = one
                    xi_S_off = (one - frac) * xi_S[j] + frac * xi_S[j+1]
                    if abs(xi_S_off) < one - mp('1e-15'):
                        Sigma_off = TOT_S_mp * xi_S_off.atanh()
                        u2_off = (Sigma_off + d_k) / two
                        u3_off = (Sigma_off - d_k) / two
                        A = A + fsig(u2_off, vm, tau, coef) * fsig(u3_off, vm, tau, coef)
            prev = nxt
    # scan δ
    for j in range(G_S):
        if abs(xi_S[j]) >= one - mp('1e-15'): continue
        Sigma_j = TOT_S_mp * xi_S[j].atanh()
        prev = P[i_u][j][0]
        for k in range(G_d - 1):
            nxt = P[i_u][j][k+1]
            dp = prev - p_target; dn = nxt - p_target
            same_zero = (dp == 0 and dn == 0)
            if not same_zero and (dp * dn) <= 0:
                den = nxt - prev
                if den != 0:
                    frac = -dp / den
                    if frac < 0: frac = zero
                    if frac > 1: frac = one
                    xi_d_off = (one - frac) * xi_d[k] + frac * xi_d[k+1]
                    if abs(xi_d_off) < one - mp('1e-15'):
                        d_off = TOT_d_mp * xi_d_off.atanh()
                        u2_off = (Sigma_j + d_off) / two
                        u3_off = (Sigma_j - d_off) / two
                        A = A + fsig(u2_off, vm, tau, coef) * fsig(u3_off, vm, tau, coef)
            prev = nxt
    return A / two

# --- agent 2/3 evidence (oblique via Σ-interp) ---
def evidence_agent_oblique(P, p_target, xi_u1, xi_S, xi_d,
                            TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, vm,
                            u_own_cell, sign_for_other):
    """For agent 2: u_other = u_3 = u_own - δ, sign_for_other = -1 (Σ = 2u_own - δ).
       For agent 3: u_other = u_2 = u_own + δ, sign_for_other = +1 (Σ = 2u_own + δ).
       Slice: P_sl[i_u, i_δ] = interp P[i_u, ?, i_δ] at Σ_required = 2*u_own + sign_for_other*δ."""
    G_u = len(xi_u1); G_d = len(xi_d)
    zero = mp(0); one = mp(1); two = mp(2)
    PI2 = arb.pi() * two
    coef = (tau / PI2).sqrt()
    # build P_slice[i_u][i_d]
    P_slice = [[zero for _ in range(G_d)] for _ in range(G_u)]
    for i in range(G_u):
        for k in range(G_d):
            if abs(xi_d[k]) >= one - mp('1e-15'):
                d_k = (one if xi_d[k] > 0 else -one) * mp('1e10')
            else:
                d_k = TOT_d_mp * xi_d[k].atanh()
            Sigma_req = two * u_own_cell + sign_for_other * d_k
            P_slice[i][k] = interp_along_Sigma(P, i, k, Sigma_req, xi_S, TOT_S_mp)
    A = zero
    # scan u_1 axis
    for k in range(G_d):
        if abs(xi_d[k]) >= one - mp('1e-15'): continue
        d_k = TOT_d_mp * xi_d[k].atanh()
        u_other = u_own_cell + sign_for_other * d_k  # u_other at fixed δ
        prev = P_slice[0][k]
        for i in range(G_u - 1):
            nxt = P_slice[i+1][k]
            dp = prev - p_target; dn = nxt - p_target
            same_zero = (dp == 0 and dn == 0)
            if not same_zero and (dp * dn) <= 0:
                den = nxt - prev
                if den != 0:
                    frac = -dp / den
                    if frac < 0: frac = zero
                    if frac > 1: frac = one
                    xi_u_off = (one - frac) * xi_u1[i] + frac * xi_u1[i+1]
                    if abs(xi_u_off) < one - mp('1e-15'):
                        u_off = TOT_u_mp * xi_u_off.atanh()
                        A = A + fsig(u_off, vm, tau, coef) * fsig(u_other, vm, tau, coef)
            prev = nxt
    # scan δ axis
    for i in range(G_u):
        if abs(xi_u1[i]) >= one - mp('1e-15'): continue
        u1_at = TOT_u_mp * xi_u1[i].atanh()
        prev = P_slice[i][0]
        for k in range(G_d - 1):
            nxt = P_slice[i][k+1]
            dp = prev - p_target; dn = nxt - p_target
            same_zero = (dp == 0 and dn == 0)
            if not same_zero and (dp * dn) <= 0:
                den = nxt - prev
                if den != 0:
                    frac = -dp / den
                    if frac < 0: frac = zero
                    if frac > 1: frac = one
                    xi_d_off = (one - frac) * xi_d[k] + frac * xi_d[k+1]
                    if abs(xi_d_off) < one - mp('1e-15'):
                        d_off = TOT_d_mp * xi_d_off.atanh()
                        u_other = u_own_cell + sign_for_other * d_off
                        A = A + fsig(u1_at, vm, tau, coef) * fsig(u_other, vm, tau, coef)
            prev = nxt
    return A / two


def phi_sigdelta_strict(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                         tau, gamma, W):
    G_u = len(xi_u1); G_S = len(xi_S); G_d = len(xi_d)
    one = mp(1); two = mp(2); vm0 = mp('-0.5'); vm1 = mp('0.5'); eps_p = mp('1e-100')
    PI2 = arb.pi() * two
    coef = (tau / PI2).sqrt()
    P_new = [[[P[i][j][k] for k in range(G_d)] for j in range(G_S)] for i in range(G_u)]
    for i in range(INNER_LO, INNER_HI):
        if abs(xi_u1[i]) >= one - mp('1e-15'): continue
        u1_cell = TOT_u_mp * xi_u1[i].atanh()
        for j in range(INNER_LO, INNER_HI):
            if abs(xi_S[j]) >= one - mp('1e-15'): continue
            Sigma_cell = TOT_S_mp * xi_S[j].atanh()
            for k in range(INNER_LO, INNER_HI):
                if abs(xi_d[k]) >= one - mp('1e-15'): continue
                d_cell = TOT_d_mp * xi_d[k].atanh()
                p_cell = P[i][j][k]
                u2_cell = (Sigma_cell + d_cell) / two
                u3_cell = (Sigma_cell - d_cell) / two
                # agent 1
                A1_0 = evidence_agent1(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm0, i)
                A1_1 = evidence_agent1(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm1, i)
                f0_u1 = fsig(u1_cell, vm0, tau, coef); f1_u1 = fsig(u1_cell, vm1, tau, coef)
                den = f0_u1 * A1_0 + f1_u1 * A1_1
                mu0 = f1_u1 * A1_1 / den if den > 0 else mp('0.5')
                # agent 2 (sign_for_other = -1)
                A2_0 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d,
                                                TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, vm0,
                                                u2_cell, -one)
                A2_1 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d,
                                                TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, vm1,
                                                u2_cell, -one)
                f0_u2 = fsig(u2_cell, vm0, tau, coef); f1_u2 = fsig(u2_cell, vm1, tau, coef)
                den = f0_u2 * A2_0 + f1_u2 * A2_1
                mu1 = f1_u2 * A2_1 / den if den > 0 else mp('0.5')
                # agent 3 (sign_for_other = +1)
                A3_0 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d,
                                                TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, vm0,
                                                u3_cell, one)
                A3_1 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d,
                                                TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, vm1,
                                                u3_cell, one)
                f0_u3 = fsig(u3_cell, vm0, tau, coef); f1_u3 = fsig(u3_cell, vm1, tau, coef)
                den = f0_u3 * A3_0 + f1_u3 * A3_1
                mu2 = f1_u3 * A3_1 / den if den > 0 else mp('0.5')
                mus = [max(eps_p, min(one - eps_p, mu)) for mu in (mu0, mu1, mu2)]
                P_new[i][j][k] = crra_clear_sym(mus, gamma, W)
    return P_new


def set_boundary(P):
    """h=0: FR=0/1 at u_1=±∞ and Σ̂=±∞, zero-order δ̂=±∞."""
    G = len(P); one = mp(1); zero = mp(0)
    for j in range(G):
        for k in range(G):
            P[0][j][k] = zero; P[G-1][j][k] = one
    for i in range(G):
        for k in range(G):
            P[i][0][k] = zero; P[i][G-1][k] = one
    for i in range(G):
        for j in range(G):
            P[i][j][0] = P[i][j][1]; P[i][j][G-1] = P[i][j][G-2]
    return P


def f_inf(A, B):
    G = len(A); m = mp(0)
    for i in range(INNER_LO, INNER_HI):
        for j in range(INNER_LO, INNER_HI):
            for k in range(INNER_LO, INNER_HI):
                d = abs(A[i][j][k] - B[i][j][k])
                if d > m: m = d
    return m


def to_np(P):
    G = len(P)
    return np.array([[[float(P[i][j][k]) for k in range(G)] for j in range(G)] for i in range(G)])


# ===== setup =====
xi_full_np = np.linspace(-1.0, 1.0, G_FULL)
xi_inner_np = xi_full_np[INNER_LO:INNER_HI]
xi_u1 = [mp(float(x)) for x in xi_full_np]
xi_S = [mp(float(x)) for x in xi_full_np]
xi_d = [mp(float(x)) for x in xi_full_np]

TOT_u_mp = mp(TOT_u); TOT_S_mp = mp(TOT_S); TOT_d_mp = mp(TOT_d)
tau = mp(TAU_F); gamma = mp(GAMMA_F); W = mp(W_F)

u_phys = TOT_u * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
S_phys = TOT_S * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
d_phys = TOT_d * np.arctanh(np.clip(xi_inner_np, -0.999999, 0.999999))
U1m, SIm, DEm = np.meshgrid(u_phys, S_phys, d_phys, indexing='ij')
U2m = 0.5*(SIm+DEm); U3m = 0.5*(SIm-DEm)
S_full = U1m + U2m + U3m
def sg(x): return 1.0/(1.0+np.exp(-x))
P_FR_in = sg(TAU_F * S_full)

# No-learning IC
mu1_NL = sg(TAU_F*U1m); mu2_NL = sg(TAU_F*U2m); mu3_NL = sg(TAU_F*U3m)
def crra_clear_f64(mu0, mu1, mu2, gf, steps=80):
    eps = 1e-30
    def dd(mu, p):
        lm = math.log(mu/(1-mu)); lp = math.log(p/(1-p))
        R = math.exp((lm-lp)/gf)
        return (R-1)/((1-p)+R*p)
    a, b = eps, 1-eps
    for _ in range(steps):
        m = (a+b)/2
        e = dd(mu0,m) + dd(mu1,m) + dd(mu2,m)
        if e > 0: a = m
        else: b = m
    return (a+b)/2
P_NL_in = np.empty_like(U1m)
G_INNER = G_FULL - 2
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL_in[i,j,k] = crra_clear_f64(mu1_NL[i,j,k], mu2_NL[i,j,k], mu3_NL[i,j,k], GAMMA_F)

# Build P as flint nested list
P_full_np = np.zeros((G_FULL,)*3)
P_full_np[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL_in
P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
P = set_boundary(P)

lg(f'flint (Σ̂, δ̂) STRICT-contour ADAPTIVE-DAMPED Picard, dps={DPS}, γ={GAMMA_F}, G={G_FULL}, NO-LEARN IC, MAX={MAX_ITER}')

omega = mp(1); res_prev = mp('1e100')
res_t = []; omega_t = []; d_FR_t = []
t0 = time.time()
for it in range(1, MAX_ITER+1):
    t_step = time.time()
    P_phi = phi_sigdelta_strict(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W)
    one = mp(1)
    P_damped = [[[(one - omega) * P[i][j][k] + omega * P_phi[i][j][k]
                   for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
    P_damped = set_boundary(P_damped)
    res_mp = f_inf(P_damped, P)
    res = float(res_mp)
    if it > 3:
        if res_mp > res_prev * mp('0.99'):
            omega = max(omega * mp('0.7'), mp('0.001'))
        elif res_mp < res_prev * mp('0.6'):
            omega = min(omega * mp('1.05'), mp(1))
    P = P_damped
    res_prev = res_mp
    res_t.append(res); omega_t.append(float(omega))
    # d_FR_rms
    Pnp = to_np(P)
    d_FR = float(np.sqrt(np.mean((Pnp[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI]
                                    - P_FR_in)**2)))
    d_FR_t.append(d_FR)
    lg(f'  iter {it:3d}  ω={float(omega):.4f}  ferr={res:.3e}  d_FR={d_FR:.3e}  ({time.time()-t_step:.1f}s)')

lg(f'Total {(time.time()-t0)/60:.1f} min')

with open('/tmp/flint_sigdelta_damp.json', 'w') as f:
    json.dump({'ferr': res_t, 'omega': omega_t, 'd_FR': d_FR_t,
                'GAMMA': GAMMA_F, 'MAX_ITER': MAX_ITER, 'dps': DPS}, f, indent=2)
lg('saved')
