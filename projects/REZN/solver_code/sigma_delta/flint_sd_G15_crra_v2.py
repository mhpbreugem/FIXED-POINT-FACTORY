"""flint sigma-delta at G_FULL=17 (G_inner=15) -- V2 fixes arb interval-comparison bug.

In v1 the contour-scan comparisons (`if frac < 0`, `if frac > 1`,
`if abs(xi_off) < 1 - 1e-15`) used arb's interval-aware operators, which
return False whenever the interval STRADDLES the comparison boundary. This
caused frac to occasionally stay at out-of-range values and produced
spurious contour points, biasing the inference toward Σ-interp errors and
giving d_FR=0.2 at G=15 (instead of float64's d_FR=8e-3).

Fix: all branch comparisons use `float(.)` to read the arb midpoint, while
arithmetic stays in arb precision.
"""
import time, math, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import flint
from flint import arb

DPS = 50
flint.ctx.prec = int(DPS * 3.33) + 20

G_FULL = 17
INNER_LO, INNER_HI = 1, G_FULL - 1
TOT_u = 2.0; TOT_S = 3.0; TOT_d = 3.0
TAU_F = 2.0; GAMMA_F = 0.1; W_F = 1.0
MAX_ITER = 40

LOG = os.path.join(HERE, 'flint_sd_G15_crra_v2.log')
open(LOG, 'w').close()
def lg(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f: f.write(line + '\n')

def mp(x):
    if isinstance(x, str): return arb(x)
    return arb(float(x))

def fsig(u, vm, tau, coef):
    d = u - vm
    return coef * (mp('-0.5') * tau * d * d).exp()

def crra_demand(mu, p, gamma, W):
    one = mp(1)
    lm = (mu / (one - mu)).log()
    lp = (p / (one - p)).log()
    R = ((lm - lp) / gamma).exp()
    return W * (R - one) / ((one - p) + R * p)

def crra_clear_sym(mus, gamma, W, steps=200):
    eps = mp('1e-40'); one = mp(1); two = mp(2); zero = mp(0)
    a, b = eps, one - eps
    for _ in range(steps):
        m = (a + b) / two
        ex = zero
        for mu in mus: ex = ex + crra_demand(mu, m, gamma, W)
        if float(ex) > 0: a = m
        else: b = m
    return (a + b) / two

def interp_along_Sigma(P, i_u, k_d, Sigma_target, xi_S, TOT_S_mp):
    G = len(xi_S); one = mp(1)
    St = float(Sigma_target)
    if St > 1e10: return P[i_u][G-1][k_d]
    if St < -1e10: return P[i_u][0][k_d]
    arg = Sigma_target / TOT_S_mp
    xi_t = arg.tanh()
    xt = float(xi_t)
    if xt <= float(xi_S[0]): return P[i_u][0][k_d]
    if xt >= float(xi_S[-1]): return P[i_u][G-1][k_d]
    for j in range(G - 1):
        xj  = float(xi_S[j]); xj1 = float(xi_S[j+1])
        if xj <= xt <= xj1:
            denom = xi_S[j+1] - xi_S[j]
            if float(denom) == 0: return P[i_u][j][k_d]
            frac = (xi_t - xi_S[j]) / denom
            return (one - frac) * P[i_u][j][k_d] + frac * P[i_u][j+1][k_d]
    return P[i_u][G-1][k_d]

def evidence_agent1(P, p_target, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm, i_u, coef):
    G_S = len(xi_S); G_d = len(xi_d); A = mp(0); one = mp(1); two = mp(2)
    pt = float(p_target)
    # scan Σ at fixed δ
    for k in range(G_d):
        prev = P[i_u][0][k]
        for j in range(G_S - 1):
            nxt = P[i_u][j+1][k]
            dp_f = float(prev) - pt; dn_f = float(nxt) - pt
            if not (dp_f == 0.0 and dn_f == 0.0) and dp_f * dn_f <= 0.0:
                dp = prev - p_target; dn = nxt - p_target
                denom = nxt - prev
                if float(denom) != 0:
                    frac = -dp / denom
                    fr_f = float(frac)
                    if fr_f < 0: frac = mp(0); fr_f = 0.0
                    elif fr_f > 1: frac = one; fr_f = 1.0
                    xi_S_off = (one - frac) * xi_S[j] + frac * xi_S[j+1]
                    if abs(float(xi_S_off)) < 1 - 1e-15:
                        Sigma_off = TOT_S_mp * xi_S_off.atanh()
                        if abs(float(xi_d[k])) < 1 - 1e-15:
                            delta_off = TOT_d_mp * xi_d[k].atanh()
                        else:
                            delta_off = mp(0)
                        u2_off = (Sigma_off + delta_off) / two
                        u3_off = (Sigma_off - delta_off) / two
                        A = A + fsig(u2_off, vm, tau, coef) * fsig(u3_off, vm, tau, coef)
    # scan δ at fixed Σ
    for j in range(G_S):
        prev = P[i_u][j][0]
        for k in range(G_d - 1):
            nxt = P[i_u][j][k+1]
            dp_f = float(prev) - pt; dn_f = float(nxt) - pt
            if not (dp_f == 0.0 and dn_f == 0.0) and dp_f * dn_f <= 0.0:
                dp = prev - p_target; dn = nxt - p_target
                denom = nxt - prev
                if float(denom) != 0:
                    frac = -dp / denom
                    fr_f = float(frac)
                    if fr_f < 0: frac = mp(0)
                    elif fr_f > 1: frac = one
                    xi_d_off = (one - frac) * xi_d[k] + frac * xi_d[k+1]
                    if abs(float(xi_d_off)) < 1 - 1e-15:
                        delta_off = TOT_d_mp * xi_d_off.atanh()
                        if abs(float(xi_S[j])) < 1 - 1e-15:
                            Sigma_off = TOT_S_mp * xi_S[j].atanh()
                        else:
                            Sigma_off = mp(0)
                        u2_off = (Sigma_off + delta_off) / two
                        u3_off = (Sigma_off - delta_off) / two
                        A = A + fsig(u2_off, vm, tau, coef) * fsig(u3_off, vm, tau, coef)
    return A / two

def evidence_agent_oblique(P, p_target, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                            tau, vm, u_cell, sign_for_other, coef):
    G_u = len(xi_u1); G_d = len(xi_d); A = mp(0); one = mp(1); two = mp(2)
    pt = float(p_target)
    # build (G_u, G_d) slice via Σ-interp
    P_slc = [[mp(0) for _ in range(G_d)] for _ in range(G_u)]
    for i in range(G_u):
        for k in range(G_d):
            if abs(float(xi_d[k])) < 1 - 1e-15:
                delta_k = TOT_d_mp * xi_d[k].atanh()
            else:
                delta_k = mp(1e10) * xi_d[k]
            Sigma_req = two * u_cell + sign_for_other * delta_k
            P_slc[i][k] = interp_along_Sigma(P, i, k, Sigma_req, xi_S, TOT_S_mp)
    # scan u_1 at fixed δ
    for k in range(G_d):
        if abs(float(xi_d[k])) >= 1 - 1e-15: continue
        delta_k = TOT_d_mp * xi_d[k].atanh()
        u_other = u_cell + sign_for_other * delta_k
        prev = P_slc[0][k]
        for i in range(G_u - 1):
            nxt = P_slc[i+1][k]
            dp_f = float(prev) - pt; dn_f = float(nxt) - pt
            if not (dp_f == 0.0 and dn_f == 0.0) and dp_f * dn_f <= 0.0:
                dp = prev - p_target; dn = nxt - p_target
                denom = nxt - prev
                if float(denom) != 0:
                    frac = -dp / denom
                    fr_f = float(frac)
                    if fr_f < 0: frac = mp(0)
                    elif fr_f > 1: frac = one
                    xi_u_off = (one - frac) * xi_u1[i] + frac * xi_u1[i+1]
                    if abs(float(xi_u_off)) < 1 - 1e-15:
                        u1_off = TOT_u_mp * xi_u_off.atanh()
                        A = A + fsig(u1_off, vm, tau, coef) * fsig(u_other, vm, tau, coef)
    # scan δ at fixed u_1
    for i in range(G_u):
        if abs(float(xi_u1[i])) >= 1 - 1e-15: continue
        u1_at = TOT_u_mp * xi_u1[i].atanh()
        prev = P_slc[i][0]
        for k in range(G_d - 1):
            nxt = P_slc[i][k+1]
            dp_f = float(prev) - pt; dn_f = float(nxt) - pt
            if not (dp_f == 0.0 and dn_f == 0.0) and dp_f * dn_f <= 0.0:
                dp = prev - p_target; dn = nxt - p_target
                denom = nxt - prev
                if float(denom) != 0:
                    frac = -dp / denom
                    fr_f = float(frac)
                    if fr_f < 0: frac = mp(0)
                    elif fr_f > 1: frac = one
                    xi_d_off = (one - frac) * xi_d[k] + frac * xi_d[k+1]
                    if abs(float(xi_d_off)) < 1 - 1e-15:
                        delta_off = TOT_d_mp * xi_d_off.atanh()
                        u_other_off = u_cell + sign_for_other * delta_off
                        A = A + fsig(u1_at, vm, tau, coef) * fsig(u_other_off, vm, tau, coef)
    return A / two

def phi_sigdelta(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W, clearing='crra'):
    one = mp(1); two = mp(2); vm0 = mp('-0.5'); vm1 = mp('0.5'); eps_p = mp('1e-40')
    coef = (tau / (two * arb.pi())).sqrt()
    G_u = len(xi_u1)
    P_new = [[[P[i][j][k] for k in range(G_u)] for j in range(G_u)] for i in range(G_u)]
    for i in range(INNER_LO, INNER_HI):
        if abs(float(xi_u1[i])) >= 1 - 1e-15: continue
        u1_cell = TOT_u_mp * xi_u1[i].atanh()
        for j in range(INNER_LO, INNER_HI):
            if abs(float(xi_S[j])) >= 1 - 1e-15: continue
            Sigma_cell = TOT_S_mp * xi_S[j].atanh()
            for k in range(INNER_LO, INNER_HI):
                if abs(float(xi_d[k])) >= 1 - 1e-15: continue
                d_cell = TOT_d_mp * xi_d[k].atanh()
                p_cell = P[i][j][k]
                u2_cell = (Sigma_cell + d_cell) / two
                u3_cell = (Sigma_cell - d_cell) / two
                A1_0 = evidence_agent1(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm0, i, coef)
                A1_1 = evidence_agent1(P, p_cell, xi_S, xi_d, TOT_S_mp, TOT_d_mp, tau, vm1, i, coef)
                f0_u1 = fsig(u1_cell, vm0, tau, coef); f1_u1 = fsig(u1_cell, vm1, tau, coef)
                den = f0_u1 * A1_0 + f1_u1 * A1_1
                mu0 = f1_u1 * A1_1 / den if float(den) > 0 else mp('0.5')
                A2_0 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                tau, vm0, u2_cell, -one, coef)
                A2_1 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                tau, vm1, u2_cell, -one, coef)
                f0_u2 = fsig(u2_cell, vm0, tau, coef); f1_u2 = fsig(u2_cell, vm1, tau, coef)
                den = f0_u2 * A2_0 + f1_u2 * A2_1
                mu1 = f1_u2 * A2_1 / den if float(den) > 0 else mp('0.5')
                A3_0 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                tau, vm0, u3_cell, one, coef)
                A3_1 = evidence_agent_oblique(P, p_cell, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp,
                                                tau, vm1, u3_cell, one, coef)
                f0_u3 = fsig(u3_cell, vm0, tau, coef); f1_u3 = fsig(u3_cell, vm1, tau, coef)
                den = f0_u3 * A3_0 + f1_u3 * A3_1
                mu2 = f1_u3 * A3_1 / den if float(den) > 0 else mp('0.5')
                # clip + clearing
                def clipmu(x):
                    xf = float(x)
                    if xf < float(eps_p): return eps_p
                    if xf > float(one - eps_p): return one - eps_p
                    return x
                mus = [clipmu(mu) for mu in (mu0, mu1, mu2)]
                if clearing == 'crra':
                    P_new[i][j][k] = crra_clear_sym(mus, gamma, W)
                else:  # cara
                    pi_ = (mus[0]/(one-mus[0])).log() + (mus[1]/(one-mus[1])).log() + (mus[2]/(one-mus[2])).log()
                    pi_ = pi_ / mp(3)
                    P_new[i][j][k] = one / (one + (-pi_).exp())
    return P_new

def set_boundary(P):
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
    m = mp(0)
    for i in range(INNER_LO, INNER_HI):
        for j in range(INNER_LO, INNER_HI):
            for k in range(INNER_LO, INNER_HI):
                d = abs(A[i][j][k] - B[i][j][k])
                if float(d) > float(m): m = d
    return m

def to_np(P):
    G = len(P)
    return np.array([[[float(P[i][j][k]) for k in range(G)] for j in range(G)] for i in range(G)])

# ----- setup -----
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
G_INNER = INNER_HI - INNER_LO
P_NL_in = np.empty_like(U1m)
for i in range(G_INNER):
    for j in range(G_INNER):
        for k in range(G_INNER):
            P_NL_in[i,j,k] = crra_clear_f64(mu1_NL[i,j,k], mu2_NL[i,j,k], mu3_NL[i,j,k], GAMMA_F)

P_full_np = np.zeros((G_FULL,)*3)
P_full_np[INNER_LO:INNER_HI, INNER_LO:INNER_HI, INNER_LO:INNER_HI] = P_NL_in
P = [[[mp(float(P_full_np[i,j,k])) for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
P = set_boundary(P)

if __name__ == '__main__':
    lg(f"CRRA flint sigma-delta V2 G_FULL={G_FULL} (G_inner={G_INNER}), dps={DPS}, gamma={GAMMA_F}, NO-LEARN IC, damped Picard.")
    lg("V2: all branch comparisons use float() (arb interval-comparison fix).")
    omega = mp(1); res_prev = mp('1e100')
    hist = {'res': [], 'omega': [], 'd_FR': [], 'sec_per_iter': []}
    t0 = time.time()
    for it in range(1, MAX_ITER+1):
        t_step = time.time()
        P_phi = phi_sigdelta(P, xi_u1, xi_S, xi_d, TOT_u_mp, TOT_S_mp, TOT_d_mp, tau, gamma, W, clearing='crra')
        one = mp(1)
        P_damp = [[[(one - omega) * P[i][j][k] + omega * P_phi[i][j][k]
                     for k in range(G_FULL)] for j in range(G_FULL)] for i in range(G_FULL)]
        P_damp = set_boundary(P_damp)
        res_mp = f_inf(P_damp, P); res = float(res_mp)
        if it > 3:
            if float(res_mp) > float(res_prev) * 0.99:   omega = max(omega * mp('0.7'), mp('0.001'))
            elif float(res_mp) < float(res_prev) * 0.6:  omega = min(omega * mp('1.05'), mp(1))
        P = P_damp; res_prev = res_mp
        Pnp = to_np(P)
        d_FR = float(np.sqrt(np.mean((Pnp[INNER_LO:INNER_HI,INNER_LO:INNER_HI,INNER_LO:INNER_HI] - P_FR_in)**2)))
        sec = time.time() - t_step
        hist['res'].append(res); hist['omega'].append(float(omega)); hist['d_FR'].append(d_FR); hist['sec_per_iter'].append(sec)
        lg(f"  iter {it:3d}  ω={float(omega):.4f}  ferr={res:.3e}  d_FR={d_FR:.3e}  ({sec:.0f}s)")
        json.dump({**hist, 'G_FULL': G_FULL, 'dps': DPS, 'gamma': GAMMA_F, 'tau': TAU_F},
                  open(os.path.join(HERE, 'flint_sd_G15_crra_v2.json'), 'w'), indent=2)
    lg(f"DONE total {(time.time()-t0)/60:.1f}m")
