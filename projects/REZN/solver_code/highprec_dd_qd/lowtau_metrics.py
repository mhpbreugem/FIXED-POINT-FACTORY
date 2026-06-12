"""Compute trading volume (TV) and value of information (Vi) for every
certified low-tau equilibrium, using the repo's definitions (port from
contour_KN_sym.sym_econ_metrics).

For each cell (tau, gamma) with certified P*(u1,u2,u3) on inner (G,G,G):
  Joint signal weight at u  =  0.5 * (f_0(u_1)f_0(u_2)f_0(u_3) + f_1(...))
  At each u, each agent k has posterior mu_k from slice integrals over P*.
  Optimal CRRA demand x*(mu_k, P; gamma, W=1).
  TV = E[|x*(mu_k, P)|]   (per-agent, expectation over u)
  Vi = CE_informed - CE_priceonly
    informed: trades with own posterior mu_k, EU at price P
    price-only: trades with mu_0 = prod_f1 / (prod_f0 + prod_f1)
                 = full-revealing-price posterior (repo convention)
  We also report Vi_PR = CE_informed - CE_kerneluninformed (using the no-
  learning baseline that conditions on NOTHING, only the unconditional prior).

For each cell we also report E[deficit_local] (a sanity cross-check).
"""
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from reznsrc.signals import f_signal
from reznsrc.demand import clear_crra as _clear

K = 3; pad = 2; UMAX = 4.0; C = 0.45
LOWTAU = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/lowtau'
EMIN15 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'
OUT = LOWTAU
W_AGENT = 1.0


def build_grid(Gi):
    du = 2*UMAX/(Gi - 1)
    Gf = Gi + 2*pad
    uf = np.array([-UMAX + (q - pad)*du for q in range(Gf)])
    lo, hi = pad, pad + Gi
    return du, uf, lo, hi


def x_crra(mu, p, gamma, W):
    eps = 1e-12
    mu = max(eps, min(1-eps, mu)); p = max(eps, min(1-eps, p))
    z = (math.log(mu/(1-mu)) - math.log(p/(1-p))) / gamma
    if z >= 0:
        e = math.exp(-z); return W*(1-e) / ((1-p)*e + p)
    e = math.exp(z); return W*(e-1) / ((1-p) + p*e)


def crra_u(w, gamma):
    if w <= 0: return -1e18
    if abs(gamma - 1) < 1e-9: return math.log(w)
    return w**(1-gamma) / (1-gamma)


def crra_inv(eu, gamma):
    if abs(gamma - 1) < 1e-9: return math.exp(eu)
    val = eu * (1 - gamma)
    if val <= 0: return 1e-18
    return val**(1/(1-gamma))


def cell_metrics(P_inner, uf, lo, hi, tau, gamma):
    """TV + value-of-information decomposition under the TRUE measure mu_FR.

    Posteriors used by each comparison agent:
      mu_informed = co-area Bayes given (own signal, price)
      mu_priceonly = co-area Bayes given (price only, integrating over all u
                     such that P(u) = p_eq) -- rational interpretation
      mu_FR = full Bayes given all 3 signals  =  prod_f1 / (prod_f0 + prod_f1)
      mu_prior = 0.5

    mu_priceonly depends only on the price value: we precompute it by
    grouping inner cube cells by their price (rounded to a fine grid) and
    aggregating sum_w * prod_f1 / sum_w * (prod_f0 + prod_f1) over each
    bucket.  In a fully-revealing equilibrium, this collapses to a delta
    of mu_priceonly = p_eq; in a PR equilibrium it differs from p_eq.

    Each agent picks the optimal CRRA position at the equilibrium price p_eq.
    Welfare is computed under the TRUE conditional probability mu_FR.
    Returns CE_*, Vi_private, Vi_public, Vi_total, Vi_FR_gap.
    """
    Gi = P_inner.shape[0]
    P_full = init_no_learning_K3(uf, np.full(3, tau), np.full(3, gamma), np.full(3, 1.0))
    P_full[lo:hi, lo:hi, lo:hi] = P_inner
    du = uf[1] - uf[0]; h = C * math.sqrt(du)

    from reznsrc.contour_K3_halo import _agent_evidence_K3_smooth
    from reznsrc.signals import f_signal as _fs

    f0_full = np.array([_fs(u, 0, tau) for u in uf])
    f1_full = np.array([_fs(u, 1, tau) for u in uf])

    # Precompute pf0, pf1, mu_FR, w_s at every inner cell
    pf0_arr = np.zeros((Gi, Gi, Gi)); pf1_arr = np.zeros((Gi, Gi, Gi))
    for i in range(Gi):
        for j in range(Gi):
            for l in range(Gi):
                pf0_arr[i,j,l] = f0_full[lo+i]*f0_full[lo+j]*f0_full[lo+l]
                pf1_arr[i,j,l] = f1_full[lo+i]*f1_full[lo+j]*f1_full[lo+l]
    w_arr = 0.5 * (pf0_arr + pf1_arr)
    mu_FR_arr = pf1_arr / np.maximum(pf0_arr + pf1_arr, 1e-30)

    # mu_priceonly(p) by binning cells. Use 200 quantile bins on P_inner.
    N_BINS = 200
    qs = np.linspace(0, 1, N_BINS + 1)
    edges = np.quantile(P_inner.ravel(), qs)
    edges[0] -= 1e-12; edges[-1] += 1e-12
    bin_centers = 0.5*(edges[:-1] + edges[1:])
    bin_idx = np.clip(np.searchsorted(edges, P_inner.ravel(), side='right') - 1,
                      0, N_BINS-1).reshape(Gi, Gi, Gi)
    num_pop_v1 = np.bincount(bin_idx.ravel(), weights=pf1_arr.ravel(), minlength=N_BINS)
    num_pop_v0 = np.bincount(bin_idx.ravel(), weights=pf0_arr.ravel(), minlength=N_BINS)
    mu_priceonly_bin = num_pop_v1 / np.maximum(num_pop_v0 + num_pop_v1, 1e-30)
    mu_priceonly_bin = np.clip(mu_priceonly_bin, 1e-12, 1-1e-12)

    # Informed posterior (price + own signal): bin-aggregate by (own_idx, price_bin)
    own_bin = np.arange(Gi)
    pair_id = own_bin[:, None, None] * N_BINS + bin_idx     # (Gi, Gi, Gi)
    num_info_v1 = np.bincount(pair_id.ravel(), weights=pf1_arr.ravel(),
                               minlength=Gi*N_BINS)
    num_info_v0 = np.bincount(pair_id.ravel(), weights=pf0_arr.ravel(),
                               minlength=Gi*N_BINS)
    mu_informed_pair = num_info_v1 / np.maximum(num_info_v0 + num_info_v1, 1e-30)
    mu_informed_pair = np.clip(mu_informed_pair, 1e-12, 1-1e-12)
    # average mu_informed for cell (i,j,l) = mu over agent 0's information set
    # (which is u_i and p_eq(i,j,l))
    mu_informed_arr = mu_informed_pair[pair_id]

    weight_sum = 0.0
    tv_sum = 0.0
    eu_informed_sum = 0.0
    eu_priceonly_sum = 0.0
    eu_FR_sum = 0.0
    eu_prior_sum = 0.0
    deficit_local_sum = 0.0

    u_in_arr = uf[lo:lo+Gi]
    sumu_arr = u_in_arr[:, None, None] + u_in_arr[None, :, None] + u_in_arr[None, None, :]
    for i in range(Gi):
        for j in range(Gi):
            for l in range(Gi):
                p_eq = float(P_inner[i, j, l])
                w_s = float(w_arr[i, j, l])
                weight_sum += w_s

                mu_FR = float(mu_FR_arr[i, j, l])
                mu_PR = float(mu_priceonly_bin[bin_idx[i, j, l]])
                mu_inf = float(mu_informed_arr[i, j, l])

                def trade_and_EU(mu):
                    x = x_crra(mu, p_eq, gamma, W_AGENT)
                    W1 = W_AGENT + x*(1 - p_eq); W0 = W_AGENT - x*p_eq
                    eu = mu_FR*crra_u(W1, gamma) + (1-mu_FR)*crra_u(W0, gamma)
                    return x, eu

                x_inf, eu_inf = trade_and_EU(mu_inf)
                _, eu_PR = trade_and_EU(mu_PR)
                _, eu_FR = trade_and_EU(mu_FR)
                _, eu_prior = trade_and_EU(0.5)

                tv_sum += w_s * abs(x_inf)
                eu_informed_sum += w_s * eu_inf
                eu_priceonly_sum += w_s * eu_PR
                eu_FR_sum += w_s * eu_FR
                eu_prior_sum += w_s * eu_prior

                pc = max(1e-12, min(1-1e-12, p_eq))
                deficit_local_sum += w_s * (math.log(pc/(1-pc)) - tau*float(sumu_arr[i,j,l]))**2

    if weight_sum <= 0: return None
    TV = tv_sum / weight_sum
    CE_informed = crra_inv(eu_informed_sum / weight_sum, gamma)
    CE_priceonly = crra_inv(eu_priceonly_sum / weight_sum, gamma)
    CE_FR = crra_inv(eu_FR_sum / weight_sum, gamma)
    CE_prior = crra_inv(eu_prior_sum / weight_sum, gamma)
    return dict(TV=float(TV),
                CE_informed=float(CE_informed),
                CE_priceonly=float(CE_priceonly),
                CE_FR=float(CE_FR),
                CE_prior=float(CE_prior),
                Vi_private=float(CE_informed - CE_priceonly),
                Vi_public=float(CE_priceonly - CE_prior),
                Vi_total=float(CE_informed - CE_prior),
                Vi_FR_gap=float(CE_FR - CE_informed),
                deficit_local_avg=float(deficit_local_sum / weight_sum))


def main():
    Gi = 21
    du, uf, lo, hi = build_grid(Gi)
    # Cells: all certified at low tau + emin15 tau=0.2, 0.5
    cells = []
    try:
        d = json.load(open(f"{LOWTAU}/lowtau.json"))
        for k, v in d.items():
            if v.get('verdict') == 'ACCEPT':
                cells.append((float(v['tau']), float(v['gamma']), 'lowtau'))
    except FileNotFoundError: pass
    try:
        d = json.load(open(f"{EMIN15}/emin15.json"))
        for k, v in d.items():
            tau_v = float(v['tau'])
            if v.get('verdict') == 'ACCEPT' and tau_v in (0.2, 0.5):
                cells.append((tau_v, float(v['gamma']), 'emin15'))
    except FileNotFoundError: pass
    print(f"To compute metrics on {len(cells)} certified cells", flush=True)

    results = {}
    for ci, (tau, gamma, src) in enumerate(sorted(cells)):
        key = f"t{tau}_g{gamma}"
        if key in results: continue
        p_src = f"{LOWTAU}/P_ld_t{tau}_g{gamma}.npy"
        if not os.path.exists(p_src):
            p_src = f"{EMIN15}/P_ld_t{tau}_g{gamma}.npy"
        if not os.path.exists(p_src):
            print(f"  missing P at {key}"); continue
        P_inner = np.load(p_src)
        t0 = time.time()
        m = cell_metrics(P_inner, uf, lo, hi, tau, gamma)
        wall = time.time() - t0
        if m is None: continue
        m['tau'] = tau; m['gamma'] = gamma; m['src'] = src; m['wall'] = wall
        results[key] = m
        if (ci+1) % 5 == 0 or ci == 0:
            print(f"  [{ci+1}/{len(cells)}] tau={tau:.2f} g={gamma:.4f}  "
                  f"TV={m['TV']:.3e}  Vi_priv={m['Vi_private']:.3e}  "
                  f"Vi_pub={m['Vi_public']:.3e}  Vi_FRgap={m['Vi_FR_gap']:.3e}  "
                  f"({wall:.0f}s)", flush=True)
        json.dump(results, open(f"{OUT}/metrics.json", 'w'), indent=2)
    print(f"\nWrote {OUT}/metrics.json with {len(results)} cells", flush=True)


if __name__ == "__main__":
    main()
