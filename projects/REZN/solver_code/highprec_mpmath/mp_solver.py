"""High-precision mpmath fixed-point solver, G=50, TOL=1e-40.

- mpmath Φ map (dps configurable), matching float64 PCHIP operator
- Frozen Newton: Jacobian from float64 phi_cell_with_jac (block-diag in p),
  refreshed when convergence stalls. Final precision set by mpmath Φ, not J.
- Warm-start each target τ from nearest float64 cached FP (0.2 τ-grid).
- Saves: float64 .npz (compat) + high-precision text dump per FP.

Usage: python mp_solver.py <tau_target> [<tau_target> ...]
"""
import sys, time, os, glob, re
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/solver_pchip_tmp")  # not used
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points")
sys.path.insert(0, "/tmp")
import numpy as np
from scipy.interpolate import PchipInterpolator
import mpmath as mp
from mp_pchip import MpPchip

DPS = 60
mp.mp.dps = DPS
CACHE = "/tmp/fp_cache"
HP_DIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highprec"
os.makedirs(HP_DIR, exist_ok=True)

gamma_f = 100.0
gamma = mp.mpf(100); A = mp.mpf('5.5'); Gp = 17
G = int(os.environ.get("MPG", "32"))
TARGET_TOL = mp.mpf('1e-40')
MAX_ITER = 90

# float64 modules for Jacobian
from compact_ift import MuField, phi_cell_with_jac, _build_basis_funcs
def _logit_f(x): return np.log(x/(1.0-x))
def _sigmoid_f(x): return 1.0/(1.0+np.exp(-x))
def _rebuild_logit(self):
    self._row_lp = [PchipInterpolator(_logit_f(np.clip(self.p_grids[i], 1e-15, 1-1e-15)),
                                       _logit_f(np.clip(self.mu_vals[i], 1e-15, 1-1e-15)),
                                       extrapolate=True) for i in range(self.G)]
    self._row = self._row_lp
def col_at_p_logit_f(self, p):
    lp = _logit_f(float(np.clip(p, 1e-15, 1-1e-15)))
    return np.array([_sigmoid_f(float(self._row_lp[i](lp))) for i in range(self.G)])
MuField._rebuild_row_interp = _rebuild_logit
MuField.col_at_p = col_at_p_logit_f
MuField.col_at_p_smooth = col_at_p_logit_f

# ---- mpmath operator ----
def sigmoid(z): return 1/(1+mp.e**(-z))
def logit(p): return mp.log(p/(1-p))
def u_of_xi(xi, tau): return (2/tau)*mp.atanh(xi)
def dudxi(xi, tau): return (2/tau)/(1-xi*xi)
def f_signal(u, v, tau):
    mean = mp.mpf('0.5') if v == 1 else mp.mpf('-0.5')
    return mp.sqrt(tau/(2*mp.pi))*mp.e**(-tau/2*(u-mean)**2)
def x_crra(mu, p):
    R = mp.e**((logit(mu)-logit(p))/gamma)
    return (R-1)/((1-p)+R*p)
def mu_no_learning(u, tau): return sigmoid(tau*u)

p_lp = [mp.mpf(-A) + 2*A*k/(Gp-1) for k in range(Gp)]
p_arr = [sigmoid(l) for l in p_lp]
p_arr_f = 1.0/(1.0+np.exp(-np.linspace(-5.5, 5.5, Gp)))
xi_grid = [mp.mpf(-0.88) + mp.mpf(1.76)*k/(G-1) for k in range(G)]
xi_grid_f = np.linspace(-0.88, 0.88, G)
dxi = xi_grid[1]-xi_grid[0]


def make_phi(tau):
    xi_anchor = mp.mpf('0.99')
    u_anchor = u_of_xi(xi_anchor, tau)
    nl_lo = mu_no_learning(-u_anchor, tau); nl_hi = mu_no_learning(u_anchor, tau)
    clip15 = mp.mpf('1e-15'); clip9 = mp.mpf('1e-9'); eps = mp.mpf('1e-4')

    def col_at_p(mu_vals, p):
        lp = logit(p); out = []
        for i in range(G):
            ly = [logit(min(max(v, clip15), 1-clip15)) for v in mu_vals[i]]
            out.append(sigmoid(MpPchip(p_lp, ly)(lp)))
        return out

    def mu_curve_at_p(mu_vals, p):
        col = col_at_p(mu_vals, p)
        return MpPchip([-xi_anchor]+xi_grid+[xi_anchor], [nl_lo]+col+[nl_hi])

    def phi_cell(mu_vals, i, j):
        xi_i = xi_grid[i]; p_j = p_arr[j]; u_i = u_of_xi(xi_i, tau)
        mu_xi = mu_curve_at_p(mu_vals, p_j)
        def demand(xi):
            m = mu_xi(xi)
            if m < clip9: m = clip9
            if m > 1-clip9: m = 1-clip9
            return x_crra(m, p_j)
        d_own = demand(xi_i)
        lo_b = -1+eps; hi_b = 1-eps
        d_lo = demand(lo_b); d_hi = demand(hi_b)
        A0 = mp.mpf(0); A1 = mp.mpf(0)
        for ip in range(G):
            xi_2 = xi_grid[ip]; target = -d_own-demand(xi_2)
            if target < d_lo or target > d_hi: continue
            w_trap = dxi*(mp.mpf('0.5') if (ip==0 or ip==G-1) else mp.mpf(1))
            # bracket with bisection (~40 iters -> ~1e-12), then secant polish (quadratic)
            a, b = lo_b, hi_b
            fa = d_lo - target; fb = d_hi - target
            for _ in range(40):
                m = (a+b)/2; fm = demand(m) - target
                if (fm > 0) == (fa > 0): a = m; fa = fm
                else: b = m; fb = fm
            # secant within [a,b]
            x0, x1 = a, b; f0, f1 = fa, fb
            for _ in range(8):
                if f1 == f0: break
                x2 = x1 - f1*(x1-x0)/(f1-f0)
                if x2 < a or x2 > b: x2 = (a+b)/2
                f2 = demand(x2) - target
                x0, f0 = x1, f1; x1, f1 = x2, f2
                if abs(f2) < mp.mpf('1e-50'): break
            xi_3 = x1
            if xi_3 <= -1+eps/2 or xi_3 >= 1-eps/2: continue
            u_2 = u_of_xi(xi_2, tau); u_3 = u_of_xi(xi_3, tau)
            w = w_trap*dudxi(xi_2, tau)
            A0 += w*f_signal(u_2,0,tau)*f_signal(u_3,0,tau)
            A1 += w*f_signal(u_2,1,tau)*f_signal(u_3,1,tau)
        f0_i = f_signal(u_i,0,tau); f1_i = f_signal(u_i,1,tau)
        den = f0_i*A0 + f1_i*A1
        if den <= 0: return mu_xi(xi_i)
        return f1_i*A1/den

    def F_full(mu_vals):
        Phi = [[phi_cell(mu_vals, i, j) for j in range(Gp)] for i in range(G)]
        F = [[Phi[i][j]-mu_vals[i][j] for j in range(Gp)] for i in range(G)]
        Finf = max(abs(F[i][j]) for i in range(G) for j in range(Gp))
        return F, Finf
    return F_full


def float_jacobian(mu_vals_f, tau):
    """Block-diag-in-p Jacobian via float64 phi_cell_with_jac; return (I-JB)^{-1} per col."""
    mf = MuField(xi_grid_f, np.tile(p_arr_f, (G, 1)), mu_vals_f.copy(), tau)
    xi_full = np.concatenate([[-0.99], mf.xi_grid, [0.99]])
    bf = _build_basis_funcs(xi_full)
    JB = np.zeros((Gp, G, G))
    for j in range(Gp):
        for i in range(G):
            try:
                _, jr = phi_cell_with_jac(mf, i, j, gamma_f, tau, bf)
                JB[j, i, :] = jr
            except Exception: pass
    I_G = np.eye(G)
    inv_blocks = []
    for j in range(Gp):
        try: inv_blocks.append(np.linalg.inv(I_G - JB[j]))
        except Exception: inv_blocks.append(I_G.copy())
    return inv_blocks


def symmetrize_mp(mu):
    """Enforce µ(ξ,p) = 1 - µ(-ξ,1-p) in mpmath."""
    out = [[None]*Gp for _ in range(G)]
    for i in range(G):
        for j in range(Gp):
            a = mu[i][j]; b = 1 - mu[G-1-i][Gp-1-j]
            out[i][j] = (a+b)/2
    return out


def _apply_step(mu, F, inv_blocks, omega):
    new = [row[:] for row in mu]
    for j in range(Gp):
        Fj = np.array([float(F[i][j]) for i in range(G)])
        dj = inv_blocks[j] @ Fj
        for i in range(G):
            v = mu[i][j] + omega*mp.mpf(float(dj[i]))
            if v < mp.mpf('1e-15'): v = mp.mpf('1e-15')
            if v > 1-mp.mpf('1e-15'): v = 1-mp.mpf('1e-15')
            new[i][j] = v
    return symmetrize_mp(new)


def solve_tau(tau_f, warm_mu_f):
    """Adaptive-damped frozen Newton. The base step overshoots by a factor c that
    varies with (τ,G) (c~1.5 to ~3.6); a fixed ω can diverge. Backtracking line
    search on ω guarantees monotone F-decrease: accept only improving steps, grow
    ω on success, shrink (and refresh Jacobian) on failure."""
    tau = mp.mpf(str(tau_f))
    F_full = make_phi(tau)
    mu = [[mp.mpf(float(warm_mu_f[i, j])) for j in range(Gp)] for i in range(G)]
    inv_blocks = float_jacobian(warm_mu_f, tau_f)
    t0 = time.perf_counter(); last_beat = t0
    F, Finf = F_full(mu)
    hist = [Finf]
    omega = mp.mpf('0.5')
    for it in range(MAX_ITER):
        now = time.perf_counter()
        if now - last_beat > 20:
            print(f"    [HB {now-t0:.0f}s] τ={tau_f} iter {it}: F={mp.nstr(Finf,4)} ω={mp.nstr(omega,3)}", flush=True)
            last_beat = now
        if Finf < TARGET_TOL: break
        mu_new = _apply_step(mu, F, inv_blocks, omega)
        Fn, Finfn = F_full(mu_new)
        if Finfn < Finf:
            mu, F, Finf = mu_new, Fn, Finfn
            omega = min(omega*mp.mpf('1.15'), mp.mpf('0.9'))
        else:
            omega = omega*mp.mpf('0.4')
            if omega < mp.mpf('0.03'):
                inv_blocks = float_jacobian(
                    np.array([[float(mu[i][j]) for j in range(Gp)] for i in range(G)]), tau_f)
                omega = mp.mpf('0.3')
        hist.append(Finf)
    wall = time.perf_counter()-t0
    return mu, hist, wall


def nearest_float64_fp(tau_target):
    """Find nearest float64 cache FP. Prefer source with G==current G, then
    nearest τ. Handles both 'mu_tauX_GN_A55.npy' and legacy 'mu_tauX_A55.npy'
    (the latter are G=10)."""
    cands = []  # (G_match_penalty, dtau, fpath, t, Gi)
    for f in glob.glob(f"{CACHE}/mu_tau*_A55.npy"):
        m = re.match(r".*mu_tau([\d.]+)_G(\d+)_A55\.npy", f)
        if m:
            t = float(m.group(1)); Gi = int(m.group(2))
        else:
            m2 = re.match(r".*mu_tau([\d.]+)_A55\.npy", f)
            if not m2: continue
            t = float(m2.group(1)); Gi = 10  # legacy unlabeled = G=10
        cands.append((0 if Gi == G else 1, abs(t-tau_target), f, t, Gi))
    if not cands: return None
    cands.sort()
    _, _, f, t, Gi = cands[0]
    return (f, t, Gi)


def save_hp(mu, tau_f, Finf, iters, wall):
    # float64 npz for compatibility
    mu_f = np.array([[float(mu[i][j]) for j in range(Gp)] for i in range(G)])
    from save_fp import save_fixed_point
    save_fixed_point(mu_f, xi_grid_f, p_arr_f,
                     gamma=gamma_f, tau=tau_f, A_logit=5.5,
                     F_final=float(Finf), newton_iters=iters, tol=1e-40,
                     solver_interp="logit_logit_pchip_MPMATH", wall_seconds=wall,
                     script="mp_solver.py", subdir="highprec",
                     note=f"mpmath dps={DPS} high-precision FP, G={G}")
    # full-precision text dump
    txt = f"{HP_DIR}/g100.0_t{tau_f:.4f}_A5.50_G{G}_Gp{Gp}_dps{DPS}.txt"
    with open(txt, "w") as fh:
        fh.write(f"# mpmath dps={DPS} tau={tau_f} G={G} Gp={Gp} F={mp.nstr(Finf,8)} iters={iters}\n")
        for i in range(G):
            for j in range(Gp):
                fh.write(mp.nstr(mu[i][j], DPS) + ("\n" if j==Gp-1 else " "))
    return txt


if __name__ == "__main__":
    targets = [float(x) for x in sys.argv[1:]] or [1.0199]
    for tau_target in targets:
        nf = nearest_float64_fp(tau_target)
        if nf is None:
            print(f"τ={tau_target}: no float64 warm start"); continue
        fpath, tsrc, Gsrc = nf
        mu_src = np.load(fpath)
        # project to G=50
        xs = np.linspace(-0.88, 0.88, Gsrc)
        warm = np.zeros((G, Gp))
        for j in range(Gp):
            warm[:, j] = np.clip(PchipInterpolator(xs, mu_src[:, j], extrapolate=True)(xi_grid_f),
                                 1e-12, 1-1e-12)
        print(f"\n=== τ={tau_target} (warm from τ={tsrc} G={Gsrc}) dps={DPS} G={G} ===", flush=True)
        mu, hist, wall = solve_tau(tau_target, warm)
        Finf = hist[-1]
        ok = Finf < TARGET_TOL
        print(f"  {'✓' if ok else '✗'} τ={tau_target}  F={mp.nstr(Finf,4)}  iters={len(hist)}  {wall:.0f}s",
              flush=True)
        if Finf < mp.mpf('1e-20'):
            txt = save_hp(mu, tau_target, Finf, len(hist), wall)
            print(f"    saved {os.path.basename(txt)}", flush=True)
    print("done")
