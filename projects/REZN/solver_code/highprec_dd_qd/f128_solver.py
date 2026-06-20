"""float128 (80-bit, ~19 digit) fixed-point solver — the midway between f64 and mpmath.
Vectorized contour (array bisection), float64 frozen Jacobian, c-estimated under-relaxation.
Reaches ~1e-17. Use: fast continuation descent below the float64 τ≈0.93 wall.

Usage: MPG=32 python f128_solver.py descend <tau_start> <tau_min> <dtau>
       MPG=32 python f128_solver.py refine <tau1> <tau2> ...
"""
import sys, os, glob, re, time
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points")
sys.path.insert(0, "/tmp")
import numpy as np
from scipy.interpolate import PchipInterpolator
from f128_pchip import F128Pchip

F = np.float128
PI = F('3.141592653589793238462643383279502884')
CACHE = "/tmp/fp_cache"
HP_DIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highprec_f128"
os.makedirs(HP_DIR, exist_ok=True)

gamma_f = 100.0
gamma = F(100); A = F('5.5'); Gp = 17
G = int(os.environ.get("MPG", "32"))
TARGET_TOL = F('1e-16')
MAX_ITER = 200

# ---- float64 modules for Jacobian ----
from compact_ift import MuField, phi_cell_with_jac, _build_basis_funcs
def _logit_f(x): return np.log(x/(1.0-x))
def _sigmoid_f(x): return 1.0/(1.0+np.exp(-x))
def _rebuild_logit(self):
    self._row_lp = [PchipInterpolator(_logit_f(np.clip(self.p_grids[i], 1e-15, 1-1e-15)),
                                       _logit_f(np.clip(self.mu_vals[i], 1e-15, 1-1e-15)),
                                       extrapolate=True) for i in range(self.G)]
    self._row = self._row_lp
def _col_logit_f(self, p):
    lp = _logit_f(float(np.clip(p, 1e-15, 1-1e-15)))
    return np.array([_sigmoid_f(float(self._row_lp[i](lp))) for i in range(self.G)])
MuField._rebuild_row_interp = _rebuild_logit
MuField.col_at_p = _col_logit_f
MuField.col_at_p_smooth = _col_logit_f

# ---- f128 helpers ----
def sigmoid(z): return 1/(1+np.exp(-z))
def logit(p): return np.log(p/(1-p))
def u_of_xi(xi, tau): return (2/tau)*np.arctanh(xi)
def dudxi(xi, tau): return (2/tau)/(1-xi*xi)
def f_signal(u, v, tau):
    mean = F('0.5') if v == 1 else F('-0.5')
    return np.sqrt(tau/(2*PI))*np.exp(-tau/2*(u-mean)**2)
def x_crra(mu, p):
    R = np.exp((logit(mu)-logit(p))/gamma)
    return (R-1)/((1-p)+R*p)
def mu_no_learning(u, tau): return sigmoid(tau*u)

p_lp = np.array([F(-A) + 2*A*k/(Gp-1) for k in range(Gp)], dtype=F)
p_arr = sigmoid(p_lp)
p_arr_f = 1.0/(1.0+np.exp(-np.linspace(-5.5, 5.5, Gp)))
xi_grid = np.array([F('-0.88') + F('1.76')*k/(G-1) for k in range(G)], dtype=F)
xi_grid_f = np.linspace(-0.88, 0.88, G)
dxi = xi_grid[1]-xi_grid[0]
CLIP15 = F('1e-15'); CLIP9 = F('1e-9'); EPS = F('1e-4')
w_trap = np.full(G, dxi, dtype=F); w_trap[0] *= F('0.5'); w_trap[-1] *= F('0.5')


def make_F(tau):
    xi_anchor = F('0.99')
    u_anchor = u_of_xi(xi_anchor, tau)
    nl_lo = mu_no_learning(-u_anchor, tau); nl_hi = mu_no_learning(u_anchor, tau)
    xi_ext = np.concatenate([[-xi_anchor], xi_grid, [xi_anchor]])

    def col_at_p(mu_vals, p):
        lp = logit(p)
        out = np.empty(G, dtype=F)
        for i in range(G):
            ly = logit(np.clip(mu_vals[i], CLIP15, 1-CLIP15))
            out[i] = F128Pchip(p_lp, ly)(lp)
        return sigmoid(out)

    def mu_curve_at_p(mu_vals, p):
        col = col_at_p(mu_vals, p)
        col_ext = np.concatenate([[nl_lo], col, [nl_hi]])
        return F128Pchip(xi_ext, col_ext)

    def demand_vec(mu_xi, xi_arr, p_j):
        m = np.clip(mu_xi(xi_arr), CLIP9, 1-CLIP9)
        return x_crra(m, p_j)

    def phi_column(mu_vals, j):
        """Return Φ[:,j] for all i (vectorized contour)."""
        p_j = p_arr[j]
        mu_xi = mu_curve_at_p(mu_vals, p_j)
        d_grid = demand_vec(mu_xi, xi_grid, p_j)        # demand at each ξ_2 (and ξ_own)
        d_lo = demand_vec(mu_xi, np.array([-1+EPS], dtype=F), p_j)[0]
        d_hi = demand_vec(mu_xi, np.array([1-EPS], dtype=F), p_j)[0]
        u2 = u_of_xi(xi_grid, tau)
        f0_2 = f_signal_arr(u2, 0); f1_2 = f_signal_arr(u2, 1)
        wj = w_trap*dudxi(xi_grid, tau)
        out = np.empty(G, dtype=F)
        for i in range(G):
            d_own = d_grid[i]
            targets = -d_own - d_grid                    # length G
            mask = (targets >= d_lo) & (targets <= d_hi)
            # vectorized bisection for all targets (only mask matters)
            a = np.full(G, F(-1)+EPS); b = np.full(G, F(1)-EPS)
            for _ in range(64):
                mmid = (a+b)/2
                fm = demand_vec(mu_xi, mmid, p_j) - targets
                pos = fm > 0
                b = np.where(pos, mmid, b)
                a = np.where(pos, a, mmid)
            xi3 = (a+b)/2
            u3 = u_of_xi(np.clip(xi3, F(-1)+EPS, F(1)-EPS), tau)
            f0_3 = f_signal_arr(u3, 0); f1_3 = f_signal_arr(u3, 1)
            contrib0 = wj*f0_2*f0_3; contrib1 = wj*f1_2*f1_3
            A0 = np.sum(np.where(mask, contrib0, F(0)))
            A1 = np.sum(np.where(mask, contrib1, F(0)))
            u_i = u_of_xi(xi_grid[i], tau)
            f0i = f_signal(u_i, 0, tau); f1i = f_signal(u_i, 1, tau)
            den = f0i*A0 + f1i*A1
            out[i] = mu_xi(xi_grid[i]) if den <= 0 else f1i*A1/den
        return out

    def f_signal_arr(u, v):
        mean = F('0.5') if v == 1 else F('-0.5')
        return np.sqrt(tau/(2*PI))*np.exp(-tau/2*(u-mean)**2)

    def F_full(mu_vals):
        Phi = np.empty((G, Gp), dtype=F)
        for j in range(Gp):
            Phi[:, j] = phi_column(mu_vals, j)
        Fmat = Phi - mu_vals
        return Fmat, np.max(np.abs(Fmat))
    return F_full


def float_jacobian(mu_vals_f, tau):
    mf = MuField(xi_grid_f, np.tile(p_arr_f, (G, 1)), mu_vals_f.copy(), tau)
    xi_full = np.concatenate([[-0.99], mf.xi_grid, [0.99]])
    bf = _build_basis_funcs(xi_full)
    inv_blocks = []
    I_G = np.eye(G)
    JB = np.zeros((Gp, G, G))
    for j in range(Gp):
        for i in range(G):
            try:
                _, jr = phi_cell_with_jac(mf, i, j, gamma_f, tau, bf); JB[j, i, :] = jr
            except Exception: pass
    for j in range(Gp):
        try: inv_blocks.append(np.linalg.inv(I_G - JB[j]))
        except Exception: inv_blocks.append(I_G.copy())
    return inv_blocks


def symmetrize(mu):
    return (mu + (1 - mu[::-1, ::-1]))/2


def _apply_step(mu, Fmat, inv_blocks, omega):
    new = mu.copy()
    for j in range(Gp):
        dj = inv_blocks[j] @ Fmat[:, j].astype(np.float64)
        new[:, j] = mu[:, j] + omega*dj.astype(F)
    new = np.clip(new, CLIP15, 1-CLIP15)
    return symmetrize(new)


def _est_omega(mu, Fmat, Finf, inv_blocks, F_full):
    mtry = _apply_step(mu, Fmat, inv_blocks, F(1))
    _, Finf1 = F_full(mtry)
    rho = Finf1/Finf if Finf > 0 else F('0.5')
    omega = 1/(1+rho)
    return min(max(omega, F('0.2')), F('0.85'))


def solve_tau(tau_f, warm_mu_f, tag=""):
    tau = F(str(tau_f))
    F_full = make_F(tau)
    mu = warm_mu_f.astype(F)
    inv_blocks = float_jacobian(warm_mu_f, tau_f)
    t0 = time.perf_counter(); last = t0
    Fmat, Finf = F_full(mu)
    hist = [Finf]
    omega = _est_omega(mu, Fmat, Finf, inv_blocks, F_full)
    for it in range(MAX_ITER):
        now = time.perf_counter()
        if now-last > 20:
            print(f"    [HB {now-t0:.0f}s] {tag}τ={tau_f} iter {it}: F={float(Finf):.3e} ω={float(omega):.3f}", flush=True)
            last = now
        if Finf < TARGET_TOL: break
        mnew = _apply_step(mu, Fmat, inv_blocks, omega)
        Fn, Finfn = F_full(mnew)
        if Finfn < Finf:
            mu, Fmat, Finf = mnew, Fn, Finfn
            if it % 25 == 24: omega = _est_omega(mu, Fmat, Finf, inv_blocks, F_full)
        else:
            omega *= F('0.5')
            if omega < F('0.05'):
                inv_blocks = float_jacobian(mu.astype(np.float64), tau_f)
                omega = _est_omega(mu, Fmat, Finf, inv_blocks, F_full)
        hist.append(Finf)
    return mu, hist, time.perf_counter()-t0


def save_fp(mu, tau_f, Finf, iters, wall):
    from save_fp import save_fixed_point
    save_fixed_point(mu.astype(np.float64), xi_grid_f, p_arr_f,
                     gamma=gamma_f, tau=tau_f, A_logit=5.5,
                     F_final=float(Finf), newton_iters=iters, tol=float(TARGET_TOL),
                     solver_interp="logit_logit_pchip_FLOAT128", wall_seconds=wall,
                     script="f128_solver.py", subdir="highprec_f128",
                     note=f"float128 80-bit FP G={G}")
    txt = f"{HP_DIR}/g100.0_t{tau_f:.4f}_A5.50_G{G}_Gp{Gp}_f128.txt"
    with open(txt, "w") as fh:
        fh.write(f"# float128 tau={tau_f} G={G} F={float(Finf):.6e} iters={iters}\n")
        for i in range(G):
            fh.write(" ".join(repr(mu[i, j]) for j in range(Gp))+"\n")
    return txt


def nearest_fp(tau_target):
    cands = []
    for f in glob.glob(f"{CACHE}/mu_tau*_A55.npy"):
        m = re.match(r".*mu_tau([\d.]+)_G(\d+)_A55\.npy", f)
        if m: t = float(m.group(1)); Gi = int(m.group(2))
        else:
            m2 = re.match(r".*mu_tau([\d.]+)_A55\.npy", f)
            if not m2: continue
            t = float(m2.group(1)); Gi = 10
        cands.append((0 if Gi == G else 1, abs(t-tau_target), f, t, Gi))
    cands.sort()
    return cands[0] if cands else None


def project(mu_src, Gsrc):
    if Gsrc == G: return mu_src
    xs = np.linspace(-0.88, 0.88, Gsrc)
    out = np.zeros((G, Gp))
    for j in range(Gp):
        out[:, j] = np.clip(PchipInterpolator(xs, mu_src[:, j], extrapolate=True)(xi_grid_f), 1e-12, 1-1e-12)
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "refine":
        for tt in [float(x) for x in sys.argv[2:]]:
            nf = nearest_fp(tt)
            _, _, f, tsrc, Gsrc = nf
            warm = project(np.load(f), Gsrc)
            print(f"\n=== refine τ={tt} (warm τ={tsrc} G={Gsrc}) G={G} f128 ===", flush=True)
            mu, hist, wall = solve_tau(tt, warm)
            ok = hist[-1] < TARGET_TOL
            print(f"  {'✓' if ok else '✗'} τ={tt} F={float(hist[-1]):.3e} iters={len(hist)} {wall:.0f}s", flush=True)
            if hist[-1] < F('1e-13'):
                print("    saved", os.path.basename(save_fp(mu, tt, hist[-1], len(hist), wall)), flush=True)
    elif mode == "descend":
        tau_start, tau_min, dtau = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        nf = nearest_fp(tau_start)
        _, _, f, tsrc, Gsrc = nf
        warm = project(np.load(f), Gsrc)
        print(f"=== DESCEND from τ={tau_start} (warm τ={tsrc} G={Gsrc}) down to {tau_min} step {dtau}, G={G} f128 ===", flush=True)
        tau = tau_start
        while tau >= tau_min - 1e-9:
            mu, hist, wall = solve_tau(tau, warm, tag="[desc] ")
            ok = hist[-1] < TARGET_TOL
            print(f"  {'✓' if ok else '✗'} τ={tau:.4f} F={float(hist[-1]):.3e} iters={len(hist)} {wall:.0f}s", flush=True)
            if hist[-1] < F('1e-13'):
                save_fp(mu, tau, hist[-1], len(hist), wall)
                warm = mu.astype(np.float64)   # continuation
            else:
                print(f"  STALL at τ={tau:.4f} (F={float(hist[-1]):.2e}) — wall may be structural; stopping", flush=True)
                break
            tau -= dtau
    print("done", flush=True)
