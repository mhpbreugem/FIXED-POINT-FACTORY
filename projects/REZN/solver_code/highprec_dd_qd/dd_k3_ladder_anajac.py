"""DD K=3 ladder with analytic Jacobian + Newton.

Per cell:
  - Warm-start float64 R4 (Anderson) to F~1e-15
  - Build analytic J at warm point (5ms via numba chain-rule)
  - Symmetric-reduce J (84x84 at G=7), LU factor
  - Newton iters: compute DD F, dx = -J^{-1} F (float64), x += dx (DD-add)
  - 1-3 iters to reach DD eps
"""
import os, sys, time, json
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code/highprec_dd_qd")
sys.path.insert(0, "/tmp/cheby_h0")
sys.path.insert(0, "/tmp")
os.environ.setdefault("NUMBA_NUM_THREADS", "6")
import numpy as np
import mpmath as mp; mp.mp.dps = 40
import scipy.linalg as sla
from itertools import combinations_with_replacement, permutations
import dd_k3_ops as DK
import dd_k3_solver as DKS
import dd_ops as DO
import dd_k3_analytic_jac as AJ
from lin_cdf_kern_tab import make_cdf_uniform_grid, make_p_grid, make_gl_for_u
from lin_cdf_richardson import richardson_weights


G = 7
G_p = 121
NQK = 16
HS = (0.5, 0.4, 0.3, 0.2)
GAMMA = 100.0
TARGET_WARM = 1e-12
TARGET_DD = 1e-25
MAX_NEWTON = 8

ROOT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder"
OUT_JSON = f"{ROOT}/ladder.json"
FPS_DIR = f"{ROOT}/fps"
os.makedirs(FPS_DIR, exist_ok=True)


def split(x):
    h = float(x); l = float(x - mp.mpf(h)); return h, l


# Symmetric reduction: 84 unknowns at G=7 (combinations_with_replacement(range(7),3))
class SymRed3:
    def __init__(self, G):
        self.G = G
        self.triples = list(combinations_with_replacement(range(G), 3))
        self.n_red = len(self.triples)
        # Map from flat (i, j, k) → reduced index
        self.flat_to_red = np.zeros((G, G, G), dtype=np.int64)
        for idx, t in enumerate(self.triples):
            for p in set(permutations(t)):
                self.flat_to_red[p[0], p[1], p[2]] = idx
    def reduce(self, P):
        return np.array([P[t[0], t[1], t[2]] for t in self.triples])
    def expand(self, vec, dtype=float):
        P = np.empty((self.G, self.G, self.G), dtype=dtype)
        for i, t in enumerate(self.triples):
            v = vec[i]
            for p in set(permutations(t)):
                P[p[0], p[1], p[2]] = v
        return P
    def reduce_jac(self, J_full):
        """Reduce J_full (G^3 x G^3) to J_red (n_red x n_red).
        J_red[t_out, t_in] = sum over output perms of t_out... actually simpler:
          dF_red[t_out] / dx_red[t_in] = sum over input perms of t_in of dPhi[t_out_rep] / dP[that perm]
        We pick t_out_rep = the canonical (sorted) triple."""
        G3 = self.G ** 3
        n = self.n_red
        J_red = np.zeros((n, n))
        for to_idx, to_t in enumerate(self.triples):
            i_out = to_t[0]*self.G*self.G + to_t[1]*self.G + to_t[2]
            row = J_full[i_out, :]    # length G^3
            # sum over input perms grouped by reduced index
            for ti_idx, ti_t in enumerate(self.triples):
                acc = 0.0
                for p in set(permutations(ti_t)):
                    i_in = p[0]*self.G*self.G + p[1]*self.G + p[2]
                    acc += row[i_in]
                J_red[to_idx, ti_idx] = acc
        return J_red


def solve_with_anajac(P_warm, tau, u_grid, p_grid, gl_u, gl_du,
                          red, mu_R4_buf, dmu_dP_buf, P_new_buf, J_full_buf,
                          target=TARGET_DD, max_iters=MAX_NEWTON, verbose=False):
    """Newton with analytic Jacobian. P_warm is float64 (close to FP)."""
    G_ = u_grid.size
    th, tl = split(mp.mpf(repr(tau)))
    gh, gl = split(mp.mpf(repr(GAMMA)))
    w_arr = DK.richardson_weights(HS)
    w_R4 = richardson_weights(HS)
    # Build analytic Jacobian at warm point
    AJ.phi_and_jac(P_warm, u_grid, p_grid, gl_u, gl_du, float(tau), float(GAMMA),
                       NQK, np.array(HS), w_R4, mu_R4_buf, dmu_dP_buf, P_new_buf, J_full_buf)
    # F_F: residual P_new - P (note: Phi - I, so Newton uses J - I)
    G3 = G_*G_*G_
    JmI = J_full_buf - np.eye(G3)
    J_red = red.reduce_jac(JmI)
    try:
        lu, piv = sla.lu_factor(J_red)
    except Exception:
        return P_warm, float('inf'), False
    # DD nail
    x_H = red.reduce(P_warm).astype(np.float64)
    x_L = np.zeros(red.n_red)
    F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                      th, tl, gh, gl, HS, w_arr)
    F_inf = float(np.max(np.abs(F0_H + F0_L)))
    best_F = F_inf; best_x_H = x_H.copy(); best_x_L = x_L.copy()
    for it in range(max_iters):
        if F_inf < target: break
        F = F0_H + F0_L
        dx = sla.lu_solve((lu, piv), -F)
        for i in range(red.n_red):
            aH, aL = DO.dd_add(x_H[i], x_L[i], dx[i], 0.0)
            x_H[i] = aH; x_L[i] = aL
        F0_H, F0_L = DKS.F_dd_red(red, x_H, x_L, u_grid, p_grid, gl_u, gl_du,
                                          th, tl, gh, gl, HS, w_arr)
        F_inf = float(np.max(np.abs(F0_H + F0_L)))
        if verbose: print(f"    Newton {it+1}: |F|={F_inf:.3e}", flush=True)
        if F_inf < best_F:
            best_F = F_inf; best_x_H = x_H.copy(); best_x_L = x_L.copy()
        if it >= 2 and F_inf > 0.9*best_F: break
    P_full = red.expand(best_x_H + best_x_L)
    return P_full, best_F, best_F < target


def main():
    u_grid = make_cdf_uniform_grid(G)
    p_grid = make_p_grid(G_p)
    gl_u, gl_du = make_gl_for_u(u_grid[0], u_grid[-1], NQK)
    print(f"=== DD K=3 ladder (analytic Jacobian) gamma={GAMMA}, G={G} ===", flush=True)
    print("JIT warmup...", flush=True); t0 = time.time()
    U1, U2, U3 = np.meshgrid(u_grid, u_grid, u_grid, indexing="ij")
    P0 = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
    PL0 = np.zeros_like(P0)
    w_arr = DK.richardson_weights(HS); w_R4 = richardson_weights(HS)
    DK.phi_dd(P0, PL0, u_grid, p_grid, gl_u, gl_du, 1.0, 0.0, GAMMA, 0.0, HS, w_arr)
    mu_R4_buf = np.empty((G_p, G)); dmu_dP_buf = np.empty((G, G_p, G, G))
    P_new_buf = np.empty_like(P0); G3 = G*G*G
    J_full_buf = np.empty((G3, G3))
    AJ.phi_and_jac(P0, u_grid, p_grid, gl_u, gl_du, 0.001, GAMMA, NQK,
                       np.array(HS), w_R4, mu_R4_buf, dmu_dP_buf, P_new_buf, J_full_buf)
    print(f"  done {time.time()-t0:.1f}s", flush=True)

    red = SymRed3(G)
    taus = [round(0.001 * i, 4) for i in range(1, 1001)]
    results = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
    P_warm = None
    if results:
        completed = sorted([float(k) for k in results.keys()])
        last = completed[-1] if completed else None
        if last is not None:
            f = f"{FPS_DIR}/tau{last:.4f}.npy"
            if os.path.exists(f):
                P_warm = np.load(f)
                print(f"resuming from tau={last}", flush=True)
                taus = [t for t in taus if t > last + 1e-6]

    n_total = 1000; n_done_seed = 1000 - len(taus)
    t_start = time.time()
    for i, tau in enumerate(taus):
        n_done = n_done_seed + i + 1
        t0 = time.time()
        if P_warm is None: P_warm = 1.0/(1.0+np.exp(-0.5*(U1+U2+U3)))
        try:
            P_warm_new, F_warm = DKS.solve_warm_f64(P_warm, u_grid, p_grid, HS,
                                                              GAMMA, tau, G_p, NQK,
                                                              target=TARGET_WARM)
        except Exception as e:
            print(f"  [{n_done}] tau={tau} warm ERROR: {e}", flush=True); continue
        t_warm = time.time() - t0
        # Newton with analytic J
        t_n = time.time()
        P_full, F, conv = solve_with_anajac(P_warm_new, tau, u_grid, p_grid,
                                                    gl_u, gl_du, red, mu_R4_buf,
                                                    dmu_dP_buf, P_new_buf, J_full_buf)
        t_nail = time.time() - t_n
        wall = time.time() - t0
        np.save(f"{FPS_DIR}/tau{tau:.4f}.npy", P_full)
        results[f"{tau:.4f}"] = dict(tau=float(tau), F=float(F), wall=wall,
                                              t_warm=t_warm, t_nail=t_nail)
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        tag = "eps" if F < TARGET_DD else "OK" if F < 1e-10 else "FAIL"
        elapsed = time.time() - t_start
        eta = elapsed * (len(taus) - i - 1) / max(1, i+1) / 60
        if n_done % 25 == 0 or F > 1e-15:
            print(f"  [{n_done:>4d}/{n_total}] tau={tau:.4f} F={F:.2e}[{tag}] "
                  f"warm={t_warm:.1f}s nail={t_nail:.1f}s ETA={eta:.0f}min",
                  flush=True)
        P_warm = P_full
    print(f"\n=== DONE === {len(results)} cells in {(time.time()-t_start)/60:.0f}min",
          flush=True)


if __name__ == "__main__":
    main()
