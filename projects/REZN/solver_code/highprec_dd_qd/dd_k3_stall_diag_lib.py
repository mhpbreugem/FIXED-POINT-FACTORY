"""Shared library for the stall-diagnosis overnight run.

Question: why does Newton-Krylov floor at F ~ 3e-2 for gamma*tau >~ 5
(K=3 CRRA REE kernel-smoothed operator)?  Candidates:
  (a) fold: sigma_min(I - J) -> 0 along the branch (branch ends),
  (b) eigenvalue of J crosses 1 with the branch continuing (bifurcation),
  (c) solver failure with a well-conditioned root present.

Tooling: G=13 (2197 unknowns) dense finite-difference Jacobian of Phi,
full dense spectrum of J and SVD of (I - J).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_2dsweep')
from reznsrc.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from scipy.optimize import newton_krylov
try:
    from scipy.optimize import NoConvergence
except ImportError:
    from scipy.optimize._nonlin import NoConvergence

K = 3; PAD = 2; UMAX = 4.0; C = 0.45
EMIN15 = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'
OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/stall_diagnosis'


def build_grid(Gi):
    du = 2*UMAX/(Gi - 1)
    Gf = Gi + 2*PAD
    uf = np.array([-UMAX + (q - PAD)*du for q in range(Gf)])
    return du, uf, PAD, PAD + Gi


def resample(P_warm, Gi):
    Gj = P_warm.shape[0]
    idx = np.linspace(0, Gj-1, Gi).astype(int)
    return P_warm[idx][:, idx][:, :, idx].copy()


class Problem:
    """Phi and F = Phi - I restricted to the inner cube, halo fixed at
    the no-learning init (same convention as the 20x5 anchor sweep)."""

    def __init__(self, tau, gamma, Gi=13):
        self.tau, self.gamma, self.Gi = tau, gamma, Gi
        self.du, self.uf, self.lo, self.hi = build_grid(Gi)
        self.h = C*np.sqrt(self.du)
        self.tv = np.full(K, tau); self.gv = np.full(K, gamma)
        self.wv = np.full(K, 1.0)
        self.P_halo = init_no_learning_K3(self.uf, self.tv, self.gv, self.wv)
        self.n = Gi**3

    def init_inner(self):
        return self.P_halo[self.lo:self.hi, self.lo:self.hi,
                           self.lo:self.hi].copy()

    def phi(self, x):
        Gi, lo, hi = self.Gi, self.lo, self.hi
        Pf = self.P_halo.copy()
        Pf[lo:hi, lo:hi, lo:hi] = x.reshape(Gi, Gi, Gi)
        Pn = phi_K3_halo_smooth(Pf, self.uf, lo, hi, self.tv, self.gv,
                                self.wv, self.h)
        return Pn[lo:hi, lo:hi, lo:hi].ravel()

    def F(self, x):
        return self.phi(x) - x

    def solve(self, x0, f_tol=1e-12, maxiter=60):
        try:
            x = newton_krylov(self.F, x0, f_tol=f_tol, maxiter=maxiter,
                              verbose=False)
        except NoConvergence as e:
            x = np.asarray(e.args[0], dtype=float)
        except Exception:
            x = np.asarray(x0, dtype=float)
        return x, float(np.max(np.abs(self.F(x))))

    def dense_jacobian(self, x, eps=1e-6):
        """Dense J = dPhi/dx by one-sided FD, column by column."""
        n = x.size
        J = np.empty((n, n))
        phi0 = self.phi(x)
        xp = x.copy()
        for j in range(n):
            xj = x[j]
            xp[j] = xj + eps
            J[:, j] = (self.phi(xp) - phi0) / eps
            xp[j] = xj
        return J


def flip_full(v3):
    """Sign-flip map u -> -u on a (G,G,G) array."""
    return v3[::-1, ::-1, ::-1]


def symmetry_defect_fp(x, Gi):
    """For the fixed point: P(u) + P(-u) - 1, sup norm."""
    P = x.reshape(Gi, Gi, Gi)
    return float(np.max(np.abs(P + flip_full(P) - 1.0)))


def symmetry_split_vec(v, Gi):
    """Eigenvector split under the symmetry. A symmetry-PRESERVING
    perturbation satisfies dP(u) = -dP(-u) (antisymmetric part);
    a symmetry-BREAKING one satisfies dP(u) = +dP(-u) (symmetric part).
    Returns (norm_breaking, norm_preserving)/norm."""
    v3 = v.reshape(Gi, Gi, Gi)
    fv = flip_full(v3)
    nb = np.linalg.norm(0.5*(v3 + fv))   # breaking component
    npres = np.linalg.norm(0.5*(v3 - fv))  # preserving component
    nn = np.linalg.norm(v3)
    return float(nb/nn), float(npres/nn)


def spectrum_report(prob, x, k=8):
    """Dense spectrum of J and SVD of (I-J); returns a JSON-able dict."""
    import scipy.linalg as sla
    t0 = time.time()
    J = prob.dense_jacobian(x)
    t_jac = time.time() - t0
    t0 = time.time()
    w, V = sla.eig(J)
    order = np.argsort(-np.abs(w))
    w = w[order]; V = V[:, order]
    A = np.eye(prob.n) - J
    sv = sla.svdvals(A)
    t_eig = time.time() - t0
    # eigenvalue closest to 1 (real distance in C)
    i1 = int(np.argmin(np.abs(w - 1.0)))
    lead = []
    for i in range(k):
        vb, vp = symmetry_split_vec(np.real(V[:, i]) if abs(np.imag(w[i])) < 1e-12
                                    else np.abs(V[:, i]), prob.Gi)
        lead.append(dict(re=float(np.real(w[i])), im=float(np.imag(w[i])),
                         mag=float(np.abs(w[i])),
                         sym_break=vb, sym_pres=vp))
    vb1, vp1 = symmetry_split_vec(np.real(V[:, i1]) if abs(np.imag(w[i1])) < 1e-12
                                  else np.abs(V[:, i1]), prob.Gi)
    return dict(
        lead_eigs=lead,
        rho=float(np.abs(w[0])),
        eig_closest_1=dict(re=float(np.real(w[i1])), im=float(np.imag(w[i1])),
                           dist=float(np.abs(w[i1] - 1.0)),
                           sym_break=vb1, sym_pres=vp1),
        sigma_min_ImJ=float(sv[-1]),
        sigma_max_ImJ=float(sv[0]),
        n_eigs_gt1=int(np.sum(np.abs(w) > 1.0)),
        n_real_eigs_gt1=int(np.sum((np.abs(np.imag(w)) < 1e-10) & (np.real(w) > 1.0))),
        t_jac=t_jac, t_eig=t_eig)


def load_warm(tau, gamma_str, Gi):
    P21 = np.load(f"{EMIN15}/P_ld_t{tau}_g{gamma_str}.npy")
    return resample(P21, Gi).ravel()


def checkpoint(obj, name):
    os.makedirs(OUT, exist_ok=True)
    tmp = f"{OUT}/{name}.tmp"
    json.dump(obj, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, f"{OUT}/{name}")
