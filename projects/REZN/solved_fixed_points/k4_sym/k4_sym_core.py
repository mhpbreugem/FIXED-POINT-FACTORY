"""Symmetry-reduced representation + operator wrappers for the K=4 study.

Identical agents => the price P[i,j,l,m] is invariant under permuting the
four signal indices. We parametrise the symmetric subspace by SORTED index
tuples (multisets) i1<=i2<=i3<=i4 over the INNER grid. At G_inner=9 there
are C(9+4-1,4) = C(12,4) = 495 such multisets.

Operations:
  - reduced -> full inner block  : broadcast each multiset value over its orbit
  - full inner block -> reduced  : symmetrize (average over orbit) + read at
                                   the sorted representative.
The Phi operator is applied on the FULL padded grid (Numba phi_K4_halo);
the halo is itself symmetric (NL IC is symmetric for identical agents).
"""
import sys
from itertools import combinations_with_replacement, permutations
import numpy as np

sys.path.insert(0, '/tmp/rezn-source')
from code.contour_K4_halo import init_no_learning_halo, phi_K4_halo
from code.metrics import revelation_deficit
from code.signals import lam

K = 4


class SymReducer:
    def __init__(self, G_inner, pad, UMAX_inner, tau, W):
        self.G_inner = G_inner
        self.pad = pad
        self.UMAX_inner = UMAX_inner
        self.G_full = G_inner + 2 * pad
        self.du = 2.0 * UMAX_inner / (G_inner - 1)
        self.u_full = np.array(
            [-UMAX_inner + (q - pad) * self.du for q in range(self.G_full)],
            dtype=np.float64)
        self.inner_lo = pad
        self.inner_hi = pad + G_inner
        self.u_inner = self.u_full[self.inner_lo:self.inner_hi].copy()
        self.tau = tau
        self.W = W
        self.inner_slc = (slice(self.inner_lo, self.inner_hi),) * K

        # multiset representatives over inner indices 0..G_inner-1
        self.multisets = list(
            combinations_with_replacement(range(G_inner), K))
        self.n_red = len(self.multisets)
        # map sorted tuple -> reduced index
        self.ms_to_red = {ms: r for r, ms in enumerate(self.multisets)}

        # Precompute, for each FULL inner cell (i,j,l,m), which reduced index
        # it maps to (sort the tuple). Vectorised expand/symmetrize use this.
        self._build_maps()

    def _build_maps(self):
        Gi = self.G_inner
        # full inner index grid -> reduced index (sorted-tuple lookup)
        red_of_cell = np.empty((Gi,) * K, dtype=np.int64)
        # orbit size of each reduced multiset (number of full cells mapping to it)
        orbit_count = np.zeros(self.n_red, dtype=np.int64)
        it = np.ndindex(*((Gi,) * K))
        for idx in it:
            ms = tuple(sorted(idx))
            r = self.ms_to_red[ms]
            red_of_cell[idx] = r
            orbit_count[r] += 1
        self.red_of_cell = red_of_cell
        self.orbit_count = orbit_count
        # flat index of one canonical representative cell per reduced index
        rep_cell = np.array([ms for ms in self.multisets], dtype=np.int64)  # (n_red,K)
        self.rep_cell = rep_cell

    # ---- reduced (n_red,) -> full inner block (Gi,)*K ----
    def expand(self, vec_red):
        return vec_red[self.red_of_cell]

    # ---- full inner block (Gi,)*K -> reduced (n_red,) via orbit-average ----
    def reduce(self, inner_block):
        acc = np.zeros(self.n_red, dtype=np.float64)
        np.add.at(acc, self.red_of_cell.ravel(), inner_block.ravel())
        return acc / self.orbit_count

    # ---- read reduced vector at sorted representatives (no averaging) ----
    def read_rep(self, inner_block):
        # value at each multiset's canonical (sorted) cell
        out = np.empty(self.n_red, dtype=np.float64)
        for r, ms in enumerate(self.multisets):
            out[r] = inner_block[ms]
        return out

    # ---- halo (NL IC) ----
    def make_halo(self, gamma):
        gv = np.full(K, gamma); tv = np.full(K, self.tau); wv = np.full(K, self.W)
        halo = init_no_learning_halo(self.u_full, tv, gv, wv)
        return halo, gv, tv, wv

    # ---- full Phi on padded grid ----
    def phi_full(self, P_full, gv, tv, wv):
        return phi_K4_halo(P_full, self.u_full, self.inner_lo, self.inner_hi,
                           tv, gv, wv)

    # ---- symmetric Phi on reduced vector: expand->phi->symmetrize->read ----
    def phi_red(self, vec_red, halo, gv, tv, wv):
        P_full = halo.copy()
        P_full[self.inner_slc] = self.expand(vec_red)
        Pn = self.phi_full(P_full, gv, tv, wv)
        inner_new = Pn[self.inner_slc]
        # symmetrize the inner block (orbit-average) then read reduced
        return self.reduce(inner_new)

    # ---- residual F_sym(P_red) = phi_red - P_red ----
    def F_red(self, vec_red, halo, gv, tv, wv):
        return self.phi_red(vec_red, halo, gv, tv, wv) - vec_red

    # ---- deficit of a reduced vector (build full inner block, regress) ----
    def deficit_red(self, vec_red, tv):
        inner = self.expand(vec_red)
        return float(revelation_deficit(inner, self.u_inner, tv, K))

    # ---- fully-revealing seed: P = lam(sum_k tau_k u_k) on inner block ----
    def fr_seed_red(self):
        tau = self.tau
        T = np.zeros((self.G_inner,) * K)
        sh1 = [1] * K
        for k in range(K):
            sh = sh1.copy(); sh[k] = self.G_inner
            T = T + (tau * self.u_inner).reshape(sh)
        P_fr = 1.0 / (1.0 + np.exp(-T))
        P_fr = np.clip(P_fr, 1e-9, 1 - 1e-9)
        return self.read_rep(P_fr)  # already symmetric, read representatives

    # ---- distance from FR manifold (analogue of K=3 d_FR) ----
    # d_FR = weighted-RMS |logit(P) - tau*sum u| / scale, but we report the
    # simpler interpretable: max & weighted-rms gap of P from lam(tau*sumu).
    def d_fr_red(self, vec_red, tv):
        inner = self.expand(vec_red)
        tau = self.tau
        T = np.zeros((self.G_inner,) * K)
        sh1 = [1] * K
        for k in range(K):
            sh = sh1.copy(); sh[k] = self.G_inner
            T = T + (tau * self.u_inner).reshape(sh)
        P_fr = 1.0 / (1.0 + np.exp(-T))
        return float(np.sqrt(np.mean((inner - P_fr) ** 2)))
