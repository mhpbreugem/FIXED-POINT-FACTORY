"""Double-double (numba) fixed-point solver — ~32 digits, target ~1e-29.
Operator matches the mpmath/float64 PCHIP map. Each p_j is a grid point so the
p-interpolation returns the column exactly; only the ξ-PCHIP is needed.
Frozen float64 Jacobian + c-estimated under-relaxation (residual kept in DD).

Usage: MPG=32 python dd_solver.py refine <tau> ...
       MPG=32 python dd_solver.py descend <tau_start> <tau_min> <dtau>
"""
import sys, os, glob, re, time
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points")
sys.path.insert(0, "/tmp")
import numpy as np
from numba import njit, prange
from scipy.interpolate import PchipInterpolator
import dd_ops as D
from dd_ops import (dd_add, dd_mul, dd_div, dd_exp, dd_log, dd_sqrt,
                    dd_atanh, dd_sigmoid, dd_logit, two_sum)

CACHE = "/tmp/fp_cache"
HP_DIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highprec_dd"
os.makedirs(HP_DIR, exist_ok=True)
gamma_f = 100.0
Gp = 17
G = int(os.environ.get("MPG", "32"))
A = 5.5
TARGET_TOL = 1e-29
MAX_ITER = 120
EPSB = 1e-4
CLIP9 = 1e-9; CLIP15 = 1e-15

p_lp = np.linspace(-A, A, Gp)
p_arr_f = 1.0/(1.0+np.exp(-p_lp))
xi_grid_f = np.linspace(-0.88, 0.88, G)
dxi = xi_grid_f[1]-xi_grid_f[0]
w_trap_f = np.full(G, dxi); w_trap_f[0] *= 0.5; w_trap_f[-1] *= 0.5


@njit
def _pchip_slopes(xhh, xhl, yh, yl, dh, dl):
    n = xhh.shape[0]
    for k in range(1, n-1):
        h0h, h0l = dd_add(xhh[k], xhl[k], -xhh[k-1], -xhl[k-1])     # DD knot spacing
        h1h, h1l = dd_add(xhh[k+1], xhl[k+1], -xhh[k], -xhl[k])
        a_h, a_l = dd_add(yh[k], yl[k], -yh[k-1], -yl[k-1])
        d0h, d0l = dd_div(a_h, a_l, h0h, h0l)
        b_h, b_l = dd_add(yh[k+1], yl[k+1], -yh[k], -yl[k])
        d1h, d1l = dd_div(b_h, b_l, h1h, h1l)
        if (d0h == 0.0 and d0l == 0.0) or (d1h == 0.0 and d1l == 0.0) \
           or ((d0h > 0.0) != (d1h > 0.0)):
            dh[k] = 0.0; dl[k] = 0.0
        else:
            th1h, th1l = dd_mul(2.0, 0.0, h1h, h1l)
            w1h, w1l = dd_add(th1h, th1l, h0h, h0l)   # 2*h1 + h0 in DD
            th0h, th0l = dd_mul(2.0, 0.0, h0h, h0l)
            w2h, w2l = dd_add(th0h, th0l, h1h, h1l)   # 2*h0 + h1 in DD
            t1h, t1l = dd_div(w1h, w1l, d0h, d0l)
            t2h, t2l = dd_div(w2h, w2l, d1h, d1l)
            sh, sl = dd_add(t1h, t1l, t2h, t2l)
            wsh, wsl = dd_add(w1h, w1l, w2h, w2l)
            dh[k], dl[k] = dd_div(wsh, wsl, sh, sl)
    for which in range(2):
        if which == 0:
            k = 0
            h0h, h0l = dd_add(xhh[1], xhl[1], -xhh[0], -xhl[0])
            h1h, h1l = dd_add(xhh[2], xhl[2], -xhh[1], -xhl[1])
            p_h, p_l = dd_add(yh[1], yl[1], -yh[0], -yl[0]); m0h, m0l = dd_div(p_h, p_l, h0h, h0l)
            q_h, q_l = dd_add(yh[2], yl[2], -yh[1], -yl[1]); m1h, m1l = dd_div(q_h, q_l, h1h, h1l)
        else:
            k = n-1
            h0h, h0l = dd_add(xhh[n-1], xhl[n-1], -xhh[n-2], -xhl[n-2])
            h1h, h1l = dd_add(xhh[n-2], xhl[n-2], -xhh[n-3], -xhl[n-3])
            p_h, p_l = dd_add(yh[n-1], yl[n-1], -yh[n-2], -yl[n-2]); m0h, m0l = dd_div(p_h, p_l, h0h, h0l)
            q_h, q_l = dd_add(yh[n-2], yl[n-2], -yh[n-3], -yl[n-3]); m1h, m1l = dd_div(q_h, q_l, h1h, h1l)
        th0h, th0l = dd_mul(2.0, 0.0, h0h, h0l)
        aw_h, aw_l = dd_add(th0h, th0l, h1h, h1l)    # 2*h0 + h1 in DD
        t_h, t_l = dd_mul(aw_h, aw_l, m0h, m0l)
        t2_h, t2_l = dd_mul(h0h, h0l, m1h, m1l)
        ddh, ddl = dd_add(t_h, t_l, -t2_h, -t2_l)
        hsh, hsl = dd_add(h0h, h0l, h1h, h1l)        # h0 + h1 in DD
        ddh, ddl = dd_div(ddh, ddl, hsh, hsl)
        if (ddh > 0.0) != (m0h > 0.0):
            dh[k] = 0.0; dl[k] = 0.0
        elif ((m0h > 0.0) != (m1h > 0.0)) and (abs(ddh) > 3*abs(m0h)):
            dh[k], dl[k] = dd_mul(3.0, 0.0, m0h, m0l)
        else:
            dh[k] = ddh; dl[k] = ddl


@njit
def _pchip_eval(xhh, xhl, yh, yl, dh, dl, qh, ql):
    n = xhh.shape[0]
    if qh <= xhh[0]:
        k = 0
    elif qh >= xhh[n-1]:
        k = n-2
    else:
        k = 0
        for kk in range(n-1):
            if xhh[kk] <= qh and qh <= xhh[kk+1]:
                k = kk; break
    hkh, hkl = dd_add(xhh[k+1], xhl[k+1], -xhh[k], -xhl[k])    # DD interval width
    qmh, qml = dd_add(qh, ql, -xhh[k], -xhl[k])                # DD query offset (knot has lo word)
    th, tl = dd_div(qmh, qml, hkh, hkl)
    omt_h, omt_l = dd_add(1.0, 0.0, -th, -tl)
    omt2_h, omt2_l = dd_mul(omt_h, omt_l, omt_h, omt_l)
    t2_h, t2_l = dd_mul(th, tl, th, tl)
    tt_h, tt_l = dd_mul(2.0, 0.0, th, tl)
    a_h, a_l = dd_add(1.0, 0.0, tt_h, tt_l)
    h00_h, h00_l = dd_mul(a_h, a_l, omt2_h, omt2_l)
    h10_h, h10_l = dd_mul(th, tl, omt2_h, omt2_l)
    nt_h, nt_l = dd_mul(-2.0, 0.0, th, tl)
    b_h, b_l = dd_add(3.0, 0.0, nt_h, nt_l)
    h01_h, h01_l = dd_mul(t2_h, t2_l, b_h, b_l)
    c_h, c_l = dd_add(th, tl, -1.0, 0.0)
    h11_h, h11_l = dd_mul(t2_h, t2_l, c_h, c_l)
    r_h, r_l = dd_mul(h00_h, h00_l, yh[k], yl[k])
    hd_h, hd_l = dd_mul(hkh, hkl, dh[k], dl[k])
    tmp_h, tmp_l = dd_mul(h10_h, h10_l, hd_h, hd_l)
    r_h, r_l = dd_add(r_h, r_l, tmp_h, tmp_l)
    tmp_h, tmp_l = dd_mul(h01_h, h01_l, yh[k+1], yl[k+1])
    r_h, r_l = dd_add(r_h, r_l, tmp_h, tmp_l)
    hd_h, hd_l = dd_mul(hkh, hkl, dh[k+1], dl[k+1])
    tmp_h, tmp_l = dd_mul(h11_h, h11_l, hd_h, hd_l)
    r_h, r_l = dd_add(r_h, r_l, tmp_h, tmp_l)
    return r_h, r_l


@njit
def _x_crra(muh, mul, ph, pl, gh, gl):
    lmh, lml = dd_logit(muh, mul)
    lph, lpl = dd_logit(ph, pl)
    dlh, dll = dd_add(lmh, lml, -lph, -lpl)
    ar_h, ar_l = dd_div(dlh, dll, gh, gl)
    Rh, Rl = dd_exp(ar_h, ar_l)
    numh, numl = dd_add(Rh, Rl, -1.0, 0.0)
    omp_h, omp_l = dd_add(1.0, 0.0, -ph, -pl)
    Rp_h, Rp_l = dd_mul(Rh, Rl, ph, pl)
    denh, denl = dd_add(omp_h, omp_l, Rp_h, Rp_l)
    return dd_div(numh, numl, denh, denl)


@njit
def _clip_mu(mh, ml):
    if mh < CLIP9: return CLIP9, 0.0
    if mh > 1.0-CLIP9: return 1.0-CLIP9, 0.0
    return mh, ml


@njit
def _demand(xq_h, xq_l, exhh, exhl, yh, yl, dh, dl, ph, pl, gh, gl):
    mh, ml = _pchip_eval(exhh, exhl, yh, yl, dh, dl, xq_h, xq_l)
    mh, ml = _clip_mu(mh, ml)
    return _x_crra(mh, ml, ph, pl, gh, gl)


@njit
def _uofxi(xh, xl, tot_h, tot_l):
    ah, al = dd_atanh(xh, xl)
    return dd_mul(tot_h, tot_l, ah, al)


@njit
def _fsig(uh, ul, mean, th, tl, coef_h, coef_l):
    # coef * exp(-tau/2 * (u-mean)^2)
    dh_, dl_ = dd_add(uh, ul, -mean, 0.0)
    d2h, d2l = dd_mul(dh_, dl_, dh_, dl_)
    ht_h, ht_l = dd_mul(-0.5, 0.0, th, tl)
    eh, el = dd_mul(ht_h, ht_l, d2h, d2l)
    exh_, exl_ = dd_exp(eh, el)
    return dd_mul(coef_h, coef_l, exh_, exl_)


@njit(parallel=True)
def phi_all(MUH, MUL, exhh, exhl, p_h, p_l, xig_h, xig_l, g_h, g_l,
            tau_h, tau_l, nl_h, nl_l, wtrap, PHIH, PHIL):
    n = exhh.shape[0]
    for j in prange(Gp):
        ph = p_h[j]; pl = p_l[j]
        yh = np.empty(n); yl = np.empty(n)
        yh[0] = nl_h[0]; yl[0] = nl_l[0]
        yh[n-1] = nl_h[1]; yl[n-1] = nl_l[1]
        for i in range(G):
            yh[i+1] = MUH[i, j]; yl[i+1] = MUL[i, j]
        dh = np.empty(n); dl = np.empty(n)
        _pchip_slopes(exhh, exhl, yh, yl, dh, dl)
        dgh = np.empty(G); dgl = np.empty(G)
        for i in range(G):
            dgh[i], dgl[i] = _demand(xig_h[i], xig_l[i], exhh, exhl, yh, yl, dh, dl, ph, pl, g_h, g_l)
        dlo_h, dlo_l = _demand(-1.0+EPSB, 0.0, exhh, exhl, yh, yl, dh, dl, ph, pl, g_h, g_l)
        dhi_h, dhi_l = _demand(1.0-EPSB, 0.0, exhh, exhl, yh, yl, dh, dl, ph, pl, g_h, g_l)
        tot_h, tot_l = dd_div(2.0, 0.0, tau_h, tau_l)
        tp_h, tp_l = dd_mul(2.0, 0.0, D.PIH, D.PIL)
        ct_h, ct_l = dd_div(tau_h, tau_l, tp_h, tp_l)
        coef_h, coef_l = dd_sqrt(ct_h, ct_l)
        f02h = np.empty(G); f02l = np.empty(G); f12h = np.empty(G); f12l = np.empty(G)
        wjh = np.empty(G); wjl = np.empty(G)
        for ip in range(G):
            u2h, u2l = _uofxi(xig_h[ip], xig_l[ip], tot_h, tot_l)
            f02h[ip], f02l[ip] = _fsig(u2h, u2l, -0.5, tau_h, tau_l, coef_h, coef_l)
            f12h[ip], f12l[ip] = _fsig(u2h, u2l, 0.5, tau_h, tau_l, coef_h, coef_l)
            x2h, x2l = dd_mul(xig_h[ip], xig_l[ip], xig_h[ip], xig_l[ip])
            omx_h, omx_l = dd_add(1.0, 0.0, -x2h, -x2l)
            du_h, du_l = dd_div(tot_h, tot_l, omx_h, omx_l)
            wjh[ip], wjl[ip] = dd_mul(wtrap[ip], 0.0, du_h, du_l)
        for i in range(G):
            downh = dgh[i]; downl = dgl[i]
            A0h = 0.0; A0l = 0.0; A1h = 0.0; A1l = 0.0
            for ip in range(G):
                th_, tl_ = dd_add(-downh, -downl, -dgh[ip], -dgl[ip])
                if th_ < dlo_h or th_ > dhi_h:
                    continue
                # bisection in FULL double-double (a,b,m as DD) so the bracket
                # refines to DD precision -> Phi stays sensitive to mu below ~1e-16
                ah = -1.0+EPSB; al = 0.0; bh = 1.0-EPSB; bl = 0.0
                for _ in range(52):
                    sh_, sl_ = dd_add(ah, al, bh, bl)
                    mh_, ml_ = dd_mul(sh_, sl_, 0.5, 0.0)
                    qh, ql = _demand(mh_, ml_, exhh, exhl, yh, yl, dh, dl, ph, pl, g_h, g_l)
                    fmh, fml = dd_add(qh, ql, -th_, -tl_)
                    if fmh > 0.0: bh = mh_; bl = ml_
                    else: ah = mh_; al = ml_
                a = ah; b = bh
                # secant refine in FULL double-double (track x0,x1 as DD pairs so
                # xi_3 reaches DD precision, not float64)
                x0h = ah; x0l = al; x1h = bh; x1l = bl
                q0h, q0l = _demand(x0h, x0l, exhh, exhl, yh, yl, dh, dl, ph, pl, g_h, g_l)
                f0h, f0l = dd_add(q0h, q0l, -th_, -tl_)
                q1h, q1l = _demand(x1h, x1l, exhh, exhl, yh, yl, dh, dl, ph, pl, g_h, g_l)
                f1h, f1l = dd_add(q1h, q1l, -th_, -tl_)
                x3h = x1h; x3l = x1l
                for _ in range(9):
                    dfh, dfl = dd_add(f1h, f1l, -f0h, -f0l)
                    if dfh == 0.0 and dfl == 0.0: break
                    dxh, dxl = dd_add(x1h, x1l, -x0h, -x0l)
                    ph_, pl_ = dd_mul(f1h, f1l, dxh, dxl)
                    sh, sl = dd_div(ph_, pl_, dfh, dfl)
                    nx_h, nx_l = dd_add(x1h, x1l, -sh, -sl)
                    if nx_h < a or nx_h > b: nx_h = 0.5*(a+b); nx_l = 0.0
                    q2h, q2l = _demand(nx_h, nx_l, exhh, exhl, yh, yl, dh, dl, ph, pl, g_h, g_l)
                    f2h, f2l = dd_add(q2h, q2l, -th_, -tl_)
                    x0h, x0l = x1h, x1l; f0h, f0l = f1h, f1l
                    x1h, x1l = nx_h, nx_l; f1h, f1l = f2h, f2l
                    x3h, x3l = nx_h, nx_l
                    if abs(f2h) < 1e-42: break
                u3h, u3l = _uofxi(x3h, x3l, tot_h, tot_l)
                f03h, f03l = _fsig(u3h, u3l, -0.5, tau_h, tau_l, coef_h, coef_l)
                f13h, f13l = _fsig(u3h, u3l, 0.5, tau_h, tau_l, coef_h, coef_l)
                p0h, p0l = dd_mul(f02h[ip], f02l[ip], f03h, f03l)
                c0h, c0l = dd_mul(wjh[ip], wjl[ip], p0h, p0l)
                A0h, A0l = dd_add(A0h, A0l, c0h, c0l)
                p1h, p1l = dd_mul(f12h[ip], f12l[ip], f13h, f13l)
                c1h, c1l = dd_mul(wjh[ip], wjl[ip], p1h, p1l)
                A1h, A1l = dd_add(A1h, A1l, c1h, c1l)
            uih, uil = _uofxi(xig_h[i], xig_l[i], tot_h, tot_l)
            f0ih, f0il = _fsig(uih, uil, -0.5, tau_h, tau_l, coef_h, coef_l)
            f1ih, f1il = _fsig(uih, uil, 0.5, tau_h, tau_l, coef_h, coef_l)
            t0h, t0l = dd_mul(f0ih, f0il, A0h, A0l)
            t1h, t1l = dd_mul(f1ih, f1il, A1h, A1l)
            denh, denl = dd_add(t0h, t0l, t1h, t1l)
            if denh <= 0.0:
                PHIH[i, j], PHIL[i, j] = _pchip_eval(exhh, exhl, yh, yl, dh, dl, xig_h[i], xig_l[i])
            else:
                PHIH[i, j], PHIL[i, j] = dd_div(t1h, t1l, denh, denl)


# ============================ driver ============================
import mpmath as mp
mp.mp.dps = 40

def _split(x):
    hi = float(x); lo = float(x - mp.mpf(hi)); return hi, lo

def _arr_split(vals):
    h = np.array([float(v) for v in vals]); l = np.array([float(v - mp.mpf(float(v))) for v in vals])
    return h, l

_xi = [mp.mpf('-0.88') + mp.mpf('1.76')*k/(G-1) for k in range(G)]
_xiext = [mp.mpf('-0.99')] + _xi + [mp.mpf('0.99')]
xig_h, xig_l = _arr_split(_xi)
exh_h, exh_l = _arr_split(_xiext)   # extended PCHIP knots as DOUBLE-DOUBLE (was float64-only)
_plp = [mp.mpf(-A) + 2*mp.mpf(A)*k/(Gp-1) for k in range(Gp)]
_parr = [1/(1+mp.e**(-l)) for l in _plp]
p_h, p_l = _arr_split(_parr)
g_h, g_l = 100.0, 0.0
wtrap = w_trap_f.copy()

def _tau_consts(tau_f):
    tau = mp.mpf(str(tau_f)); th, tl = _split(tau)
    u_anchor = (2/tau)*mp.atanh(mp.mpf('0.99'))
    nl_lo = 1/(1+mp.e**(tau*u_anchor)); nl_hi = 1/(1+mp.e**(-tau*u_anchor))
    nl_h = np.array([_split(nl_lo)[0], _split(nl_hi)[0]])
    nl_l = np.array([_split(nl_lo)[1], _split(nl_hi)[1]])
    return th, tl, nl_h, nl_l

@njit
def _residual(PHIH, PHIL, MUH, MUL):
    RH = np.empty((G, Gp)); RL = np.empty((G, Gp)); mx = 0.0
    for i in range(G):
        for j in range(Gp):
            rh, rl = dd_add(PHIH[i, j], PHIL[i, j], -MUH[i, j], -MUL[i, j])
            RH[i, j] = rh; RL[i, j] = rl
            a = abs(rh+rl)
            if a > mx: mx = a
    return RH, RL, mx

@njit
def _symmetrize(MUH, MUL):
    OH = np.empty((G, Gp)); OL = np.empty((G, Gp))
    for i in range(G):
        for j in range(Gp):
            brh, brl = dd_add(1.0, 0.0, -MUH[G-1-i, Gp-1-j], -MUL[G-1-i, Gp-1-j])
            sh, sl = dd_add(MUH[i, j], MUL[i, j], brh, brl)
            OH[i, j], OL[i, j] = dd_mul(sh, sl, 0.5, 0.0)
    return OH, OL

def F_full(MUH, MUL, tc):
    th, tl, nl_h, nl_l = tc
    PHIH = np.empty((G, Gp)); PHIL = np.empty((G, Gp))
    phi_all(MUH, MUL, exh_h, exh_l, p_h, p_l, xig_h, xig_l, g_h, g_l, th, tl, nl_h, nl_l, wtrap, PHIH, PHIL)
    return _residual(PHIH, PHIL, MUH, MUL)

from compact_ift import MuField, phi_cell_with_jac, _build_basis_funcs
def _lf(x): return np.log(x/(1.0-x))
def _sf(x): return 1.0/(1.0+np.exp(-x))
def _rl(self):
    self._row_lp = [PchipInterpolator(_lf(np.clip(self.p_grids[i],1e-15,1-1e-15)),
                    _lf(np.clip(self.mu_vals[i],1e-15,1-1e-15)), extrapolate=True) for i in range(self.G)]
    self._row = self._row_lp
def _cap(self, p):
    lp = _lf(float(np.clip(p,1e-15,1-1e-15)))
    return np.array([_sf(float(self._row_lp[i](lp))) for i in range(self.G)])
MuField._rebuild_row_interp = _rl; MuField.col_at_p = _cap; MuField.col_at_p_smooth = _cap

def float_jac(mu_f, tau_f):
    mf = MuField(xi_grid_f, np.tile(p_arr_f, (G, 1)), mu_f.copy(), tau_f)
    xf = np.concatenate([[-0.99], mf.xi_grid, [0.99]]); bf = _build_basis_funcs(xf)
    inv = []; I = np.eye(G)
    for j in range(Gp):
        JB = np.zeros((G, G))
        for i in range(G):
            try: _, jr = phi_cell_with_jac(mf, i, j, gamma_f, tau_f, bf); JB[i,:] = jr
            except Exception: pass
        try: inv.append(np.linalg.inv(I - JB))
        except Exception: inv.append(I.copy())
    return np.array(inv)   # (Gp, G, G)

@njit
def _dd_matstep(MUH, MUL, RH, RL, INV, omega):
    # DD Newton step: dj = INV @ R (matvec + mu-update done in double-double, so
    # corrections are DD-precise -> floor reaches ~DD precision, not float64's ~1e-16).
    NH = MUH.copy(); NL = MUL.copy()
    for j in range(Gp):
        for i in range(G):
            aH = 0.0; aL = 0.0
            for k in range(G):
                pH, pL = dd_mul(INV[j, i, k], 0.0, RH[k, j], RL[k, j])
                aH, aL = dd_add(aH, aL, pH, pL)
            dH, dL = dd_mul(omega, 0.0, aH, aL)
            h, l = dd_add(NH[i, j], NL[i, j], dH, dL)
            if h < CLIP15: h = CLIP15; l = 0.0
            if h > 1.0-CLIP15: h = 1.0-CLIP15; l = 0.0
            NH[i, j] = h; NL[i, j] = l
    OH = np.empty((G, Gp)); OL = np.empty((G, Gp))
    for i in range(G):
        for j in range(Gp):
            brh, brl = dd_add(1.0, 0.0, -NH[G-1-i, Gp-1-j], -NL[G-1-i, Gp-1-j])
            sh, sl = dd_add(NH[i, j], NL[i, j], brh, brl)
            OH[i, j], OL[i, j] = dd_mul(sh, sl, 0.5, 0.0)
    return OH, OL

def _step(MUH, MUL, RH, RL, INV, omega):
    return _dd_matstep(MUH, MUL, RH, RL, INV, omega)

def _est_omega(MUH, MUL, RH, RL, INV, tc, Finf):
    NH, NL = _step(MUH, MUL, RH, RL, INV, 1.0)
    _, _, F1 = F_full(NH, NL, tc)
    rho = F1/Finf if Finf > 0 else 0.5
    return min(max(1.0/(1.0+rho), 0.03), 0.9)

def solve(tau_f, warm_f, tag=""):
    tc = _tau_consts(tau_f)
    MUH = warm_f.astype(np.float64).copy(); MUL = np.zeros((G, Gp))
    INV = float_jac(warm_f, tau_f)
    t0 = time.perf_counter(); last = t0
    RH, RL, Finf = F_full(MUH, MUL, tc)
    hist = [Finf]
    omega = _est_omega(MUH, MUL, RH, RL, INV, tc, Finf)
    for it in range(MAX_ITER):
        now = time.perf_counter()
        print(f"    [{now-t0:6.0f}s] {tag}tau={tau_f:.4f} it{it:3d}: F={Finf:.3e} w={omega:.3f}", flush=True); last = now
        if Finf < TARGET_TOL: break
        # full-Newton during the hard descent: refresh Jacobian at the current mu
        # (the frozen warm-start Jacobian is a poor model in the stiff low-tau regime)
        if Finf > 1e-7 and it % 2 == 1:
            INV = float_jac(MUH.copy(), tau_f); omega = _est_omega(MUH, MUL, RH, RL, INV, tc, Finf)
        NH, NL = _step(MUH, MUL, RH, RL, INV, omega)
        rh2, rl2, F2 = F_full(NH, NL, tc)
        if F2 < Finf:
            MUH, MUL, RH, RL, Finf = NH, NL, rh2, rl2, F2
            if it % 25 == 24: omega = _est_omega(MUH, MUL, RH, RL, INV, tc, Finf)
        else:
            omega *= 0.5
            if omega < 0.05:
                # already deep and stalling -> stop (no futile refresh grind)
                if Finf < 1e-25:
                    break
                INV = float_jac(MUH.copy(), tau_f); omega = _est_omega(MUH, MUL, RH, RL, INV, tc, Finf)
        hist.append(Finf)
        # early-stop: F floored (improved <2x over last 4 iters) and already deep
        if it >= 6 and Finf < 1e-25 and hist[-5] < 2.0*Finf:
            break
    return MUH, MUL, hist, time.perf_counter()-t0

def save_dd(MUH, MUL, tau_f, Finf, iters, wall):
    from save_fp import save_fixed_point
    save_fixed_point(MUH.copy(), xi_grid_f, p_arr_f, gamma=gamma_f, tau=tau_f, A_logit=A,
                     F_final=float(Finf), newton_iters=iters, tol=TARGET_TOL,
                     solver_interp="logit_logit_pchip_DOUBLEDOUBLE", wall_seconds=wall,
                     script="dd_solver.py", subdir="highprec_dd", note=f"double-double numba G={G}")
    txt = f"{HP_DIR}/g100.0_t{tau_f:.4f}_A5.50_G{G}_Gp{Gp}_dd.txt"
    with open(txt, "w") as fh:
        fh.write(f"# double-double tau={tau_f} G={G} F={float(Finf):.4e} iters={iters}\n")
        for i in range(G):
            fh.write(" ".join(f"{MUH[i,j]!r}+{MUL[i,j]!r}" for j in range(Gp))+"\n")
    # full double-double binary (hi+lo, reloadable at 32-digit precision)
    np.savez(f"{HP_DIR}/dd_t{tau_f:.4f}_G{G}.npz",
             mu_hi=MUH, mu_lo=MUL, xi=xi_grid_f, p=p_arr_f,
             tau=tau_f, gamma=gamma_f, F_final=float(Finf), iters=iters)
    return txt

def nearest_fp(tt):
    cands = []
    for f in glob.glob(f"{CACHE}/mu_tau*_A55.npy"):
        m = re.match(r".*mu_tau([\d.]+)_G(\d+)_A55\.npy", f)
        if m: t=float(m.group(1)); Gi=int(m.group(2))
        else:
            m2=re.match(r".*mu_tau([\d.]+)_A55\.npy", f)
            if not m2: continue
            t=float(m2.group(1)); Gi=10
        cands.append((abs(t-tt), 0 if Gi==G else 1, f, t, Gi))
    cands.sort(); return cands[0] if cands else None

def git_push(tau_f, Finf):
    import subprocess
    R="/home/user/FIXED-POINT-FACTORY"
    subprocess.run(["git","add","projects/REZN/solved_fixed_points/highprec_dd/"], cwd=R)
    subprocess.run(["git","commit","-q","-m",f"DD sweep: FP tau={tau_f:.4f} F={float(Finf):.2e} (double-double 32-digit)"], cwd=R)
    for attempt in range(4):
        if subprocess.run(["git","push","-u","origin","claude/study-fixed-point-economics-y12PB"], cwd=R).returncode==0: break
        time.sleep(2*(2**attempt))

def project(mu, Gs):
    if Gs==G: return mu
    xs=np.linspace(-0.88,0.88,Gs); out=np.zeros((G,Gp))
    for j in range(Gp): out[:,j]=np.clip(PchipInterpolator(xs,mu[:,j],extrapolate=True)(xi_grid_f),1e-12,1-1e-12)
    return out

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "refine":
        for tt in [float(x) for x in sys.argv[2:]]:
            _,_,f,ts,Gs = nearest_fp(tt); warm = project(np.load(f), Gs)
            print(f"\n=== refine tau={tt} (warm {ts} G{Gs}) G={G} DD ===", flush=True)
            MUH,MUL,hist,wall = solve(tt, warm)
            ok = hist[-1] < TARGET_TOL
            print(f"  {'OK' if ok else 'x'} tau={tt} F={hist[-1]:.3e} it={len(hist)} {wall:.0f}s", flush=True)
            if hist[-1] < 1e-20: print("   saved", os.path.basename(save_dd(MUH,MUL,tt,hist[-1],len(hist),wall)), flush=True)
    elif mode == "descend":
        ts0, tmin, dt = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        ddnpz = f"{HP_DIR}/dd_t{ts0:.4f}_G{G}.npz"
        if os.path.exists(ddnpz):
            warm = np.load(ddnpz)["mu_hi"]; print(f"=== DESCEND {ts0}->{tmin} step {dt} (warm from saved DD {ts0}) G={G} ===", flush=True)
        else:
            _,_,f,ts,Gs = nearest_fp(ts0); warm = project(np.load(f), Gs)
            print(f"=== DESCEND {ts0}->{tmin} step {dt} (warm {ts} G{Gs}) G={G} DD ===", flush=True)
        tau = ts0
        while tau >= tmin - 1e-9:
            MUH,MUL,hist,wall = solve(tau, warm, tag="[desc] ")
            # retry from nearest float64 FP if continuation stalled
            if hist[-1] >= 1e-12:
                nf = nearest_fp(tau)
                if nf is not None and abs(nf[3]-tau) < 0.08:
                    print(f"  continuation stalled (F={hist[-1]:.2e}); retry warm from float64 tau={nf[3]} G{nf[4]}", flush=True)
                    warm2 = project(np.load(nf[2]), nf[4])
                    M2,L2,h2,w2 = solve(tau, warm2, tag="[retry] ")
                    if h2[-1] < hist[-1]: MUH,MUL,hist,wall = M2,L2,h2,w2
            print(f"  tau={tau:.4f} F={hist[-1]:.3e} it={len(hist)} {wall:.0f}s", flush=True)
            if hist[-1] < 1e-12:
                save_dd(MUH,MUL,tau,hist[-1],len(hist),wall); git_push(tau, hist[-1]); warm = MUH.copy()
            else:
                print(f"  STALL at tau={tau:.4f} (F={hist[-1]:.2e}) -> stopping (structural?)", flush=True); break
            tau -= dt
    print("done", flush=True)
