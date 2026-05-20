"""Quad-double (numba, ~64 digits) fixed-point solver + continuation sweep.
Operator matches the float64/DD PCHIP map; QD scalars are stored as the last axis
(size 4) of arrays. Frozen float64 Jacobian + c-estimated under-relaxation.

Usage: MPG=32 python qd_solver.py descend <tau_start> <tau_min> <dtau>
       MPG=32 python qd_solver.py refine <tau> ...
"""
import sys, os, glob, re, time
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solver_code")
sys.path.insert(0, "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points")
sys.path.insert(0, "/tmp")
import numpy as np
from numba import njit, prange
from scipy.interpolate import PchipInterpolator
import qd_ops as Q
from qd_ops import (qd_add, qd_sub, qd_mul, qd_div, qd_mul_d, qd_sqr, qd_sqrt,
                    qd_exp, qd_log, qd_atanh, qd_sigmoid, qd_logit, PI)

CACHE = "/tmp/fp_cache"
HP_DIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/highprec_qd"
os.makedirs(HP_DIR, exist_ok=True)
gamma_f = 100.0
Gp = 17
G = int(os.environ.get("MPG", "32"))
A = 5.5
TARGET_TOL = 1e-40
MAX_ITER = 150
EPSB = 1e-4
CLIP9 = 1e-9; CLIP15 = 1e-15
PI0, PI1, PI2, PI3 = PI

p_lp = np.linspace(-A, A, Gp)
p_arr_f = 1.0/(1.0+np.exp(-p_lp))
xi_grid_f = np.linspace(-0.88, 0.88, G)
dxi = xi_grid_f[1]-xi_grid_f[0]
w_trap_f = np.full(G, dxi); w_trap_f[0] *= 0.5; w_trap_f[-1] *= 0.5


@njit
def _slopes(xh, Y, D):
    n = xh.shape[0]
    for k in range(1, n-1):
        h0 = xh[k]-xh[k-1]; h1 = xh[k+1]-xh[k]
        a0, a1, a2, a3 = qd_sub(Y[k,0],Y[k,1],Y[k,2],Y[k,3], Y[k-1,0],Y[k-1,1],Y[k-1,2],Y[k-1,3])
        d00,d01,d02,d03 = qd_div(a0,a1,a2,a3, h0,0.,0.,0.)
        b0,b1,b2,b3 = qd_sub(Y[k+1,0],Y[k+1,1],Y[k+1,2],Y[k+1,3], Y[k,0],Y[k,1],Y[k,2],Y[k,3])
        d10,d11,d12,d13 = qd_div(b0,b1,b2,b3, h1,0.,0.,0.)
        if (d00==0. and d01==0.) or (d10==0. and d11==0.) or ((d00>0.)!=(d10>0.)):
            D[k,0]=0.;D[k,1]=0.;D[k,2]=0.;D[k,3]=0.
        else:
            w1=2*h1+h0; w2=h1+2*h0
            t10,t11,t12,t13 = qd_div(w1,0.,0.,0., d00,d01,d02,d03)
            t20,t21,t22,t23 = qd_div(w2,0.,0.,0., d10,d11,d12,d13)
            s0,s1,s2,s3 = qd_add(t10,t11,t12,t13, t20,t21,t22,t23)
            D[k,0],D[k,1],D[k,2],D[k,3] = qd_div(w1+w2,0.,0.,0., s0,s1,s2,s3)
    for which in range(2):
        if which==0:
            k=0; h0=xh[1]-xh[0]; h1=xh[2]-xh[1]
            u0,u1,u2,u3 = qd_sub(Y[1,0],Y[1,1],Y[1,2],Y[1,3], Y[0,0],Y[0,1],Y[0,2],Y[0,3])
            m00,m01,m02,m03 = qd_div(u0,u1,u2,u3, h0,0.,0.,0.)
            v0,v1,v2,v3 = qd_sub(Y[2,0],Y[2,1],Y[2,2],Y[2,3], Y[1,0],Y[1,1],Y[1,2],Y[1,3])
            m10,m11,m12,m13 = qd_div(v0,v1,v2,v3, h1,0.,0.,0.)
        else:
            k=n-1; h0=xh[n-1]-xh[n-2]; h1=xh[n-2]-xh[n-3]
            u0,u1,u2,u3 = qd_sub(Y[n-1,0],Y[n-1,1],Y[n-1,2],Y[n-1,3], Y[n-2,0],Y[n-2,1],Y[n-2,2],Y[n-2,3])
            m00,m01,m02,m03 = qd_div(u0,u1,u2,u3, h0,0.,0.,0.)
            v0,v1,v2,v3 = qd_sub(Y[n-2,0],Y[n-2,1],Y[n-2,2],Y[n-2,3], Y[n-3,0],Y[n-3,1],Y[n-3,2],Y[n-3,3])
            m10,m11,m12,m13 = qd_div(v0,v1,v2,v3, h1,0.,0.,0.)
        t0,t1,t2,t3 = qd_mul_d(m00,m01,m02,m03, 2*h0+h1)
        s0,s1,s2,s3 = qd_mul_d(m10,m11,m12,m13, h0)
        e0,e1,e2,e3 = qd_sub(t0,t1,t2,t3, s0,s1,s2,s3)
        e0,e1,e2,e3 = qd_div(e0,e1,e2,e3, h0+h1,0.,0.,0.)
        if (e0>0.)!=(m00>0.):
            D[k,0]=0.;D[k,1]=0.;D[k,2]=0.;D[k,3]=0.
        elif ((m00>0.)!=(m10>0.)) and (abs(e0)>3*abs(m00)):
            D[k,0],D[k,1],D[k,2],D[k,3] = qd_mul_d(m00,m01,m02,m03, 3.0)
        else:
            D[k,0]=e0;D[k,1]=e1;D[k,2]=e2;D[k,3]=e3


@njit
def _peval(xh, Y, D, q0, q1, q2, q3):
    n = xh.shape[0]
    if q0 <= xh[0]: k = 0
    elif q0 >= xh[n-1]: k = n-2
    else:
        k = 0
        for kk in range(n-1):
            if xh[kk] <= q0 and q0 <= xh[kk+1]:
                k = kk; break
    hk = xh[k+1]-xh[k]
    qm0,qm1,qm2,qm3 = qd_sub(q0,q1,q2,q3, xh[k],0.,0.,0.)
    t0,t1,t2,t3 = qd_div(qm0,qm1,qm2,qm3, hk,0.,0.,0.)
    om0,om1,om2,om3 = qd_sub(1.,0.,0.,0., t0,t1,t2,t3)
    o20,o21,o22,o23 = qd_sqr(om0,om1,om2,om3)
    s20,s21,s22,s23 = qd_sqr(t0,t1,t2,t3)
    tt0,tt1,tt2,tt3 = qd_mul_d(t0,t1,t2,t3, 2.0)
    a0,a1,a2,a3 = qd_add(1.,0.,0.,0., tt0,tt1,tt2,tt3)
    h00_0,h00_1,h00_2,h00_3 = qd_mul(a0,a1,a2,a3, o20,o21,o22,o23)
    h10_0,h10_1,h10_2,h10_3 = qd_mul(t0,t1,t2,t3, o20,o21,o22,o23)
    nt0,nt1,nt2,nt3 = qd_mul_d(t0,t1,t2,t3, -2.0)
    b0,b1,b2,b3 = qd_add(3.,0.,0.,0., nt0,nt1,nt2,nt3)
    h01_0,h01_1,h01_2,h01_3 = qd_mul(s20,s21,s22,s23, b0,b1,b2,b3)
    c0,c1,c2,c3 = qd_sub(t0,t1,t2,t3, 1.,0.,0.,0.)
    h11_0,h11_1,h11_2,h11_3 = qd_mul(s20,s21,s22,s23, c0,c1,c2,c3)
    r0,r1,r2,r3 = qd_mul(h00_0,h00_1,h00_2,h00_3, Y[k,0],Y[k,1],Y[k,2],Y[k,3])
    hd0,hd1,hd2,hd3 = qd_mul_d(D[k,0],D[k,1],D[k,2],D[k,3], hk)
    tm0,tm1,tm2,tm3 = qd_mul(h10_0,h10_1,h10_2,h10_3, hd0,hd1,hd2,hd3)
    r0,r1,r2,r3 = qd_add(r0,r1,r2,r3, tm0,tm1,tm2,tm3)
    tm0,tm1,tm2,tm3 = qd_mul(h01_0,h01_1,h01_2,h01_3, Y[k+1,0],Y[k+1,1],Y[k+1,2],Y[k+1,3])
    r0,r1,r2,r3 = qd_add(r0,r1,r2,r3, tm0,tm1,tm2,tm3)
    hd0,hd1,hd2,hd3 = qd_mul_d(D[k+1,0],D[k+1,1],D[k+1,2],D[k+1,3], hk)
    tm0,tm1,tm2,tm3 = qd_mul(h11_0,h11_1,h11_2,h11_3, hd0,hd1,hd2,hd3)
    return qd_add(r0,r1,r2,r3, tm0,tm1,tm2,tm3)


@njit
def _xcrra(m0,m1,m2,m3, p0,p1,p2,p3, g0,g1,g2,g3):
    lm0,lm1,lm2,lm3 = qd_logit(m0,m1,m2,m3)
    lp0,lp1,lp2,lp3 = qd_logit(p0,p1,p2,p3)
    dl0,dl1,dl2,dl3 = qd_sub(lm0,lm1,lm2,lm3, lp0,lp1,lp2,lp3)
    ar0,ar1,ar2,ar3 = qd_div(dl0,dl1,dl2,dl3, g0,g1,g2,g3)
    R0,R1,R2,R3 = qd_exp(ar0,ar1,ar2,ar3)
    nu0,nu1,nu2,nu3 = qd_sub(R0,R1,R2,R3, 1.,0.,0.,0.)
    om0,om1,om2,om3 = qd_sub(1.,0.,0.,0., p0,p1,p2,p3)
    Rp0,Rp1,Rp2,Rp3 = qd_mul(R0,R1,R2,R3, p0,p1,p2,p3)
    de0,de1,de2,de3 = qd_add(om0,om1,om2,om3, Rp0,Rp1,Rp2,Rp3)
    return qd_div(nu0,nu1,nu2,nu3, de0,de1,de2,de3)


@njit
def _demand(q0,q1,q2,q3, xh, Y, D, p0,p1,p2,p3, g0,g1,g2,g3):
    m0,m1,m2,m3 = _peval(xh, Y, D, q0,q1,q2,q3)
    if m0 < CLIP9: m0=CLIP9;m1=0.;m2=0.;m3=0.
    elif m0 > 1.0-CLIP9: m0=1.0-CLIP9;m1=0.;m2=0.;m3=0.
    return _xcrra(m0,m1,m2,m3, p0,p1,p2,p3, g0,g1,g2,g3)


@njit
def _uofxi(x0,x1,x2,x3, to0,to1,to2,to3):
    a0,a1,a2,a3 = qd_atanh(x0,x1,x2,x3)
    return qd_mul(to0,to1,to2,to3, a0,a1,a2,a3)


@njit
def _fsig(u0,u1,u2,u3, mean, t0,t1,t2,t3, c0,c1,c2,c3):
    d0,d1,d2,d3 = qd_sub(u0,u1,u2,u3, mean,0.,0.,0.)
    dd0,dd1,dd2,dd3 = qd_sqr(d0,d1,d2,d3)
    ht0,ht1,ht2,ht3 = qd_mul_d(t0,t1,t2,t3, -0.5)
    e0,e1,e2,e3 = qd_mul(ht0,ht1,ht2,ht3, dd0,dd1,dd2,dd3)
    ex0,ex1,ex2,ex3 = qd_exp(e0,e1,e2,e3)
    return qd_mul(c0,c1,c2,c3, ex0,ex1,ex2,ex3)


@njit(parallel=True)
def phi_all(MU, exh, P, XI, g0,g1,g2,g3, t0,t1,t2,t3, NL, wtrap, PHI):
    n = exh.shape[0]
    for j in prange(Gp):
        p0=P[j,0];p1=P[j,1];p2=P[j,2];p3=P[j,3]
        Y = np.zeros((n,4)); D = np.zeros((n,4))
        Y[0,0]=NL[0,0];Y[0,1]=NL[0,1];Y[0,2]=NL[0,2];Y[0,3]=NL[0,3]
        Y[n-1,0]=NL[1,0];Y[n-1,1]=NL[1,1];Y[n-1,2]=NL[1,2];Y[n-1,3]=NL[1,3]
        for i in range(G):
            Y[i+1,0]=MU[i,j,0];Y[i+1,1]=MU[i,j,1];Y[i+1,2]=MU[i,j,2];Y[i+1,3]=MU[i,j,3]
        _slopes(exh, Y, D)
        DG = np.zeros((G,4))
        for i in range(G):
            DG[i,0],DG[i,1],DG[i,2],DG[i,3] = _demand(XI[i,0],XI[i,1],XI[i,2],XI[i,3], exh,Y,D, p0,p1,p2,p3, g0,g1,g2,g3)
        dlo0,dlo1,dlo2,dlo3 = _demand(-1.0+EPSB,0.,0.,0., exh,Y,D, p0,p1,p2,p3, g0,g1,g2,g3)
        dhi0,dhi1,dhi2,dhi3 = _demand(1.0-EPSB,0.,0.,0., exh,Y,D, p0,p1,p2,p3, g0,g1,g2,g3)
        to0,to1,to2,to3 = qd_div(2.,0.,0.,0., t0,t1,t2,t3)
        tp0,tp1,tp2,tp3 = qd_mul_d(PI0,PI1,PI2,PI3, 2.0)
        ct0,ct1,ct2,ct3 = qd_div(t0,t1,t2,t3, tp0,tp1,tp2,tp3)
        co0,co1,co2,co3 = qd_sqrt(ct0,ct1,ct2,ct3)
        F02=np.zeros((G,4)); F12=np.zeros((G,4)); WJ=np.zeros((G,4))
        for ip in range(G):
            u0,u1,u2,u3 = _uofxi(XI[ip,0],XI[ip,1],XI[ip,2],XI[ip,3], to0,to1,to2,to3)
            F02[ip,0],F02[ip,1],F02[ip,2],F02[ip,3] = _fsig(u0,u1,u2,u3, -0.5, t0,t1,t2,t3, co0,co1,co2,co3)
            F12[ip,0],F12[ip,1],F12[ip,2],F12[ip,3] = _fsig(u0,u1,u2,u3, 0.5, t0,t1,t2,t3, co0,co1,co2,co3)
            x20,x21,x22,x23 = qd_sqr(XI[ip,0],XI[ip,1],XI[ip,2],XI[ip,3])
            ox0,ox1,ox2,ox3 = qd_sub(1.,0.,0.,0., x20,x21,x22,x23)
            du0,du1,du2,du3 = qd_div(to0,to1,to2,to3, ox0,ox1,ox2,ox3)
            WJ[ip,0],WJ[ip,1],WJ[ip,2],WJ[ip,3] = qd_mul_d(du0,du1,du2,du3, wtrap[ip])
        for i in range(G):
            dn0=DG[i,0];dn1=DG[i,1];dn2=DG[i,2];dn3=DG[i,3]
            A00=0.;A01=0.;A02=0.;A03=0.; A10=0.;A11=0.;A12=0.;A13=0.
            for ip in range(G):
                tg0,tg1,tg2,tg3 = qd_add(-dn0,-dn1,-dn2,-dn3, -DG[ip,0],-DG[ip,1],-DG[ip,2],-DG[ip,3])
                if tg0 < dlo0 or tg0 > dhi0: continue
                a=-1.0+EPSB; b=1.0-EPSB
                for _ in range(44):
                    m=0.5*(a+b)
                    q0,q1,q2,q3 = _demand(m,0.,0.,0., exh,Y,D, p0,p1,p2,p3, g0,g1,g2,g3)
                    fm0 = q0 - tg0
                    if fm0 > 0.0: b=m
                    else: a=m
                x0=a; x1=b
                q0,q1,q2,q3 = _demand(x0,0.,0.,0., exh,Y,D, p0,p1,p2,p3, g0,g1,g2,g3)
                f00,f01,f02_,f03 = qd_sub(q0,q1,q2,q3, tg0,tg1,tg2,tg3)
                q0,q1,q2,q3 = _demand(x1,0.,0.,0., exh,Y,D, p0,p1,p2,p3, g0,g1,g2,g3)
                f10,f11,f12_,f13 = qd_sub(q0,q1,q2,q3, tg0,tg1,tg2,tg3)
                x3_0=x1;x3_1=0.;x3_2=0.;x3_3=0.
                for _ in range(9):
                    df0,df1,df2,df3 = qd_sub(f10,f11,f12_,f13, f00,f01,f02_,f03)
                    if df0==0.0: break
                    dx0,dx1,dx2,dx3 = qd_mul_d(f10,f11,f12_,f13, x1-x0)
                    sx0,sx1,sx2,sx3 = qd_div(dx0,dx1,dx2,dx3, df0,df1,df2,df3)
                    nx0,nx1,nx2,nx3 = qd_sub(x1,0.,0.,0., sx0,sx1,sx2,sx3)
                    if nx0 < a or nx0 > b: nx0=0.5*(a+b);nx1=0.;nx2=0.;nx3=0.
                    q0,q1,q2,q3 = _demand(nx0,nx1,nx2,nx3, exh,Y,D, p0,p1,p2,p3, g0,g1,g2,g3)
                    g00,g01,g02,g03 = qd_sub(q0,q1,q2,q3, tg0,tg1,tg2,tg3)
                    x0=x1; f00,f01,f02_,f03 = f10,f11,f12_,f13
                    x1=nx0; f10,f11,f12_,f13 = g00,g01,g02,g03
                    x3_0=nx0;x3_1=nx1;x3_2=nx2;x3_3=nx3
                    if abs(g00) < 1e-62: break
                u30,u31,u32,u33 = _uofxi(x3_0,x3_1,x3_2,x3_3, to0,to1,to2,to3)
                f030,f031,f032,f033 = _fsig(u30,u31,u32,u33, -0.5, t0,t1,t2,t3, co0,co1,co2,co3)
                f130,f131,f132,f133 = _fsig(u30,u31,u32,u33, 0.5, t0,t1,t2,t3, co0,co1,co2,co3)
                pr0,pr1,pr2,pr3 = qd_mul(F02[ip,0],F02[ip,1],F02[ip,2],F02[ip,3], f030,f031,f032,f033)
                cc0,cc1,cc2,cc3 = qd_mul(WJ[ip,0],WJ[ip,1],WJ[ip,2],WJ[ip,3], pr0,pr1,pr2,pr3)
                A00,A01,A02,A03 = qd_add(A00,A01,A02,A03, cc0,cc1,cc2,cc3)
                pr0,pr1,pr2,pr3 = qd_mul(F12[ip,0],F12[ip,1],F12[ip,2],F12[ip,3], f130,f131,f132,f133)
                cc0,cc1,cc2,cc3 = qd_mul(WJ[ip,0],WJ[ip,1],WJ[ip,2],WJ[ip,3], pr0,pr1,pr2,pr3)
                A10,A11,A12,A13 = qd_add(A10,A11,A12,A13, cc0,cc1,cc2,cc3)
            ui0,ui1,ui2,ui3 = _uofxi(XI[i,0],XI[i,1],XI[i,2],XI[i,3], to0,to1,to2,to3)
            f0i0,f0i1,f0i2,f0i3 = _fsig(ui0,ui1,ui2,ui3, -0.5, t0,t1,t2,t3, co0,co1,co2,co3)
            f1i0,f1i1,f1i2,f1i3 = _fsig(ui0,ui1,ui2,ui3, 0.5, t0,t1,t2,t3, co0,co1,co2,co3)
            tt00,tt01,tt02,tt03 = qd_mul(f0i0,f0i1,f0i2,f0i3, A00,A01,A02,A03)
            tt10,tt11,tt12,tt13 = qd_mul(f1i0,f1i1,f1i2,f1i3, A10,A11,A12,A13)
            de0,de1,de2,de3 = qd_add(tt00,tt01,tt02,tt03, tt10,tt11,tt12,tt13)
            if de0 <= 0.0:
                PHI[i,j,0],PHI[i,j,1],PHI[i,j,2],PHI[i,j,3] = _peval(exh,Y,D, XI[i,0],XI[i,1],XI[i,2],XI[i,3])
            else:
                PHI[i,j,0],PHI[i,j,1],PHI[i,j,2],PHI[i,j,3] = qd_div(tt10,tt11,tt12,tt13, de0,de1,de2,de3)


# ============================ driver ============================
import mpmath as mp
mp.mp.dps = 80

def _split4(x):
    x = mp.mpf(x); c = []
    for _ in range(4):
        h = float(x); c.append(h); x = x - mp.mpf(h)
    return c

_xi = [mp.mpf('-0.88') + mp.mpf('1.76')*k/(G-1) for k in range(G)]
_xiext = [mp.mpf('-0.99')] + _xi + [mp.mpf('0.99')]
exh = np.array([float(v) for v in _xiext])
XI = np.array([_split4(v) for v in _xi])
_plp = [mp.mpf(-A) + 2*mp.mpf(A)*k/(Gp-1) for k in range(Gp)]
P = np.array([_split4(1/(1+mp.e**(-l))) for l in _plp])
GAM = _split4(mp.mpf(100))
wtrap = w_trap_f.copy()

def _tau_consts(tau_f):
    tau = mp.mpf(str(tau_f)); T = _split4(tau)
    ua = (2/tau)*mp.atanh(mp.mpf('0.99'))
    nl_lo = 1/(1+mp.e**(tau*ua)); nl_hi = 1/(1+mp.e**(-tau*ua))
    NL = np.array([_split4(nl_lo), _split4(nl_hi)])
    return T, NL

@njit
def _residual(PHI, MU):
    R = np.zeros((G, Gp, 4)); mx = 0.0
    for i in range(G):
        for j in range(Gp):
            r0,r1,r2,r3 = qd_sub(PHI[i,j,0],PHI[i,j,1],PHI[i,j,2],PHI[i,j,3], MU[i,j,0],MU[i,j,1],MU[i,j,2],MU[i,j,3])
            R[i,j,0]=r0;R[i,j,1]=r1;R[i,j,2]=r2;R[i,j,3]=r3
            a = abs(r0+r1)
            if a > mx: mx = a
    return R, mx

@njit
def _symmetrize(MU):
    O = np.zeros((G, Gp, 4))
    for i in range(G):
        for j in range(Gp):
            br0,br1,br2,br3 = qd_sub(1.,0.,0.,0., MU[G-1-i,Gp-1-j,0],MU[G-1-i,Gp-1-j,1],MU[G-1-i,Gp-1-j,2],MU[G-1-i,Gp-1-j,3])
            s0,s1,s2,s3 = qd_add(MU[i,j,0],MU[i,j,1],MU[i,j,2],MU[i,j,3], br0,br1,br2,br3)
            O[i,j,0],O[i,j,1],O[i,j,2],O[i,j,3] = qd_mul_d(s0,s1,s2,s3, 0.5)
    return O

def F_full(MU, tc):
    T, NL = tc
    PHI = np.zeros((G, Gp, 4))
    phi_all(MU, exh, P, XI, GAM[0],GAM[1],GAM[2],GAM[3], T[0],T[1],T[2],T[3], NL, wtrap, PHI)
    return _residual(PHI, MU)

from compact_ift import MuField, phi_cell_with_jac, _build_basis_funcs
def _lf(x): return np.log(x/(1.0-x))
def _sf(x): return 1.0/(1.0+np.exp(-x))
def _rl(self):
    self._row_lp=[PchipInterpolator(_lf(np.clip(self.p_grids[i],1e-15,1-1e-15)),_lf(np.clip(self.mu_vals[i],1e-15,1-1e-15)),extrapolate=True) for i in range(self.G)]
    self._row=self._row_lp
def _cap(self,p):
    lp=_lf(float(np.clip(p,1e-15,1-1e-15)))
    return np.array([_sf(float(self._row_lp[i](lp))) for i in range(self.G)])
MuField._rebuild_row_interp=_rl; MuField.col_at_p=_cap; MuField.col_at_p_smooth=_cap

def float_jac(mu_f, tau_f):
    mf = MuField(xi_grid_f, np.tile(p_arr_f,(G,1)), mu_f.copy(), tau_f)
    xf = np.concatenate([[-0.99], mf.xi_grid, [0.99]]); bf=_build_basis_funcs(xf)
    inv=[]; I=np.eye(G)
    for j in range(Gp):
        JB=np.zeros((G,G))
        for i in range(G):
            try: _,jr=phi_cell_with_jac(mf,i,j,gamma_f,tau_f,bf); JB[i,:]=jr
            except Exception: pass
        try: inv.append(np.linalg.inv(I-JB))
        except Exception: inv.append(I.copy())
    return inv

def _qd_from_f(mu_f):
    MU = np.zeros((G, Gp, 4)); MU[:,:,0] = mu_f
    return MU

def _step(MU, R, inv, omega):
    N = MU.copy()
    for j in range(Gp):
        dj = inv[j] @ R[:, j, 0]
        for i in range(G):
            r0,r1,r2,r3 = qd_add(N[i,j,0],N[i,j,1],N[i,j,2],N[i,j,3], omega*dj[i],0.,0.,0.)
            if r0 < CLIP15: r0=CLIP15;r1=0.;r2=0.;r3=0.
            if r0 > 1-CLIP15: r0=1-CLIP15;r1=0.;r2=0.;r3=0.
            N[i,j,0]=r0;N[i,j,1]=r1;N[i,j,2]=r2;N[i,j,3]=r3
    return _symmetrize(N)

def _est_omega(MU, R, inv, tc, Finf):
    N = _step(MU, R, inv, 1.0)
    _, F1 = F_full(N, tc)
    rho = F1/Finf if Finf > 0 else 0.5
    return min(max(1.0/(1.0+rho), 0.2), 0.85)

def solve(tau_f, warm_f, tag=""):
    tc = _tau_consts(tau_f)
    MU = _qd_from_f(warm_f.astype(np.float64))
    inv = float_jac(warm_f, tau_f)
    t0 = time.perf_counter(); last = t0
    R, Finf = F_full(MU, tc)
    hist = [Finf]
    omega = _est_omega(MU, R, inv, tc, Finf)
    for it in range(MAX_ITER):
        now = time.perf_counter()
        # report after every iteration (each F-eval ~12-15s ~ matches 15s cadence)
        print(f"    [{now-t0:6.0f}s] {tag}tau={tau_f:.4f} it{it:3d}: F={Finf:.3e} w={omega:.3f}", flush=True)
        if Finf < TARGET_TOL: break
        N = _step(MU, R, inv, omega)
        R2, F2 = F_full(N, tc)
        if F2 < Finf:
            MU, R, Finf = N, R2, F2
            if it % 25 == 24: omega = _est_omega(MU, R, inv, tc, Finf)
        else:
            omega *= 0.5
            if omega < 0.05:
                inv = float_jac(MU[:,:,0].copy(), tau_f); omega = _est_omega(MU, R, inv, tc, Finf)
        hist.append(Finf)
    return MU, hist, time.perf_counter()-t0

def save_qd(MU, tau_f, Finf, iters, wall):
    from save_fp import save_fixed_point
    save_fixed_point(MU[:,:,0].copy(), xi_grid_f, p_arr_f, gamma=gamma_f, tau=tau_f, A_logit=A,
                     F_final=float(Finf), newton_iters=iters, tol=TARGET_TOL,
                     solver_interp="logit_logit_pchip_QUADDOUBLE", wall_seconds=wall,
                     script="qd_solver.py", subdir="highprec_qd", note=f"quad-double numba G={G}")
    txt = f"{HP_DIR}/g100.0_t{tau_f:.4f}_A5.50_G{G}_Gp{Gp}_qd.txt"
    with open(txt, "w") as fh:
        fh.write(f"# quad-double tau={tau_f} G={G} F={float(Finf):.4e} iters={iters} (4 comps per value)\n")
        for i in range(G):
            for j in range(Gp):
                fh.write(f"{MU[i,j,0]!r} {MU[i,j,1]!r} {MU[i,j,2]!r} {MU[i,j,3]!r}\n")
    np.save(f"{HP_DIR}/mu_qd_t{tau_f:.4f}_G{G}.npy", MU)
    return txt

def nearest_fp(tt):
    cands=[]
    for f in glob.glob(f"{CACHE}/mu_tau*_A55.npy"):
        m=re.match(r".*mu_tau([\d.]+)_G(\d+)_A55\.npy", f)
        if m: t=float(m.group(1)); Gi=int(m.group(2))
        else:
            m2=re.match(r".*mu_tau([\d.]+)_A55\.npy", f)
            if not m2: continue
            t=float(m2.group(1)); Gi=10
        cands.append((abs(t-tt), 0 if Gi==G else 1, f, t, Gi))
    cands.sort()
    if not cands: return None
    d, _, f, t, Gi = cands[0]
    return (None, None, f, t, Gi)

def project(mu, Gs):
    if Gs==G: return mu
    xs=np.linspace(-0.88,0.88,Gs); out=np.zeros((G,Gp))
    for j in range(Gp): out[:,j]=np.clip(PchipInterpolator(xs,mu[:,j],extrapolate=True)(xi_grid_f),1e-12,1-1e-12)
    return out

def git_push(tau_f, Finf):
    import subprocess
    R="/home/user/FIXED-POINT-FACTORY"
    subprocess.run(["git","add","projects/REZN/solved_fixed_points/highprec_qd/"], cwd=R)
    subprocess.run(["git","commit","-q","-m",f"QD sweep: FP tau={tau_f:.4f} F={float(Finf):.2e} (quad-double 64-digit)"], cwd=R)
    for attempt in range(4):
        r=subprocess.run(["git","push","-u","origin","claude/study-fixed-point-economics-y12PB"], cwd=R)
        if r.returncode==0: break
        time.sleep(2*(2**attempt))

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "refine":
        for tt in [float(x) for x in sys.argv[2:]]:
            _,_,f,ts,Gs = nearest_fp(tt); warm=project(np.load(f),Gs)
            print(f"\n=== refine tau={tt} (warm {ts} G{Gs}) G={G} QD ===", flush=True)
            MU,hist,wall = solve(tt, warm)
            print(f"  {'OK' if hist[-1]<TARGET_TOL else 'x'} tau={tt} F={hist[-1]:.3e} it={len(hist)} {wall:.0f}s", flush=True)
            if hist[-1] < 1e-20: print("   saved", os.path.basename(save_qd(MU,tt,hist[-1],len(hist),wall)), flush=True)
    elif mode == "descend":
        ts0, tmin, dt = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        _,_,f,ts,Gs = nearest_fp(ts0); warm=project(np.load(f),Gs)
        print(f"=== QD DESCEND {ts0}->{tmin} step {dt} (warm {ts} G{Gs}) G={G} ===", flush=True)
        tau = ts0; nok = 0
        while tau >= tmin - 1e-9:
            MU,hist,wall = solve(tau, warm, tag="[desc] ")
            ok = hist[-1] < 1e-25
            print(f"  {'OK' if ok else 'STALL'} tau={tau:.4f} F={hist[-1]:.3e} it={len(hist)} {wall:.0f}s", flush=True)
            if ok:
                save_qd(MU, tau, hist[-1], len(hist), wall); git_push(tau, hist[-1])
                warm = MU[:,:,0].copy(); nok += 1
            else:
                print(f"  STOP: stall at tau={tau:.4f} (F={hist[-1]:.2e}) after {nok} successful pts", flush=True); break
            tau -= dt
    print("done", flush=True)
