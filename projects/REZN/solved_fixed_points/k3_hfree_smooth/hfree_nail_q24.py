"""Self-consistent Nq=24 nail: residual AND Jacobian use the SAME h-free
operator (Nq=24 Gauss-Legendre) -> EXACT FD Jacobian -> true quadratic Newton
with full steps. NO kernel/bandwidth/smoothing parameter (Nq is a quadrature
resolution that converges as ->inf, not a smoothing knob). 54s/iter at G=9.
Chunked + checkpointed: python3 hfree_nail_q24.py MAXIT
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import sys, json, time
import numpy as np
sys.path.insert(0, "/tmp")
import hfree_operator as H

OUT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth"
UMAX=4.0; TAU=2.0; GAMMA=0.1; NQ=24; SUB=4; G=9
tau=np.full(3,TAU); gam=np.full(3,GAMMA); W=np.full(3,1.0)
gn,gw=H.gauss_legendre(NQ,-UMAX,UMAX)
ui=np.linspace(-UMAX,UMAX,G); N=G**3
LOG=open("/tmp/hfree.log","a")
def log(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); LOG.write(s+"\n"); LOG.flush()
def F(x):
    P=x.reshape((G,G,G)); return (H.phi_hfree(P,ui,gn,gw,tau,gam,W,SUB)-P).ravel()
def metrics(P):
    U1,U2,U3=np.meshgrid(ui,ui,ui,indexing="ij"); T=TAU*(U1+U2+U3)
    Pc=np.clip(P,1e-12,1-1e-12); y=np.log(Pc/(1-Pc)).ravel()
    a=np.polyfit(T.ravel(),y,1); pr=a[0]*T.ravel()+a[1]
    deficit=float(np.sum((y-pr)**2)/np.sum((y-y.mean())**2))
    P_FR=1/(1+np.exp(-T)); dFR=float(np.sqrt(np.mean((P-P_FR)**2)))
    X=np.column_stack([U1.ravel(),U2.ravel(),U3.ravel(),np.ones(U1.size)]); coef,*_=np.linalg.lstsq(X,y,rcond=None)
    return dict(deficit=deficit,d_FR=dFR,slope_T=float(a[0]),b1=float(coef[0]),b2=float(coef[1]),b3=float(coef[2]))

CK="/tmp/P_ckpt_q24.npy"; TR="/tmp/traj_q24.json"
def main(maxit):
    if os.path.exists(CK):
        x=np.load(CK).ravel(); traj=json.load(open(TR))
        log(f"[resume q24] from ||F||~{traj[-1]:.3e} (it {len(traj)-1})")
    else:
        P0=np.load(f"/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_limit/P_inner_G{G}.npy")
        x=P0.ravel(); traj=[float(np.max(np.abs(F(x))))]
        log(f"[start q24] warm kernel-PR -> ||F||={traj[-1]:.3e}")
    Fx=F(x); nrm=float(np.max(np.abs(Fx))); eps=1e-7
    for it in range(maxit):
        t0=time.time(); f0=Fx; J=np.empty((N,N))
        for k in range(N):
            xp=x.copy(); dk=eps*max(1.0,abs(x[k])); xp[k]+=dk
            J[:,k]=(F(xp)-f0)/dk
        try: dx=np.linalg.solve(J,-Fx)
        except np.linalg.LinAlgError: dx,*_=np.linalg.lstsq(J,-Fx,rcond=None)
        alpha=1.0; best=None
        for _ in range(50):
            xn=np.clip(x+alpha*dx,1e-12,1-1e-12); Fn=F(xn); nn=float(np.max(np.abs(Fn)))
            if best is None or nn<best[2]: best=(xn,Fn,nn,alpha)
            if nn<(1-1e-4*alpha)*nrm: break
            alpha*=0.5
            if alpha<1e-13: break
        xn,Fn,nn,alpha=best; prev=nrm; x=xn; Fx=Fn; nrm=nn; traj.append(nrm)
        rate=np.log(nrm)/np.log(prev) if (0<prev<1 and nrm>0) else float("nan")
        log(f"[q24] it={len(traj)-1} ||F||inf={nrm:.6e} alpha={alpha:.3g} jac_t={time.time()-t0:.0f}s rate~{rate:.2f}")
        np.save(CK,x.reshape((G,G,G))); json.dump(traj,open(TR,"w"))
        if nrm<1e-13: break
        if len(traj)>4 and nrm>0.9*prev: log("  floor reached"); break
    m=metrics(x.reshape((G,G,G)))
    log(f"[q24] end ||F||={nrm:.3e} deficit={m['deficit']:.5f} d_FR={m['d_FR']:.4f} slopeT={m['slope_T']:.4f} b1={m['b1']:.3f}")
    np.save(f"{OUT}/P_nailed_G9_q24.npy",x.reshape((G,G,G)))
    return traj,m

if __name__=="__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else 3)
