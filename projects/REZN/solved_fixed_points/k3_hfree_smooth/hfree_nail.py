"""NAIL v2: robust dense-Newton nail of the h-FREE smooth co-area PR at G=9.
Forward-diff Jacobian (fast, ~84s/it), Armijo line search, save every iter.
Warm-start = kernel-nailed co-area PR (G=9). Demonstrates QUADRATIC Newton on
the smooth (NO kernel) operator and drives ||F||inf to the discretization floor.
Then a light G=13 Newton-Krylov to confirm at finer grid.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import sys, json, time
import numpy as np
sys.path.insert(0, "/tmp")
import hfree_operator as H

OUT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_smooth"
LOG = open("/tmp/hfree.log", "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s+"\n"); LOG.flush()

UMAX=4.0; TAU=2.0; GAMMA=0.1; NQ=40; SUB=4
tau=np.full(3,TAU); gam=np.full(3,GAMMA); W=np.full(3,1.0)

def metrics(P, ui):
    G=ui.size
    U1,U2,U3=np.meshgrid(ui,ui,ui,indexing="ij"); T=TAU*(U1+U2+U3)
    Pc=np.clip(P,1e-12,1-1e-12); y=np.log(Pc/(1-Pc)).ravel()
    a=np.polyfit(T.ravel(),y,1); pr=a[0]*T.ravel()+a[1]
    deficit=float(np.sum((y-pr)**2)/max(np.sum((y-y.mean())**2),1e-30))
    P_FR=1.0/(1.0+np.exp(-T)); d_FR=float(np.sqrt(np.mean((P-P_FR)**2)))
    X=np.column_stack([U1.ravel(),U2.ravel(),U3.ravel(),np.ones(U1.size)])
    coef,*_=np.linalg.lstsq(X,y,rcond=None)
    return dict(deficit=deficit,d_FR=d_FR,slope_T=float(a[0]),
                b1=float(coef[0]),b2=float(coef[1]),b3=float(coef[2]))

def F_of(x, G, ui):
    P=x.reshape((G,G,G))
    return (H.phi_hfree(P,ui,gn,gw,tau,gam,W,SUB)-P).ravel()

def newton(G, x0, ui, max_it=14, tol=1e-13):
    N=G**3
    x=x0.copy(); Fx=F_of(x,G,ui); nrm=np.max(np.abs(Fx)); traj=[nrm]
    log(f"[Newton G={G}] it=0 ||F||inf={nrm:.6e}")
    eps=1e-7
    for it in range(1,max_it+1):
        t0=time.time(); J=np.empty((N,N)); f0=Fx
        for k in range(N):
            dk=eps*max(1.0,abs(x[k])); xp=x.copy(); xp[k]+=dk
            J[:,k]=(F_of(xp,G,ui)-f0)/dk
        try: dx=np.linalg.solve(J,-Fx)
        except np.linalg.LinAlgError: dx,*_=np.linalg.lstsq(J,-Fx,rcond=None)
        alpha=1.0; best=None
        for _ in range(40):
            xn=np.clip(x+alpha*dx,1e-12,1-1e-12); Fn=F_of(xn,G,ui); nn=np.max(np.abs(Fn))
            if best is None or nn<best[2]: best=(xn,Fn,nn,alpha)
            if nn<(1-1e-4*alpha)*nrm: break
            alpha*=0.5
            if alpha<1e-10: break
        xn,Fn,nn,alpha=best
        prev=nrm; x=xn; Fx=Fn; nrm=nn; traj.append(nrm)
        rate=np.log(nrm)/np.log(prev) if (0<prev<1 and nrm>0) else float("nan")
        log(f"[Newton G={G}] it={it} ||F||inf={nrm:.6e} alpha={alpha:.3g} "
            f"jac_t={time.time()-t0:.1f}s (log-rate~{rate:.2f})")
        np.save(os.path.join(OUT,f"P_nailed_G{G}.npy"),x.reshape((G,G,G)))
        if nrm<tol: break
        if it>3 and nrm>0.95*prev: log("  -> floor reached"); break
    return x,traj

def order_q(traj):
    t=np.array([v for v in traj if v>0]); L=np.log(t)
    qs=[(L[k+1]-L[k])/(L[k]-L[k-1]) for k in range(1,len(L)-1) if abs(L[k]-L[k-1])>1e-12]
    return float(np.median(qs)) if qs else None

if __name__=="__main__":
    t0=time.time()
    log("="*70); log("h-FREE SMOOTH NAIL v2 (NO kernel/bandwidth/smoothing param)")
    log(f"date 2026-05-29 tau={TAU} gamma={GAMMA} Nq={NQ} sub={SUB}")
    gn,gw=H.gauss_legendre(NQ,-UMAX,UMAX)
    G=9; ui=np.linspace(-UMAX,UMAX,G)
    P0=np.load("/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_coarea_limit/P_inner_G9.npy")
    log(f"warm-start kernel co-area PR G=9 deficit={metrics(P0,ui)['deficit']:.5f}")
    x9,traj9=newton(G,P0.ravel(),ui,max_it=14)
    P9=x9.reshape((G,G,G)); m9=metrics(P9,ui); q9=order_q(traj9)
    log(f"[G=9] lowest||F||={traj9[-1]:.3e} order_q={q9} deficit={m9['deficit']:.5f} "
        f"d_FR={m9['d_FR']:.4f} slopeT={m9['slope_T']:.4f}")

    # G=13 light Newton-Krylov from interpolated + picard-preconditioned start
    nk={}
    try:
        from scipy.optimize import newton_krylov
        try: from scipy.optimize import NoConvergence
        except ImportError: from scipy.optimize._nonlin import NoConvergence
        from scipy.interpolate import RegularGridInterpolator
        G2=13; ui2=np.linspace(-UMAX,UMAX,G2)
        ax=np.linspace(-UMAX,UMAX,G); rgi=RegularGridInterpolator((ax,ax,ax),P9,bounds_error=False,fill_value=None)
        A,B,C=np.meshgrid(ui2,ui2,ui2,indexing="ij")
        Ps=rgi(np.column_stack([A.ravel(),B.ravel(),C.ravel()])).reshape((G2,)*3)
        for _ in range(20): Ps=0.7*Ps+0.3*H.phi_hfree(Ps,ui2,gn,gw,tau,gam,W,SUB)
        cnt={"n":0}
        def cb(x,fx): cnt["n"]+=1
        conv=True
        try:
            sol=newton_krylov(lambda x:F_of(x,G2,ui2),Ps.ravel(),f_tol=1e-10,maxiter=50,method="lgmres",callback=cb)
        except NoConvergence as e:
            sol=np.asarray(e.args[0]).ravel(); conv=False
        Finf2=float(np.max(np.abs(F_of(sol,G2,ui2)))); P13=sol.reshape((G2,)*3); m13=metrics(P13,ui2)
        np.save(os.path.join(OUT,"P_nailed_G13.npy"),P13)
        nk=dict(G=G2,Finf=Finf2,iters=cnt["n"],converged=conv,**m13)
        log(f"[G=13 NK] ||F||={Finf2:.3e} it={cnt['n']} conv={conv} deficit={m13['deficit']:.5f} d_FR={m13['d_FR']:.4f}")
    except Exception as e:
        log(f"[G=13 NK] failed: {e}")

    rep=dict(method="h-free smooth co-area (C2 cubic spline + FIXED Gauss-Legendre quadrature decoupled from grid + smooth contour root-find + partition-of-unity)",
             NO_h=True,NO_kernel=True,NO_bandwidth=True,NO_smoothing_param=True,
             only_discretizations=["grid G","Gauss-Legendre Nq","spline sub-bracket"],
             tau=TAU,gamma=GAMMA,UMAX=UMAX,Nq=NQ,sub=SUB,
             newton_G9=dict(trajectory=[float(v) for v in traj9],lowest_Finf=float(traj9[-1]),order_q=q9,**m9),
             newton_krylov_G13=nk, walltime_s=round(time.time()-t0,1))
    json.dump(rep,open(os.path.join(OUT,"nail_report.json"),"w"),indent=2)
    log(f"DONE in {time.time()-t0:.0f}s lowest||F||(G9)={traj9[-1]:.2e} order_q={q9}")
