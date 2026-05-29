"""G->inf limit of the h-free PR deficit using the ROBUST original solver
(orbit-average residual, which converges) + SELF-CONTINUATION warm-start."""
import os,sys,time,json,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from hfree_fast_solver import solve_hfree_pr, SymReducer3, UMAX
from scipy.interpolate import RegularGridInterpolator
OUT=os.path.join(os.path.dirname(HERE),'k3_hfree_ginf'); os.makedirs(OUT,exist_ok=True)
rows=[]; prevP=None; prevui=None
print("h-free PR deficit vs G (ROBUST orbit-average solver + self-continuation) tau=2 gamma=0.1",flush=True)
for G in [9,13,17,21,25]:
    t=time.time(); x0=None
    if prevP is not None:
        itp=RegularGridInterpolator((prevui,)*3,prevP,bounds_error=False,fill_value=None)
        ug=np.linspace(-UMAX,UMAX,G); X1,X2,X3=np.meshgrid(ug,ug,ug,indexing='ij')
        P0=np.clip(itp(np.stack([X1.ravel(),X2.ravel(),X3.ravel()],1)).reshape(G,G,G),1e-9,1-1e-9)
        red=SymReducer3(G); x0=red.reduce(P0)
    try:
        P,sol,Finf,info=solve_hfree_pr(G,2.0,0.1,x0_red=x0,f_tol=1e-9)
        ui=np.linspace(-UMAX,UMAX,G); prevP,prevui=P,ui
        r=dict(G=G,deficit=info['deficit'],d_FR=info['d_FR'],Finf=float(Finf),iters=info['iters'],conv=info['converged'],sec=round(time.time()-t,1))
        rows.append(r); print(f"  G={G:2d}: deficit={r['deficit']:.4f} d_FR={r['d_FR']:.4f} ||F||={Finf:.1e} it={r['iters']} conv={r['conv']} ({r['sec']}s)",flush=True)
        json.dump({'rows':rows,'solver':'robust orbit-average + self-continuation'},open(OUT+'/ginf_robust.json','w'),indent=2)
    except Exception as e:
        print(f"  G={G}: ERR {type(e).__name__}: {e}",flush=True)
if len(rows)>=3:
    Gs=np.array([r['G'] for r in rows]); D=np.array([r['deficit'] for r in rows])
    for p,lab in [(1,'1/G'),(2,'1/G^2')]:
        A=np.column_stack([np.ones_like(Gs,float),1.0/Gs**p]); c,*_=np.linalg.lstsq(A,D,rcond=None)
        print(f"  extrap deficit(G->inf)[{lab}]={c[0]:.4f}  (kernel co-area limit was ~0.27)",flush=True); rows.append({f'extrap_{lab}':float(c[0])})
    json.dump({'rows':rows},open(OUT+'/ginf_robust.json','w'),indent=2)
print("DONE",flush=True)
