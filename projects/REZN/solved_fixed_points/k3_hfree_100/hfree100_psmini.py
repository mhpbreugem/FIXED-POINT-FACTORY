import os,sys
os.environ.setdefault("NUMBA_NUM_THREADS","4")
sys.path.insert(0,"/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/k3_hfree_100")
from flint import arb
import hfree_operator_arb as Ha
UMAX=4.0;NQ=40;SUB=4;G=9
def one_cell(dec):
    Ha.set_precision(dec)
    ui=[arb(-4)+arb(i)*arb(1) for i in range(G)]
    Pa=[[[Ha.lam(arb("0.4")*(ui[i]+ui[j]+ui[l])) for l in range(G)] for j in range(G)] for i in range(G)]
    gn,gw=Ha.gauss_legendre(NQ,-UMAX,UMAX)
    twopi=Ha._twopi()
    # just agent0 slice + bayes + clear for cell (3,5,6)
    i,j,l=3,5,6
    p=Pa[i][j][l]
    S0=[[Pa[i][jj][ll] for ll in range(G)] for jj in range(G)]
    A0,A1=Ha.slice_evidence(S0,arb(1),arb(-4),p,gn,gw,arb(2),arb(2),SUB,twopi)
    mu=Ha.bayes(ui[i],arb(2),A0,A1,twopi)
    return A0,A1,mu
import time
res={}
for dec in [110,160,210,260]:
    t=time.time()
    res[dec]=one_cell(dec)
    print(f"dec={dec} done {time.time()-t:.1f}s A0={res[dec][0].str(12)}",flush=True)
ref=res[260]
for dec in [110,160,210]:
    Ha.set_precision(300)
    dA0=abs(arb(res[dec][0].str(120,radius=False))-arb(ref[0].str(120,radius=False)))
    dmu=abs(arb(res[dec][2].str(120,radius=False))-arb(ref[2].str(120,radius=False)))
    print(f"dec={dec}: dA0={float(dA0):.3e}  dmu={float(dmu):.3e}",flush=True)
