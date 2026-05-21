import json, glob, math, numpy as np, csv, time
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FIG='/home/user/FIXED-POINT-FACTORY/projects/REZN/figures'
ML=open('/tmp/master.log','a')
def log(m): ML.write(f"{time.strftime('%H:%M:%S')} {m}\n"); ML.flush()

def onemR2(P,u,tau):
    G=len(u)
    f0=np.sqrt(tau/(2*np.pi))*np.exp(-tau/2*(u+0.5)**2); f1=np.sqrt(tau/(2*np.pi))*np.exp(-tau/2*(u-0.5)**2)
    T=[];LP=[];W=[]
    for i in range(G):
     for j in range(G):
      for k in range(G):
        p=min(max(float(P[i,j,k]),1e-12),1-1e-12); T.append(tau*(u[i]+u[j]+u[k])); LP.append(math.log(p/(1-p))); W.append(0.5*(f0[i]*f0[j]*f0[k]+f1[i]*f1[j]*f1[k]))
    T=np.array(T);LP=np.array(LP);W=np.array(W); sl,ic=np.polyfit(T,LP,1,w=np.sqrt(W)); pred=sl*T+ic
    m=np.average(LP,weights=W); return float(np.average((LP-pred)**2,weights=W)/np.average((LP-m)**2,weights=W)), float(sl)

rows=[]
for fn in sorted(glob.glob('/tmp/ckpts/*.npz')):
    try:
        d=np.load(fn,allow_pickle=True)
        if 'P_inner' not in d: continue
        g=round(float(d['gamma_vec'][0]),3); tau=round(float(d['tau_vec'][0]),3)
        r,sl=onemR2(d['P_inner'],d['u_grid_inner'],tau); rows.append((g,tau,r,sl,len(d['u_grid_inner'])))
    except Exception as e: log(f"  ckpt {fn} ERR {e}")
rows.sort()
with open(f'{FIG}/deficit_metrics.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['gamma','tau','one_minus_R2','slope','G'])
    for r in rows: w.writerow([r[0],r[1],f"{r[2]:.6f}",f"{r[3]:.6f}",r[4]])
log(f"PHASE2: 1-R2 computed for {len(rows)} checkpoints -> deficit_metrics.csv")

GAMMA_COLORS={0.1:(0.55,0,0),0.13:(0.7,0.3,0.1),0.25:(0.7,0.11,0.11),0.35:(0.72,0.53,0.04),0.5:(0,0,0),0.75:(0.3,0.3,0.3),1.0:(0.11,0.35,0.02)}
# vs tau per gamma
by_g=defaultdict(list)
for g,t,r,sl,G in rows: by_g[g].append((t,r))
fig,ax=plt.subplots(figsize=(8,5.2),dpi=140)
for g in sorted(by_g):
    p=sorted(by_g[g]);
    if len(p)<1: continue
    a=np.array(p); ax.plot(a[:,0],a[:,1],'o-',color=GAMMA_COLORS.get(g,(0.5,0.5,0.5)),ms=5,lw=1.6,label=f'γ = {g:g}')
ax.set_xlabel(r'$\tau$',fontsize=11); ax.set_ylabel(r'$1-R^2$  (price-informativeness deficit)',fontsize=11)
ax.set_title('Revelation deficit $1-R^2$ vs $\\tau$  (production REE checkpoints, K=3)',fontsize=12)
ax.grid(True,ls=':',lw=0.4,alpha=0.6); ax.legend(title='risk aversion',fontsize=9,ncol=2); ax.set_ylim(bottom=0)
plt.tight_layout(); plt.savefig(f'{FIG}/fig6_deficit_vs_tau.png',dpi=145); plt.close()
log("PHASE2: wrote fig6_deficit_vs_tau.png")
# vs gamma per tau
by_t=defaultdict(list)
for g,t,r,sl,G in rows: by_t[t].append((g,r))
fig,ax=plt.subplots(figsize=(8,5.2),dpi=140)
for t in sorted(by_t):
    p=sorted(by_t[t])
    if len(p)<2: continue
    a=np.array(p); ax.plot(a[:,0],a[:,1],'o-',ms=5,lw=1.5,label=f'τ = {t:g}')
ax.set_xlabel(r'$\gamma$',fontsize=11); ax.set_ylabel(r'$1-R^2$',fontsize=11); ax.set_xscale('log')
ax.set_title('Revelation deficit $1-R^2$ vs $\\gamma$',fontsize=12)
ax.grid(True,ls=':',lw=0.4,alpha=0.6,which='both'); ax.legend(fontsize=9,ncol=2); ax.set_ylim(bottom=0)
plt.tight_layout(); plt.savefig(f'{FIG}/fig6_deficit_vs_gamma.png',dpi=145); plt.close()
log("PHASE2: wrote fig6_deficit_vs_gamma.png")
print("done phase2",flush=True)
