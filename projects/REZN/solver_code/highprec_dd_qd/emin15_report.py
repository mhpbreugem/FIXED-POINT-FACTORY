"""Generate figures + LaTeX for the emin15 certification report, then build PDF."""
import json, subprocess, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/emin15'
BUILD = '/tmp/emin15_build'
os.makedirs(BUILD, exist_ok=True)
r = json.load(open(f"{OUT}/emin15.json"))

recs = []
for v in r.values():
    recs.append(dict(tau=float(v['tau']), gamma=float(v['gamma']),
                     F64=float(v['F64']) if v['F64'] not in (None, 'inf') else np.inf,
                     F_ld=(float(v['F_ld']) if v.get('F_ld') not in (None, 'None') else None),
                     slope=float(v.get('slope', np.nan)), deficit=float(v.get('deficit', np.nan)),
                     rescued=bool(v.get('rescued', False)), verdict=v['verdict']))
taus = sorted(set(x['tau'] for x in recs))
gammas = sorted(set(x['gamma'] for x in recs))
n_acc = sum(1 for x in recs if x['verdict'] == 'ACCEPT')
n_res = sum(1 for x in recs if x['rescued'])

# ---------- Figure 1: F_ld waterfall ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cmap = plt.get_cmap('viridis')
for ti, tau in enumerate(taus):
    c = cmap(ti/max(1, len(taus)-1))
    row = sorted([x for x in recs if x['tau'] == tau], key=lambda x: x['gamma'])
    g = [x['gamma'] for x in row]
    f64 = [x['F64'] for x in row]
    fld = [x['F_ld'] if x['F_ld'] is not None else np.nan for x in row]
    axes[0].loglog(g, f64, 'o--', color=c, alpha=0.4, markersize=4)
    axes[0].loglog(g, fld, 'o-', color=c, label=f'$\\tau={tau}$', markersize=5)
for ax in axes[:1]:
    ax.axhline(1e-15, color='r', ls=':', label='accept bar $10^{-15}$')
    ax.axhline(4e-15, color='gray', ls=':', alpha=0.6, label='float64 floor')
axes[0].set_xlabel(r'$\gamma$'); axes[0].set_ylabel(r'$\|F\|_\infty$')
axes[0].set_title('Residual: float64 chain (dashed) vs longdouble polish (solid)')
axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=0.3, which='both')
# verdict map
A = np.full((len(taus), len(gammas)), np.nan)
for x in recs:
    i = taus.index(x['tau']); j = gammas.index(x['gamma'])
    A[i, j] = {'ACCEPT': 2, 'REJECT': 0}[x['verdict']] + (1 if x['rescued'] else 0)
im = axes[1].imshow(A, aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=3,
                    extent=[np.log10(gammas[0]), np.log10(gammas[-1]),
                            -0.5, len(taus)-0.5])
axes[1].set_yticks(range(len(taus))); axes[1].set_yticklabels([f'{t}' for t in taus])
axes[1].set_xlabel(r'$\log_{10}\gamma$'); axes[1].set_ylabel(r'$\tau$')
axes[1].set_title(f'Verdict map: {n_acc}/100 ACCEPT (green; +1 tint = rescued)')
plt.tight_layout(); plt.savefig(f"{BUILD}/fig_residuals.png", dpi=130); plt.close()

# ---------- Figure 2: certified deficit map ----------
fig, ax = plt.subplots(figsize=(8, 5))
for ti, tau in enumerate(taus):
    c = cmap(ti/max(1, len(taus)-1))
    row = sorted([x for x in recs if x['tau'] == tau and x['verdict'] == 'ACCEPT'],
                 key=lambda x: x['gamma'])
    if not row: continue
    ax.loglog([x['gamma'] for x in row], [max(x['deficit'], 1e-9) for x in row],
              'o-', color=c, label=f'$\\tau={tau}$', markersize=5)
gref = np.logspace(-1.3, 1.5, 50)
ax.loglog(gref, 0.04/gref, '--', color='gray', alpha=0.4)
ax.text(0.1, 0.1, r'$\propto 1/\gamma$', color='gray')
ax.set_xlabel(r'$\gamma$'); ax.set_ylabel('revelation deficit $1-R^2$')
ax.set_title(f'Deficit over the {n_acc} ACCEPTED cells (longdouble-certified)')
ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.savefig(f"{BUILD}/fig_deficit.png", dpi=130); plt.close()

# ---------- LaTeX table rows ----------
rows = []
for tau in taus:
    row = sorted([x for x in recs if x['tau'] == tau], key=lambda x: x['gamma'])
    acc = [x for x in row if x['verdict'] == 'ACCEPT']
    res = [x for x in row if x['rescued']]
    fmax = max([x['F_ld'] for x in acc if x['F_ld'] is not None], default=np.nan)
    rows.append(f"{tau:.1f} & {len(acc)}/20 & {len(res)} & "
                f"{(f'{fmax:.1e}' if acc else '--')} & "
                f"{(f'{min(x['deficit'] for x in acc):.2e}' if acc else '--')} & "
                f"{(f'{max(x['deficit'] for x in acc):.3f}' if acc else '--')} \\\\")
table = "\n".join(rows)

stats = dict(n_acc=n_acc, n_res=n_res,
             worst_acc=max([x['F_ld'] for x in recs
                            if x['verdict'] == 'ACCEPT' and x['F_ld'] is not None],
                           default=float('nan')))

tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx,booktabs,amsmath,xcolor,hyperref}
\title{emin15 Certification of the $20\times5$ Joint-Limit Anchor Sweep\\
\large K=3 CRRA REE: accept only $\|F\|_\infty \le 10^{-15}$, strict $h\to0$ schedule}
\author{Fixed-Point Factory --- overnight session}
\date{\today}
\begin{document}\maketitle

\section*{Policy}
A cell $(\tau,\gamma)$ is \textbf{ACCEPTED} only if the self-consistency residual
$\|\Phi(P)-P\|_\infty \le 10^{-15}$ \emph{evaluated in 80-bit extended precision}
(longdouble, $\approx18.9$ digits). The float64 production solver cannot reach this bar:
its CRRA market-clearing bisection breaks at interval width $10^{-14}$ and its
accumulated rounding floors the residual at $\approx4\times10^{-15}$. The bar therefore
forces an extended-precision polish stage. The kernel bandwidth stays on the strict
joint-limit schedule $h = 0.45\sqrt{\Delta u} \to 0$ as the grid refines; no fixed-$h$
result is certified.

\section*{Method}
\begin{enumerate}
\item \textbf{float64 chain} (as in the 20$\times$5 sweep): $G$-ladder $\{9,13,17,21\}$,
$\gamma$-continuation warm start along each $\tau$ row, Newton--Krylov, $f_{tol}=10^{-12}$.
\item \textbf{longdouble polish} at $G=21$: a pure-numpy port of the kernel co-area operator
(Gaussian evidence, Bayes posterior, CRRA clearing by 90-step vectorized bisection,
width $\sim10^{-27}$). Jacobian-free Newton--GMRES(40), finite-difference step $3\times10^{-10}$
(longdouble $\sqrt{\varepsilon}$). The port was validated against the numba float64 operator:
agreement $3.8\times10^{-15}$, exactly the float64 floor.
\item \textbf{$\tau$-continuation rescue} for cells whose float64 chain stalls
($F>10^{-8}$, the $\gamma\tau\gtrsim5$ corner): warm-start from the highest solved
lower-$\tau$ row at the same $\gamma$ and march $\tau$ upward in $0.1$ steps at $G=21$.
Rescued cells re-enter the polish stage; cells that still stall are \textbf{REJECTED} ---
their slope/deficit numbers are quarantined.
\end{enumerate}
One longdouble operator application at $G=21$ costs $2.1$\,s; one Newton step typically
takes the residual from $\sim5\times10^{-14}$ to $\sim3\times10^{-17}$ (quadratic
convergence; the smooth kernel operator is analytic, so Newton behaves).

\section*{Results}
\begin{center}
\begin{tabular}{cccccc}
\toprule
$\tau$ & accepted & rescued & worst $\|F\|_\infty$ & min deficit & max deficit \\
\midrule
%TABLE%
\bottomrule
\end{tabular}
\end{center}

\textbf{%NACC%/100 cells ACCEPTED} at $\|F\|_\infty\le10^{-15}$ (worst accepted residual
%WORST%); %NRES% stalled cells rescued by $\tau$-continuation.

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{fig_residuals.png}
\caption{Left: residual before (dashed, float64 floor $\approx4\times10^{-15}$) and after
(solid) the longdouble polish; the accept bar is $10^{-15}$. Right: verdict map.}
\end{figure}

\begin{figure}[h]\centering
\includegraphics[width=0.75\textwidth]{fig_deficit.png}
\caption{Revelation deficit $1-R^2$ over accepted cells only.}
\end{figure}

\section*{What acceptance does and does not certify}
Acceptance certifies that $P$ is a genuine fixed point of the \emph{discretized}
operator at $(G{=}21,\,h{=}0.45\sqrt{\Delta u})$ to 15+ digits --- it removes solver
error entirely. It does \emph{not} by itself certify proximity to the $h\to0,G\to\infty$
limit: per the Fable~5 referee report on this sweep, the joint-limit schedule converges
no faster than $O(h)$ empirically, and discretization floors (from $1.5\times10^{-6}$ at
$\tau{=}0.2$ to $\sim3\times10^{-2}$ at $\tau{=}2$) bound how small a deficit the $G{=}21$
grid can resolve. The certified cells are exact anchors of their discrete problems;
continuum claims still require the $G$-ladder extrapolation budget quoted there.
\end{document}
"""
tex = tex.replace('%TABLE%', table).replace('%NACC%', str(n_acc))
tex = tex.replace('%NRES%', str(n_res)).replace('%WORST%', f"{stats['worst_acc']:.1e}")
open(f"{BUILD}/emin15.tex", "w").write(tex)
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'emin15.tex'],
               cwd=BUILD, capture_output=True)
out = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'emin15.tex'],
                     cwd=BUILD, capture_output=True)
ok = os.path.exists(f"{BUILD}/emin15.pdf")
print(f"PDF built: {ok}")
if ok:
    subprocess.run(['cp', f"{BUILD}/emin15.pdf", f"{OUT}/emin15_certification.pdf"])
    subprocess.run(['cp', f"{BUILD}/fig_residuals.png", f"{BUILD}/fig_deficit.png", OUT])
    print(f"copied to {OUT}")
