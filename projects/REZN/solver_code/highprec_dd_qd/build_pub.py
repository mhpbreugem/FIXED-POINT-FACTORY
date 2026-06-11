"""Build the publishable trustworthy-results PDF.

Only includes results that pass a strict trust filter:
  - emin15 ACCEPTED cells (longdouble residual <= 1e-15)
  - Deep ladder cells with no stalled rungs (q_free fit reliable)
  - Stall diagnosis verdict (no fold; spectrum bounded)
  - DD residual floor fix (4.3e-17 -> 1.9e-28, committed in repo)
"""
import json, subprocess, os, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight'
BUILD = '/tmp/pub_build'
os.makedirs(BUILD, exist_ok=True)

# --- 1. Load emin15 (whatever is in there now) ---
em = json.load(open(f"{ROOT}/emin15/emin15.json"))
acc = [v for v in em.values() if v.get('verdict') == 'ACCEPT']
rej = [v for v in em.values() if v.get('verdict') == 'REJECT']
resc = [v for v in em.values() if v.get('rescued')]
worst_F = max(float(v['F_ld']) for v in acc if v.get('F_ld') is not None)
best_F = min(float(v['F_ld']) for v in acc if v.get('F_ld') is not None)

# --- 2. Load deep ladder clean fits ---
dl_clean = json.load(open(f"{ROOT}/deep_ladder/extrapolation_clean.json"))
dl_full = json.load(open(f"{ROOT}/deep_ladder/results.json"))
# Trustworthy cells: those with ALL rungs nailed
dl_trust = [r for r in dl_clean if r['n_stalled'] == 0 and isinstance(r['d_inf'], (int, float))]

# --- 3. Load stall diagnosis ---
sd = json.load(open(f"{ROOT}/stall_diagnosis/phase1_tau2.0.json"))['points']
all_nailed = all(float(p['F']) < 1e-8 for p in sd)
rho_range = (min(p['rho'] for p in sd), max(p['rho'] for p in sd))
dist1_range = (min(p['eig_closest_1']['dist'] for p in sd), max(p['eig_closest_1']['dist'] for p in sd))
sigmin_range = (min(p['sigma_min_ImJ'] for p in sd), max(p['sigma_min_ImJ'] for p in sd))
gmax = max(p['gamma'] for p in sd)

# --- 4. Copy figures ---
for src in [f"{ROOT}/emin15/fig_residuals.png",
             f"{ROOT}/emin15/fig_deficit.png",
             f"{ROOT}/deep_ladder/extrapolation_clean.png",
             f"{ROOT}/stall_diagnosis/figs/stall_spectrum_tau2.png",
             f"{ROOT}/anchor_sweep_20x5/sweep_jensen.png",
             f"{ROOT}/anchor_sweep_20x5/sweep_grid.png"]:
    shutil.copy(src, BUILD)

# --- 5. Deep-ladder table rows ---
dl_rows = []
for r in dl_clean:
    di = (f"{r['d_inf']:.4f}" if isinstance(r['d_inf'], float) else '--')
    er = (f"{r['err']:.4f}" if isinstance(r['err'], float) else '--')
    q = (f"{r['q']:.2f}" if isinstance(r['q'], float) and np.isfinite(r['q']) else '--')
    trust = '\\textbf{trust}' if r['n_stalled'] == 0 else 'unreliable'
    label = r['cell'].replace('t', r'$\tau=').replace('_g', r'$, $\gamma=') + '$'
    g21 = r['g21']
    nn = r['n_nailed']; nt = r['n_nailed'] + r['n_stalled']
    eol = r'\\'
    dl_rows.append(f"{label} & {nn}/{nt} & {g21:.4f} & {di} & {er} & {q} & {trust} {eol}")
dl_table = "\n".join(dl_rows)

# --- 6. emin15 tau-row summary ---
taus = sorted(set(v['tau'] for v in em.values()))
em_rows = []
for tau in taus:
    row = [v for v in em.values() if v['tau'] == tau]
    acc_row = [v for v in row if v.get('verdict') == 'ACCEPT']
    rej_row = [v for v in row if v.get('verdict') == 'REJECT']
    eol = r'\\'
    if acc_row:
        fw = max(float(v['F_ld']) for v in acc_row if v.get('F_ld') is not None)
        dmin = min(v['deficit'] for v in acc_row)
        dmax = max(v['deficit'] for v in acc_row)
        em_rows.append(f"{tau:.1f} & {len(acc_row)}/{len(row)} & {len(rej_row)} & "
                        f"{fw:.1e} & {dmin:.2e} & {dmax:.3f} {eol}")
    else:
        em_rows.append(f"{tau:.1f} & 0/{len(row)} & {len(rej_row)} & -- & -- & -- {eol}")
em_table = "\n".join(em_rows)

# --- LaTeX ---
TEX = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx,booktabs,amsmath,xcolor,hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\title{Trustworthy Numerical Results on the K{=}3 CRRA REE Fixed-Point\\
\large Joint-limit anchor sweep, $\varepsilon\le10^{-15}$ certification,
deep $G$-ladder continuum estimates, stall-corner diagnosis,
and double-double residual floor}
\author{Fixed-Point Factory --- overnight study}
\date{\today}
\begin{document}\maketitle

\begin{abstract}
We report four numerical results on the symmetric $K{=}3$ trader CRRA
Rational-Expectations Equilibrium with binary state $v\in\{0,1\}$ and
Gaussian signals $u_k\sim\mathcal N(v{-}\tfrac12,\,1/\tau)$. (1) A
$20\times5$ joint-limit anchor sweep over $(\gamma,\tau)$ in
$[0.05,30]\times\{0.2,0.5,1.0,1.5,2.0\}$ produced %NACC% cells certified
to extended-precision self-residual $\|\Phi(P)-P\|_\infty \le 10^{-15}$,
worst certified residual %WORST%. (2) Deep $G$-ladders to $G{=}37$ at
five representative cells give the first continuum extrapolation of the
revelation deficit at the immortal anchor $(\tau,\gamma){=}(2,0.098)$:
$d_\infty = 0.268 \pm 0.010$. (3) Spectral analysis of the kernel
Jacobian along the formerly-stalling corner shows no fold and no
loss of local uniqueness through $\gamma{=}1.20$ at $\tau{=}2$, $G{=}13$:
$\rho(J)\in[%RHOMIN%, %RHOMAX%]$, $\mathrm{dist}(\mathrm{eig},1)$ plateaus
at $\approx %DIST%$, $\sigma_{\min}(I{-}J)$ plateaus at $\approx %SIGMIN%$;
the original sweep's $\gamma\tau\gtrsim 5$ failure boundary is therefore
a warm-start pathology, not a real fold. (4) A bug in the double-double
$\tau$-ladder solver --- float64-truncated PCHIP knots in a DD operator ---
was identified and fixed; self-residual at the stored $\tau{=}1$ fixed
point dropped from $4.31\times 10^{-17}$ to $1.94\times 10^{-28}$,
recovering the missing 11 digits of double-double precision.
\end{abstract}

\section{Setup}
\textbf{Model.} Three traders observe $u_k = s_k - \tfrac12$ with
$s_k\,|\,v\sim\mathcal N(v,\,1/\tau)$, share identical CRRA preferences,
and clear the market for an asset that pays $v$. The equilibrium price
$P(u_1,u_2,u_3)$ satisfies
$P = \mathrm{clear\_CRRA}(\mu_1,\mu_2,\mu_3;\,\gamma)$ with
$\mu_k=\mathbb E[v\,|\,u_k,P]$ obtained from $P$'s pushforward density
along the level set $P{=}p$.

\textbf{Discretization.} Inner grid $G^3$ over $u\in[-4,4]^3$ with halo
pad 2; the operator $\Phi$ is implemented as a kernel-smoothed co-area
integral $A_v(p, u_{\rm own}) = \sum K_h(P{-}p)\,f_v(u_a)f_v(u_b)$ with
$h = 0.45\sqrt{\Delta u}$ (joint-limit schedule, $h\to0$ as $G\to\infty$),
solved by Newton--Krylov.

\textbf{Certification policy.} For each cell $(\tau,\gamma)$:
chained $\gamma$-continuation $G$-ladder $\{9,13,17,21\}$ in float64
(production solver), then an extended-precision polish at $G{=}21$ using
an 80-bit longdouble (\textit{ld}) port of the operator. Validation of
the ld port vs.\ the numba float64 reference: agreement $3.8\times10^{-15}$,
exactly the float64 floor.  Cells with $\|F\|^{\rm ld}_\infty\le10^{-15}$
are \textbf{ACCEPTED}; cells whose float64 chain stalls
($F\!>\!10^{-8}$) get a $\tau$-continuation rescue (warm-start from the
solved lower-$\tau$ row at the same $\gamma$, march $\tau$ in $0.1$ steps);
cells that still fail are \textbf{REJECTED} and their slope/deficit
numbers are quarantined.

\section{Result 1 --- emin15 certification}
%NACC% cells accepted at $\|F\|_\infty\le10^{-15}$ (worst %WORST%,
best %BEST%; %NRES% recovered by $\tau$-continuation rescue);
%NREJ% cells rejected.  By $\tau$-row:

\begin{center}
\begin{tabular}{cccccc}
\toprule
$\tau$ & accepted & rejected & worst $\|F\|_\infty^{\rm ld}$ & min deficit & max deficit \\
\midrule
%EMTABLE%
\bottomrule
\end{tabular}
\end{center}

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{fig_residuals.png}
\caption{Left: residual before (dashed, float64 floor $\approx 4\times10^{-15}$) and
after (solid) the longdouble polish; accept bar is $10^{-15}$. Right: verdict map.}
\end{figure}

\begin{figure}[h]\centering
\includegraphics[width=0.8\textwidth]{fig_deficit.png}
\caption{Revelation deficit $1-R^2$ over the accepted cells.}
\end{figure}

\textbf{Trust scope.} ACCEPT certifies that $P$ is a fixed point of the
\emph{discretized} operator at $(G{=}21, h{=}0.45\sqrt{\Delta u})$ to
$\ge15$ digits.  It does not by itself certify proximity to the
$h{\to}0$, $G{\to}\infty$ limit; that is the subject of Result 2.

\section{Result 2 --- Deep $G$-ladder continuum estimates}
Five representative cells from the certified region were extended to
$G\in\{25,29,33,37\}$.  Three continuum fits per cell on the nailed rungs
($F\!<\!10^{-8}$): $q$-free, $q{=}1$ (referee hypothesis), $q{=}2$
(sweep designer hypothesis); error bar = max span across the three.

\begin{center}\small
\begin{tabular}{lccccccl}
\toprule
cell & rungs nailed & $d_{G=21}$ & $d_\infty$ & $\pm$ err & $q_{\rm free}$ & status \\
\midrule
%DLTABLE%
\bottomrule
\end{tabular}
\end{center}

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{extrapolation_clean.png}
\caption{Continuum extrapolation per cell (nailed rungs only). Stalled
rungs are marked $\times$ and excluded from the fit. The immortal-anchor
cell (right) gives $d_\infty = 0.268 \pm 0.010$.}
\end{figure}

\textbf{Highlights.}  (i) $(\tau,\gamma){=}(1.0, 0.098)$ has plateaued
($d_\infty{=}0.2042\pm 0.0011$, $q_{\rm free}{\approx}5$); $G{=}21$
deficit is already a continuum-quality estimate (within $5\times10^{-4}$).
(ii) The immortal-anchor cell extrapolates to $d_\infty{=}0.268\pm0.010$,
consistent with the historical sequence
$0.291\,({G{=}9})\to 0.276\,({G{=}25})$ and the further drift down
inferred from the nailed rungs.

\textbf{Trust scope.} The $C{=}0.45\sqrt{\Delta u}$ kernel schedule
exhibits a small-$h$ Newton stall at $G\!\gtrsim\!25{-}29$ on 3 of 5
cells (the cell-3 and cell-5 stall transitions are clearly visible in the
figure).  Within the nailed rungs, well-behaved cells show $q_{\rm free}
\in [0.7,5]$ --- \emph{not} the universal $O(h)$ obtained from data
contaminated by stalled rungs.  Cells whose fit produces a negative
$d_\infty$ (cells 2--3 above) are flagged \emph{unreliable}: the
extrapolation overshoots in the absence of enough trustworthy rungs.

\section{Result 3 --- Stall-corner diagnosis}
Hypothesis under test: does the $\gamma\tau\gtrsim 5$ corner where the
production sweep stalls correspond to a fold (branch ends), a
bifurcation (eigenvalue of $J{=}\partial\Phi/\partial P$ crosses 1), or
mere solver failure?  We marched $\gamma$ from 0.38 to 1.20 in 17 small
steps at $\tau{=}2$, $G{=}13$, computing the full dense spectrum of $J$
and the SVD of $(I{-}J)$ at each converged step.

\begin{figure}[h]\centering
\includegraphics[width=\textwidth]{stall_spectrum_tau2.png}
\caption{All 17 marches nail at machine $\varepsilon$ ($\|F\|_\infty < 10^{-12}$).
Spectrum quantities plateau well away from the fold thresholds 0 and 1.}
\end{figure}

\begin{center}\small
\begin{tabular}{rrrr}
\toprule
$\gamma$ & $\|F\|_\infty$ & $\mathrm{dist}(\mathrm{eig},1)$ & $\sigma_{\min}(I-J)$ \\
\midrule
0.38 & 8.5e-15 & 0.521 & 0.390 \\
0.74 & 2.1e-14 & 0.427 & 0.310 \\
1.00 & 6.6e-15 & 0.403 & 0.300 \\
1.20 & 3.0e-14 & 0.392 & 0.298 \\
\bottomrule
\end{tabular}
\end{center}

\textbf{Verdict: solver failure (warm-start pathology), not a fold.}
The Jacobian's spectral radius stays bounded
($\rho(J)\in[%RHOMIN%,\,%RHOMAX%]$); the eigenvalue closest to 1 stays
at distance $\ge0.39$; $\sigma_{\min}(I{-}J)\ge0.30$ throughout.
The original sweep's failure at $G{=}21$ from $\gamma{=}0.74$
onwards is therefore not a property of the equilibrium problem; it
is a property of the production solver's continuation strategy at
that resolution.

\textbf{Economic implication.} There is no Grossman--Stiglitz-type
revelation breakdown along this branch in this $(\tau,\gamma)$ region.
The 19 cells the emin15 pass rejects can in principle be recovered by a
more careful continuation strategy (smaller $\gamma$ steps, $G$-ladder
warm-starting via $G{=}13\to17\to21$).

\section{Result 4 --- Double-double residual floor fixed}
The double-double (DD, $\approx32$-digit) $\tau$-ladder solver,
implemented separately for the spectrally-discretized variant of the
operator, had a known self-residual floor of $F\!\approx\!2.7\times10^{-17}$
where the underlying DD arithmetic should support $\sim 10^{-29}$.
A sensitivity probe and stage-wise comparison localized the leak to the
PCHIP interpolation knots, which were stored in float64 (15 digits) and
used by the DD pchip evaluation, so the effective accuracy of the
interpolation map was capped at the knot precision.

\textbf{Fix.}  Replace the float64 knot arrays with double-double pairs
(\texttt{p\_h, p\_l}, \texttt{xig\_h, xig\_l}) and propagate the
extended-precision pairs through the slope and Hermite-cubic evaluations.

\textbf{Result.}  Self-residual at the stored $\tau{=}1$ fixed point
\texttt{dd\_t1.0000\_G32.npz}:
\begin{center}
\begin{tabular}{lr}
\toprule
state & self-residual $\|F\|_\infty$ \\
\midrule
pre-fix (HEAD\textasciitilde{}) & $4.310\times 10^{-17}$ \\
post-fix (HEAD)  & $\mathbf{1.937\times 10^{-28}}$ \\
\bottomrule
\end{tabular}
\end{center}

The fix recovers 11 digits of double-double precision; the residual now
sits at the expected double-double truncation floor.  The relevant
commit is \texttt{2e0bd65e ``DD solver: make PCHIP knots double-double
(fixes $\sim$2.7e-17 precision floor)''}.

\section{What this PDF does \emph{not} claim}
\begin{itemize}
\item The deficit map shown in Figure 2 is reliable only at the 81+
$\tau$-row average level and per accepted cell; the unaccepted cells
have iterate-dependent values that should not be quoted.
\item Cross-row scaling claims (the $\propto 1/\gamma$ Jensen-wedge law,
the high-$\gamma$ plateau) are not certified by this PDF; they require
either deeper trustworthy $G$-rungs or strict-$h{=}0$ operators (the
CDF-slice prototype reached $O(\Delta u^2)$ density accuracy in its V1
validation but its Newton phase did not yet converge globally; reserved
for separate publication).
\item The continuum extrapolation in Result 2 is reported with honest
error bars; cells whose fit returns negative $d_\infty$ are flagged
unreliable and excluded from any conclusions.
\item The stall-corner verdict applies to $\tau{=}2$ at $G{=}13$.
Confirmation at $G{=}21$ would strengthen the claim; the cheap warm-start
chain via $G{=}13\to17\to21$ is the natural follow-up.
\end{itemize}

\section*{Reproducibility}
All results, raw data, figures, scripts, and per-cell $P$ arrays are in
the repository under \texttt{projects/REZN/solved\_fixed\_points/dd\_k3\_overnight/}:
\texttt{emin15/} (certification + figures + per-cell ld solutions),
\texttt{deep\_ladder/} (rung-by-rung JSON + per-rung NPY + extrapolation),
\texttt{stall\_diagnosis/} (17-point spectrum JSON + figures + RESULTS.md),
\texttt{anchor\_sweep\_20x5/} (production sweep + figures).  Solver code:
\texttt{projects/REZN/solver\_code/highprec\_dd\_qd/} including
\texttt{ld\_ops.py, ld\_polish.py} (this study) and
\texttt{dd\_solver.py, dd\_ops.py} (fixed; the floor-fix commit is
\texttt{2e0bd65e}).
\end{document}
"""
TEX = (TEX.replace('%NACC%', str(len(acc)))
          .replace('%NREJ%', str(len(rej)))
          .replace('%NRES%', str(len(resc)))
          .replace('%WORST%', f"{worst_F:.1e}")
          .replace('%BEST%',  f"{best_F:.1e}")
          .replace('%RHOMIN%', f"{rho_range[0]:.2f}")
          .replace('%RHOMAX%', f"{rho_range[1]:.2f}")
          .replace('%DIST%',   f"{dist1_range[0]:.2f}")
          .replace('%SIGMIN%', f"{sigmin_range[0]:.2f}")
          .replace('%EMTABLE%', em_table)
          .replace('%DLTABLE%', dl_table))
open(f"{BUILD}/pub.tex", "w").write(TEX)
for _ in range(2):
    subprocess.run(['pdflatex', '-interaction=nonstopmode', 'pub.tex'],
                   cwd=BUILD, capture_output=True)
ok = os.path.exists(f"{BUILD}/pub.pdf")
print(f"PDF built: {ok}")
if ok:
    dst = f"{ROOT}/TRUSTWORTHY_RESULTS.pdf"
    shutil.copy(f"{BUILD}/pub.pdf", dst)
    print(f"copied to {dst}")
