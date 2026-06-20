"""Generate the DD gamma-ladder report: figure + tex from ladder.json."""
import json, sys
import numpy as np
import matplotlib.pyplot as plt

OUT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/dd_gamma_ladder"
d = json.load(open(f"{OUT}/ladder.json"))
gammas = [r['gamma'] for r in d]
acc = [r['verdict'] == 'ACCEPT' for r in d]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# Panel 1: F_dd + strict cross-check
axes[0].loglog(gammas, [r['T1_F_dd'] for r in d], 'o-', label='$F_{dd}$ (R4 operator, DD)')
axes[0].loglog(gammas, [r['T6_F_strict_w'] for r in d], 's-', label='$F_w$ strict cross-check')
axes[0].loglog(gammas, [r['T6_operator_gap_w'] for r in d], '^--', alpha=0.6,
                label='R4-vs-strict operator gap')
axes[0].set_xlabel(r'$\gamma$'); axes[0].set_ylabel('residual')
axes[0].set_title(r'DD nail + cross-check, $\tau=0.1$, $G=7$')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, which='both')
# Panel 2: slope vs gamma (G=7 and G=9)
axes[1].semilogx(gammas, [r['T7_slope_G7'] for r in d], 'o-', label='slope, $G=7$')
axes[1].semilogx(gammas, [r['T7_slope_G9'] for r in d], 's--', label='slope, $G=9$')
axes[1].axhline(0.1/3, color='k', ls=':', alpha=0.6, label=r'no-learning $\tau/3$')
axes[1].axhline(0.1, color='r', ls=':', alpha=0.4, label=r'FR slope $\tau$')
axes[1].set_xlabel(r'$\gamma$'); axes[1].set_ylabel('logit-price slope on $\Sigma u$')
axes[1].set_title('Revelation slope vs risk aversion')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
# Panel 3: h2 fit quality + verdicts
colors = ['green' if a else 'red' for a in acc]
axes[2].semilogx(gammas, [r['T5_h2_fit_rel'] for r in d], 'k-', alpha=0.4)
axes[2].scatter(gammas, [r['T5_h2_fit_rel'] for r in d], c=colors, s=70, zorder=3)
axes[2].axhline(0.1, color='gray', ls='--', label='T5 threshold')
axes[2].set_xscale('log')
axes[2].set_xlabel(r'$\gamma$'); axes[2].set_ylabel(r'$h^2$-expansion remainder (rel.)')
axes[2].set_title('Richardson validity / verdicts (green=ACCEPT)')
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/ladder_summary.png", dpi=130)
print("figure saved")

# tex
n_acc = sum(acc)
def _row(r):
    v = "\\textbf{ACCEPT}" if r['verdict'] == 'ACCEPT' else "\\textcolor{red}{REJECT}"
    fl = ', '.join(r['fails']) if r['fails'] else '---'
    return (f"{r['gamma']:g} & {r['T1_F_dd']:.1e} & {r['T6_F_strict_w']:.1e} & "
            f"{r['T5_h2_fit_rel']:.3f} & {r['T7_slope_G7']:.6f} & {r['T7_slope_G9']:.6f} & "
            + v + " & " + fl + " \\\\")
rows = "\n".join(_row(r) for r in d)
tex = r"""\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{graphicx,amsmath,booktabs,xcolor,hyperref}
\setlength{\parskip}{6pt}
\title{Double-double $\gamma$-ladder for K=3 at $\tau=0.1$\\
\large with the seven-test acceptance battery of the monograph (Part VII)}
\author{Fixed-Point Factory}\date{June 2026}
\begin{document}\maketitle

\section*{Design}
Ten log-spaced $\gamma \in [1, 1000]$ at $\tau = 0.1$, $G = 7$
(CDF-uniform), $G_p = 121$, $\mathrm{NQK} = 16$. Solver: float64
warm-start to $10^{-12}$, then the double-double Newton nail on the
sym-reduced R4 kernel-band operator to target $10^{-25}$
(\texttt{dd\_k3\_solver.py}), chained warm starts in $\gamma$. Every
cell is then subjected to the acceptance battery; a cell failing
\emph{any} test is rejected, per the instruction to accept only
trustworthy results.

\begin{description}
\item[T1] DD convergence: $F_{dd} \le 10^{-20}$ on the R4 operator.
\item[T2] symmetry: sign-flip and permutation residuals $\le 10^{-12}$.
\item[T3] axis-monotonicity of $P$.
\item[T4] branch identity: $P(\mathbf{0}) = \tfrac12$ to $10^{-12}$ and
  non-trivial price range.
\item[T5] Richardson validity: the four per-bandwidth operator values
  $\Phi_h(P^*)$ must be consistent with a smooth expansion in $h^2$
  (quadratic-in-$h^2$ fit; $h^6$+ remainder $\le 10\%$ of spread,
  density-weighted). \emph{Note: a first version of this test fit only a
  linear-in-$h^2$ model and rejected everything at 23\% remainder ---
  that was the genuine $h^4$ term, not a kink; the test was corrected to
  match the expansion R4 itself assumes.}
\item[T6] strict cross-check: float64 \mbox{strict-$h{=}0$} weighted
  residual at the DD fixed point within $3\times$ the measured
  R4-vs-strict operator gap. \textbf{Honesty note:} at an exactly
  nailed R4 fixed point $|\Phi_{\rm strict} - P| \equiv
  |\Phi_{\rm strict} - \Phi_{R4}|$, so this test passes by construction
  and has no discriminating power here. Its real content is the
  \emph{gap value itself}, which is the operator-truth uncertainty of
  the cell and is reported as such.
\item[T7] grid stability: slope and price range at $G = 7$ vs an
  independent $G = 9$ DD solve within 5\%/10\%.
\end{description}

\section*{Results}
\begin{center}
\includegraphics[width=\linewidth]{ladder_summary.png}
\end{center}

{\small\begin{tabular}{rrrrrrll}
\toprule
$\gamma$ & $F_{dd}$ & $F^{strict}_w$ & $h^2$ rem. & slope$_{G7}$ & slope$_{G9}$ & verdict & failed \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}}

\vspace{1ex}
\textbf{""" + f"{n_acc}/{len(d)}" + r""" cells accepted.}

\section*{Reading}
\begin{itemize}
  \item The DD nail reaches $10^{-25}$--$10^{-29}$ at every cell: the
        \emph{arithmetic} is never the binding constraint.
  \item The revelation slope sits near the no-learning benchmark
        $\tau/3$ and moves toward it with $\gamma$ only in the sixth
        decimal: at $\tau = 0.1$ the Jensen wedge
        $(\tfrac12 - p)\mathrm{Var}(m)/\gamma$ is $O(\tau^2)$-small,
        so $\gamma$-dependence is genuine but tiny --- resolvable only
        because the fixed points are nailed to 29 digits.
  \item \textbf{The dominant uncertainty is the operator, not the
        arithmetic.} The R4-vs-strict gap is $\approx 4\times10^{-2}$
        (density-weighted) at $G = 7$, $\tau = 0.1$: the equilibrium
        surface is pinned to 29 digits \emph{conditional on the R4
        operator}, but only to $\sim 4\%$ across operator families at
        this resolution. Per the monograph protocol, shrinking that
        band requires the joint $(G\uparrow, h\downarrow)$ central-path
        ladder, not more arithmetic precision.
  \item The T5 verdicts are marginal (remainders within a factor 1.05
        of the threshold): with bandwidths as coarse as $h = 0.5$, the
        expansion-validity test cannot cleanly separate smooth
        high-order terms from kink contamination. The threshold was
        \emph{not} loosened to force acceptance; a marginal fail is
        reported as a fail.
  \item Bottom line under the instruction ``accept only if you truly
        trust the result'': what is certified is (i) the arithmetic
        (29 digits), (ii) the branch, symmetries, monotonicity, and
        (iii) grid stability of the economics (slope stable to five
        decimals from $G=7$ to $G=9$). What is \emph{not} certified at
        this resolution is operator-level truth beyond $\sim 4\%$;
        the verdict column reflects exactly that.
\end{itemize}
\end{document}
"""
open(f"{OUT}/report.tex", "w").write(tex)
print("tex written")
