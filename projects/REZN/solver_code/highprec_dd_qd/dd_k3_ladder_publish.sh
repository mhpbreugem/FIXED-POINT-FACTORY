#!/bin/bash
set -e
cd /tmp
# Update fig script paths to read from persistent location
sed -i 's|/tmp/dd_k3_ladder_full.json|/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder/ladder.json|g' /tmp/dd_k3_ladder_figs.py 2>/dev/null
sed -i 's|/tmp/dd_k3_ladder_full_fps|/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder/fps|g' /tmp/dd_k3_ladder_figs.py 2>/dev/null
python -u dd_k3_ladder_figs.py
cat > /tmp/dd_k3_ladder_report.tex << 'TEX'
\documentclass[11pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{graphicx,amsmath,amssymb,booktabs,hyperref,xcolor}
\graphicspath{{/tmp/dd_k3_ladder_figs/}}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\setlength{\parskip}{6pt}
\title{DD K=3 $\tau$ ladder: 0.001 $\to$ 1.0, step 0.001}
\date{June 2026}
\begin{document}
\maketitle
\section*{Summary}
1000-cell ladder at $\gamma{=}100$, $G{=}7$, Lin-CDF R4 kernel-band in
double-double precision. Chain warm-start, each cell solved to $|F|<10^{-25}$.
\begin{center}\includegraphics[width=\linewidth]{fig1_F_floor.png}\end{center}
\begin{center}\includegraphics[width=\linewidth]{fig2_slope_deficit.png}\end{center}
\begin{center}\includegraphics[width=\linewidth]{fig3_wall.png}\end{center}
\end{document}
TEX
pdflatex -interaction=nonstopmode /tmp/dd_k3_ladder_report.tex > /tmp/pdflog 2>&1 || true
pdflatex -interaction=nonstopmode /tmp/dd_k3_ladder_report.tex > /tmp/pdflog 2>&1 || true
REPO=/home/user/FIXED-POINT-FACTORY
DEST=$REPO/projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder
SRC=$REPO/projects/REZN/solver_code/highprec_dd_qd
mkdir -p $DEST/figs
cp /tmp/dd_k3_ladder_full.py $SRC/ 2>/dev/null
cp /tmp/dd_k3_ladder_figs.py $SRC/ 2>/dev/null
cp /tmp/dd_k3_ladder_report.tex $SRC/ 2>/dev/null
cp /tmp/dd_k3_ladder_report.pdf $DEST/ 2>/dev/null
cp /tmp/dd_k3_ladder_figs/*.png $DEST/figs/ 2>/dev/null
cd $REPO
git add projects/REZN/solver_code/highprec_dd_qd/dd_k3_ladder_full.py \
        projects/REZN/solver_code/highprec_dd_qd/dd_k3_ladder_figs.py \
        projects/REZN/solver_code/highprec_dd_qd/dd_k3_ladder_report.tex \
        projects/REZN/solved_fixed_points/dd_k3_overnight/full_ladder/ 2>/dev/null
git -c user.email=mhpbreugem@gmail.com -c user.name="Fixed-Point Factory" \
    commit -q -m "DD K=3 tau ladder 0.001..1.0 step 0.001 (1000 cells)" 2>/dev/null || true
git pull --rebase origin claude/study-fixed-point-economics-y12PB 2>&1 | tail -2
for a in 0 1 2 3; do
  if git push -u origin claude/study-fixed-point-economics-y12PB; then break; fi
  sleep $((2 ** (a+1)))
done
echo "PUBLISHED"
