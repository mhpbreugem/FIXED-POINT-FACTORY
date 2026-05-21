import os

P02="(-3.500,-0.584)(-3.146,-0.586)(-2.793,-0.589)(-2.439,-0.590)(-2.086,-0.599)(-1.803,-0.637)(-1.520,-0.706)(-1.243,-0.813)(-1.036,-0.955)(-0.884,-1.123)(-0.754,-1.379)(-0.672,-1.623)(-0.615,-1.944)(-0.592,-2.227)(-0.590,-2.581)(-0.588,-2.934)(-0.585,-3.288)(-0.584,-3.500)"
P03="(-3.500,2.086)(-2.934,1.810)(-2.793,1.463)(-2.414,1.096)(-2.115,0.601)(-1.683,0.247)(-1.291,-0.106)(-0.884,-0.509)(-0.509,-0.884)(-0.106,-1.291)(0.247,-1.683)(0.601,-2.115)(1.096,-2.414)(1.463,-2.793)(1.810,-2.934)(2.086,-3.500)"
P05="(-3.500,2.580)(-3.146,2.155)(-2.722,1.793)(-2.298,1.421)(-1.944,0.990)(-1.567,0.601)(-1.187,0.177)(-0.813,-0.199)(-0.413,-0.601)(-0.035,-0.976)(0.389,-1.359)(0.762,-1.732)(1.191,-2.086)(1.591,-2.462)(1.941,-2.934)(2.298,-3.288)(2.580,-3.500)"
P07="(-3.500,2.917)(-3.076,2.491)(-2.622,2.086)(-2.439,1.713)(-2.015,1.288)(-1.591,0.984)(-1.181,0.530)(-0.742,0.153)(-0.289,-0.247)(0.136,-0.672)(0.515,-1.167)(0.976,-1.520)(1.237,-1.964)(1.662,-2.389)(2.055,-2.652)(2.486,-3.005)(2.864,-3.447)(2.917,-3.500)"
P08="(-1.768,3.500)(-1.752,2.793)(-1.732,2.141)(-1.545,1.591)(-1.237,1.199)(-0.865,0.884)(-0.476,0.530)(-0.110,0.177)(0.234,-0.177)(0.597,-0.530)(0.899,-0.884)(1.237,-1.274)(1.662,-1.587)(2.157,-1.735)(2.864,-1.752)(3.500,-1.768)"

preamble = r"""\documentclass[12pt,border=2mm]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots,fillbetween}
\pgfplotsset{compat=1.18}
\definecolor{BCred}{rgb}{0.7, 0.11, 0.11}
\definecolor{BCblue}{rgb}{0.0, 0.20, 0.42}
\definecolor{BCgreen}{rgb}{0.11, 0.35, 0.02}
\pgfplotsset{every axis/.append style={width=7.5cm, height=7.5cm, scale only axis,
  ticklabel style={font=\small, /pgf/number format/fixed},
  yticklabel style={text width=10mm, align=right},
  label style={font=\normalsize}, title style={font=\normalsize},
  legend style={draw=none, font=\small}}}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=1 by 1}]
"""

axes = r"xmin=-3.5, xmax=3.5, ymin=-3.5, ymax=3.5, xlabel={$u_2$}, ylabel={$u_3$}, axis equal image"

footer = r"""\end{groupplot}
\end{tikzpicture}
\end{document}
"""

T = 0.4  # tension

styles = {}

# 11: Red gradient
styles[11] = rf"""\nextgroupplot[{axes}, title={{11. Red gradient}}]
\addplot[thick,color=BCred!20,smooth,tension={T}] coordinates {{{P02}}};
\addplot[thick,color=BCred!40,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCred!40,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thick,color=BCred!20,smooth,tension={T}] coordinates {{{P08}}};
"""

# 12: Five distinct colors
styles[12] = rf"""\nextgroupplot[{axes}, title={{12. Five colors}}, legend pos=north east]
\addplot[very thick,color=blue!70,smooth,tension={T}] coordinates {{{P02}}};
\addplot[very thick,color=teal,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[very thick,color=orange,smooth,tension={T}] coordinates {{{P07}}};
\addplot[very thick,color=purple!70,smooth,tension={T}] coordinates {{{P08}}};
\legend{{$p\!=\!0.2$,$p\!=\!0.3$,$p\!=\!0.5$,$p\!=\!0.7$,$p\!=\!0.8$}}
"""

# 13: Monochrome weight hierarchy
styles[13] = rf"""\nextgroupplot[{axes}, title={{13. Monochrome weights}}]
\addplot[thin,color=black!35,smooth,tension={T}] coordinates {{{P02}}};
\addplot[semithick,color=black!55,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=black,smooth,tension={T}] coordinates {{{P05}}};
\addplot[semithick,color=black!55,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thin,color=black!35,smooth,tension={T}] coordinates {{{P08}}};
"""

# 14: CARA overlay
styles[14] = rf"""\nextgroupplot[{axes}, title={{14. With CARA reference}}]
\addplot[thin,color=BCblue!25,dashed] coordinates {{(-3.5,0.42)(3.5,-6.58)}};
\addplot[thin,color=BCblue!25,dashed] coordinates {{(-3.5,1.23)(3.5,-5.77)}};
\addplot[thin,color=BCblue!25,dashed] coordinates {{(-3.5,2.50)(3.5,-4.50)}};
\addplot[thin,color=BCblue!25,dashed] coordinates {{(-3.5,3.50)(1.72,-3.50)}};
\addplot[thin,color=BCblue!25,dashed] coordinates {{(-2.73,3.5)(3.5,-2.73)}};
\addplot[thick,color=BCred!25,smooth,tension={T}] coordinates {{{P02}}};
\addplot[thick,color=BCred!45,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCred!45,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thick,color=BCred!25,smooth,tension={T}] coordinates {{{P08}}};
"""

# 15: Cool-to-warm
styles[15] = rf"""\nextgroupplot[{axes}, title={{15. Cool to warm}}]
\addplot[very thick,color=blue!80,smooth,tension={T}] coordinates {{{P02}}};
\addplot[very thick,color=blue!40!red!20,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=black!60,smooth,tension={T}] coordinates {{{P05}}};
\addplot[very thick,color=red!20!blue!40,smooth,tension={T}] coordinates {{{P07}}};
\addplot[very thick,color=red!80,smooth,tension={T}] coordinates {{{P08}}};
"""

# 16: All same color, labeled inline
styles[16] = rf"""\nextgroupplot[{axes}, title={{16. Labeled inline}}]
\addplot[thick,color=BCred!30,smooth,tension={T}] coordinates {{{P02}}};
\addplot[thick,color=BCred!50,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCred!50,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thick,color=BCred!30,smooth,tension={T}] coordinates {{{P08}}};
\node[font=\tiny,color=BCred!40,rotate=-80] at (axis cs:-1.5,-1.0) {{0.2}};
\node[font=\tiny,color=BCred!60,rotate=-48] at (axis cs:-1.4,0.5) {{0.3}};
\node[font=\scriptsize,color=BCred,rotate=-40] at (axis cs:-0.3,0.2) {{0.5}};
\node[font=\tiny,color=BCred!60,rotate=-35] at (axis cs:0.8,0.5) {{0.7}};
\node[font=\tiny,color=BCred!40,rotate=-8] at (axis cs:1.5,1.0) {{0.8}};
"""

# 17: Shaded band between extremes
styles[17] = rf"""\nextgroupplot[{axes}, title={{17. Shaded band}}]
\addplot[thick,color=BCred!20,smooth,tension={T},name path=lo] coordinates {{{P02}}};
\addplot[thick,color=BCred!20,smooth,tension={T},name path=hi] coordinates {{{P08}}};
\addplot[BCred!6] fill between[of=lo and hi];
\addplot[thick,color=BCred!40,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCred!40,smooth,tension={T}] coordinates {{{P07}}};
"""

# 18: Subtle academic
styles[18] = rf"""\nextgroupplot[{axes}, title={{18. Subtle}}]
\addplot[semithick,color=BCred!15,smooth,tension={T}] coordinates {{{P02}}};
\addplot[semithick,color=BCred!28,smooth,tension={T}] coordinates {{{P03}}};
\addplot[thick,color=BCred!75,smooth,tension={T}] coordinates {{{P05}}};
\addplot[semithick,color=BCred!28,smooth,tension={T}] coordinates {{{P07}}};
\addplot[semithick,color=BCred!15,smooth,tension={T}] coordinates {{{P08}}};
"""

# 19: Higher tension (smoother)
styles[19] = rf"""\nextgroupplot[{axes}, title={{19. Extra smooth (tension=0.7)}}]
\addplot[thick,color=BCred!25,smooth,tension=0.7] coordinates {{{P02}}};
\addplot[thick,color=BCred!45,smooth,tension=0.7] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension=0.7] coordinates {{{P05}}};
\addplot[thick,color=BCred!45,smooth,tension=0.7] coordinates {{{P07}}};
\addplot[thick,color=BCred!25,smooth,tension=0.7] coordinates {{{P08}}};
"""

# 20: Blue-based with p=0.5 emphasized
styles[20] = rf"""\nextgroupplot[{axes}, title={{20. Blue, p=0.5 bold}}]
\addplot[thick,color=BCblue!20,smooth,tension={T}] coordinates {{{P02}}};
\addplot[thick,color=BCblue!40,smooth,tension={T}] coordinates {{{P03}}};
\addplot[ultra thick,color=BCblue,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCblue!40,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thick,color=BCblue!20,smooth,tension={T}] coordinates {{{P08}}};
\node[font=\scriptsize,color=BCblue] at (axis cs:-0.3,0.3) {{$p\!=\!0.5$}};
"""

for num, body in styles.items():
    content = preamble + body + footer
    with open(f"style{num}.tex", "w") as f:
        f.write(content)
    print(f"Created style{num}.tex")
