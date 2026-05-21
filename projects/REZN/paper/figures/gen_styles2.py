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
T = 0.4
styles = {}

# 21: Thick p=0.5 only, rest very faint
styles[21] = rf"""\nextgroupplot[{axes}, title={{21. Focus on $p\!=\!0.5$}}]
\addplot[semithick,color=BCred!12,smooth,tension={T}] coordinates {{{P02}}};
\addplot[semithick,color=BCred!12,smooth,tension={T}] coordinates {{{P03}}};
\addplot[ultra thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[semithick,color=BCred!12,smooth,tension={T}] coordinates {{{P07}}};
\addplot[semithick,color=BCred!12,smooth,tension={T}] coordinates {{{P08}}};
\node[font=\small,color=BCred] at (axis cs:-0.5,0.5) {{$p\!=\!0.5$}};
"""

# 22: Warm earth tones
styles[22] = rf"""\nextgroupplot[{axes}, title={{22. Earth tones}}]
\addplot[very thick,color={{rgb,255:red,139;green,90;blue,43}},smooth,tension={T}] coordinates {{{P02}}};
\addplot[very thick,color={{rgb,255:red,180;green,120;blue,60}},smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color={{rgb,255:red,120;green,60;blue,30}},smooth,tension={T}] coordinates {{{P05}}};
\addplot[very thick,color={{rgb,255:red,180;green,120;blue,60}},smooth,tension={T}] coordinates {{{P07}}};
\addplot[very thick,color={{rgb,255:red,139;green,90;blue,43}},smooth,tension={T}] coordinates {{{P08}}};
"""

# 23: Gray background, white contours
styles[23] = rf"""\nextgroupplot[{axes}, title={{23. Gray field}}, axis background/.style={{fill=black!8}}]
\addplot[thick,color=BCred!30,smooth,tension={T}] coordinates {{{P02}}};
\addplot[thick,color=BCred!50,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCred!50,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thick,color=BCred!30,smooth,tension={T}] coordinates {{{P08}}};
"""

# 24: Double band fill (low + high regions)
styles[24] = rf"""\nextgroupplot[{axes}, title={{24. Double shaded bands}}]
\addplot[draw=none,smooth,tension={T},name path=p02] coordinates {{{P02}}};
\addplot[draw=none,smooth,tension={T},name path=p03] coordinates {{{P03}}};
\addplot[draw=none,smooth,tension={T},name path=p07] coordinates {{{P07}}};
\addplot[draw=none,smooth,tension={T},name path=p08] coordinates {{{P08}}};
\addplot[BCred!5] fill between[of=p02 and p03];
\addplot[BCred!5] fill between[of=p07 and p08];
\addplot[thick,color=BCred!30,smooth,tension={T}] coordinates {{{P02}}};
\addplot[thick,color=BCred!45,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCred!45,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thick,color=BCred!30,smooth,tension={T}] coordinates {{{P08}}};
"""

# 25: Dashed outer, solid inner
styles[25] = rf"""\nextgroupplot[{axes}, title={{25. Dashed outer, solid inner}}]
\addplot[thick,color=BCred!35,dashed,smooth,tension={T}] coordinates {{{P02}}};
\addplot[thick,color=BCred!60,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCred!60,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thick,color=BCred!35,dashed,smooth,tension={T}] coordinates {{{P08}}};
"""

# 26: Teal/navy academic
styles[26] = rf"""\nextgroupplot[{axes}, title={{26. Navy academic}}]
\addplot[thick,color=BCblue!25,smooth,tension={T}] coordinates {{{P02}}};
\addplot[thick,color=BCblue!45,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCblue,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCblue!45,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thick,color=BCblue!25,smooth,tension={T}] coordinates {{{P08}}};
\node[font=\scriptsize,color=BCblue!40] at (axis cs:-2.0,-1.3) {{$p\!=\!0.2$}};
\node[font=\scriptsize,color=BCblue] at (axis cs:-0.3,0.3) {{$p\!=\!0.5$}};
\node[font=\scriptsize,color=BCblue!40] at (axis cs:2.0,1.3) {{$p\!=\!0.8$}};
"""

# 27: Extra smooth tension=0.8 + gradient
styles[27] = rf"""\nextgroupplot[{axes}, title={{27. Ultra smooth (t=0.8)}}]
\addplot[thick,color=BCred!20,smooth,tension=0.8] coordinates {{{P02}}};
\addplot[thick,color=BCred!40,smooth,tension=0.8] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension=0.8] coordinates {{{P05}}};
\addplot[thick,color=BCred!40,smooth,tension=0.8] coordinates {{{P07}}};
\addplot[thick,color=BCred!20,smooth,tension=0.8] coordinates {{{P08}}};
"""

# 28: Markers on all curves, no fill
styles[28] = rf"""\nextgroupplot[{axes}, title={{28. Markers everywhere}}]
\addplot[thick,color=BCred!30,smooth,tension={T},mark=o,mark size=1pt,mark repeat=3] coordinates {{{P02}}};
\addplot[thick,color=BCred!50,smooth,tension={T},mark=square,mark size=1pt,mark repeat=2] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T},mark=*,mark size=1.2pt,mark repeat=2] coordinates {{{P05}}};
\addplot[thick,color=BCred!50,smooth,tension={T},mark=triangle,mark size=1.2pt,mark repeat=2] coordinates {{{P07}}};
\addplot[thick,color=BCred!30,smooth,tension={T},mark=diamond,mark size=1pt,mark repeat=3] coordinates {{{P08}}};
"""

# 29: Mixed red/blue for symmetric pairs
styles[29] = rf"""\nextgroupplot[{axes}, title={{29. Symmetric color pairs}}]
\addplot[very thick,color=BCblue!50,smooth,tension={T}] coordinates {{{P02}}};
\addplot[very thick,color=BCblue!80,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=black,smooth,tension={T}] coordinates {{{P05}}};
\addplot[very thick,color=BCred!80,smooth,tension={T}] coordinates {{{P07}}};
\addplot[very thick,color=BCred!50,smooth,tension={T}] coordinates {{{P08}}};
"""

# 30: Clean gradient + CARA + labels
styles[30] = rf"""\nextgroupplot[{axes}, title={{30. Complete: gradient + CARA + labels}}]
\addplot[thin,color=black!15,dashed] coordinates {{(-3.5,0.42)(3.5,-6.58)}};
\addplot[thin,color=black!15,dashed] coordinates {{(-3.5,1.23)(3.5,-5.77)}};
\addplot[thin,color=black!15,dashed] coordinates {{(-3.5,2.50)(3.5,-4.50)}};
\addplot[thin,color=black!15,dashed] coordinates {{(-3.5,3.50)(1.72,-3.50)}};
\addplot[thin,color=black!15,dashed] coordinates {{(-2.73,3.5)(3.5,-2.73)}};
\addplot[thick,color=BCred!20,smooth,tension={T}] coordinates {{{P02}}};
\addplot[thick,color=BCred!40,smooth,tension={T}] coordinates {{{P03}}};
\addplot[very thick,color=BCred,smooth,tension={T}] coordinates {{{P05}}};
\addplot[thick,color=BCred!40,smooth,tension={T}] coordinates {{{P07}}};
\addplot[thick,color=BCred!20,smooth,tension={T}] coordinates {{{P08}}};
\node[font=\tiny,color=BCred!35] at (axis cs:-2.2,-1.3) {{0.2}};
\node[font=\tiny,color=BCred!55] at (axis cs:-1.5,0.3) {{0.3}};
\node[font=\scriptsize,color=BCred] at (axis cs:-0.3,0.3) {{0.5}};
\node[font=\tiny,color=BCred!55] at (axis cs:0.8,0.5) {{0.7}};
\node[font=\tiny,color=BCred!35] at (axis cs:1.8,1.3) {{0.8}};
"""

for num, body in styles.items():
    content = preamble + body + footer
    with open(f"style{num}.tex", "w") as f:
        f.write(content)
    print(f"Created style{num}.tex")
