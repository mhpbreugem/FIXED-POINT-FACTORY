#!/usr/bin/env python3
"""Restructure main.tex for cleaner narrative flow."""

with open('main.tex', 'r') as f:
    lines = f.readlines()

# Define line ranges (0-indexed) for each block
preamble = lines[0:110]        # lines 1-110: preamble + title page
intro = lines[110:274]         # lines 111-274: Introduction

# Model subsections
model_head = lines[274:296]    # lines 275-296: Model section head
model_info = lines[296:333]    # lines 297-333: Information structure  
model_pref = lines[333:427]    # lines 334-427: Preferences and demands
model_equil = lines[427:472]   # lines 428-472: Equilibrium
model_agg = lines[472:524]     # lines 473-524: Aggregation space → MOVE TO §3

# No-learning subsections
nolearn_head = lines[524:549]  # lines 525-549: No-learning section head
nolearn_cara = lines[549:640]  # lines 550-640: CARA → FR
nolearn_noncara = lines[640:660] # lines 641-660: Non-CARA → PR
nolearn_jensen = lines[660:687] # lines 661-687: Jensen gap
nolearn_smooth = lines[687:721] # lines 688-721: Smooth transition
nolearn_robust = lines[721:759] # lines 722-759: Robustness → MOVE TO §5

# REE subsections
ree_head = lines[759:788]     # lines 760-788: REE section head
ree_fp = lines[788:816]       # lines 789-816: Price-function FP → CONDENSE
ree_contour = lines[816:856]  # lines 817-856: Contour integration → CONDENSE
ree_why = lines[856:901]      # lines 857-901: Why CARA straight (keep)
ree_exist = lines[901:979]    # lines 902-979: Existence + convergence + Prop 5
ree_magnitudes = lines[979:1044] # lines 980-1044: Magnitudes

# Welfare
welfare_head = lines[1044:1051] # lines 1045-1051: Welfare head
welfare_volume = lines[1051:1095] # lines 1052-1095: Trade volume
welfare_value = lines[1095:1123] # lines 1096-1123: Value of info
welfare_gs = lines[1123:1169]  # lines 1124-1169: GS paradox

# Mechanisms → MERGE INTO WELFARE
mechanisms = lines[1169:1268]  # lines 1170-1268

# Discussion
discuss_head = lines[1268:1270] # section head
discuss_cara = lines[1270:1308] # lines 1271-1308: What CARA does → MERGE INTO §3
discuss_lit = lines[1308:1344] # lines 1309-1344: Literature
discuss_vanish = lines[1344:1393] # lines 1345-1393: Vanishing noise
discuss_ext = lines[1393:1408] # lines 1394-1408: Extensions

conclusion = lines[1408:1442] # lines 1409-1442
proofs = lines[1442:1777]     # lines 1443-1777
numerics = lines[1777:]       # lines 1778-end

# =====================================================================
# REASSEMBLE in new order
# =====================================================================

new = []

# Preamble + title page (unchanged)
new.extend(preamble)

# §1 Introduction (unchanged)
new.extend(intro)

# §2 Model (WITHOUT aggregation space)
new.extend(model_head)
new.extend(model_info)
new.extend(model_pref)
new.extend(model_equil)

# §3 The Knife-Edge (aggregation space + no-learning + "what CARA does")
new.append('\n')
new.append('% =====================================================================\n')
new.append('\\section{The Knife-Edge}\\label{sec:knife-edge}\n')
new.append('\n')
new.append('This section establishes the paper\'s central result: CARA is the\n')
new.append('unique preference class that generates full revelation. Any departure\n')
new.append('from CARA---however small---breaks it.\n')
new.append('\n')

# 3.1 The aggregation space (from §2.4)
new.extend(model_agg)

# 3.2 CARA → FR (from §3.1)
new.extend(nolearn_cara)

# 3.3 Non-CARA → PR (from §3.2)  
new.extend(nolearn_noncara)

# 3.4 Jensen gap (from §3.3)
new.extend(nolearn_jensen)

# 3.5 Smooth transition (from §3.4)
new.extend(nolearn_smooth)

# §4 REE — streamlined
new.append('\n')
new.append('% =====================================================================\n')

# REE head (keep the intro paragraphs)
new.extend(ree_head)

# 4.1 The contour mechanism (merge "why CARA straight" with brief FP description)
new.append('\\subsection{The contour mechanism}\\label{sec:contour-mechanism}\n')
new.append('\n')
new.append('The no-learning results of Section~\\ref{sec:knife-edge} assume agents\n')
new.append('ignore prices. In a rational-expectations equilibrium (REE), each agent\n')
new.append('observes the market-clearing price and updates her posterior. The\n')
new.append('question is whether the Jensen gap survives this learning.\n')
new.append('\n')
new.append('The equilibrium is a fixed point of a contour-integration map\n')
new.append('$\\Phi$. Agent~$k$, knowing her own signal $u_k$ and the price $p$,\n')
new.append('extracts the contour $\\{(u_j,u_l): P(u_k,u_j,u_l)=p\\}$ from the\n')
new.append('price function, integrates the signal density along this contour to\n')
new.append('form her posterior, and clears the market. The full numerical\n')
new.append('procedure is described in Appendix~\\ref{app:numerics}.\n')
new.append('\n')

# Keep "why CARA straight" subsection content (this is the key insight)
new.extend(ree_why)

# 4.2 Existence and magnitudes (merge existence + magnitudes)
new.extend(ree_exist)
new.extend(ree_magnitudes)

# §5 Economic Implications (welfare + GS + mechanisms + robustness)
new.append('\n')
new.append('% =====================================================================\n')
new.append('\\section{Economic Implications}\\label{sec:implications}\n')
new.append('\n')
new.append('This section translates the informational result into welfare.\n')
new.append('The distinction between full and partial revelation matters\n')
new.append('precisely because trade volume, the value of information, and the\n')
new.append('resolution of the Grossman--Stiglitz paradox all depend on it.\n')
new.append('\n')

# 5.1 Trade volume
new.extend(welfare_volume)

# 5.2 Value of info
new.extend(welfare_value)

# 5.3 GS paradox
new.extend(welfare_gs)

# 5.4 Sources of larger deficits (mechanisms)
new.extend(mechanisms)

# 5.5 Robustness (from §3.5)
new.extend(nolearn_robust)

# §6 Related Literature and Extensions (compact)
new.append('\n')
new.append('% =====================================================================\n')
new.append('\\section{Related Literature and Extensions}\\label{sec:literature}\n')
new.append('\n')

# Literature connection
new.extend(discuss_lit)

# What CARA does (brief, merged)
new.extend(discuss_cara)

# Vanishing noise
new.extend(discuss_vanish)

# Extensions
new.extend(discuss_ext)

# §7 Conclusion
new.extend(conclusion)

# Appendix A: Proofs
new.extend(proofs)

# Appendix B: Numerics (add the contour methodology here)
new.append('\n')
new.append('% =====================================================================\n')
# Keep existing numerics appendix
new.extend(numerics)

with open('main.tex', 'w') as f:
    f.writelines(new)

print(f"Restructured: {len(new)} lines (was {len(lines)})")
print("Removed: §2.4 aggregation space moved to §3")
print("Removed: §3 no-learning section head (merged into §3 knife-edge)")
print("Removed: §4.1-4.2 FP+contour (condensed to 1 para)")
print("Removed: §5 welfare head (new §5 economic implications)")
print("Removed: §6 mechanisms section (merged into §5)")
print("Removed: §7 discussion head (new §6)")
