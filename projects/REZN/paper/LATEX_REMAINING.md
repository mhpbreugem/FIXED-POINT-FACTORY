# REMAINING EDITORIAL TASKS FOR CLAUDE CODE (LATEX)
# The paper (main.tex) is structurally complete: 0 todos, 33 pages.
# These are polish items.

## 1. Proofread the introduction
Read lines 97-260. Check for:
- Consistency with the new results (1-R²≈0.10, not 0.03)
- References to figures by correct label
- Flow and readability

## 2. Check all figure/table cross-references
Compile and verify no "??" appear in the PDF.
All \ref{} should resolve.

## 3. Consider moving convergence figure to appendix
Fig 5 (fig5_convergence) shows Picard vs Anderson convergence paths.
This is technical detail. Consider moving to Appendix B and referencing
from main text with one sentence.

## 4. Check figure captions for consistency
All gray-background figures should say "Gray background: projected
values pending final computation." Verify this is consistent.

## 5. Final compile check
Run: pdflatex + bibtex + pdflatex + pdflatex
Verify: 0 warnings, 0 undefined references, all citations resolve.
Push: main.tex + main.pdf

## 6. Abstract review
Does the abstract mention the key numbers? It should state:
- CARA is knife-edge
- 1-R² = 0.10 at γ=0.5 (or "of order 10%")
- Machine-precision convergence
- No noise traders
