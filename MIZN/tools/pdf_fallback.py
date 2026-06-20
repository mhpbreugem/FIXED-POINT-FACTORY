"""Matplotlib-PDF fallback for environments without pdflatex.

Produces a 2-3 page PDF with summary + metrics + the same figures the
LaTeX report would embed.  Called from runner if pdflatex is missing or
fails.
"""
from __future__ import annotations
from pathlib import Path
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _text_page(pdf, lines, title=None, fontsize=10):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    y = 0.96
    if title is not None:
        ax.text(0.5, y, title, ha='center', va='top',
                 fontsize=14, weight='bold')
        y -= 0.05
    for ln in lines:
        ax.text(0.06, y, ln, ha='left', va='top',
                 fontsize=fontsize, family='monospace')
        y -= 0.022
    pdf.savefig(fig); plt.close(fig)


def _image_page(pdf, img_path: Path, title: str):
    if not img_path.exists():
        return
    img = plt.imread(str(img_path))
    fig = plt.figure(figsize=(8.5, 11))
    ax_title = fig.add_axes([0.05, 0.93, 0.9, 0.04]); ax_title.set_axis_off()
    ax_title.text(0.5, 0.5, title, ha='center', va='center',
                    fontsize=12, weight='bold')
    ax_img = fig.add_axes([0.08, 0.08, 0.84, 0.82])
    ax_img.imshow(img); ax_img.set_axis_off()
    pdf.savefig(fig); plt.close(fig)


def render_pdf(rdir: Path, ctx: dict):
    out = rdir / 'report.pdf'
    plots = rdir / 'artifacts' / 'plots'
    with PdfPages(out) as pdf:
        # Title page
        cfg_lines = ['Configuration:', '']
        for k, v in ctx['config'].items():
            cfg_lines.append(f'    {k:>12s} = {v}')
        _text_page(pdf, [
            f"Compiled: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
            "",
            "Scenario summary",
            "----------------",
            f"  scenario : {ctx['scenario']}",
            f"  model    : {ctx['active_variants']['model']}",
            f"  grid     : {ctx['active_variants']['grid']}",
            f"  step2    : {ctx['active_variants']['step2']}",
            f"  solver   : {ctx['active_variants']['solver']}",
            f"  iters    : {ctx['iters']}",
            f"  Finf     : {ctx['Finf']:.3e}",
            f"  walltime : {ctx['walltime_s']:.3f} s",
            "",
            "Metrics",
            "-------",
            f"  slope_T            = {ctx['slope_T']:.6f}",
            f"  deficit (1-R^2)    = {ctx['deficit']:.6e}",
            f"  d_FR               = {ctx['d_FR']:.4f}",
            f"  max|P-P_analytic|  = {ctx['d_analytic']:.3e}",
            "",
            *cfg_lines,
            "",
            f"Repo SHA: {ctx['repo_sha'][:10]} "
            f"({'CLEAN' if not ctx['repo_dirty'] else 'DIRTY'})",
            f"Tracked source files: {len(ctx['files'])}",
        ], title=f"MIZN run report: {ctx['scenario']}")
        _image_page(pdf, plots / 'convergence.png',
                      'Convergence: ||F||_inf vs iteration')
        _image_page(pdf, plots / 'P_slices.png',
                      'Fixed-point price slices')
        # Manifest table
        files = ctx['files']
        rows = []
        for f, info in sorted(files.items()):
            rows.append(f"  {info['git_blob'][:10]}  {info['sha256'][:10]}  {f}")
        _text_page(pdf, [
            "Per-file manifest  (git_blob[:10], sha256[:10], path)",
            "",
            *rows,
        ], title='Version manifest', fontsize=8)
