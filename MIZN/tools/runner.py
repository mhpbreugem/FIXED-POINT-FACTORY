"""The single entry point all scripts call.

`run(scenario, params, solver_fn)` creates a timestamped folder under
runs/, captures git/env state, dumps config + manifest, runs the solver,
saves arrays + plots, computes metrics, renders a Jinja2 LaTeX report,
and appends an index row.  pdflatex is invoked if available; absence is
not fatal (the .tex is always emitted).
"""
from __future__ import annotations
import json, os, subprocess, sys, time, hashlib, datetime
from dataclasses import asdict
from pathlib import Path
import numpy as np
import yaml

from mizn.config import Params
from mizn.diagnostics import compute_metrics, make_plots

from .manifest import build_manifest


REPO_ROOT = Path(__file__).resolve().parent.parent       # MIZN/
RUNS_DIR  = REPO_ROOT / 'runs'


def _utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')


def _short_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         text=True, cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'nogit'


def _capture_git(rdir: Path):
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                        text=True, cwd=REPO_ROOT).strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                           text=True, cwd=REPO_ROOT).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'],
                                           text=True, cwd=REPO_ROOT)
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'MIZN'],
                                         text=True, cwd=REPO_ROOT)
        (rdir / 'git_state.txt').write_text(
            f"sha:    {sha}\nbranch: {branch}\ndirty:  {bool(status.strip())}\n"
            f"\n--- status --\n{status}\n--- diff --\n{diff}\n")
    except Exception as e:
        (rdir / 'git_state.txt').write_text(f"git unavailable: {e}\n")


def _capture_env(rdir: Path):
    import platform
    info = [
        f"python:     {sys.version.split()[0]}",
        f"platform:   {platform.platform()}",
        f"machine:    {platform.machine()}",
        f"cpu_count:  {os.cpu_count()}",
    ]
    for pkg in ('numpy', 'scipy', 'matplotlib', 'numba', 'jinja2', 'yaml'):
        try:
            mod = __import__(pkg)
            v = getattr(mod, '__version__', '?')
        except Exception:
            v = 'not-installed'
        info.append(f"{pkg:>12s}: {v}")
    (rdir / 'env.txt').write_text('\n'.join(info) + '\n')


def _render_latex(rdir: Path, manifest: dict, metrics: dict,
                    params: Params, scenario: str, iters: int,
                    Finf: float, walltime: float):
    """Render a small standalone LaTeX report (and try to compile to PDF)."""
    template_path = REPO_ROOT / 'tools' / 'report_template.tex.j2'
    try:
        from jinja2 import Template
        tmpl = Template(template_path.read_text())
    except Exception as e:
        (rdir / 'report.tex').write_text(f"% jinja2 unavailable: {e}\n")
        return
    ctx = dict(
        scenario=scenario,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config=asdict(params),
        Finf=Finf, iters=iters, walltime_s=walltime,
        slope_T=metrics['slope_T'],
        deficit=metrics['deficit'],
        d_FR=metrics['d_FR'],
        d_analytic=metrics.get('d_analytic', float('nan')),
        files=manifest['files'],
        repo_sha=manifest['repo_sha'],
        repo_dirty=manifest['repo_dirty'],
        active_variants=dict(model=params.model, grid=params.grid,
                              step2=params.step2, solver=params.solver),
    )
    (rdir / 'report.tex').write_text(tmpl.render(**ctx))
    notes = rdir / 'notes.tex'
    if not notes.exists():
        notes.write_text("% append manual notes here\n")
    # try pdflatex first; if it's missing or the .pdf wasn't produced, fall
    # back to a matplotlib-rendered PDF so reports are always readable.
    try:
        subprocess.run(['pdflatex', '-interaction=nonstopmode', 'report.tex'],
                         cwd=rdir, capture_output=True, timeout=30)
    except Exception:
        pass
    if not (rdir / 'report.pdf').exists():
        from .pdf_fallback import render_pdf
        render_pdf(rdir, ctx)


def _append_index(rid: str, scenario: str, params: Params, metrics: dict,
                    Finf: float, iters: int):
    RUNS_DIR.mkdir(exist_ok=True)
    idx_md = RUNS_DIR / 'INDEX.md'
    idx_js = RUNS_DIR / 'index.json'
    is_new = not idx_md.exists()
    if is_new:
        idx_md.write_text(
            '| timestamp | scenario | model | grid | step2 | G | gamma | ||F||_inf | slope_T | deficit | report |\n'
            '|---|---|---|---|---|---|---|---|---|---|---|\n')
    idx_md.open('a').write(
        f"| {rid[:16]} | {scenario} | {params.model} | {params.grid} "
        f"| {params.step2} | {params.G} | {params.gamma} "
        f"| {Finf:.2e} | {metrics['slope_T']:.5f} "
        f"| {metrics['deficit']:.5f} "
        f"| [pdf]({rid}/report.pdf) |\n")
    rows = json.loads(idx_js.read_text()) if idx_js.exists() else []
    rows.append(dict(run_id=rid, scenario=scenario,
                      model=params.model, grid=params.grid, step2=params.step2,
                      G=params.G, gamma=params.gamma,
                      Finf=Finf, iters=iters, **metrics))
    idx_js.write_text(json.dumps(rows, indent=2, default=str))


def run(scenario: str, params: Params, solver_fn) -> str:
    """Run one experiment, return run_id."""
    rid = f"{_utc_stamp()}__{scenario}__{_short_sha()}"
    rdir = RUNS_DIR / rid
    (rdir / 'artifacts' / 'plots').mkdir(parents=True, exist_ok=True)

    _capture_git(rdir)
    _capture_env(rdir)
    manifest = build_manifest()
    manifest['run_id'] = rid
    (rdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (rdir / 'config.yaml').write_text(yaml.safe_dump(asdict(params)))

    t0 = time.time()
    P, hist, iters = solver_fn(params)
    walltime = time.time() - t0

    np.save(rdir / 'artifacts' / 'P_final.npy', P)
    np.save(rdir / 'artifacts' / 'ferr_history.npy', np.asarray(hist))
    make_plots(P, hist, params, rdir / 'artifacts' / 'plots')
    metrics = compute_metrics(P, params)
    Finf = float(hist[-1])
    json.dump(dict(**metrics, Finf=Finf, iters=int(iters),
                     walltime_s=walltime),
                open(rdir / 'results.json', 'w'), indent=2, default=str)

    _render_latex(rdir, manifest, metrics, params, scenario,
                    iters, Finf, walltime)
    _append_index(rid, scenario, params, metrics, Finf, iters)
    return rid
