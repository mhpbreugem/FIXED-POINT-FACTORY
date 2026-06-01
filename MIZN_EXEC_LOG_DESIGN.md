# MIZN — per-run execution log & LaTeX report system

Every `scripts/*.py` execution produces:

1. A **timestamped subfolder** in `runs/` with config, version manifest, results.
2. A **compiled LaTeX PDF report** with summary, metrics, plots, version table.
3. A **one-line entry** appended to `runs/INDEX.md` + `runs/index.json`.

This makes every result **reproducible** (manifest pins every file version) and
**reviewable** (PDF readable offline, no UI needed).

## Per-run directory layout

```
runs/2026-06-01T14-23-18Z__cara_default__a4f7c2/
├── config.yaml          full Params dump (the exact dict that ran)
├── manifest.json        ★ per-file version (git_blob + sha256) + active variants
├── git_state.txt        repo SHA, branch, clean/dirty; diff if dirty
├── env.txt              python ver, numpy/numba/scipy versions, OS, CPU count
├── stdout.log           captured prints / progress
├── results.json         metrics: ‖F‖, slope, deficit, iters, walltime
├── artifacts/
│   ├── P_final.npy
│   ├── ferr_history.npy
│   └── plots/           *.png from diagnostics.plotting
├── report.tex           ★ auto-generated from Jinja2 template
└── report.pdf           ← compiled, this is what you read
```

**Naming convention** `<UTC timestamp>__<scenario>__<short-git-sha>/`:
- timestamps sort chronologically;
- scenario tag (e.g. `cara_default`, `crra_g0.1_G9`, `gamma_sweep_adaptive`)
  makes folders scannable by eye;
- short SHA pins the code state for instant lookup.

## The version manifest (the key reproducibility piece)

`manifest.json` records, for every `.py` file in `src/mizn/` that gets imported
during the run, BOTH:

- **`git_blob`** = `git hash-object file.py`. Ties the file to repo history;
  you can `git checkout <git_blob>` for that exact file version.
- **`sha256`** = content hash of the file as actually read at runtime. If the
  working tree was dirty (uncommitted edits), this records what *actually
  executed*; the diff is captured in `git_state.txt`.

Example:
```json
{
  "run_id": "2026-06-01T14-23-18Z__cara_default__a4f7c2",
  "repo_sha": "a4f7c2d8...",
  "repo_dirty": false,
  "files": {
    "src/mizn/config.py":         {"git_blob": "8af3...", "sha256": "9c1b...", "lines": 47},
    "src/mizn/main.py":           {"git_blob": "1de4...", "sha256": "4f88...", "lines": 32},
    "src/mizn/grid/linear_u.py":  {"git_blob": "ee20...", "sha256": "7a02...", "lines": 18},
    "src/mizn/step2_learning/gaussian.py":
                                  {"git_blob": "b733...", "sha256": "c1d0...", "lines": 84},
    "...": "..."
  },
  "active_variants": {
    "model":   "cara",
    "grid":    "linear_u",
    "step2":   "gaussian",
    "step3":   "cara",
    "step5":   "cara",
    "solver":  "picard"
  }
}
```

**Reproduction**: anyone gets `manifest.json` + `config.yaml`, does
`git checkout <repo_sha>`, runs the same `scripts/*.py` with the same config,
gets the same numbers. If the repo was dirty, `git_state.txt` has the diff to
apply.

## The wrapper (`tools/runner.py`)

Every `scripts/*.py` calls **`runner.run(scenario, params, solver_fn)`**.
It does all the boilerplate:

```python
def run(scenario: str, params: Params, solver_fn):
    rid = make_run_id(scenario)
    rdir = Path("runs") / rid
    rdir.mkdir(parents=True)

    capture_git_state(rdir)                 # → git_state.txt
    capture_env(rdir)                       # → env.txt
    manifest = build_manifest(params)       # walks src/mizn/, hashes
    (rdir/"manifest.json").write_text(json.dumps(manifest, indent=2))
    (rdir/"config.yaml").write_text(yaml.dump(asdict(params)))

    with tee_stdout(rdir/"stdout.log"):
        t0 = time.time()
        P, Finf, iters = solver_fn(params)
        walltime = time.time() - t0

    metrics = compute_metrics(P, params)
    np.save(rdir/"artifacts/P_final.npy", P)
    make_plots(P, params, out=rdir/"artifacts/plots")
    json.dump({**metrics, "Finf": Finf, "iters": iters,
               "walltime_s": walltime},
              open(rdir/"results.json","w"), indent=2)

    render_latex(rdir, manifest, metrics, params)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"],
                   cwd=rdir, check=False)
    append_index(rid, scenario, metrics)
    return rid
```

Standardization for free: every script gets the same logging.

## The LaTeX report template (`tools/report_template.tex.j2`)

Jinja2 template (~80 lines) producing a 1–2 page PDF per run:

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref,longtable}
\title{Run report: {{ run.scenario }}}
\author{auto-generated}
\date{{{ run.timestamp }}}

\begin{document}\maketitle

\section*{1.~Summary}
{{ run.summary_paragraph }}

\section*{2.~Configuration}
\begin{tabular}{ll}\toprule
parameter & value \\\midrule
{% for k,v in run.config.items() %}{{ k|escape }} & {{ v }} \\
{% endfor %}\bottomrule\end{tabular}

\section*{3.~Result metrics}
\begin{tabular}{lr}\toprule
metric & value \\\midrule
$\|F\|_\infty$         & {{ "%.3e"|format(run.Finf) }} \\
slope$_T$              & {{ "%.5f"|format(run.slope_T) }} \\
deficit ($1-R^2$)      & {{ "%.5f"|format(run.deficit) }} \\
$d_\mathrm{FR}$        & {{ "%.4f"|format(run.d_FR) }} \\
iterations             & {{ run.iters }} \\
walltime (s)           & {{ "%.1f"|format(run.walltime_s) }} \\
\bottomrule\end{tabular}

\section*{4.~Convergence}
\includegraphics[width=\linewidth]{artifacts/plots/convergence.png}

\section*{5.~Fixed point}
\includegraphics[width=\linewidth]{artifacts/plots/P_slices.png}

\section*{6.~Version manifest (sub-component versions)}
\small\begin{longtable}{lll}\toprule
file & git\_blob & sha256 \\\midrule
{% for f, info in run.files.items() %}\texttt{ {{- f -}} } & \texttt{ {{- info.git_blob[:10] -}} } & \texttt{ {{- info.sha256[:10] -}} } \\
{% endfor %}\bottomrule\end{longtable}

Active variants: \texttt{ {{ run.active_variants }} }.
Repo SHA: \texttt{ {{ run.repo_sha[:10] }} }
({{ 'CLEAN' if not run.repo_dirty else '\\textbf{DIRTY}' }}).

\section*{7.~Notes (manual)}
\input{notes.tex}

\end{document}
```

`notes.tex` lets you append observations *after the run*; the rest is fully auto.

## Top-level index

**`runs/INDEX.md`** (one row per run, auto-appended):
```markdown
| timestamp | scenario | model | grid | step2 | G | γ | ‖F‖ | slope | deficit | report |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-01T14:23 | cara_default | cara | linear_u | gaussian | — | — | 2e-16 | 1.000 | 0.000 | [pdf](2026…/report.pdf) |
| 2026-06-01T14:51 | hfree_g0.1_G9 | crra | linear_u | coarea_h0 | 9 | 0.1 | 9.4e-16 | 0.364 | 0.172 | [pdf](2026…/report.pdf) |
```

**`runs/index.json`** is the machine-readable mirror.

**`tools/index_filter.py`** lets you do
`python -m tools.index_filter "model=crra,grid=linear_u,Finf<1e-11"`
and get back the matching report paths.

## What to git-track vs ignore

```
# .gitignore
runs/*/artifacts/*.npy          # large binaries — regeneratable
runs/*/stdout.log               # noisy
runs/*/report.pdf               # regeneratable from .tex (optional: keep them)
```

**Keep tracked** (so reports become part of repo history):
`report.tex`, `manifest.json`, `results.json`, `config.yaml`, `git_state.txt`,
`env.txt`, `INDEX.md`, `index.json`. PDFs and big arrays are downloadable
artifacts.

If PDFs are precious for sharing: track them too (usually < 1 MB).

## Build order (incremental, doesn't block other work)

1. **`tools/manifest.py` first** — walks `src/mizn/`, returns dict with
   `git_blob` + `sha256`. Test on one file.
2. **`tools/runner.py` skeleton** — captures dirs, env, stdout. No LaTeX yet.
3. Wire `scripts/nail_anchor.py` to call `runner.run(...)`. Check layout.
4. Add `tools/report_template.tex.j2` minimal (config + metrics only). Render. Compile.
5. Add plot regeneration into `diagnostics/plotting.py`, include in report.
6. Add `tools/index_filter.py`.
7. CI: workflow runs the full test suite on every push + nails a `cara_default`
   on green main to verify reproducibility (PDF artifact uploaded).

Each step ~80–150 lines. Each PR-able alone.

## Why not MLflow / W&B / DVC / Sacred?

| Tool | Pros | Cons | Verdict |
|---|---|---|---|
| **This (file-based)** | tiny deps, git-native, PDFs work offline, transparent | maintain `runner.py` (~120 LOC) | ★ recommended |
| MLflow | UI, model registry, search | server/DB, not git-native, no LaTeX | overkill for math |
| Weights & Biases | beautiful UI | cloud-locked, no LaTeX | wrong tool for papers |
| DVC | tracks large data | adds remote-storage config | combine *with* this if data grows |
| Hydra | structured config mgmt | no logging, no reports | use *with* this for many scenarios |
| Sacred / Sumatra | similar idea | maintenance status varies | this design is "modern Sumatra, lighter" |

The format is open — you can graduate to MLflow later if needed (you'd just
ingest the existing JSON files). Starting file-based costs nothing and gives
PDFs your collaborators can read on day one.

## Subtle point worth knowing

This design treats **every script execution as a publishable artifact**.
The runs subfolder is *the* canonical record. Notebooks are scratch; runs are
permanent. This is the right discipline for paper-grade research and is the
only way the prior session's findings (γ-sweep, fold, etc.) would be
re-verifiable in 6 months.
