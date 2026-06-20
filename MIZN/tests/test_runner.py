"""Smoke test for tools.runner end-to-end."""
import json, sys
from pathlib import Path

# tools/ is at the repo root, sibling of src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mizn.config import Params
from mizn.main import solve
from tools.runner import run, RUNS_DIR


def test_runner_writes_full_artifact_tree(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.runner.RUNS_DIR', tmp_path)
    rid = run('smoke_test', Params(G=5, tol=1e-10, maxit=20), solver_fn=solve)
    rdir = tmp_path / rid
    assert rdir.exists()
    for f in ('config.yaml', 'manifest.json', 'git_state.txt',
               'env.txt', 'results.json', 'report.tex', 'report.pdf'):
        assert (rdir / f).exists(), f"missing artifact: {f}"
    res = json.loads((rdir / 'results.json').read_text())
    assert res['Finf'] < 1e-9
    assert res['iters'] >= 1
    # manifest tracks > 10 source files
    mf = json.loads((rdir / 'manifest.json').read_text())
    assert len(mf['files']) > 10
    # index appended
    assert (tmp_path / 'INDEX.md').exists()
    assert (tmp_path / 'index.json').exists()
    idx = json.loads((tmp_path / 'index.json').read_text())
    assert idx[0]['scenario'] == 'smoke_test'
