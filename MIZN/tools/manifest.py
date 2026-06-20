"""Per-file version manifest: git_blob + sha256 for every .py in src/mizn/.

Lets you tie each run to *exactly* the source it executed under, even if
the working tree was dirty.  Combined with a `git_state.txt` diff this is
a 1-file reproducibility receipt.
"""
from __future__ import annotations
import hashlib, subprocess
from pathlib import Path


SRC = Path(__file__).resolve().parent.parent / 'src' / 'mizn'


def _git_blob(path: Path) -> str:
    try:
        out = subprocess.check_output(['git', 'hash-object', str(path)],
                                        text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ''


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_manifest() -> dict:
    files = {}
    for p in sorted(SRC.rglob('*.py')):
        rel = p.relative_to(SRC.parent.parent)  # MIZN-root relative
        try:
            n_lines = sum(1 for _ in p.open())
        except Exception:
            n_lines = -1
        files[str(rel)] = dict(git_blob=_git_blob(p),
                                sha256=_sha256(p),
                                lines=n_lines)
    repo_sha = ''
    repo_dirty = False
    try:
        repo_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                             text=True, stderr=subprocess.DEVNULL).strip()
        repo_dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], text=True).strip())
    except Exception:
        pass
    return dict(repo_sha=repo_sha, repo_dirty=repo_dirty, files=files)
