"""
progress.py — Worker progress reporter (REST API edition).

Solvers import this and call `update(iter=N, ftol=X)` periodically.
A background thread PUTs the latest snapshot to
    projects/$PROJECT/progress/$TASK_ID.json
once per `interval` seconds via the GitHub Contents API.

Why REST API instead of git push?
Many workers writing their own progress files in parallel used to collide
at `git push` time, dropping snapshots and (worse) leaving the worker's
local tree diverged from main.  Each progress file is owned by exactly one
worker, so the only conflict path is the file's own previous SHA — which
we cache from the last successful PUT.  No git operations at all.

Usage from a solver:
    from core.progress import ProgressReporter
    reporter = ProgressReporter(
        project="REZN", task_id="g400_t1000", worker_id="solver-1",
        branch="main", interval=60,
    )
    reporter.start()
    for it in range(max_iter):
        ftol = compute_residual(...)
        reporter.update(iter=it, ftol=ftol)
        if ftol < target_tol:
            break
    reporter.stop()    # final flush + thread join + delete from repo
"""
from __future__ import annotations

import base64
import datetime
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else Path(".")


def _gh_token() -> str:
    t = os.environ.get("GITHUB_TOKEN", "")
    if t:
        return t
    r = subprocess.run(["git", "remote", "get-url", "origin"],
                       capture_output=True, text=True)
    m = re.search(r"https://([A-Za-z0-9_]+)@github\.com", r.stdout.strip())
    return m.group(1) if m else ""


def _gh_repo() -> tuple[str, str]:
    r = subprocess.run(["git", "remote", "get-url", "origin"],
                       capture_output=True, text=True)
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+?)(?:\.git)?$", r.stdout.strip())
    return (m.group(1), m.group(2)) if m else ("", "")


class ProgressReporter:
    def __init__(
        self,
        project: str,
        task_id: str,
        worker_id: str,
        branch: str = "main",
        interval: int = 60,
        repo_root: Optional[Path] = None,
    ):
        self.project = project
        self.task_id = task_id
        self.worker_id = worker_id
        self.branch = branch
        self.interval = interval
        self.repo_root = Path(repo_root) if repo_root else _repo_root()
        self.api_path = f"projects/{project}/progress/{task_id}.json"

        # API auth — fall back to no-op mode if no token (local testing)
        self._token = _gh_token()
        self._owner, self._repo = _gh_repo()
        self._api_enabled = bool(self._token and self._owner and self._repo)

        # Cached SHA of the latest version of the file on the remote.
        # None means "file doesn't exist on remote yet" (first PUT omits sha).
        self._remote_sha: Optional[str] = None

        self._state: dict = {
            "task_id":      task_id,
            "worker_id":    worker_id,
            "started_at":   _utcnow_iso(),
            "last_update":  _utcnow_iso(),
            "iter":         None,
            "ftol":         None,
            "ftol_history": [],
            "extra":        {},
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── public API ────────────────────────────────────────────────────────────────────
    def update(self, iter: Optional[int] = None, ftol: Optional[float] = None,
               **extra) -> None:
        """Called by the solver inner loop. Cheap: just updates memory."""
        with self._lock:
            if iter is not None:
                self._state["iter"] = int(iter)
            if ftol is not None:
                ftol_str = str(ftol) if isinstance(ftol, str) else f"{float(ftol):.6e}"
                self._state["ftol"] = ftol_str
                hist = self._state["ftol_history"]
                if not hist or hist[-1] != ftol_str:
                    hist.append(ftol_str)
                    if len(hist) > 8:
                        hist.pop(0)
            if extra:
                self._state["extra"].update(extra)
            self._state["last_update"] = _utcnow_iso()

    def start(self) -> None:
        """Start the background flusher thread."""
        if self._thread is not None:
            return
        if not self._api_enabled:
            print("[progress] GITHUB_TOKEN missing — progress reporting disabled",
                  flush=True)
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, delete: bool = True) -> None:
        """Stop the flusher. If delete=True, remove the progress file from remote."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=15)
            self._thread = None
        if delete and self._api_enabled:
            self._api_delete()

    # ── internals ────────────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._flush()
            for _ in range(self.interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
        # Final flush — always runs so the last state is written before stop()/delete()
        self._flush()

    def _flush(self) -> None:
        """PUT the current state to GitHub. Caches the new file SHA on success."""
        with self._lock:
            snapshot = dict(self._state)
        body = json.dumps(snapshot, indent=2).encode()
        message = (f"progress {self.task_id} "
                   f"iter={snapshot.get('iter')} ftol={snapshot.get('ftol')} "
                   f"({self.worker_id})")
        # 3 attempts: 200/201 = success, 409 = stale SHA (refresh and retry),
        # transient errors = backoff and retry.
        for attempt in range(3):
            try:
                status, new_sha = self._api_put(body, message, self._remote_sha)
            except Exception as exc:
                print(f"[progress] flush exc attempt {attempt}: {exc}", flush=True)
                time.sleep(2 * (attempt + 1))
                continue
            if status in (200, 201):
                if new_sha:
                    self._remote_sha = new_sha
                return
            if status == 409:
                # Lost the SHA somehow (e.g. cleanup deleted the file).
                # Refresh and try again on next loop iteration.
                self._remote_sha = self._api_get_sha()
                time.sleep(1 + attempt)
                continue
            if status == 404:
                # File missing on remote; clear cached SHA so next PUT creates it.
                self._remote_sha = None
                continue
            print(f"[progress] PUT rc={status} attempt {attempt} — check token/permissions",
                  flush=True)
            time.sleep(2 * (attempt + 1))

    def _api_put(self, body: bytes, message: str,
                 sha: Optional[str]) -> tuple[int, Optional[str]]:
        url = (f"https://api.github.com/repos/{self._owner}/{self._repo}"
               f"/contents/{self.api_path}")
        payload: dict = {
            "message": message,
            "content": base64.b64encode(body).decode(),
            "branch":  self.branch,
        }
        if sha:
            payload["sha"] = sha
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="PUT",
            headers={"Authorization": f"token {self._token}",
                     "Content-Type":  "application/json",
                     "Accept":        "application/vnd.github+json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                resp = json.loads(r.read().decode())
                return r.status, resp.get("content", {}).get("sha")
        except urllib.error.HTTPError as e:
            try:
                body_txt = e.read().decode(errors="replace")[:400]
                print(f"[progress] PUT {e.code} body: {body_txt}", flush=True)
            except Exception:
                pass
            return e.code, None

    def _api_get_sha(self) -> Optional[str]:
        url = (f"https://api.github.com/repos/{self._owner}/{self._repo}"
               f"/contents/{self.api_path}?ref={self.branch}")
        req = urllib.request.Request(
            url, headers={"Authorization": f"token {self._token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode()).get("sha")
        except urllib.error.HTTPError:
            return None
        except Exception:
            return None

    def _api_delete(self) -> None:
        sha = self._remote_sha or self._api_get_sha()
        if not sha:
            return  # nothing to delete
        url = (f"https://api.github.com/repos/{self._owner}/{self._repo}"
               f"/contents/{self.api_path}")
        payload = {
            "message": f"progress cleanup {self.task_id} ({self.worker_id})",
            "sha":     sha,
            "branch":  self.branch,
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="DELETE",
            headers={"Authorization": f"token {self._token}",
                     "Content-Type":  "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=20).read()
        except Exception as exc:
            print(f"[progress] cleanup failed (non-fatal): {exc}", flush=True)
