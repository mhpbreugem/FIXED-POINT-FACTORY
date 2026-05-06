"""
progress.py — Worker progress reporter.
Solvers import this and call `report(iter=N, ftol=X)` periodically.
A background thread writes the latest state to
    projects/$PROJECT/progress/$TASK_ID.json
once per `interval` seconds, commits it, and pushes.
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
    reporter.stop()    # final flush + thread join
Design notes:
  - State is in-memory; only the latest snapshot is pushed every minute.
  - File contents are small (~200 bytes) and one file per task.
  - Push failures are retried with rebase; never blocks the solver.
  - Stop deletes the file (the task is done or bailed; supervisor
    doesn't need to see stale progress).
"""
from __future__ import annotations
import datetime
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else Path(".")
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
        self.progress_dir = self.repo_root / "projects" / project / "progress"
        self.progress_file = self.progress_dir / f"{task_id}.json"
        self._state: dict = {
            "task_id": task_id,
            "worker_id": worker_id,
            "started_at": _utcnow_iso(),
            "last_update": _utcnow_iso(),
            "iter": None,
            "ftol": None,
            "extra": {},
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    # -------- public API --------------------------------------------------
    def update(self, iter: Optional[int] = None, ftol: Optional[float] = None,
               **extra) -> None:
        """Called by the solver inner loop. Cheap: just updates memory."""
        with self._lock:
            if iter is not None:
                self._state["iter"] = int(iter)
            if ftol is not None:
                # store as string to preserve very small mp values
                self._state["ftol"] = (
                    str(ftol) if isinstance(ftol, str) else f"{float(ftol):.6e}"
                )
            if extra:
                self._state["extra"].update(extra)
            self._state["last_update"] = _utcnow_iso()
    def start(self) -> None:
        """Start the background flusher thread."""
        if self._thread is not None:
            return
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    def stop(self, delete: bool = True) -> None:
        """Stop the flusher. If delete=True, remove the progress file."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        if delete and self.progress_file.exists():
            try:
                self.progress_file.unlink()
                self._git_commit_push(
                    f"progress cleanup {self.task_id} ({self.worker_id})",
                    delete=True,
                )
            except Exception as e:
                print(f"[progress] cleanup failed: {e}")
    # -------- internals ---------------------------------------------------
    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._flush()
            # Sleep in small ticks so stop() returns quickly
            for _ in range(self.interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
        # Final flush before exit
        self._flush()
    def _flush(self) -> None:
        with self._lock:
            snapshot = dict(self._state)
        try:
            self.progress_file.write_text(json.dumps(snapshot, indent=2))
            self._git_commit_push(
                f"progress {self.task_id} iter={snapshot.get('iter')} "
                f"ftol={snapshot.get('ftol')} ({self.worker_id})"
            )
        except Exception as e:
            # Never let progress reporting kill the solver.
            print(f"[progress] flush failed (non-fatal): {e}")
    def _git_commit_push(self, message: str, delete: bool = False) -> None:
        rel = f"projects/{self.project}/progress/{self.task_id}.json"
        cwd = str(self.repo_root)
        # Pull first to minimize conflicts
        subprocess.run(["git", "pull", "--rebase", "origin", self.branch],
                       cwd=cwd, capture_output=True, timeout=30)
        if delete:
            subprocess.run(["git", "rm", "-f", "--ignore-unmatch", rel],
                           cwd=cwd, capture_output=True)
        else:
            subprocess.run(["git", "add", rel], cwd=cwd, capture_output=True)
        # Anything to commit?
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"],
                              cwd=cwd, capture_output=True)
        if diff.returncode == 0:
            return  # nothing changed
        subprocess.run(["git", "commit", "-m", message, "--quiet"],
                       cwd=cwd, capture_output=True)
        # Push with retries
        for attempt in range(3):
            push = subprocess.run(["git", "push", "origin", self.branch, "--quiet"],
                                  cwd=cwd, capture_output=True, timeout=30)
            if push.returncode == 0:
                return
            subprocess.run(["git", "pull", "--rebase", "origin", self.branch],
                           cwd=cwd, capture_output=True, timeout=30)
            time.sleep(2 * (attempt + 1))
