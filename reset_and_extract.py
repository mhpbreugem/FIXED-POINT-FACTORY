#!/usr/bin/env python3
"""reset_and_extract.py — two patches for the queue.

  1. Reset all 'bailed' tasks back to 'ready' so the workflow can re-pick them.
  2. For every K=3 .npz checkpoint on disk, extract the converged residual
     (last finite stage_F_inf) and inject as result['F_max'] in the queue.
     Also extract 1-R² fresh via weighted regression in logit space.
     If the checkpoint has no corresponding queue entry, ADD a new done entry.

Idempotent. Run from repo root.
"""
import json
import csv
from pathlib import Path
from typing import Optional
import numpy as np

QUEUE = Path("projects/REZN/TASK_QUEUE.json")
CKPT_DIR = Path("projects/REZN/checkpoints")
SEED = Path("results/full_ree/seed_g050_t0200_3d.npz")


def signal_density(u: np.ndarray, v: int, tau: float) -> np.ndarray:
    """Centered-signal density: N(v - 0.5, 1/tau)."""
    return np.sqrt(tau / (2 * np.pi)) * np.exp(-tau / 2 * (u - (v - 0.5))**2)


def extract_metrics(npz_path: Path) -> Optional[dict]:
    """Return {gamma, tau, F_max, '1-R2'} or None if checkpoint is unreadable."""
    try:
        d = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    if 'P_full' not in d.files:
        return None
    K = int(d.get('K', 3))
    if K != 3:
        return None

    out = {}
    out['gamma'] = float(d['gamma_vec'][0])
    out['tau'] = float(d['tau_vec'][0])

    # F_max: last finite entry in stage_F_inf
    if 'stage_F_inf' in d.files:
        stages = d['stage_F_inf']
        finite = stages[np.isfinite(stages)]
        if len(finite) > 0:
            out['F_max'] = float(finite[-1])

    # 1-R^2 via weighted logit regression (matches solver convention)
    P = d['P_full']
    u = d['u_full']
    pad = int(d['pad'])
    G_inner = int(d['G_inner'])
    P_inner = P[pad:pad+G_inner, pad:pad+G_inner, pad:pad+G_inner]
    u_inner = u[pad:pad+G_inner]
    tau = out['tau']
    eps = 1e-9

    Tstar, logit_p, w = [], [], []
    for i in range(G_inner):
        for j in range(G_inner):
            for l in range(G_inner):
                p = P_inner[i, j, l]
                if not (eps < p < 1 - eps):
                    continue
                T = tau * (u_inner[i] + u_inner[j] + u_inner[l])
                lp = float(np.log(p / (1 - p)))
                w_v1 = (signal_density(u_inner[i], 1, tau)
                        * signal_density(u_inner[j], 1, tau)
                        * signal_density(u_inner[l], 1, tau))
                w_v0 = (signal_density(u_inner[i], 0, tau)
                        * signal_density(u_inner[j], 0, tau)
                        * signal_density(u_inner[l], 0, tau))
                Tstar.append(T); logit_p.append(lp); w.append(0.5 * (w_v0 + w_v1))

    if len(Tstar) >= 4:
        Ts = np.asarray(Tstar); lp = np.asarray(logit_p); w = np.asarray(w)
        W = np.sqrt(w)
        A = np.column_stack([np.ones_like(Ts), Ts])
        coef, *_ = np.linalg.lstsq(A * W[:, None], lp * W, rcond=None)
        a, b = coef
        resid = lp - (a + b * Ts)
        ss_res = float(np.sum(w * resid**2))
        mean_lp = float(np.sum(w * lp) / np.sum(w))
        ss_tot = float(np.sum(w * (lp - mean_lp)**2))
        if ss_tot > 0:
            out['1-R2'] = float(1.0 - (1 - ss_res / ss_tot))
            out['slope'] = float(b)

    return out


def main() -> None:
    q = json.loads(QUEUE.read_text())
    by_id = {t['id']: t for t in q['tasks']}

    # (1) Reset bailed tasks
    reset_count = 0
    for t in q['tasks']:
        if t['status'] == 'bailed':
            t['status'] = 'ready'
            t['claimed_by'] = None
            t['claimed_at'] = None
            t['completed_at'] = None
            existing = (t.get('note') or '')
            if 'RESET from bailed' not in existing:
                t['note'] = (existing + ' | RESET from bailed').strip(' |')
            reset_count += 1
    print(f"[1/2] Reset {reset_count} bailed tasks to ready.")

    # (2) Extract F_max + 1-R^2 from each checkpoint
    files = sorted(CKPT_DIR.glob("g*.npz"))
    if SEED.exists():
        files.append(SEED)

    updated = 0
    added = 0
    skipped = 0
    for ckpt in files:
        stem = ckpt.stem
        if stem.startswith("symK") or stem.startswith("extract_"):
            continue
        m = extract_metrics(ckpt)
        if m is None:
            skipped += 1
            continue

        ckpt_rel = str(ckpt.relative_to(Path('.')))

        if stem in by_id:
            t = by_id[stem]
            if t.get('result') is None:
                t['result'] = {}
            # Don't overwrite values the solver set; only fill gaps.
            for k in ('F_max', '1-R2', 'slope'):
                if k in m and t['result'].get(k) is None:
                    t['result'][k] = m[k]
            # Fix checkpoint pointer if stale
            if not Path(t.get('checkpoint') or '').exists():
                t['checkpoint'] = ckpt_rel
                t['status'] = 'done'
            updated += 1
        else:
            new = {
                "id": stem,
                "kind": "ree",
                "gamma": m['gamma'],
                "tau": m['tau'],
                "G": 12,
                "K": 3,
                "depends_on": [],
                "deps_satisfy": "any",
                "status": "done",
                "claimed_by": None,
                "claimed_at": None,
                "completed_at": "2026-05-09T20:00:00Z",
                "checkpoint": ckpt_rel,
                "result": {k: v for k, v in m.items() if k in ('F_max', '1-R2', 'slope')},
                "note": "Added from on-disk checkpoint (orphan: no prior queue entry)."
            }
            q['tasks'].append(new)
            by_id[stem] = new
            added += 1

    print(f"[2/2] Updated {updated} existing tasks; added {added} orphan-checkpoint tasks.")
    if skipped:
        print(f"      Skipped {skipped} unreadable / non-K=3 checkpoints.")

    QUEUE.write_text(json.dumps(q, indent=2) + "\n")
    have_fmax = sum(1 for t in q['tasks'] if (t.get('result') or {}).get('F_max') is not None)
    have_r2 = sum(1 for t in q['tasks'] if (t.get('result') or {}).get('1-R2') is not None)
    print(f"\nFinal queue: {len(q['tasks'])} tasks total.")
    print(f"  with F_max: {have_fmax}")
    print(f"  with 1-R²:  {have_r2}")


if __name__ == "__main__":
    main()
