# REZN — REE with No Noise

**Paper**: "On the Possibility of Informationally Inefficient Markets Without Noise"  
**Author**: Matthijs Breugem (Nyenrode Business University)  
**SSRN**: https://ssrn.com/abstract=6693198  
**Status**: Preliminary and Incomplete (figures being computed)

---

## What this project computes

The paper proves that CARA (exponential) utility is the **unique** preference
class that gives full revelation of information through prices. Every other
CRRA preference gives **partial** revelation — without noise traders, supply
shocks, or random endowments. The Grossman-Stiglitz paradox is resolved without
invoking irrationality.

The numerical evidence: for each (γ, τ) pair, solve a posterior-function fixed
point on a 2-D grid and measure weighted 1-R² of the REE price on the sufficient
statistic T* = τ(u₁ + u₂ + u₃). Under CARA: 1-R² = 0 exactly. Under CRRA: 1-R² > 0.

---

## Model

- Binary asset: v ∈ {0,1}, prior P(v=1) = ½
- K = 3 CRRA agents with common risk aversion γ
- Signals: s_k = v + ε_k, ε_k iid N(0, 1/τ); centred u_k = s_k − ½
- NO noise traders, NO supply shocks, NO random endowments
- Market clearing among the three agents determines price
- Unknown: μ*(u, p) = P(v=1 | own signal u, price p) — the REE posterior function

Full equations: see `EQUATIONS.md`.

---

## Task queue status

See `TASK_QUEUE.json` for the current state.

| γ     | τ-sweep range | Status |
|-------|---------------|--------|
| 0.25  | 0.3–3.0       | γ=2.0 root ready; chain blocked on it |
| 0.5   | 0.3–3.0       | ✓ done (7 points); τ≥4 bailed (boundary) |
| 1.0   | 0.3–4.0       | ✓ done (8 points); τ≥5 bailed |
| 2.0   | 0.3–5.0       | root ready; chain blocked on it |
| 4.0   | 0.3–10.0+     | τ≤7 done; τ=10 ready; τ=15 blocked |

Lognormal Fig R2 tasks (no-learning, no warm-start needed): ready for γ ∈ {0.5, 1.0, 4.0}.

---

## Key numerical findings

- **Fig 4B (γ-sweep at τ=2)**: weighted 1-R² decreases monotonically with γ.
  Amplification (REE/no-learning ratio) increases with γ.
- **γ=4.0 τ-sweep**: 1-R² has a minimum near τ=2 (1-R²≈0.016), rising on both sides.
- **Verified anchor** (γ=0.5, τ=2): 1-R² ≈ 0.085, slope ≈ 0.543 (weighted; stable across grid choices).

---

## Files

| File                      | Purpose |
|---------------------------|---------|
| `EQUATIONS.md`            | Complete model and fixed-point system |
| `CHECKPOINT_FORMAT.md`    | JSON schema for solver output files |
| `TASK_QUEUE.json`         | Task list with statuses and results |
| `SOLVER_INSTRUCTIONS.md`  | REZN-specific solver addendum |
| `solver_code/`            | Solver implementation (solve.py entry point) |
| `checkpoints/`            | Symlink target or local checkpoint storage |
| `heartbeats/`             | VM heartbeat files (one per active worker) |

---

## Running a solver VM

```bash
GITHUB_TOKEN=ghp_xxx \
PROJECT=REZN \
BRANCH=main \
VM_NAME=solver-1 \
bash core/create_gcp_vm.sh
```

Monitor:

```bash
python3 core/supervisor.py --project REZN --pull
```

---

## Seed checkpoint

The reference starting point is in the REZN repository at:

    results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json

γ = 0.5, τ = 2, G = 20, UMAX = 5, dps = 300, ||F|| = 7.4e-119.
All subsequent checkpoints warm-start from this or from each other.
