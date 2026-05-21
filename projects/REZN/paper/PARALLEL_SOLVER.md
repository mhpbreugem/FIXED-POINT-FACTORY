# PARALLEL SOLVER INSTRUCTIONS — 4 TERMINALS
# Updated 2026-05-03
# Repo: github.com/mhpbreugem/REZN
# Each terminal: `git pull origin main` first, then execute its task

---

## SHARED PARAMETERS (ALL TERMINALS)
```
G = 20, UMAX = 5, trim = 0.05
Precision: mp50 (50 decimal digits)
Tolerance: ||F||∞ < 1e-25 (NOT machine precision)
Weighting: ex-ante probability w = 0.5*(f0³ + f1³)
Seed: results/full_ree/posterior_v3_G20_umax5_notrim_mp300.json (γ=0.5 τ=2)
Task 2 checkpoints: results/full_ree/ (γ=1.0, 2.0, 4.0 at τ=2, already converged)
```

---

## TERMINAL 1 (EXISTING) — γ=0.5 τ-sweep (high τ)

You already have τ=0.3,0.5,0.8,1.0,1.5,2.0,3.0 converged.
τ=4.0 is stuck (boundary issue). Try:

**Option A:** Skip τ=4.0, try τ=5.0 warm-started from τ=3.0
**Option B:** Use mp30 instead of mp50 for τ≥4 (fewer digits, faster bisection)
**Option C:** Reduce UMAX to 4 for high-τ runs (denser grid where it matters)

Remaining τ values needed: [4.0, 5.0, 7.0, 10.0, 15.0]

**Warm-start chain:**
```
τ=3.0 (done) → τ=5.0 → τ=7.0 → τ=10.0 → τ=15.0
```

If all high-τ bail, that's fine — the γ=0.5 curve has 7 points already
which is enough for the figure. Move on to Fig R2 (lognormal, no-learning).

**Save each to:** results/full_ree/task3_g050_tXXXX_mp50.json
**Measure:** weighted 1-R² using the formula in FIGURES_TODO.md

---

## TERMINAL 2 (NEW) — γ=1.0 τ-sweep

`git pull origin main && read FIGURES_TODO.md`

**Seed:** Use the γ=1.0 τ=2 checkpoint from Task 2.
Find it at: results/full_ree/ (look for G20 γ=1.0 or g1.0 mp50 checkpoint)

**τ values:** [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
τ=2.0 is already done (from Task 2). Walk up and down:

```
τ=2.0 (done) → τ=1.5 → τ=1.0 → τ=0.8 → τ=0.5 → τ=0.3   (walk down)
τ=2.0 (done) → τ=3.0 → τ=4.0 → τ=5.0 → τ=7.0 → τ=10.0 → τ=15.0  (walk up)
```

Each step: interpolate previous τ's converged μ* as initial guess.
Run Newton-Krylov (or LM) to ||F|| < 1e-25.
Measure weighted 1-R² at convergence.

**Save each to:** results/full_ree/task3_g100_tXXXX_mp50.json
**Format:** same as existing Task 2 checkpoints (G, tau, gamma, dps, F_max, u_grid, p_grid, mu_strings)

**After all τ done, also save:**
results/full_ree/fig4A_g100_tau_sweep.json with:
```json
{"gamma": 1.0, "points": [{"tau": 0.3, "1-R2": ...}, ...]}
```

**If a τ point bails (||F|| > 0.1 after 15 iterations):** skip it, move to next.

---

## TERMINAL 3 (NEW) — γ=4.0 τ-sweep

`git pull origin main && read FIGURES_TODO.md`

**Seed:** Use the γ=4.0 τ=2 checkpoint from Task 2.

**τ values:** [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]

Same warm-start chain strategy as Terminal 2.

γ=4.0 is closest to CARA so convergence should be FASTEST here.
The 1-R² values will be small (~0.001-0.05) — that's correct.

**Save each to:** results/full_ree/task3_g400_tXXXX_mp50.json

**After all τ done, also save:**
results/full_ree/fig4A_g400_tau_sweep.json

---

## TERMINAL 4 (NEW) — Figure extraction & no-learning tasks

`git pull origin main && read FIGURES_TODO.md`

This terminal does NOT run convergence. It extracts figures from
existing checkpoints and computes no-learning results.

### Task 4a: Fig 5 REDO — asymmetric triples
Load seed (G=20 umax=5 γ=0.5 τ=2 mp300).
For Fig 5, use ASYMMETRIC triples:
- Fix u₁ = closest grid point to 1.0
- Fix u₂ = closest grid point to -1.0
- Vary u₃ from -3 to +3 (50 evenly-spaced points)
- T* = τ(u₁ + u₂ + u₃) varies
- At each u₃: compute p_FR = Λ(T*/3), p_NL (solve MC with private priors), p_REE (solve MC with μ*)
- Save: results/full_ree/fig5_G20_asymmetric_pgfplots.tex

### Task 4b: Fig 6B REDO — from G=18 umax=4
The G=20 umax=5 posteriors have artifacts (flat regions, kinks).
Use the G=18 umax=4 mp300 checkpoint instead:
results/full_ree/posterior_v3_G18_mp300_notrim.json

- Fix u₁ = closest to +1, u₂ = closest to -1
- Vary u₃ to sweep T* from -3 to +4 (50 points)
- Compute μ₁, μ₂, p at each T*
- Save: results/full_ree/fig6B_G18_pgfplots.tex

### Task 4c: Fig R2 — lognormal no-learning
Repeat the knife-edge computation but with lognormal payoff:
v ~ LogNormal(0, 1) instead of v ∈ {0,1}.
- γ ∈ {0.5, 1.0, 4.0}
- τ sweep: 15-20 values from 0.1 to 10
- Compute no-learning weighted 1-R² at each (γ, τ)
- Save: results/full_ree/figR2_G20_pgfplots.tex

### Task 4d: Fig 10 — convergence path
Extract iteration history from the mp300 convergence run.
Plot ||F||∞ vs iteration number.
Save: results/full_ree/fig10_convergence_pgfplots.tex

---

## GIT WORKFLOW FOR PARALLEL TERMINALS

CRITICAL: To avoid merge conflicts, each terminal works on DIFFERENT files.

Terminal 1: results/full_ree/task3_g050_*
Terminal 2: results/full_ree/task3_g100_*
Terminal 3: results/full_ree/task3_g400_*
Terminal 4: results/full_ree/fig5_*, fig6B_G18_*, figR2_*, fig10_*

Before each `git push`:
```
git pull origin main --rebase
git add results/full_ree/YOUR_FILES_ONLY
git commit -m "Terminal N: description"
git push origin YOUR_BRANCH
```

Each terminal should work on its own branch:
- Terminal 1: claude/add-claude-documentation-xGy0S (existing)
- Terminal 2: claude/task3-gamma1
- Terminal 3: claude/task3-gamma4
- Terminal 4: claude/extractions

---

## EXPECTED TIMELINE
- Terminal 1 (γ=0.5 high-τ): 3-6 hours or bail
- Terminal 2 (γ=1.0 full sweep): 6-10 hours
- Terminal 3 (γ=4.0 full sweep): 3-6 hours (fastest, closest to CARA)
- Terminal 4 (extractions): 1-2 hours (no convergence needed)

## WHEN DONE
Push all results. I will pull from all branches and assemble Fig 4A.

---

## STATUS UPDATE (2026-05-03)

### Terminal 2 (γ=1.0): DONE for τ≤4. Stop.
τ=0.3 to 3.0 have ||F|| < 1e-12 — excellent.
τ=4.0 has ||F|| = 2.3e-5 — acceptable.
τ≥5 did NOT converge (||F|| > 0.001) — DISCARD those points.
→ You have 8 usable points. STOP this terminal.

### Terminal 3 (γ=4.0): REDO WITH CORRECT SOLVER
You used K3 staggered NK float64. That is the WRONG solver.
The correct solver is the posterior-method v3 (code in python/).
||F|| = 0.001-0.013 everywhere — nothing converged.
→ REDO all 12 τ values using posterior-method v3 with mp50.
→ Warm-start from the γ=4.0 Task 2 checkpoint.
→ γ=4.0 is closest to CARA — should converge fastest.

### Terminal 1 (γ=0.5 high-τ): STOP, MOVE ON
τ≥4 won't converge at G=20. Accept 7 points (τ=0.3 to 3.0).
→ Switch to helping Terminal 3 or doing Fig R2.

### Terminal 4 (extractions): ✅ COMPLETE
All figures extracted. No further work needed.

## WHAT WE HAVE FOR FIG 4A

| τ | γ=0.5 | γ=1.0 | γ=4.0 |
|---|---|---|---|
| 0.3 | 0.037 ✅ | 0.030 ✅ | 🔲 REDO |
| 0.5 | 0.038 ✅ | 0.026 ✅ | 🔲 |
| 0.8 | 0.047 ✅ | 0.031 ✅ | 🔲 |
| 1.0 | 0.052 ✅ | 0.033 ✅ | 🔲 |
| 1.5 | 0.064 ✅ | 0.040 ✅ | 🔲 |
| 2.0 | 0.088 ✅ | 0.046 ✅ | 0.016 ✅ (Task 2) |
| 3.0 | 0.138 ✅ | 0.066 ✅ | 🔲 |
| 4.0 | ❌ stuck | 0.090 ⚠️ | 🔲 |

We can build Fig 4A now with γ=0.5 (7 pts) and γ=1.0 (7-8 pts).
γ=4.0 needs Terminal 3 redo.
