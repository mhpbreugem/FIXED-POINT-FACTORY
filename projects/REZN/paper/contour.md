# The Contour Method — How It Works

## Setup

Agent 1 has signal u₁ = 1. The realized signals are (u₁, u₂, u₃) = (1, −1, 1). The price is p = P[i,j,l], say 0.648.

Agent 1 knows two things: her own signal u₁ = 1, and the price p = 0.648. She does NOT know u₂ or u₃.

She knows the price function P. So she knows that whatever u₂, u₃ are, they must satisfy P(1, u₂, u₃) = 0.648. This defines a curve in (u₂, u₃) space — the **contour**.

Her job: integrate the signal density along this curve under each state v ∈ {0,1} to form her posterior.

## The Slice

She pulls out her slice of P: fix the first index at i (her signal), get P[i, :, :]. This is a G × G matrix — prices as a function of (u₂, u₃).

For G = 5 with grid [−2, −1, 0, 1, 2], her slice looks like:

```
        u₃= -2     -1      0      1      2
u₂=-2  0.331   0.331   0.342   0.496   0.645
u₂=-1  0.342   0.352   0.500   0.648 ← 0.658
u₂= 0  0.496   0.500   0.595   0.727   0.738
u₂=+1  0.645   0.648   0.727   0.881   0.902
u₂=+2  0.655   0.658   0.738   0.902   0.931
```

The arrow marks the actual realization (u₂, u₃) = (−1, 1) with price 0.648.

## Pass A: Sweep u₂ on Grid, Find u₃* Off Grid

For each grid value of u₂, look at the row P[i, j', :] as a 1D function of u₃. Find where it crosses 0.648.

**u₂ = −2:** row is [0.331, 0.331, 0.342, 0.496, 0.645]. All below 0.648. No crossing.

**u₂ = −1:** row is [0.342, 0.352, 0.500, 0.648, 0.658]. Crosses 0.648 between u₃ = 0 (0.500) and u₃ = 1 (0.648). Linear interpolation:

    u₃* = 0 + (0.648 − 0.500)/(0.648 − 0.500) × 1 = 1.000

So the crossing is at (u₂, u₃*) = (−1, 1). This IS the actual realization — as expected.

**u₂ = 0:** row is [0.496, 0.500, 0.595, 0.727, 0.738]. Crosses between u₃ = −1 (0.500) and u₃ = 0 (0.595):

    u₃* = −1 + (0.648 − 0.500)/(0.595 − 0.500) × 1 ≈ 0.558

This is **off grid**. The conjectured signal u₃* = 0.558 is not one of the grid points. That's the whole point.

Continue for u₂ = 1: find u₃* ≈ −1.0. For u₂ = 2: no crossing (all prices above 0.648).

Each crossing gives a pair (u₂_grid, u₃_off-grid).

## Pass B: Sweep u₃ on Grid, Find u₂* Off Grid

Same procedure but on columns. For each grid value of u₃, look at P[i, :, l'] as a function of u₂, find where it crosses 0.648.

This gives pairs (u₂_off-grid, u₃_grid).

The two passes trace the same contour from orthogonal directions. Averaging removes directional bias.

## The Integral

At each crossing point (u₂ᶜ, u₃ᶜ) — one coordinate on grid, one off — compute the joint signal density under each state:

    f_v(u₂ᶜ) · f_v(u₃ᶜ)

These are the **prior** (ex ante) densities, not posteriors. They measure how likely this particular pair of other-agent signals is under state v.

Sum over all crossings:

    Pass A sum:  S_v^A = Σ_{j'} f_v(u_{j'}_grid) · f_v(u_{l'}_off-grid)
    Pass B sum:  S_v^B = Σ_{l'} f_v(u_{j'}_off-grid) · f_v(u_{l'}_grid)

Average:

    A_v = (S_v^A + S_v^B) / 2

## Why No Jacobian?

Each crossing contributes one term to the sum. We are not integrating a continuous function with du — we are summing over discrete sweep points. The grid spacing is the same for every crossing, so it factors out of the ratio A₁/A₀. The Jacobian |∂P/∂u₃| would rescale each term by how steep the price function is at that crossing, but since it enters both A₁ and A₀ identically, it cancels in the posterior. (This is approximate — strictly the Jacobian differs between v=0 and v=1 — but at finite G the sum-without-Jacobian is a valid discrete approximation.)

## Bayes' Rule

Agent 1 combines the contour evidence with her own signal:

    μ₁ = f₁(u₁) · A₁ / (f₀(u₁) · A₀ + f₁(u₁) · A₁)

Three likelihoods enter:

1. f_v(u₁) — her own signal (actual, known)
2. f_v(u₂ᶜ) — conjectured signal of agent 2 (from the crossing)
3. f_v(u₃ᶜ) — conjectured signal of agent 3 (from the crossing)

The contour integral A_v aggregates (2) and (3) over all consistent conjectures. The own signal (1) multiplies the whole thing.

## Why CARA Gives Full Revelation

Under CARA, the price is logit(p) = (1/3) Σ τuₖ. The contour P(u₁, u₂, u₃) = p is τu₂ + τu₃ = const — a straight line. Every crossing has the same T* = τ(u₁ + u₂ + u₃). The density ratio f₁(u₂ᶜ)f₁(u₃ᶜ) / f₀(u₂ᶜ)f₀(u₃ᶜ) = exp(τ(u₂ᶜ + u₃ᶜ)) = exp(T* − τu₁) is constant along the contour. So A₁/A₀ reveals T* exactly. Combined with the own signal, the agent learns everything.

## Why CRRA Gives Partial Revelation

Under CRRA, the price aggregates in probability space: p ≈ (1/3) Σ Λ(τuₖ). The contour Λ(τu₂) + Λ(τu₃) = const is curved (because Λ is nonlinear). Different crossings have different T*. Some crossings favor v=1, others favor v=0. The agent cannot tell them apart from the price alone. The integral A₁/A₀ is an average over these different T* values — informative but not fully revealing.

## Market Clearing

All three agents do this independently on their own slices. Agent 2 uses P[:, j, :], agent 3 uses P[:, :, l]. Each gets her own posterior μ₁, μ₂, μ₃. Then solve:

    Σₖ xₖ(μₖ, p_new) = 0

for the new price. Under CARA, all three posteriors are equal (full revelation) so the new price equals the common posterior. Under CRRA, they disagree (partial revelation) and the market clearing price is a nonlinear compromise.

## The Fixed Point

The new prices P_new feed back into the contour computation. Iterate until P = Φ(P). Picard does this with damping. Anderson/Newton does it in one shot.

## Why the System Decouples

There is only ONE unknown object: the price array P[i,j,l]. Everything else — the root-found signals, the contour integrals, the posteriors — is a FUNCTION of P. They are not independent unknowns. They are intermediate computations.

Think of it as a single function Φ that maps a price array to a new price array:

    P_new = Φ(P)

Here is exactly what Φ does, step by step, given P as input:

    Step 1: For each grid point (i,j,l), read off the price p = P[i,j,l].

    Step 2: For agent 1 at signal u_i, extract her slice P[i,:,:].
            Sweep u₂ on grid, for each one solve P[i, u₂, u₃*] = p for u₃*.
            This is a 1D root-find. The root u₃* is determined by P — it is not free.
            
    Step 3: At each crossing, evaluate f_v(u₂) · f_v(u₃*) and sum.
            This gives A_v. Again, determined by P.

    Step 4: Compute μ₁ = f₁(u₁)·A₁ / (f₀(u₁)·A₀ + f₁(u₁)·A₁).
            Determined by P.

    Step 5: Repeat steps 2-4 for agents 2 and 3 on their own slices.

    Step 6: Solve market clearing Σ x_k(μ_k, p_new) = 0 for p_new.
            This is the output: P_new[i,j,l] = p_new.

Every step takes P as input and produces a deterministic output. There are no choices, no degrees of freedom, no additional unknowns. The root-found u₃* is not an unknown — it is the unique solution of P[i, u₂, u₃*] = p, which is fully determined once you know P.

The equilibrium condition is:

    P = Φ(P)

This is G³ scalar equations in G³ scalar unknowns. Solve it with:

- **Picard:** P^(n+1) = α·Φ(P^(n)) + (1−α)·P^(n), iterate.
- **Anderson:** Same Φ evaluations, but use past residuals to build a quasi-Newton step.
- **Newton:** Solve F(P) = P − Φ(P) = 0 directly. The Jacobian ∂F/∂P is G³ × G³.

In all three cases, ONE call to Φ means: loop over all G³ grid points, for each do the 2-pass contour (6G root-finds), compute 3 posteriors, solve 1 market clearing. Then you have the new P. That is one "iteration" or one "function evaluation."

The table of equations D1-D6, E1 in the session summary lists the intermediate steps INSIDE Φ. They are not separate equations to be solved simultaneously with P. They are the recipe for computing Φ(P) given P.
