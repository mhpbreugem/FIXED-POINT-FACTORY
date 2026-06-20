# Referee Report — 20x5 (gamma, tau) Anchor Sweep, K=3 CRRA REE (phi_K3_halo_smooth)

**Referee:** independent re-adjudication (Fable), 2026-06-10
**Data:** `/home/user/FIXED-POINT-FACTORY/projects/REZN/solved_fixed_points/dd_k3_overnight/anchor_sweep_20x5/grid.json` (100 cells, sweep complete, total wall ~67 min)
**Supporting evidence used:** deep-ladder anchor `coarea_limit_repro/results.json` (tau=2, gamma=0.1, G=9..25), independent cold-start runs `anchor_grid/grid.json` (9 cells), method cross-check `anchor_crosscheck/results.json`, solver source `/tmp/big_sweep.py`, live log `/tmp/big_sweep.log`. No solver was re-run; judgments are from recorded evidence only.

## 1. Verdict criteria

A cell is judged on whether its reported (slope, deficit) can be trusted as an estimate of the joint-limit (G -> inf, h = 0.45*sqrt(du) -> 0) anchor value, not merely on its own Newton residual (pitfall 1).

- **C1** `F_final < 1e-10` at G=21 (true convergence; the sweep's own `ok` bar of 1e-8 is looser).
- **C2** every ladder rung converged (`F < 1e-8` at G = 9, 13, 17, 21).
- **C3** resolution: conservative h-linear Richardson bound `err = s3 * h21/(h17-h21) = 8.47*|d21-d17|` must be < 15% of d21; ladder steps monotone in sign (unless all steps <= 2% of the value) and not growing (s3 <= 1.5*s2, waived when the last step is <= 1% of the value).
- **C4** not floor-contaminated: where a row exhibits a deficit floor (its minimum-deficit converged cell has local d log(deficit)/d log(gamma) > -0.5 at high gamma), any cell with deficit < 3x that floor is flagged.
- **C5** no kink in log(deficit) along the gamma-continuation chain between converged neighbours (|second difference| < 1.0).
- **C6** not downstream of a stalled cell in the warm-start chain (pitfall 4; `big_sweep.py` passes the stalled iterate's `P_final` to the next gamma even when `F ~ 1e-2`).

**CERTIFIED** = C1-C6. **PROVISIONAL** = C1, C2, C6 hold (a genuine machine-eps fixed point of the *discretized* operator at G=21) but the deficit/slope cannot be certified as joint-limit estimates (C3/C4/C5 failed). **REJECTED** = stalled Newton on the final ladder, or contaminated warm start.

**Calibration of C3 against the only fully-trusted object.** The deep anchor (tau=2, gamma=0.1) has ladder steps 0.0074, 0.0025, 0.0021 (G9->13->17->21) on deficit 0.2793, i.e. Richardson err 6.2%; it passes with margin. Its *observed* G21->G25 step (0.0028) exceeded the h^2/du-Richardson prediction (0.0014) by 2x, showing empirical convergence is no faster than O(h) here — the 8.47x multiplier is the matching, not an excessive, bound. For the steadily contracting tau=0.2 ladders (step ratio ~0.70, consistent with O(h^2)=O(du) decay) the geometric tail estimate is ~2.4x the last step instead of 8.47x; even under that friendlier estimator the mid-row tau=0.2 cells sit at 11-15% and the verdict map changes by at most ~4 borderline cells. Threshold sensitivity: moving the bar from 15% to 25% would add only 3 cells (t0.2 g0.137 @18%, t0.5 g0.192 @16%, t2.0 g0.192 @20%).

## 2. Verdict table (all 100 cells)

F = final Newton residual at G=21; err = conservative h-extrapolation error bound on deficit (C3). 

| tau | gamma | F_final | slope | deficit | verdict | reason |
|----:|------:|--------:|------:|--------:|:--------|:-------|
| 0.2 | 0.0500 | 1.2e-14 | 0.0766 | 1.698e-01 | CERT | F=1e-14; Richardson err ~2.5% |
| 0.2 | 0.0700 | 8.9e-15 | 0.0746 | 1.491e-01 | CERT | F=9e-15; Richardson err ~5.1% |
| 0.2 | 0.0980 | 6.8e-15 | 0.0726 | 1.183e-01 | CERT | F=7e-15; Richardson err ~9.8% |
| 0.2 | 0.1373 | 7.8e-15 | 0.0708 | 7.870e-02 | PROV | under-resolved: Richardson err ~18% |
| 0.2 | 0.1922 | 6.0e-15 | 0.0695 | 4.036e-02 | PROV | under-resolved: Richardson err ~30% |
| 0.2 | 0.2692 | 7.9e-15 | 0.0688 | 1.534e-02 | PROV | under-resolved: Richardson err ~42% |
| 0.2 | 0.3769 | 6.3e-15 | 0.0686 | 4.717e-03 | PROV | under-resolved: Richardson err ~48% |
| 0.2 | 0.5278 | 6.8e-15 | 0.0688 | 1.421e-03 | PROV | under-resolved: Richardson err ~47% |
| 0.2 | 0.7391 | 8.2e-15 | 0.0691 | 5.251e-04 | PROV | under-resolved: Richardson err ~42% |
| 0.2 | 1.0350 | 7.0e-15 | 0.0694 | 2.559e-04 | PROV | under-resolved: Richardson err ~39% |
| 0.2 | 1.4493 | 6.3e-15 | 0.0697 | 1.442e-04 | PROV | under-resolved: Richardson err ~43% |
| 0.2 | 2.0294 | 7.9e-15 | 0.0700 | 8.277e-05 | PROV | under-resolved: Richardson err ~49% |
| 0.2 | 2.8418 | 6.3e-15 | 0.0702 | 4.616e-05 | PROV | under-resolved: Richardson err ~55% |
| 0.2 | 3.9793 | 6.2e-15 | 0.0704 | 2.482e-05 | PROV | under-resolved: Richardson err ~62% |
| 0.2 | 5.5722 | 8.4e-15 | 0.0705 | 1.296e-05 | PROV | under-resolved: Richardson err ~64% |
| 0.2 | 7.8027 | 6.1e-15 | 0.0706 | 6.734e-06 | PROV | under-resolved: Richardson err ~53% |
| 0.2 | 10.9261 | 6.7e-15 | 0.0706 | 3.647e-06 | PROV | within 3x of row deficit floor (~1.5e-06); smoothing bias suppresses curvature |
| 0.2 | 15.2997 | 6.7e-15 | 0.0707 | 2.231e-06 | PROV | under-resolved: Richardson err ~73%, ladder step sign reversal, ladder steps growing (ratio 8.1); within 3x of row deficit floor (~1.5e-06); smoothing bias suppresses curvature |
| 0.2 | 21.4241 | 8.4e-15 | 0.0707 | 1.659e-06 | PROV | under-resolved: Richardson err ~181%, ladder step sign reversal, ladder steps growing (ratio 2.2); within 3x of row deficit floor (~1.5e-06); smoothing bias suppresses curvature |
| 0.2 | 30.0000 | 6.3e-15 | 0.0707 | 1.489e-06 | PROV | under-resolved: Richardson err ~264%, ladder steps growing (ratio 1.7); within 3x of row deficit floor (~1.5e-06); smoothing bias suppresses curvature |
| 0.5 | 0.0500 | 7.4e-15 | 0.1435 | 1.860e-01 | CERT | F=7e-15; Richardson err ~2.3% |
| 0.5 | 0.0700 | 7.4e-15 | 0.1447 | 1.731e-01 | CERT | F=7e-15; Richardson err ~3.2% |
| 0.5 | 0.0980 | 5.1e-13 | 0.1464 | 1.557e-01 | CERT | F=5e-13; Richardson err ~5.1% |
| 0.5 | 0.1373 | 2.1e-13 | 0.1486 | 1.329e-01 | CERT | F=2e-13; Richardson err ~8.7% |
| 0.5 | 0.1922 | 1.1e-13 | 0.1518 | 1.045e-01 | PROV | under-resolved: Richardson err ~16% |
| 0.5 | 0.2692 | 8.2e-14 | 0.1560 | 7.263e-02 | PROV | under-resolved: Richardson err ~27% |
| 0.5 | 0.3769 | 8.1e-14 | 0.1609 | 4.341e-02 | PROV | under-resolved: Richardson err ~42% |
| 0.5 | 0.5278 | 4.3e-14 | 0.1663 | 2.297e-02 | PROV | under-resolved: Richardson err ~54% |
| 0.5 | 0.7391 | 7.3e-14 | 0.1716 | 1.193e-02 | PROV | under-resolved: Richardson err ~63% |
| 0.5 | 1.0350 | 5.6e-14 | 0.1765 | 6.577e-03 | PROV | under-resolved: Richardson err ~71% |
| 0.5 | 1.4493 | 1.2e-13 | 0.1808 | 3.793e-03 | PROV | under-resolved: Richardson err ~81% |
| 0.5 | 2.0294 | 1.6e-13 | 0.1844 | 2.203e-03 | PROV | under-resolved: Richardson err ~82% |
| 0.5 | 2.8418 | 2.0e-13 | 0.1874 | 1.288e-03 | PROV | under-resolved: Richardson err ~57%; within 3x of row deficit floor (~5.0e-04); smoothing bias suppresses curvature |
| 0.5 | 3.9793 | 2.2e-13 | 0.1896 | 8.007e-04 | PROV | under-resolved: Richardson err ~24%, ladder step sign reversal; within 3x of row deficit floor (~5.0e-04); smoothing bias suppresses curvature |
| 0.5 | 5.5722 | 2.6e-13 | 0.1914 | 5.763e-04 | PROV | under-resolved: Richardson err ~163%, ladder step sign reversal, ladder steps growing (ratio 210.4); within 3x of row deficit floor (~5.0e-04); smoothing bias suppresses curvature |
| 0.5 | 7.8027 | 2.7e-13 | 0.1927 | 5.007e-04 | PROV | under-resolved: Richardson err ~303%, ladder step sign reversal, ladder steps growing (ratio 3.1); within 3x of row deficit floor (~5.0e-04); smoothing bias suppresses curvature |
| 0.5 | 10.9261 | 3.0e-13 | 0.1936 | 5.001e-04 | PROV | under-resolved: Richardson err ~390%, ladder steps growing (ratio 2.3); within 3x of row deficit floor (~5.0e-04); smoothing bias suppresses curvature |
| 0.5 | 15.2997 | 3.1e-13 | 0.1943 | 5.313e-04 | PROV | under-resolved: Richardson err ~426%, ladder steps growing (ratio 2.1); within 3x of row deficit floor (~5.0e-04); smoothing bias suppresses curvature |
| 0.5 | 21.4241 | 3.1e-13 | 0.1948 | 5.715e-04 | PROV | under-resolved: Richardson err ~437%, ladder steps growing (ratio 1.9); within 3x of row deficit floor (~5.0e-04); smoothing bias suppresses curvature |
| 0.5 | 30.0000 | 3.2e-13 | 0.1952 | 6.100e-04 | PROV | under-resolved: Richardson err ~437%, ladder steps growing (ratio 1.9); within 3x of row deficit floor (~5.0e-04); smoothing bias suppresses curvature |
| 1.0 | 0.0500 | 7.2e-15 | 0.2186 | 2.176e-01 | CERT | F=7e-15; Richardson err ~2.5% |
| 1.0 | 0.0700 | 6.6e-15 | 0.2213 | 2.121e-01 | CERT | F=7e-15; Richardson err ~3.8% |
| 1.0 | 0.0980 | 6.9e-15 | 0.2250 | 2.046e-01 | CERT | F=7e-15; Richardson err ~2.8% |
| 1.0 | 0.1373 | 6.9e-15 | 0.2301 | 1.936e-01 | CERT | F=7e-15; Richardson err ~0.4% |
| 1.0 | 0.1922 | 7.0e-15 | 0.2375 | 1.777e-01 | CERT | F=7e-15; Richardson err ~4.2% |
| 1.0 | 0.2692 | 7.0e-15 | 0.2479 | 1.552e-01 | CERT | F=7e-15; Richardson err ~14.3% |
| 1.0 | 0.3769 | 6.9e-15 | 0.2621 | 1.263e-01 | PROV | under-resolved: Richardson err ~32% |
| 1.0 | 0.5278 | 6.8e-15 | 0.2798 | 9.421e-02 | PROV | under-resolved: Richardson err ~58% |
| 1.0 | 0.7391 | 6.9e-15 | 0.2997 | 6.442e-02 | PROV | under-resolved: Richardson err ~89% |
| 1.0 | 1.0350 | 6.8e-15 | 0.3199 | 4.144e-02 | PROV | under-resolved: Richardson err ~117% |
| 1.0 | 1.4493 | 6.8e-15 | 0.3388 | 2.613e-02 | PROV | under-resolved: Richardson err ~128%; within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.0 | 2.0294 | 6.7e-15 | 0.3552 | 1.690e-02 | PROV | under-resolved: Richardson err ~96%; within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.0 | 2.8418 | 6.6e-15 | 0.3690 | 1.188e-02 | PROV | under-resolved: ladder step sign reversal; within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.0 | 3.9793 | 6.8e-15 | 0.3800 | 9.628e-03 | PROV | under-resolved: Richardson err ~176%, ladder step sign reversal, ladder steps growing (ratio 4.7); within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.0 | 5.5722 | 6.7e-15 | 0.3887 | 9.078e-03 | PROV | under-resolved: Richardson err ~331%, ladder step sign reversal, ladder steps growing (ratio 5.1); within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.0 | 7.8027 | 7.5e-15 | 0.3953 | 9.456e-03 | PROV | under-resolved: Richardson err ~425%, ladder step sign reversal, ladder steps growing (ratio 3.0); within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.0 | 10.9261 | 8.5e-15 | 0.4002 | 1.024e-02 | PROV | under-resolved: Richardson err ~465%, ladder steps growing (ratio 2.5); within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.0 | 15.2997 | 9.5e-15 | 0.4039 | 1.112e-02 | PROV | under-resolved: Richardson err ~477%, ladder steps growing (ratio 2.3); within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.0 | 21.4241 | 1.0e-14 | 0.4065 | 1.193e-02 | PROV | under-resolved: Richardson err ~477%, ladder steps growing (ratio 2.2); within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.0 | 30.0000 | 1.1e-14 | 0.4084 | 1.261e-02 | PROV | under-resolved: Richardson err ~472%, ladder steps growing (ratio 2.1); within 3x of row deficit floor (~9.1e-03); smoothing bias suppresses curvature |
| 1.5 | 0.0500 | 1.8e-14 | 0.2921 | 2.553e-01 | CERT | F=2e-14; Richardson err ~0.5% |
| 1.5 | 0.0700 | 1.1e-14 | 0.2947 | 2.521e-01 | CERT | F=1e-14; Richardson err ~1.4% |
| 1.5 | 0.0980 | 7.5e-15 | 0.2986 | 2.475e-01 | CERT | F=8e-15; Richardson err ~2.9% |
| 1.5 | 0.1373 | 7.0e-15 | 0.3046 | 2.401e-01 | CERT | F=7e-15; Richardson err ~4.9% |
| 1.5 | 0.1922 | 1.4e-14 | 0.3134 | 2.280e-01 | CERT | F=1e-14; Richardson err ~10.5% |
| 1.5 | 0.2692 | 1.8e-13 | 0.3264 | 2.089e-01 | PROV | under-resolved: Richardson err ~25%, ladder steps growing (ratio 1.6) |
| 1.5 | 0.3769 | 7.1e-15 | 0.3447 | 1.812e-01 | PROV | under-resolved: Richardson err ~53%, ladder steps growing (ratio 1.7) |
| 1.5 | 0.5278 | 7.0e-15 | 0.3690 | 1.471e-01 | PROV | under-resolved: Richardson err ~93%, ladder steps growing (ratio 1.6) |
| 1.5 | 0.7391 | 6.8e-15 | 0.3976 | 1.122e-01 | PROV | under-resolved: Richardson err ~134% |
| 1.5 | 1.0350 | 7.1e-15 | 0.4287 | 8.145e-02 | PROV | under-resolved: Richardson err ~158% |
| 1.5 | 1.4493 | 7.1e-15 | 0.4599 | 5.784e-02 | PROV | under-resolved: Richardson err ~148% |
| 1.5 | 2.0294 | 1.1e-13 | 0.4892 | 4.221e-02 | PROV | under-resolved: Richardson err ~81% |
| 1.5 | 2.8418 | 2.6e-13 | 0.5150 | 3.334e-02 | PROV | under-resolved: Richardson err ~49%, ladder step sign reversal |
| 1.5 | 3.9793 | 3.1e-02 | 0.5367 | 2.907e-02 | REJ  | stalled Newton (F_final=3e-02) |
| 1.5 | 5.5722 | 9.6e-04 | 0.5540 | 2.804e-02 | REJ  | stalled Newton (F_final=1e-03); downstream of stall (contaminated warm start) |
| 1.5 | 7.8027 | 3.1e-02 | 0.5678 | 2.838e-02 | REJ  | stalled Newton (F_final=3e-02); downstream of stall (contaminated warm start) |
| 1.5 | 10.9261 | 3.1e-02 | 0.5776 | 3.059e-02 | REJ  | stalled Newton (F_final=3e-02); downstream of stall (contaminated warm start) |
| 1.5 | 15.2997 | 3.2e-02 | 0.5865 | 3.050e-02 | REJ  | stalled Newton (F_final=3e-02); downstream of stall (contaminated warm start) |
| 1.5 | 21.4241 | 4.6e-02 | 0.5909 | 3.112e-02 | REJ  | stalled Newton (F_final=5e-02); downstream of stall (contaminated warm start) |
| 1.5 | 30.0000 | 5.0e-02 | 0.5952 | 3.416e-02 | REJ  | stalled Newton (F_final=5e-02); downstream of stall (contaminated warm start) |
| 2.0 | 0.0500 | 1.4e-13 | 0.3638 | 2.844e-01 | CERT | F=1e-13; Richardson err ~2.7% |
| 2.0 | 0.0700 | 5.9e-14 | 0.3663 | 2.828e-01 | CERT | F=6e-14; Richardson err ~3.6% |
| 2.0 | 0.0980 | 4.4e-14 | 0.3703 | 2.796e-01 | CERT | F=4e-14; Richardson err ~6.0% |
| 2.0 | 0.1373 | 3.4e-13 | 0.3766 | 2.731e-01 | CERT | F=3e-13; Richardson err ~10.4% |
| 2.0 | 0.1922 | 7.0e-15 | 0.3866 | 2.613e-01 | PROV | under-resolved: Richardson err ~20% |
| 2.0 | 0.2692 | 6.6e-13 | 0.4029 | 2.394e-01 | PROV | under-resolved: Richardson err ~50%, ladder steps growing (ratio 2.1) |
| 2.0 | 0.3769 | 7.8e-14 | 0.4267 | 2.068e-01 | PROV | under-resolved: Richardson err ~106%, ladder steps growing (ratio 2.3) |
| 2.0 | 0.5278 | 1.9e-14 | 0.4515 | 1.774e-01 | PROV | under-resolved: Richardson err ~129%, ladder steps growing (ratio 1.6) |
| 2.0 | 0.7391 | 3.2e-02 | 0.4787 | 1.483e-01 | REJ  | stalled Newton (F_final=3e-02) |
| 2.0 | 1.0350 | 3.1e-02 | 0.5373 | 2.496e-01 | REJ  | stalled Newton (F_final=3e-02); downstream of stall (contaminated warm start) |
| 2.0 | 1.4493 | 2.7e-02 | 0.5493 | 9.021e-02 | REJ  | stalled Newton (F_final=3e-02); downstream of stall (contaminated warm start) |
| 2.0 | 2.0294 | 4.0e-02 | 0.5885 | 6.864e-02 | REJ  | stalled Newton (F_final=4e-02); downstream of stall (contaminated warm start) |
| 2.0 | 2.8418 | 4.6e-01 | 0.6170 | 2.269e-01 | REJ  | stalled Newton (F_final=5e-01); downstream of stall (contaminated warm start) |
| 2.0 | 3.9793 | 2.5e-02 | 0.6607 | 4.220e-02 | REJ  | stalled Newton (F_final=3e-02); downstream of stall (contaminated warm start) |
| 2.0 | 5.5722 | 3.0e-02 | 0.6895 | 3.604e-02 | REJ  | stalled Newton (F_final=3e-02); downstream of stall (contaminated warm start) |
| 2.0 | 7.8027 | 9.1e-02 | 0.7148 | 2.886e-02 | REJ  | stalled Newton (F_final=9e-02); downstream of stall (contaminated warm start) |
| 2.0 | 10.9261 | 5.9e-02 | 0.7325 | 3.316e-02 | REJ  | stalled Newton (F_final=6e-02); downstream of stall (contaminated warm start) |
| 2.0 | 15.2997 | 4.7e-02 | 0.7444 | 3.524e-02 | REJ  | stalled Newton (F_final=5e-02); downstream of stall (contaminated warm start) |
| 2.0 | 21.4241 | 3.5e-02 | 0.7570 | 3.496e-02 | REJ  | stalled Newton (F_final=3e-02); downstream of stall (contaminated warm start) |
| 2.0 | 30.0000 | 3.7e-02 | 0.7653 | 3.568e-02 | REJ  | stalled Newton (F_final=4e-02); downstream of stall (contaminated warm start) |

**Counts: 22 CERTIFIED, 59 PROVISIONAL, 19 REJECTED.** The certified set is exactly the low-gamma head of each row (gamma <= 0.098 for tau=0.2; <= 0.137 for tau=0.5; <= 0.269 for tau=1.0; <= 0.192 for tau=1.5; <= 0.137 for tau=2.0).

## 3. Per-row Jensen fits (deficit ~ gamma^p; Jensen wedge predicts p = -1 at fixed tau, high gamma)

Fits are OLS of log(deficit) on log(gamma); +/- is the standard error of the slope.

| tau | CERTIFIED cells (all in low-gamma regime) | CERTIFIED with gamma >= 1 | exploratory tail, gamma >= 1 (PROVISIONAL, converged-F, non-floor, monotone ladder) |
|----:|:--|:--|:--|
| 0.2 | p = -0.54 +/- 0.09 (n=3, g in [0.05, 0.098]) | none | p = **-1.80 +/- 0.03** (n=7, g in [1.04, 7.80]) |
| 0.5 | p = -0.33 +/- 0.04 (n=4, g in [0.05, 0.137]) | none | p = **-1.63 +/- 0.01** (n=3, g in [1.04, 2.03]) |
| 1.0 | p = -0.19 +/- 0.03 (n=6, g in [0.05, 0.269]) | none | insufficient (n=1; floor at 9.1e-3 swallows the tail) |
| 1.5 | p = -0.08 +/- 0.01 (n=5, g in [0.05, 0.192]) | none | p = **-0.98 +/- 0.02** (n=3, g in [1.04, 2.03]) |
| 2.0 | p = -0.04 +/- 0.01 (n=4, g in [0.05, 0.137]) | none | insufficient (row stalls from g = 0.74) |

**Is p = -1 (Jensen) supported? Not at certification grade — there is not a single CERTIFIED cell with gamma >= 1 in the entire sweep**, so the 1/gamma law cannot be certified from this run. The certified low-gamma fits measure the saturation regime (deficit -> finite cap as gamma -> 0) and are *expected* to be shallow; they are not tests of Jensen. At exploratory (uncertified) grade the tail powers are tau-dependent: tau=1.5 agrees with -1 (p = -0.98 +/- 0.02, but n=3 over less than one decade of gamma), while tau=0.2 and tau=0.5 decay *significantly steeper* than -1 (p ~ -1.8 and -1.6, with formal SEs that ignore the 40-80% systematic h-extrapolation error on each point). Honest statement: the sweep is *consistent with* the deficit approaching const/gamma behaviour as tau grows, but it neither pins the exponent nor confirms universality of -1; at low tau the apparent exponent is steeper than Jensen until the discretization floor takes over.

## 4. Audits

### 4(a) tau=2.0 row vs the trusted anchor (deficit ~0.28 at gamma=0.1)

Sweep cell t2.0 g=0.0980 vs deep-ladder reproduction at gamma=0.100, rung by rung:

| G | sweep deficit | anchor deficit | diff | sweep slope | anchor slope |
|--:|--:|--:|--:|--:|--:|
| 9 | 0.291341 | 0.291199 | +0.000142 | 0.3222 | 0.3224 |
| 13 | 0.284000 | 0.283827 | +0.000173 | 0.3408 | 0.3410 |
| 17 | 0.281559 | 0.281339 | +0.000220 | 0.3564 | 0.3567 |
| 21 | 0.279564 | 0.279281 | +0.000283 | 0.3703 | 0.3706 |
| 25 | — | 0.276481 | | | 0.3838 |

Agreement is to ~0.1% at every rung, and the sign/size of the difference is exactly what the gamma offset (0.098 vs 0.100, d(deficit)/d(gamma) ~ -0.16 locally) predicts. **The sweep's tau=2.0 chain starts in the correct basin and reproduces the anchor.** The four certified tau=2.0 cells (g <= 0.137) inherit this verification by chain continuity. Beyond g=0.528 the row is dead: g=0.269-0.528 are under-resolved (Richardson err 50-129%, growing ladder steps), and from g=0.739 every cell stalls (F ~ 1e-2 even after the from-scratch fallback), exactly as the historical record warned (pitfall 5). Forensic confirmation that stalled numbers are meaningless: the independent cold-start run at (tau=2, g=1.0) also stalls but lands on deficit 0.1235 vs the chained sweep's 0.2496 at g=1.035 — two non-converged iterates, two different answers (pitfall 1 in action).

### 4(b) Fallback-branch usage

`big_sweep.py` runs the no-learning-init fallback ladder whenever the chained ladder ends with F > 1e-8, and keeps whichever residual is smaller — but **grid.json does not record which branch produced the stored ladder** (a bookkeeping deficiency; recommend adding a `fallback_used` flag). Consequences of the audit:
- All 19 stalled cells ran the fallback by construction; none reached F < 1e-8 either way, so no stalled cell is rescued. One partial success: t1.5 g=5.57 reached F = 9.6e-4 (vs ~3e-2 for its neighbours), still 8 orders short of certification.
- A *silent* fallback (chain stalls, fallback converges) would show as a converged cell with ~2x wall time. Scanning converged cells for wall > 2.5x row median flags only t1.5 g=2.8418 (72 s vs median 19 s). Its slope (0.5150) and deficit (0.03334) sit smoothly between neighbours (0.4892/0.04221 and the stalled 0.5367/0.02907), and its ladder is fully converged, so whether or not it came from the fallback it is continuous with the chain. No discontinuity attributable to a basin switch via fallback was found anywhere among converged cells.
- Independent cold-start cross-checks (anchor_grid, 7 converged cells at tau = 0.5, 1.0, 2.0 x g = 0.1, 1.0, 10.0) match the chained sweep to within the gamma-offset between 0.1/1.0/10.0 and the sweep's 0.098/1.035/10.93 nodes (e.g. t1.0 g=1.0: 0.043425 cold vs 0.041443 chained at g=1.035; local d(deficit)/d(gamma) ~ -0.054 predicts -0.0019, observed -0.0020). **No basin-hopping along any converged continuation chain.**

### 4(c) tau=0.2 row: are the ~1e-6 deficits real?

Three regimes in the ladder behaviour:
- **g <= ~8 (deficit >= ~7e-6): real but imprecise.** Ladders shrink monotonically with sustained step ratio ~0.70 (consistent with O(h^2) = O(du) error decay), so a genuine positive curvature signal is being resolved and the joint-limit deficit is nonzero. However the conservative error bound is 40-65% of the value (geometric: ~12-15%), so these deficits are order-of-magnitude estimates, PROVISIONAL not certified.
- **g >= ~11 (deficit < 3x floor ~ 1.5e-6): grid artifact territory.** Ladder steps reverse sign and *grow* with G (e.g. g=30: 6.4e-7 -> 7.5e-7 -> 1.03e-6 -> 1.49e-6). The kernel smoothing at h = 0.28-0.45 *suppresses* log-odds curvature, so near the floor the measured deficit is biased low and rises as the grid refines — the G=21 operator cannot resolve curvature below ~1.5e-6. These cells' deficits measure the discretization, not the economics.
- The row's slope saturates at 0.0707 across five decades of deficit: revelation at tau=0.2 is essentially complete for g >= ~0.5, and the deficit is a tiny residual-curvature diagnostic. Jensen-product check: deficit*gamma falls from 2.6e-4 (g=1.04) to 4.0e-5 (g=10.9) — clearly NOT constant, i.e. the tau=0.2 tail genuinely decays faster than 1/gamma over this window (then the floor turns the product back up; the upturn at g >= 15 is pure artifact).

Same floor structure across rows: floor ~ 1.5e-6 (tau=0.2), 5.0e-4 (tau=0.5), 9.1e-3 (tau=1.0), and the converged part of tau=1.5 bottoms out ~3.3e-2 right where the row starts stalling. The floor grows steeply with tau and swallows the entire high-gamma half of the table for tau >= 1.0.

## 5. Additional findings

- **Residual quality:** all 81 converged cells sit at F <= 6.6e-13 (most at ~7e-15), far below the 1e-8 `ok` bar; C1 rejected nobody that C2 didn't.
- **Chain smoothness (C5):** no converged cell anywhere triggers the log-deficit kink test; slope is monotone increasing in gamma within every row's converged segment. The continuation scheme behaved well wherever Newton converged.
- **The stalls are a solver/operator property, not a continuation artifact:** cold starts stall at the same (tau, gamma) cells (anchor_grid t2.0 g=1.0 and g=10.0).
- **Provenance gap:** stalled cells report `deficit`/`slope` of a non-converged iterate with no uncertainty flag other than `ok=false`; downstream consumers should treat `ok=false` rows as missing data, and grid.json should record fallback usage.

## 6. Summary

This sweep **establishes**: (i) machine-precision fixed points of the kernel-smoothed, finite-G operator at 81 of 100 (gamma, tau) cells; (ii) a certified low-gamma anchor surface — 22 cells with gamma <= ~0.27 whose deficits are pinned to better than ~15% (mostly <= 5%) in the joint limit, including an exact rung-by-rung reproduction of the trusted (tau=2, gamma=0.1) anchor; (iii) correct basin tracking of the gamma-continuation chains, verified against independent cold starts at seven cells; and (iv) a clean qualitative picture: deficit decreases in gamma and increases in tau, slope increases in both. It does **not** establish: (i) the Jensen 1/gamma law — no certified cell reaches the asymptotic regime, and exploratory tail exponents range from -1.8 (tau=0.2) to -0.98 (tau=1.5); (ii) trustworthy deficits anywhere in the mid-gamma transition region (conservative h-extrapolation errors 16-160%; the h = 0.45*sqrt(du) schedule shrinks h too slowly at G <= 21); (iii) anything at all for tau >= 1.5, gamma >= ~3 (and tau=2.0, gamma >= 0.74), where Newton stalls at F ~ 1e-2 from both warm and cold starts and the recorded numbers are iterate-dependent; (iv) high-gamma deficits for any tau, which are clamped at a tau-dependent discretization floor (1.5e-6 / 5e-4 / 9e-3 / ~3e-2) with curvature-suppression bias. The 'plateau' a reader might see in the tau >= 1.0 tails is the floor and the stalls, not economics. To certify the Jensen exponent this needs: deeper ladders (G >= 25-33) or a faster h-schedule in the window gamma in [1, 10] for tau in [0.5, 1.5], and a different solver strategy (damping, homotopy in h, or the DD/strict-h0 pipeline) for the tau >= 1.5 stall region.

---
*Generated by the independent referee session; analysis scripts: /tmp/referee_analysis.py, /tmp/make_report.py; machine-readable verdicts: /tmp/referee_verdicts.json.*
