# ANNOTATED PROOFS — Personal Reference
## "On the Possibility of Informationally Inefficient Markets Without Noise"
### Matthijs Breugem

---

# How to Read This Document

Every proof is broken into: (1) **What we're proving** in plain English, (2) **The setup** — what objects are in play, (3) **The argument** — each step explained in detail, (4) **Why this step works** — the mathematical justification, and (5) **What could go wrong** — where the proof would fail if assumptions changed.

---

# PART I: DEMAND DERIVATIONS

## Lemma 1: CARA Demand

### What we're proving
Under exponential (CARA) utility $U(W) = -e^{-\alpha W}/\alpha$, the optimal demand for a binary asset paying 1 with probability $\mu$ at price $p$ is:
$$x = \frac{\text{logit}(\mu) - \text{logit}(p)}{\alpha}$$
This is LINEAR in the log-odds gap. No wealth effects.

### The argument in detail

**Step 1: Write the first-order condition.**
The agent maximises $\mu \cdot U(W + x(1-p)) + (1-\mu) \cdot U(W - xp)$. Taking the derivative w.r.t. $x$ and setting it to zero:
$$\mu(1-p) U'(W + x(1-p)) = (1-\mu) p \cdot U'(W - xp)$$

*Intuition:* The left side is the marginal gain from buying one more unit (you gain $1-p$ in state $v=1$ with probability $\mu$). The right side is the marginal cost (you lose $p$ in state $v=0$ with probability $1-\mu$).

**Step 2: Substitute CARA marginal utility.**
$U'(W) = e^{-\alpha W}$ (the constant $\alpha$ in front cancels). So:
$$\mu(1-p) e^{-\alpha(W + x(1-p))} = (1-\mu) p \cdot e^{-\alpha(W - xp)}$$

**Step 3: Cancel the wealth terms.**
The $e^{-\alpha W}$ appears on BOTH sides and cancels! This is the key CARA property — no wealth effects. We get:
$$\mu(1-p) e^{-\alpha x(1-p)} = (1-\mu) p \cdot e^{+\alpha xp}$$

**Step 4: Collect the exponentials.**
Divide both sides by $(1-\mu) p$ and collect $e^{-\alpha x}$:
$$\frac{\mu(1-p)}{(1-\mu)p} = e^{\alpha x(1-p) + \alpha xp} = e^{\alpha x}$$

The left side is $e^{\text{logit}(\mu) - \text{logit}(p)}$ (definition of logit). So:
$$e^{\alpha x} = e^{\text{logit}(\mu) - \text{logit}(p)}$$

Taking logs: $\alpha x = \text{logit}(\mu) - \text{logit}(p)$, giving $x = [\text{logit}(\mu) - \text{logit}(p)]/\alpha$. QED.

### Why this matters
The demand is a LINEAR function of the log-odds gap $z = \text{logit}(\mu) - \text{logit}(p)$. When you sum demands in market clearing, you get a LINEAR equation in log-odds. This means the equilibrium price aggregates posteriors in log-odds space — which is exactly where the Bayesian sufficient statistic lives.

### What could go wrong
If $U$ is not CARA, $e^{-\alpha W}$ does NOT cancel in Step 3. You get wealth-dependent demand, which is nonlinear in $z$.

---

## Lemma 2: CRRA Demand

### What we're proving
Under CRRA utility $U(W) = W^{1-\gamma}/(1-\gamma)$, the demand is:
$$x = \frac{W(R-1)}{(1-p) + Rp}, \quad R = \exp\left(\frac{\text{logit}(\mu) - \text{logit}(p)}{\gamma}\right)$$

### The argument

**Step 1: FOC with CRRA.**
$U'(W) = W^{-\gamma}$, so the FOC becomes:
$$\mu(1-p)(W + (1-p)x)^{-\gamma} = (1-\mu)p(W - px)^{-\gamma}$$

**Step 2: Rearrange the ratio.**
$$\frac{(W + (1-p)x)^{-\gamma}}{(W - px)^{-\gamma}} = \frac{(1-\mu)p}{\mu(1-p)}$$

The left side is $(W_1/W_0)^{-\gamma}$ where $W_v$ is terminal wealth in state $v$. Take the $(-1/\gamma)$-th power:
$$\frac{W + (1-p)x}{W - px} = \left(\frac{\mu(1-p)}{(1-\mu)p}\right)^{1/\gamma} = R$$

where $R = \exp(z/\gamma)$ and $z = \text{logit}(\mu) - \text{logit}(p)$.

**Step 3: Solve for $x$.**
$W + (1-p)x = R(W - px)$, so $W + (1-p)x = RW - Rpx$, giving:
$$x[(1-p) + Rp] = W(R-1)$$
$$x = \frac{W(R-1)}{(1-p) + Rp}$$

### Key difference from CARA
- $R$ depends exponentially on $z/\gamma$, not linearly on $z$.
- $W$ appears in the numerator — there ARE wealth effects.
- The demand is a NONLINEAR function of $z$: it curves, it saturates.
- When $\gamma \to \infty$: $R = e^{z/\gamma} \to 1 + z/\gamma$, so $x \to Wz/\gamma \cdot 1/(1+pz/\gamma) \to Wz/\gamma$, which is the CARA limit (Lemma 3).

---

## Lemma 3: CARA as the $\gamma \to \infty$ limit

### The argument
Let $\delta = z/\gamma$ where $z = \text{logit}(\mu) - \text{logit}(p)$. As $\gamma \to \infty$, $\delta \to 0$.

Taylor expand $R = e^{\delta} = 1 + \delta + \delta^2/2 + ...$

Then:
$$x^{\text{CRRA}} = \frac{W(\delta + O(\delta^2))}{1 + p\delta + O(\delta^2)} = W\delta[1 + O(\delta)]$$

Multiply by $\gamma$: $\gamma x^{\text{CRRA}} = W \cdot z \cdot [1 + O(1/\gamma)]$

So $\gamma x^{\text{CRRA}} \to W \cdot z$ as $\gamma \to \infty$, which is CARA demand with $\alpha = \gamma/W$.

### Why this matters
CARA is NOT an independent preference class — it is the limiting case of CRRA as $\gamma \to \infty$. The transition from PR to FR is continuous and smooth, not a sudden jump.

---

# PART II: THE KNIFE-EDGE RESULTS

## Theorem 1: CARA is the UNIQUE log-odds-linear preference

### What we're proving
If demand can be written as $x^* = z/\alpha(p,W)$ where $z = \text{logit}(\mu) - \text{logit}(p)$, then $U$ must be CARA. No other utility function gives demand that is linear in the log-odds gap.

### The argument in detail

This is the most important proof in the paper. It works by showing that if demand is log-odds-linear, then $U''(W) = 0$ for the log-marginal-utility, which forces CARA.

**Step 1: Set up the functional equation.**
Define $h(W) = \ln U'(W)$. This is well-defined because $U' > 0$ (strict monotonicity). The FOC gives:
$$h(W + (1-p)x) - h(W - px) = -z$$
where $x = z/\alpha(p,W)$.

*Why $h$?* Taking logs of the FOC ratio $U'(W_1)/U'(W_0) = e^{-z}$ gives $h(W_1) - h(W_0) = -z$. This is a functional equation in $h$.

**Step 2: First derivative in $z$ at $z = 0$.**
At $z = 0$, $x = 0$, so both arguments of $h$ equal $W$. Differentiating:
$$h'(W) \cdot \frac{1-p}{\alpha} + h'(W) \cdot \frac{p}{\alpha} = -1$$

(The chain rule gives $(1-p)/\alpha$ from the first term and $p/\alpha$ from the second, with $-p/\alpha$ getting a minus sign that flips to plus.)

Simplifying: $h'(W)/\alpha = -1$, so $\alpha = -h'(W)$.

*Key insight:* $\alpha$ depends ONLY on $W$, not on $p$. This is because the $p$-dependence cancelled in the sum $(1-p)/\alpha + p/\alpha = 1/\alpha$.

**Step 3: Second derivative in $z$ at $z = 0$.**
Differentiating again:
$$h''(W) \cdot \left[\frac{(1-p)^2}{\alpha^2} - \frac{p^2}{\alpha^2}\right] = 0$$

The right side is 0 because $-z$ is LINEAR in $z$, so its second derivative vanishes.

The bracket equals $(1-2p)/\alpha^2$. This must hold for ALL $p \in (0,1)$.

For $p \neq 1/2$: the bracket is nonzero, so $h''(W) = 0$.

**Step 4: $h$ is affine, so $U$ is CARA.**
$h''(W) = 0$ everywhere means $h(W) = -\alpha_0 W + C$ for constants $\alpha_0 > 0$ and $C$.

Therefore $U'(W) = e^h = e^C \cdot e^{-\alpha_0 W}$, which is CARA marginal utility.

Integrating: $U(W) = a - b e^{-\alpha_0 W}$ with $b > 0$. This is CARA.

### Why the proof is elegant
It uses a remarkably simple trick: differentiate the functional equation TWICE in $z$ and evaluate at $z = 0$. The first derivative pins down $\alpha$. The second forces $h'' = 0$, which forces CARA. No heavy machinery needed.

### Where it would fail
If we only required demand to be log-odds-linear at a SINGLE price $p = 1/2$, the proof fails at Step 3 (the bracket is zero). We need it for ALL prices.

---

## Proposition 1: CARA = Full Revelation (always)

### What we're proving
For any number of agents $K \geq 1$, any risk aversion parameters $\alpha_k$, any signal precisions $\tau_k$, any wealth levels $W_k$: the CARA equilibrium price is a function of $T^* = \sum \tau_k u_k$ only. Hence it reveals $T^*$, and the equilibrium is fully revealing.

### The argument

**Step 1: Substitute CARA demand into market clearing.**
$$\sum_k \frac{\text{logit}(\mu_k) - \text{logit}(p)}{\alpha_k} = 0$$

**Step 2: Solve for logit(p).**
$$\text{logit}(p) = \sum_k w_k \cdot \text{logit}(\mu_k), \quad w_k = \frac{1/\alpha_k}{\sum_j 1/\alpha_j}$$

**Step 3: Substitute the posterior.**
$\text{logit}(\mu_k) = \tau_k u_k$ (from the signal structure). So:
$$\text{logit}(p) = \sum_k w_k \tau_k u_k$$

This is a LINEAR combination of signals. It is a FUNCTION of $T^* = \sum \tau_k u_k$ (with possibly different weights). Observing this price is informationally equivalent to observing $T^*$.

### Why the linear aggregation matters
The key is that CARA demand is linear in log-odds, and the log-odds IS the Bayesian sufficient statistic. Market clearing sums demands, and the sum of log-odds terms is the sufficient statistic. Any other utility creates a nonlinear demand, so market clearing sums a nonlinear function of log-odds — which is NOT the sufficient statistic.

---

## Proposition 2: Non-CARA = Partial Revelation

### What we're proving
Under log utility ($\gamma = 1$), the price $p = \bar{\mu} = (1/K) \sum \Lambda(\tau u_k)$ is NOT a function of $T^*$ alone.

### The argument (by counterexample)

Fix $T^* = 0$. Consider two signal realisations:
- Realisation A: $(u_1, u_2, u_3) = (0, 0, 0)$. Price: $p_A = \Lambda(0) = 1/2$.
- Realisation B: $(u_1, u_2, u_3) = (\delta, -\delta, 0)$. Price: $p_B = [\Lambda(\tau\delta) + \Lambda(-\tau\delta) + 1/2]/3$.

Both have $T^* = 0$, but $p_A \neq p_B$ because $\Lambda$ is NONLINEAR (concave-convex):
$$\frac{\Lambda(\tau\delta) + \Lambda(-\tau\delta)}{2} \neq \Lambda(0) = \frac{1}{2}$$

by Jensen's inequality (strict unless $\delta = 0$).

### Why this is Jensen's inequality
$\Lambda$ (the logistic function) is concave on $(0, \infty)$ and convex on $(-\infty, 0)$. The average $[\Lambda(x) + \Lambda(-x)]/2$ is strictly less than $\Lambda(0) = 1/2$ for $x \neq 0$. This IS the Jensen gap — the curvature of $\Lambda$ means averaging in probability space does not equal the probability of the average.

---

## Proposition 3: Jensen Gap Expansion

### What we're proving
The price deviation from CARA is:
$$\Delta p = -\frac{\tau^3}{48K}\left[\sum u_k^3 - \frac{(\sum u_k)^3}{K^2}\right] + O(\tau^5)$$

### The argument

**Step 1: Taylor expand $\Lambda(\tau u_k)$ around 0.**
$$\Lambda(x) = \frac{1}{2} + \frac{x}{4} - \frac{x^3}{48} + O(x^5)$$

(This comes from $\Lambda'(0) = 1/4$, $\Lambda''(0) = 0$, $\Lambda'''(0) = -1/8$.)

**Step 2: Average over $k$.**
$$p = \frac{1}{K}\sum_k \Lambda(\tau u_k) = \frac{1}{2} + \frac{\tau \bar{u}}{4} - \frac{\tau^3}{48K}\sum_k u_k^3 + O(\tau^5)$$

**Step 3: Expand the CARA price.**
$$p^{\text{CARA}} = \Lambda(\tau \bar{u}) = \frac{1}{2} + \frac{\tau \bar{u}}{4} - \frac{\tau^3}{48}\bar{u}^3 + O(\tau^5)$$

**Step 4: Subtract.**
$$\Delta p = p - p^{\text{CARA}} = -\frac{\tau^3}{48}\left[\frac{1}{K}\sum u_k^3 - \bar{u}^3\right]$$

Using $\bar{u} = (\sum u_k)/K$, we get the stated formula.

### Interpretation
The Jensen gap is $O(\tau^3)$ — it vanishes at small signal precision but grows cubically. The bracket is the difference between the average of cubes and the cube of the average — a measure of dispersion of the signals.

---

# PART III: REE SURVIVAL

## Proposition 5: REE Partial Revelation

### What we're proving
The partial revelation found in the no-learning equilibrium SURVIVES when agents learn from prices. The converged rational-expectations equilibrium (REE) has strictly positive $1 - R^2$.

### The argument (two parts)

**Part (a): Inductive argument that the contour stays curved.**

The proof uses the Picard iteration $\mu^{n+1} = \Phi(\mu^n)$ where $\Phi$ is the contour-integration Bayes update.

**Base case ($n = 0$):** The demand is $d^0(u) = x(\Lambda(\tau u), p)$. Since CRRA demand $x(\mu, p)$ has nonzero second derivative in $z = \text{logit}(\mu) - \text{logit}(p)$ for finite $\gamma$, and $z = \tau u - \text{logit}(p)$ is affine in $u$, the composition $d^0(u)$ is nonlinear in $u$. Therefore the contour $\{(u_2, u_3) : d^0(u_2) + d^0(u_3) = -d^0(u_1)\}$ is CURVED.

*Under CARA:* $x$ is AFFINE in $z$, so $d^0$ is affine, contour is STRAIGHT. That's why CARA gives FR.

**Inductive step:** Suppose $d^n$ is nonlinear (contour is curved). Then:

(i) $T^* = \tau(u_2 + u_3)$ varies along the curved contour. The likelihood ratio $\exp(\tau(u_2 + u_3))$ is therefore non-constant on the contour.

(ii) The contour integrals $A_v$ average this non-constant likelihood ratio with the signal density. The resulting $\log(A_1/A_0)$ depends on $u_1$ — different own signals generate different contour shapes.

(iii) The updated posterior $\mu^{n+1} = \Lambda(\tau u_1 + \log(A_1/A_0))$ depends on $u_1$ beyond the price. The demand $d^{n+1}(u) = x(\mu^{n+1}(u,p), p)$ is the composition of the nonlinear CRRA demand with $\mu^{n+1}(u,p)$, which is generically nonlinear.

*Why "generically"?* For $d^{n+1}$ to be linear, the curvature of $\mu^{n+1}$ would need to exactly cancel the curvature of the CRRA demand at every $u$ simultaneously. This is a codimension-one condition (one equation for a continuous family of values) — it holds only for CARA.

**Taking the limit:** If $\mu^n \to \mu^*$, the nonlinearity bound passes to the limit by continuity. Hence $d^*$ is nonlinear, the contour is curved, and the equilibrium is partially revealing.

**Part (b): Convergence (numerical).**
The Picard sequence converges numerically at G=15 to machine precision ($\|F\|_\infty < 10^{-14}$), and at G=20 to 119 digits of precision ($\|F\|_\infty < 10^{-119}$). The deficit $1 - R^2$ stabilises across grid refinement.

### What this proof does NOT show
- It does not prove uniqueness of the REE (there could be other fixed points).
- It does not prove the Picard iteration converges for all $\gamma$ (numerically verified, not proved).
- The "generic nonlinearity" in step (iii) is not proved for ALL iterates — the codimension argument is plausibility, not proof.

---

# PART IV: WELFARE AND GS

## Proposition 6: Positive Trade

### Under CARA REE
All posteriors $\mu_k = p = \Lambda(T^*)$, so $\text{logit}(\mu_k) - \text{logit}(p) = 0$ for all $k$. CARA demand is $x = 0/\alpha = 0$. No trade — this is the Milgrom-Stokey no-trade theorem in action.

### Under CRRA REE
Posteriors DISAGREE (from Proposition 5): $\mu_k \neq p$ for some $k$ on a full-measure set. Since CRRA demand is nonzero whenever $\mu_k \neq p$, and market clearing requires $\sum x_k = 0$, there must be both buyers and sellers. Trade volume $V = (1/2)\sum |x_k|$ is strictly positive.

---

## Proposition 7: Value of Information

### Under CARA
The price already reveals $T^*$, so learning an additional signal $u_0$ adds nothing: $\mu^+ = \mu^- = \Lambda(T^*)$. The agent's certainty equivalent doesn't change. $V_{\text{CARA}}(\tau) = 0$.

### Under CRRA
The price does NOT fully reveal $T^*$. An additional signal $u_0$ shifts the posterior: $\mu^+ \neq \mu^-$. The agent can then trade more profitably. $V_{\text{CRRA}}(\tau) > 0$.

*Non-negativity:* The agent can always IGNORE the signal (replicate $\mu^-$), so $V \geq 0$.
*Strict positivity:* On a positive-measure set, the signal genuinely helps, so $V > 0$.

---

## Proposition 8: GS Resolution

### Under CARA (paradox)
If a fraction $\lambda > 0$ of agents acquire signals at cost $c > 0$, the price fully reveals $T^*$ (Prop 1). But then $V = 0$ (Prop 7), so each acquirer earns $-c < 0$. They'd prefer not to acquire. Contradiction — no positive $\lambda$ is an equilibrium.

But $\lambda = 0$ isn't an equilibrium either (no one is informed, so the first acquirer would get infinite rent). This is the Grossman-Stiglitz paradox: no equilibrium exists.

### Under CRRA (resolution)
$V_{\text{CRRA}}(\tau, 0^+) > 0$ (first acquirer gets full informational rent).
$V_{\text{CRRA}}(\tau, \lambda)$ is continuous and weakly decreasing in $\lambda$ (more acquirers → more informative price → less rent).
By the intermediate value theorem: for any $c \in (0, \bar{c})$, there exists $\lambda^*(c) \in (0,1)$ where the marginal acquirer is indifferent. This IS the equilibrium.

### Why this works
The paradox is resolved because prices are only PARTIALLY revealing. There's always some informational rent left for acquirers. The amount decreases with $\lambda$ but never hits zero (unlike CARA where it hits zero immediately at $\lambda > 0$).

---

# PART V: THE CONTOUR METHOD

## How it works geometrically

Agent 1 knows $u_1$ and observes price $p$. She knows the price function $P$, so the set of possible $(u_2, u_3)$ pairs is:
$$\mathcal{C} = \{(u_2, u_3) : P(u_1, u_2, u_3) = p\}$$

This is a CURVE in $(u_2, u_3)$ space — the contour.

She integrates the signal density along this curve:
$$A_v = \int_{\mathcal{C}} f_v(u_2) \cdot f_v(u_3^*(u_2)) \, du_2$$

Her posterior is then:
$$\mu_1 = \frac{f_1(u_1) \cdot A_1}{f_0(u_1) \cdot A_0 + f_1(u_1) \cdot A_1}$$

### Under CARA: the contour is a STRAIGHT LINE
$\text{logit}(p) = (1/3)(\tau u_1 + \tau u_2 + \tau u_3)$, so the contour is $\tau u_2 + \tau u_3 = \text{const}$. Every point on this line has the SAME $T^*$. The density ratio $A_1/A_0 = \exp(\tau(u_2 + u_3)) = \exp(T^* - \tau u_1)$ is CONSTANT on the contour. The agent learns $T^*$ exactly from the price. → Full revelation.

### Under CRRA: the contour is CURVED
The price aggregates in probability space, not log-odds. The contour $\Lambda(\tau u_2) + \Lambda(\tau u_3) = \text{const}$ is curved because $\Lambda$ is nonlinear. Different points on the contour have different $T^*$. The agent cannot extract $T^*$ from the price alone. → Partial revelation.

### The curvature IS the Jensen gap
The curvature of the contour is literally the geometric manifestation of Jensen's inequality. A convex/concave contour means $\sum \Lambda(\tau u_k)/K \neq \Lambda(\sum \tau u_k / K)$. The degree of curvature determines the degree of partial revelation.

---

# SUMMARY OF THE LOGICAL CHAIN

$$\text{CARA demand linear in log-odds}$$
$$\downarrow$$
$$\text{Market clearing aggregates in log-odds space}$$
$$\downarrow$$
$$\text{Log-odds = Bayesian sufficient statistic}$$
$$\downarrow$$
$$\text{Price reveals T* → Full revelation}$$

vs.

$$\text{Non-CARA demand NONLINEAR in log-odds}$$
$$\downarrow$$
$$\text{Market clearing aggregates in a DIFFERENT space}$$
$$\downarrow$$
$$\text{Different space ≠ sufficient statistic (Jensen gap)}$$
$$\downarrow$$
$$\text{Price does NOT reveal T* → Partial revelation}$$

And the UNIQUENESS theorem closes the loop:
$$\text{Log-odds-linear demand} \iff \text{CARA utility}$$

So CARA is not just SUFFICIENT for FR — it is NECESSARY. Every other expected-utility preference gives PR.
