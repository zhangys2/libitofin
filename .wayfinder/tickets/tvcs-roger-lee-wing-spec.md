---
title: Roger Lee Wing Splicing Specification
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: "Roger Lee Wing Researcher (fcb576b3)"
map: ../maps/total-variance-cubic-smile.md
resolution: "Adopted Roger Lee asymptotic linear wing bounds beta_R in [0, 2 - epsilon] (right) and beta_L in [-2 + epsilon, 0] (left). Spliced via an exterior Smoothstep C2 Transition Bridge (w'(u) = w'_b + (beta - w'_b)(3u^2 - 2u^3)) over width delta_k = max(delta_k_adjacent, sigma_atm * sqrt(T)). Strictly preserves C2 continuity everywhere (w''(k_b) = 0) and leaves interior regularized least-squares fit 100% untouched. Proved lim g(k) = (4 - beta^2)/16 >= 0 under Lee bounds. Formulated O(1) evaluation struct RogerLeeWing and integrated with SmileSection."
---

# Roger Lee Wing Splicing Specification

## Question

What exact splicing formula and continuity conditions ($C^1$ vs $C^2$) should be used to extend total variance $w(k)$ into extreme strikes $|k| \to \infty$ while strictly guaranteeing Roger Lee's bound $\beta \in [0, 2]$?

## Resolution

### 1. Roger Lee Bounds on Asymptotic Slopes
Roger Lee's (2004) moment formula establishes that for critical moment explosion exponents $p^* = \sup\{p \ge 0 : \mathbb{E}[S_T^{1+p}] < \infty\}$ and $q^* = \sup\{q \ge 0 : \mathbb{E}[S_T^{-q}] < \infty\}$:
$$\limsup_{k \to +\infty} \frac{w(k)}{k} = \Psi(p^*) \in [0, 2], \quad \limsup_{k \to -\infty} \frac{w(k)}{|k|} = \Psi(q^*) \in [0, 2]$$
where $\Psi(x) = 2 - 4(\sqrt{x^2+x} - x) \in (0, 2]$.
- **Right wing asymptotic slope:** $\beta_R = \text{clamp}(w'(k_{\max}), 0.0, 2.0 - \epsilon)$ with $\epsilon = 10^{-4}$.
- **Left wing asymptotic slope:** $\beta_L = \text{clamp}(w'(k_{\min}), -2.0 + \epsilon, 0.0)$.

### 2. Smoothstep $C^2$ Transition Bridge
Clamping a boundary slope directly introduces a jump discontinuity $\Delta w' = \beta - w'_b$, injecting a Dirac delta $\Delta w' \delta(k - k_b)$ into $w''(k)$ and driving Durrleman's density $g(k) \to -\infty$ (localized butterfly arbitrage).
Re-solving interior knots distorts the least-squares fit against liquid market quotes.

**Solution:** An exterior Smoothstep $C^2$ Bridge over transition width $\Delta k = \max(k_{\text{boundary}} - k_{\text{adjacent}}, \sigma_{\text{atm}}\sqrt{T})$:
Normalized coordinate $u = \frac{|k - k_b|}{\Delta k} \in [0, 1]$.
Hermite transition polynomial:
$$w'(u) = w'_b + (\beta - w'_b)(3u^2 - 2u^3)$$

#### Total Variance and Derivatives:
- **Transition Zone ($0 \le u \le 1$)**:
  $$w(k) = w_b + \text{sgn} \cdot \Delta k \left[ w'_b u + (\beta - w'_b)\left(u^3 - \frac{1}{2} u^4\right) \right]$$
  $$w'(k) = w'_b + (\beta - w'_b)(3u^2 - 2u^3)$$
  $$w''(k) = \text{sgn} \cdot \frac{\beta - w'_b}{\Delta k} 6u(1 - u)$$
- **Linear Wing ($u > 1$)**:
  $$w(k) = w(k_1) + \beta (k - k_1), \quad w'(k) = \beta, \quad w''(k) = 0$$

#### Boundary Continuity:
- At knot $u = 0$ ($k = k_b$): $w = w_b$, $w' = w'_b$, $w'' = 0$ (exact $C^2$ match with interior natural cubic spline).
- At wing interface $u = 1$ ($k = k_1$): $w = w_1$, $w' = \beta$, $w'' = 0$ (exact $C^2$ match with linear wing).
- Result: **Strict $C^2$ continuity everywhere on $(-\infty, +\infty)$ with zero interior distortion.**

### 3. Data Structures and API
```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RogerLeeWing {
    boundary_k: Real,
    boundary_w: Real,
    boundary_slope: Real,
    asymptotic_slope: Real,
    transition_width: Real,
    is_right_wing: bool,
}

impl RogerLeeWing {
    pub fn total_variance(&self, k: Real) -> Real;
    pub fn derivative(&self, k: Real) -> Real;
    pub fn second_derivative(&self, k: Real) -> Real;
    pub fn durrleman_density(&self, k: Real) -> Real;
}
```
Integrated into `TotalVarianceCubicSmileSection` and `SmileSection::volatility_impl(strike)` via:
$$\sigma(K) = \sqrt{\frac{\max(w(\ln(K/F)), 0)}{T}}$$
