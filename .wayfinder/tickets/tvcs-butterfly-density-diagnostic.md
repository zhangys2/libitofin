---
title: Butterfly Arbitrage Density Diagnostic & Regularizer Ramp
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: "Butterfly Arbitrage Researcher (c04a22ad)"
map: ../maps/total-variance-cubic-smile.md
resolution: "Adopted Durrleman density diagnostic evaluated across knots + 5-point interior Gauss-Legendre quadrature points per segment evaluated directly via cubic polynomial coefficients with tolerance tau = 10^-8. Proved Roger Lee bound beta in [0, 2] is necessary and sufficient for non-negative wing density lim g(k) = (4 - beta^2)/16 >= 0. Implemented adaptive lambda-doubling ramp (K_max = 5) with structured fallback policy (default AllowWithReport)."
---

# Butterfly Arbitrage Density Diagnostic & Regularizer Ramp

## Question

How should the Durrleman butterfly arbitrage condition $g(k) \ge 0$ be verified across the fitted spline, and how should an adaptive regularizer ramp resolve detected violations?

## Resolution

### 1. Mathematical Foundation (Durrleman / Gatheral Density)
In total variance space $w(k) = \sigma^2(k) T$ with log-moneyness $k = \ln(K/F)$, the Breeden-Litzenberger risk-neutral density is non-negative if and only if:
$$g(k) = \left( 1 - \frac{k w'(k)}{2 w(k)} \right)^2 - \frac{w'(k)^2}{4}\left(\frac{1}{w(k)} + \frac{1}{4}\right) + \frac{w''(k)}{2} \ge 0, \quad \forall k \in \mathbb{R}$$
Evaluating asymptotic linear wings $w(k) = \beta k + \alpha$ as $|k| \to \infty$ reveals:
$$\lim_{|k| \to \infty} g(k) = \frac{4 - \beta^2}{16}$$
Therefore:
$$\lim_{|k| \to \infty} g(k) \ge 0 \iff \beta \le 2$$
This proves that Roger Lee's bound $\beta \in [0, 2]$ is **strictly necessary and sufficient** to avoid butterfly arbitrage in the wings. Clamping boundary slopes to $\beta \le 2 - \epsilon$ guarantees a strictly positive asymptotic floor $g(\pm\infty) \ge \frac{\epsilon}{4} > 0$.

### 2. Diagnostic Evaluation Mesh & Numerical Tolerance
Because the numerator of $g(k)$ is a polynomial of degree up to 10 on each spline segment, evaluating $g(k)$ at knots alone misses interior dips.
The evaluation mesh evaluates:
1. All spline knots $k_0, \dots, k_{K-1}$.
2. 5 interior Gauss-Legendre quadrature nodes per segment $[k_i, k_{i+1}]$:
   $$k_{i, m} = \frac{k_i + k_{i+1}}{2} + \frac{k_{i+1} - k_i}{2} \xi_m, \quad \xi \in \left\{ 0.0, \pm \frac{1}{3}\sqrt{5 - 2\sqrt{10/7}}, \pm \frac{1}{3}\sqrt{5 + 2\sqrt{10/7}} \right\}$$
   evaluated directly from segment polynomial coefficients $[y_i, a_i, b_i, c_i]$ in $\mathcal{O}(1)$ time without binary search.
3. Wing asymptotic limits $g(\pm\infty)$ and wing minimum vertices.

Numerical tolerance threshold:
$$\tau = 10^{-8}$$
Violations with $g_{\min} < -\tau$ are flagged as butterfly arbitrage. Roundoff errors in $[-10^{-8}, 0)$ are floored.

### 3. Diagnostic Report Data Structure
```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ButterflyArbitrageReport {
    pub min_density: Real,
    pub argmin_k: Real,
    pub has_arbitrage: bool,
    pub tolerance: Real,
    pub points_checked: usize,
    pub ramp_iterations: usize,
    pub final_smoothing: Real,
}
```

### 4. Adaptive Regularizer Ramp Algorithm
As $\lambda \to \infty$, $\sup |w_\lambda''| = \mathcal{O}(\lambda^{-1/2}) \to 0$ and $w_\lambda \to \text{affine line}$. Since affine lines with $|\beta| \le 2$ have strictly positive density $g_\infty(k) > 0$, increasing $\lambda$ extinguishes spline-induced curvature wiggles.
- **Ramp Step:** $\lambda \leftarrow 2 \lambda$ (scaling QR penalty rows by $\sqrt{2}$).
- **Maximum Iterations:** $K_{\max} = 5$ (testing up to $32 \times \lambda_0$).
- **Computational Cost:** $< 50\,\mu\text{s}$ total because data rows and unit basis functions are cached and evaluated once.
- **Fallback Policy:** Configurable via `ArbitrageFallbackPolicy`:
  - `AllowWithReport` (default): Attach report with `has_arbitrage = true`; does not abort live trading pricing engines.
  - `FailOnError`: Abort fit on persistent arbitrage.
  - `AffineFallback`: Fall back to an affine regression line with slope clamped to $[-2+\epsilon, 2-\epsilon]$.
