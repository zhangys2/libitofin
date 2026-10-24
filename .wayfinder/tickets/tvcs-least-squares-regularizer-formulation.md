---
title: Total Variance Least-Squares Regularizer Formulation
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: "Total Variance QR Solver Researcher (2f476018)"
map: ../maps/total-variance-cubic-smile.md
resolution: "Adopted IV-normalized linear residual (w(k_i) - w_i)/(2 sigma_i T) with target sigma_i/2 and curvature scaling lambda_eff = lambda * sigma_atm / (4 sqrt(T)). Integrated analytically via exact 2-point Gauss-Legendre quadrature per segment. 9 standardized knots k_j = x_j sigma_atm sqrt(T). Affine nullspace requires N >= 2 distinct observations for full-rank QR solve in O(N) time."
---

# Total Variance Least-Squares Regularizer Formulation

## Question

How should the least-squares data fidelity term and curvature penalty be formulated in total variance space ($k \mapsto w(k)$) with QR matrix decomposition?

## Resolution

### 1. IV-Normalized Linear Residual Formulation
Unweighted total variance residuals $(w(k_i) - w_i)^2 \approx 4 \sigma_i^2 T^2 (\sigma - \sigma_i)^2$ vanish as $T \to 0$ (causing fixed smoothing to over-flatten short expiries) and overweight wing quotes $4\times$ to $9\times$ vs ATM.
To maintain strict implied volatility distance and scale invariance across expiries $T \in (0, 10]$:
$$\Delta_i = \frac{w(k_i) - w_i}{2 \sigma_i T} = \frac{w(k_i)}{2 \sigma_i T} - \frac{\sigma_i}{2}$$
This residual is **strictly linear** in the knot ordinates $\mathbf{w}$, with observation target scalar $b_i = \frac{\sigma_i}{2}$.

### 2. Curvature Regularizer Scaling
To match `CubicSmileSection`'s dimensionless smoothing parameter $\lambda$ (default $0.01$):
$$\mathcal{J}(w) = \frac{1}{N} \sum_{i=1}^N \left( \frac{w(k_i) - w_i}{2 \sigma_i T} \right)^2 + \lambda_{\text{eff}} \int_{k_{\min}}^{k_{\max}} (w''(k))^2 dk$$
where
$$\lambda_{\text{eff}} = \lambda \cdot \frac{\sigma_{\text{atm}}}{4 \sqrt{T}}$$
Both data fidelity and regularization terms have exact dimension $[1/\text{year}]$, ensuring scale invariance as $T \to 0$ or $T \to \infty$.

### 3. Exact Integration of Curvature Penalty
For natural cubic splines, $w''(k)$ is piecewise linear on each interval $[k_j, k_{j+1}]$, so $(w''(k))^2$ is quadratic (degree 2).
The exact analytical integral on segment $j$ is:
$$\int_{k_j}^{k_{j+1}} (w''(k))^2 dk = \frac{h_j}{3} \left( M_j^2 + M_j M_{j+1} + M_{j+1}^2 \right)$$
Because 2-point Gauss-Legendre quadrature is exact for polynomials up to degree $2(2)-1 = 3$, it evaluates this integral with **zero truncation error**.
Each segment $j \in \{0, \dots, K-2\}$ introduces 2 augmented penalty rows with zero target:
$$A_{\text{row}, l} = \sqrt{\lambda_{\text{eff}} \frac{h_j}{2}} \cdot \phi_l''(x_{j, q}), \quad b_{\text{row}} = 0.0$$

### 4. Standardized Knot Placement & Affine Nullspace
- Standardized knots: $k_j = x_j \cdot (\sigma_{\text{atm}} \sqrt{T})$ using the 9-point grid:
  $$\mathbf{x} = [-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0]$$
- The nullspace of $\int (w'')^2 dk$ is the 2D affine line space $\mathcal{N} = \{ a + b k \}$.
- For $\lambda > 0$, any $N \ge 2$ distinct in-range observations guarantee full column rank ($K = 9$) without artificial synthetic quotes or ATM pinning.

### 5. QR Factorization Performance
- Dimension: $(N + 16) \times 9$.
- FLOP count: $\approx 162 N + 2106$ FLOPs ($< 1.5\,\mu\text{s}$ for $N=30$).
- Column pivoting with rank tolerance $|r_{ii}| > 128 \epsilon_{\text{mach}} \max |r_{jj}|$ eliminates rank deficiency without squaring condition numbers.
