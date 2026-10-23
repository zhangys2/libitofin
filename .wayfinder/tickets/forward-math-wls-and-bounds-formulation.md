---
title: Implied Forward Mathematical Formulation & WLS Objective
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: "Math & Market Formulations Researcher"
map: ../maps/implied-forward-from-options.md
resolution: "Closed-form Weighted Least Squares (WLS) with weights w_i = 1 / (Delta C_i^2 + Delta P_i^2 + epsilon^2) estimating (F*, D*) simultaneously via alpha = D*F, beta = -D. Provides dual output: spread-weighted point estimate F* and strict + robust synthetic bid-ask bounds [F_bid, F_ask]."
---

# Implied Forward Mathematical Formulation & WLS Objective

## Question

What exact mathematical objective, weighting scheme, and robust loss function should `ImpliedForward` use to calculate the weighted point estimate $(F^*, D^*)$ and synthetic bid-ask bounds $[F_{\text{bid}}, F_{\text{ask}}]$ across discrete option strike pairs with bid-ask spreads?

## Resolution

Resolved 2026-10-07.

### 1. Objective Function & Weighted Least Squares (WLS) Formulation

For European options at expiration $T$, put-call parity holds for mid prices:
$$y_i \equiv C_{\text{mid}, i} - P_{\text{mid}, i} = D \cdot F - D \cdot K_i = \alpha + \beta K_i$$
where $\alpha = D \cdot F$ and $\beta = -D$.

To downweight noisy, illiquid wing quotes and prioritize tight ATM quotes:
- Combined bid-ask spread: $\Delta_i = (C_{\text{ask}, i} - C_{\text{bid}, i}) + (P_{\text{ask}, i} - P_{\text{bid}, i})$.
- Spread floor regularizer: $\epsilon_i = 10^{-4} \cdot \max(K_i, 1.0)$ to prevent division by zero at minimum tick increments.
- Inverse-variance weights:
  $$w_i = \frac{1}{(C_{\text{ask}, i} - C_{\text{bid}, i})^2 + (P_{\text{ask}, i} - P_{\text{bid}, i})^2 + \epsilon_i^2}$$
  with normalized weights $\bar{w}_i = \frac{w_i}{\sum_{j=1}^N w_j}$.

### 2. Parameter Estimation Modes

#### Mode A: Joint Estimation ($(F^*, D^*)$ both unknown)
When the discount factor or financing rate is not externally provided:
$$\bar{K} = \sum_{i=1}^N \bar{w}_i K_i, \quad \bar{y} = \sum_{i=1}^N \bar{w}_i y_i$$
$$\text{Var}_w(K) = \sum_{i=1}^N \bar{w}_i (K_i - \bar{K})^2, \quad \text{Cov}_w(K, y) = \sum_{i=1}^N \bar{w}_i (K_i - \bar{K})(y_i - \bar{y})$$
- Slope: $\beta = \frac{\text{Cov}_w(K, y)}{\text{Var}_w(K)}$
- Intercept: $\alpha = \bar{y} - \beta \bar{K}$
- Implied Discount Factor: $D^* = -\beta$
- Implied Forward Price: $F^* = \frac{\alpha}{D^*} = \bar{K} + \frac{\bar{y}}{D^*}$
- Implied Cost of Carry: $r - q = -\frac{\ln(D^*)}{T}$

Requires $N \ge 2$ distinct strike pairs with non-zero $\text{Var}_w(K)$.

#### Mode B: Anchored Mode ($D$ externally specified)
When external discount factor $D$ (or risk-free rate $r$) is supplied:
$$F_i = K_i + \frac{C_{\text{mid}, i} - P_{\text{mid}, i}}{D}$$
- WLS Point Estimate: $F^* = \sum_{i=1}^N \bar{w}_i F_i = \bar{K} + \frac{\bar{y}}{D}$.
- Robust Point Estimate: Weighted median of $F_i$ across the top $M$ tightest pairs (resilient against quote spikes).

### 3. Synthetic Forward Bid-Ask Bounds

For each strike $i$, synthetic conversion/reversal arbitrage bounds are:
$$F_{\text{bid}, i} = K_i + \frac{C_{\text{bid}, i} - P_{\text{ask}, i}}{D^*}$$
$$F_{\text{ask}, i} = K_i + \frac{C_{\text{ask}, i} - P_{\text{bid}, i}}{D^*}$$

The engine calculates:
1. **Strict No-Arbitrage Interval**:
   $$F_{\text{bid}}^{\text{strict}} = \max_{i \in \text{ATM}} F_{\text{bid}, i}, \quad F_{\text{ask}}^{\text{strict}} = \min_{i \in \text{ATM}} F_{\text{ask}, i}$$
   If $F_{\text{bid}}^{\text{strict}} > F_{\text{ask}}^{\text{strict}}$, a quote crossing / arbitrage exists.
2. **Robust Spread Interval**:
   $$F_{\text{bid}}^{\text{robust}} = \sum_{i=1}^N \bar{w}_i F_{\text{bid}, i}, \quad F_{\text{ask}}^{\text{robust}} = \sum_{i=1}^N \bar{w}_i F_{\text{ask}, i}$$
   Guarantees $F_{\text{bid}}^{\text{robust}} \le F^* \le F_{\text{ask}}^{\text{robust}}$ as long as individual quotes are not crossed.
