---
title: American Option Strike Filtering & Early Exercise Bounds
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: "Math & Market Formulations Researcher"
map: ../maps/implied-forward-from-options.md
resolution: "Model-free strike corridor |ln(K/S)| <= kappa * sigma_atm * sqrt(T) plus Merton bounds [F_min, F_max]. Amended 2026-10-27: the corridor is the flagged fallback of the model-based solver. Fast is the Ju level. The dividend adjustment and the repo-rate diagnostic were replaced by the dividend schedule and ForwardStatus::WarningImplausibleCarry."
---

# American Option Strike Filtering & Early Exercise Bounds

## Question

What quantitative criteria and analytical no-arbitrage bounds should `ImpliedForward` use to extract forward price estimates from American-style equity/ETF options without full numerical de-Americanization?

## Resolution

Resolved 2026-10-07.

### 1. Model-Free ATM Strike Filtering Window

American early exercise premium is strictly zero for out-of-the-money options ($K > S$ calls, $K < S$ puts), and bounded near the money.
To isolate options where the early exercise premium is smaller than the market bid-ask spread:
- Restrict candidate pairs to an ATM log-moneyness window:
  $$\left| \ln \left( \frac{K_i}{S} \right) \right| \le \kappa \cdot \sigma_{\text{atm}} \sqrt{T}$$
  with default multiplier $\kappa = 0.5$ (or roughly delta range $[0.40, 0.60]$).
- Exclude deep ITM calls ($K < S e^{-2\sigma\sqrt{T}}$) and deep ITM puts ($K > S e^{2\sigma\sqrt{T}}$) from the parity regression.

### 2. Merton Analytical No-Arbitrage Bounds

For American options with spot $S$, discrete dividend schedule $\{Div_j, t_j\}_{t_j \le T}$, and discount factor $D$:
- American call lower bound: $C_{\text{amer}} \ge C_{\text{eur}} \ge \max(S - \text{PV}(Div) - D \cdot K, 0)$.
- American put lower bound: $P_{\text{amer}} \ge P_{\text{eur}} \ge \max(D \cdot K - S + \text{PV}(Div), 0)$.
- Early exercise premium for put is bounded by: $\mathcal{E}_P(K) \le K(1 - D)$.

Model-free forward bounds from American quotes:
$$F_{\min} = \max_{i \in \text{ATM}} \left[ K_i + \frac{C_{\text{bid}, i} - P_{\text{ask}, i}}{D} \right]$$
$$F_{\max} = \min_{i \in \text{ATM}} \left[ K_i + \frac{C_{\text{ask}, i} - (P_{\text{bid}, i} - K_i(1 - D))}{D} \right]$$

### 3. Ex-Dividend Handling & Confidence Status

1. **Clean Period (No dividend before $T$)**:
   - Strike-restricted parity regression operates with high confidence (`Confidence::High`).
2. **Dividend Within Expiry Period ($t_{\text{ex}} \le T$)**:
   - Subtract discounted cash dividend $\text{PV}(Div) = \sum_{t_j \le T} Div_j e^{-r t_j}$ from spot/forward basis:
     $$F_{\text{clean}} = F - \text{PV}(Div) \cdot e^{r T}$$
   - Flag result as `Confidence::AmericanDividendAdjusted`.
3. **High Borrow Fee / Hard-to-Borrow**:
   - If implied repo rate $q = r - \frac{1}{T}\ln(F/S)$ is anomalously high ($q > 0.15$), flag `Diagnostics::NegativeCarryDetected`.

## Amendment (2026-10-25, updated 2026-10-27)

The decision to replace iterative PDE inversion with this model-free filter is reopened. The corridor is the flagged fallback when the discount factor is absent, the model path does not converge, or the pricer errors. It does not remove the early-exercise premium, so its bias grows with rates, dividends and maturity. Fast is the Ju level in [De-Americanization Method & Tier Structure](forward-deamericanization-method-and-tiering.md).

The dividend adjustment in section 3 (`Confidence::AmericanDividendAdjusted`) and the repo-rate diagnostic (`Diagnostics::NegativeCarryDetected`) were replaced by the dividend schedule on `AmericanModelConfig` and by `ForwardStatus::WarningImplausibleCarry` on `carry_flag_range`.
