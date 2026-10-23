---
title: Arbitrage Diagnostics & Crossed-Quote Reporting
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: "Math & Market Formulations Researcher"
map: ../maps/implied-forward-from-options.md
resolution: "Designed ForwardDiagnostics with ForwardStatus flags (Valid, CrossedQuotes, BoxSpreadArbitrage, SpotDeviationExceeded). Returns result on market anomalies with status flags rather than hard error; reserves hard error for fatal degenerate cases. Outlier rejection prunes crossed individual quotes and residual > 3*RMSE."
---

# Arbitrage Diagnostics & Crossed-Quote Reporting

## Question

What diagnostic indicators, validation rules, and threshold checks should `ImpliedForward` return alongside the calculated forward and discount factor to alert callers to quote errors, stale feeds, and arbitrage violations?

## Resolution

Resolved 2026-10-07.

### 1. Diagnostic Status & Structured Reporting

The engine will not fail catastrophically on dirty market data; instead, it provides structured diagnostics alongside the estimates so streaming tickers can fall back to spot or last-known marks gracefully.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardStatus {
    /// Clean fit, all arbitrage and consistency bounds satisfied
    Valid,
    /// Strict synthetic bid-ask quotes are crossed (F_bid_strict > F_ask_strict)
    WarningCrossedSyntheticQuotes,
    /// Box-spread parity violated across adjacent liquid strikes
    WarningBoxSpreadArbitrage,
    /// Implied discount factor outside plausible interval [0.50, 1.20]
    WarningImplausibleDiscountFactor,
    /// Forward deviates from spot beyond threshold |F/S - 1.0| > delta_max
    WarningSpotDeviationExceeded,
    /// American options early exercise spread widened bounds
    DegradedAmericanBounds,
}

#[derive(Debug, Clone)]
pub struct ForwardDiagnostics {
    pub status: ForwardStatus,
    pub total_pairs_received: usize,
    pub pairs_used: usize,
    pub pairs_pruned: usize,
    pub is_crossed: bool,
    pub box_arbitrage_violations: usize,
    pub wls_rmse: f64,
    pub min_spread: f64,
    pub max_spread: f64,
}
```

### 2. Validation & Arbitrage Checks

1. **Crossed Synthetic Forward Check**:
   - For every strike $i$, verify $F_{\text{bid}, i} \le F_{\text{ask}, i}$. If individual bid exceeds ask, the underlying option ticker has crossed quotes ($C_{\text{bid}} > C_{\text{ask}}$ or $P_{\text{bid}} > P_{\text{ask}}$); drop that pair immediately.
   - For the aggregate bounds across all ATM strikes:
     $$F_{\text{bid}}^{\text{strict}} = \max_{i \in \text{ATM}} F_{\text{bid}, i}, \quad F_{\text{ask}}^{\text{strict}} = \min_{i \in \text{ATM}} F_{\text{ask}, i}$$
     If $F_{\text{bid}}^{\text{strict}} > F_{\text{ask}}^{\text{strict}}$, mark `is_crossed = true` and `status = WarningCrossedSyntheticQuotes`. The robust weighted bounds $F_{\text{bid}}^{\text{robust}} \le F^* \le F_{\text{ask}}^{\text{robust}}$ remain available.

2. **Box-Spread Parity Check**:
   - For adjacent strikes $K_1 < K_2$, no-arbitrage requires:
     $$(C_{\text{bid}, 1} - P_{\text{ask}, 1}) - (C_{\text{ask}, 2} - P_{\text{bid}, 2}) \le D(K_2 - K_1) \le (C_{\text{ask}, 1} - P_{\text{bid}, 1}) - (C_{\text{bid}, 2} - P_{\text{ask}, 2})$$
   - Violations indicate cross-strike mispricings or asynchronous latency skew. Counted in `box_arbitrage_violations`.

3. **Discount Factor Sanity Thresholds**:
   - If unanchored regression yields $D^* \le 0$ or $D^* > 1.20$ (e.g. extreme negative interest rate artifacts), reject the unanchored slope, fallback to $D = 1.0$, and flag `WarningImplausibleDiscountFactor`.

4. **Spot Deviation Threshold**:
   - If `spot` is provided and $|F^* / S - 1.0| > \delta_{\max}$ (default $\delta_{\max} = 0.05$, configurable), flag `WarningSpotDeviationExceeded`.

### 3. Outlier Pruning Pipeline

Before the final WLS fit:
1. **Sanity Filter**: Discard pairs with non-finite or non-positive strikes/prices, or $ask < bid$.
2. **Spread Ceiling**: Discard pairs where total spread $(ask_C - bid_C) + (ask_P - bid_P) > 0.25 \cdot K$.
3. **Two-Pass Residual Pruning**:
   - Fit initial WLS regression on surviving pairs.
   - Compute residual $r_i = y_i - (\alpha + \beta K_i)$ and $\text{RMSE} = \sqrt{\sum \bar{w}_i r_i^2}$.
   - If any pair has $|r_i| > 3.5 \cdot \text{RMSE}$, prune and refit once.
