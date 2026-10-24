---
title: Inverse Crypto Option Parity & Cash Flow Mechanics
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: "Math & Market Formulations Researcher"
map: ../maps/implied-forward-from-options.md
resolution: "Solved natively in coin space via linear regression Delta_btc = D_coin - (D_coin / F) * K, producing F* = -alpha / beta directly with zero circular dependence. Synthetic bounds mapped via F_bid = K / (1 - (C_ask - P_bid)) and F_ask = K / (1 - (C_bid - P_ask)). Fallback to listed future mark when option liquidity is thin."
---

# Inverse Crypto Option Parity & Cash Flow Mechanics

## Question

How should the engine mathematically formulate put-call parity for coin-margined (inverse) cryptocurrency options (e.g. Deribit BTC/ETH) without numerical instability or circular dependence on the forward price?

## Resolution

Resolved 2026-10-07.

### 1. Coin-Space Numeraire Linear Parity

Deribit inverse options have payoffs settled in cryptocurrency (BTC/ETH):
- Call payoff: $\frac{\max(S_T - K, 0)}{S_T}$ in BTC.
- Put payoff: $\frac{\max(K - S_T, 0)}{S_T}$ in BTC.
- Payoff difference at expiry:
  $$\text{Payoff difference} = \frac{S_T - K}{S_T} = 1 - \frac{K}{S_T}$$

Taking the expectation under the coin forward measure:
$$C_{\text{btc}}(K) - P_{\text{btc}}(K) = D_{\text{coin}} \left( 1 - \frac{K}{F} \right) = D_{\text{coin}} - \left( \frac{D_{\text{coin}}}{F} \right) K$$
where $D_{\text{coin}}$ is the coin-currency discount factor ($D_{\text{coin}} \approx 1$ under zero coin borrow rates).

Notice this is **strictly linear in strike $K$**:
$$y_i = C_{\text{btc}, i} - P_{\text{btc}, i} = \alpha + \beta K_i$$
with:
$$\alpha = D_{\text{coin}}, \quad \beta = -\frac{D_{\text{coin}}}{F}$$

### 2. Solving $F^*$ Directly Without Spot Dependency

From the linear regression parameters:
$$F^* = -\frac{\alpha}{\beta}$$
- No circular dependence on forward price $F$.
- No requirement to know the underlying index spot price $S_{\text{index}}$ to estimate the forward.
- The implied coin discount factor $D_{\text{coin}} = \alpha$ provides an immediate sanity check: in healthy crypto markets, $\alpha \in [0.98, 1.02]$.

### 3. Synthetic Bid-Ask Bounds in Inverse Markets

Under coin pricing:
- Long synthetic forward (buy Call ask, sell Put bid in BTC):
  $$1 - \frac{K}{F_{\text{bid}, i}} = C_{\text{ask}, \text{btc}, i} - P_{\text{bid}, \text{btc}, i} \implies F_{\text{bid}, i} = \frac{K_i}{1 - (C_{\text{ask}, \text{btc}, i} - P_{\text{bid}, \text{btc}, i})}$$
- Short synthetic forward (sell Call bid, buy Put ask in BTC):
  $$F_{\text{ask}, i} = \frac{K_i}{1 - (C_{\text{bid}, \text{btc}, i} - P_{\text{ask}, \text{btc}, i})}$$

Singularity check: require $1 - (C_{\text{ask}} - P_{\text{bid}}) > 0$; pairs violating this are rejected as extreme misquotes.

### 4. Integration with Listed Futures Quotes

Deribit lists dated futures (e.g. `BTC-28MAR25`).
- When a dated future has a valid, tight quote:
  - The future's mark/mid price serves as an anchor.
  - The option-implied $F^*$ validates the future against basis manipulation.
- When dated futures are unavailable or wide, the option-implied $F^*$ provides the primary forward.
