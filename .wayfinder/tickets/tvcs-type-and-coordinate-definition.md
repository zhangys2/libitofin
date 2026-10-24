---
title: Total Variance Cubic Smile Type & Coordinate Definition
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
map: ../maps/total-variance-cubic-smile.md
---

# Total Variance Cubic Smile Type & Coordinate Definition

## Question

How should total-variance cubic interpolation be exposed in the type system, and what coordinate and state representation should it use?

## Context

`CubicSmileSection` currently interpolates annualized Black implied volatility $\sigma(x)$ against standardized standard-deviation moneyness $x = \frac{\ln(K/F)}{\sigma_{\text{atm}}\sqrt{T}}$.
In contrast, total variance $w(k) = \sigma^2(k) T$ against log-moneyness $k = \ln(K/F)$ is the natural space for no-arbitrage wing asymptotics (Roger Lee moment formula) and butterfly arbitrage verification (Durrleman condition).

## Options Considered

- **A. Dedicated Type (`TotalVarianceCubicSmileSection`)**: A standalone struct implementing `SmileSection`, with internal spline operating on $k \mapsto w(k)$.
- **B. Enum Representation in `CubicSmileSection`**: Adding a flag or enum `SmileInterpolationTarget::ImpliedVolatility | TotalVariance`.
- **C. Full 2D Surface Only**: Deferring 1D slices and writing a full 2D surface model directly.

## Resolution

Confirmed 2026-10-01 (via Round 1 alignment):

- **Dedicated struct**: `TotalVarianceCubicSmileSection` implementing `SmileSection`.
- **Coordinate Space**:
  - Independent variable (abscissa): Log-moneyness $k = \ln(K / F)$.
  - Dependent variable (ordinate): Total implied variance $w(k) = \sigma^2(k) T$.
- **Trait compatibility**:
  - `SmileSection::volatility(strike)` derives $\sigma(K) = \sqrt{\max(w(\ln(K/F)), 0) / T}$.
  - `SmileSection::variance(strike)` returns $w(\ln(K/F))$.
- **Scope**: Focused on 1D single-expiry slice first; multi-expiry surface calendar arbitrage deferred.
