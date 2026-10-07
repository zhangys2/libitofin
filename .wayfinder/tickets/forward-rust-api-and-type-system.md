---
title: Rust Core API, Structs, and Trait Representation
type: Task (AFK)
status: closed
blocked_by: []
claimed_by: "Math & Market Formulations Researcher"
map: ../maps/implied-forward-from-options.md
resolution: "Designed ImpliedForward engine in crates/libitofin/src/termstructures/forward/implied_forward.rs with OptionQuotePair, MarketConvention, ImpliedForwardConfig, ImpliedForwardResult, and ForwardDiagnostics. Zero heap allocation in inner loop, typed ForwardError, and re-exported under libitofin::termstructures::forward."
---

# Rust Core API, Structs, and Trait Representation

## Question

What public API types, input quote structs, configuration builders, and output representations should `crates/libitofin` expose for the unified `ImpliedForward` engine, and where in the crate hierarchy should they reside?

## Resolution

Resolved 2026-10-07.

### 1. Module Placement

Create a dedicated forward term-structure module:
- `crates/libitofin/src/termstructures/forward/mod.rs`
- `crates/libitofin/src/termstructures/forward/implied_forward.rs`

Re-exported in `crates/libitofin/src/termstructures/mod.rs` and `crates/libitofin/src/lib.rs` as:
```rust
pub mod forward {
    pub use implied_forward::*;
}
```

### 2. Core Data Types

```rust
/// Market convention defining the quoting currency and payoff structure
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MarketConvention {
    /// European cash or physically settled vanilla options (e.g. SPX, NDX, SX5E)
    #[default]
    EuropeanVanilla,
    /// Coin-margined inverse options (e.g. Deribit BTC/ETH) where quotes and payoffs are in cryptocurrency
    CryptoInverse,
    /// American equity options with early-exercise strike corridor filtering
    AmericanEquity {
        spot: f64,
        atm_vol: Option<f64>,
    },
}

/// A two-sided quote pair at a single strike
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionQuotePair {
    pub strike: f64,
    pub call_bid: f64,
    pub call_ask: f64,
    pub put_bid: f64,
    pub put_ask: f64,
}

impl OptionQuotePair {
    pub fn new(strike: f64, call_bid: f64, call_ask: f64, put_bid: f64, put_ask: f64) -> Self {
        Self { strike, call_bid, call_ask, put_bid, put_ask }
    }
}
```

### 3. Engine Configuration & Builder

```rust
#[derive(Debug, Clone)]
pub struct ImpliedForwardConfig {
    /// External discount factor D = exp(-r*T). If None, solved jointly from quotes.
    pub discount_factor: Option<f64>,
    /// Underlying spot price for validation and American filtering
    pub spot: Option<f64>,
    /// Minimum required valid pairs (default: 2 for joint, 1 for anchored)
    pub min_pairs: usize,
    /// Maximum pairs used for estimation (default: 50)
    pub max_pairs: usize,
    /// Spread floor regularizer (default: 1e-4)
    pub spread_floor: f64,
    /// Maximum allowable relative deviation from spot |F/S - 1.0| (default: 0.10)
    pub max_spot_deviation: Option<f64>,
    /// Multiplier for American ATM corridor: |ln(K/S)| <= kappa * sigma * sqrt(T) (default: 0.5)
    pub american_kappa: f64,
}

impl Default for ImpliedForwardConfig {
    fn default() -> Self {
        Self {
            discount_factor: None,
            spot: None,
            min_pairs: 2,
            max_pairs: 50,
            spread_floor: 1e-4,
            max_spot_deviation: Some(0.10),
            american_kappa: 0.5,
        }
    }
}
```

### 4. Output Result & Error Types

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct ImpliedForwardResult {
    /// Weighted point estimate F*
    pub forward: f64,
    /// Strict synthetic no-arbitrage bid bound (max over ATM strikes)
    pub forward_bid_strict: f64,
    /// Strict synthetic no-arbitrage ask bound (min over ATM strikes)
    pub forward_ask_strict: f64,
    /// Robust synthetic spread-weighted bid
    pub forward_bid_robust: f64,
    /// Robust synthetic spread-weighted ask
    pub forward_ask_robust: f64,
    /// Implied or anchored discount factor D
    pub discount_factor: f64,
    /// Implied annualized cost of carry (r - q) = -ln(D) / T
    pub implied_carry_rate: Option<f64>,
    /// Diagnostics on quote quality, crossed quotes, and box arbitrage
    pub diagnostics: ForwardDiagnostics,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ForwardError {
    #[error("insufficient valid strike pairs: expected at least {required}, found {found}")]
    InsufficientPairs { required: usize, found: usize },
    #[error("degenerate strike distribution: strike variance is zero")]
    DegenerateStrikes,
    #[error("invalid discount factor: D = {0}")]
    InvalidDiscountFactor(f64),
    #[error("invalid expiry: T must be strictly positive, got {0}")]
    InvalidExpiry(f64),
}
```

### 5. High-Level Calculation Entrypoints

```rust
pub struct ImpliedForward;

impl ImpliedForward {
    /// Calculate implied forward and discount factor from quote pairs
    pub fn calculate(
        quotes: &[OptionQuotePair],
        convention: MarketConvention,
        config: &ImpliedForwardConfig,
        expiry_years: f64,
    ) -> Result<ImpliedForwardResult, ForwardError> {
        // Implementation following WLS + synthetic bounds + diagnostics
    }
}
```
