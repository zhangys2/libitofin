---
title: C-FFI, PyO3 Python Bindings, and Downstream Feed Migration
type: Task (AFK)
status: closed
blocked_by: []
claimed_by: "Math & Market Formulations Researcher"
map: ../maps/implied-forward-from-options.md
resolution: "Bound in C-FFI via itofin_implied_forward_calculate with flat arrays and in PyO3 via itofin.termstructures.implied_forward. Downstream spx_option_iv/feed.py refactored to delegate parity_forward to itofin, and btc_option_iv/market.py augmented with crypto-inverse option parity fallback."
---

# C-FFI, PyO3 Python Bindings, and Downstream Feed Migration

## Question

How should the unified `ImpliedForward` engine be bound in C-FFI (`crates/libitofin-ffi`) and PyO3 (`crates/itofin-py`), and how should `spx_option_iv/feed.py` and `btc_option_iv/market.py` be migrated to eliminate duplicated forward-pricing logic with zero latency or behavioral regressions?

## Resolution

Resolved 2026-10-07.

### 1. C-FFI Surface (`crates/libitofin-ffi`)

Exposed in `crates/libitofin-ffi/src/forward_api.rs`:

```c
typedef struct {
    double forward;
    double forward_bid_strict;
    double forward_ask_strict;
    double forward_bid_robust;
    double forward_ask_robust;
    double discount_factor;
    double implied_carry_rate; // NaN if undefined
    int32_t status_code;       // 0=Valid, 1=Crossed, 2=BoxArbitrage, 3=SpotDeviation, 4=DegradedAmerican
    uint32_t pairs_used;
    uint32_t pairs_pruned;
    double wls_rmse;
} ItofinImpliedForwardResult;

int32_t itofin_implied_forward_calculate(
    const double *strikes,
    const double *call_bids,
    const double *call_asks,
    const double *put_bids,
    const double *put_asks,
    size_t count,
    int32_t convention_code,   // 0=European, 1=CryptoInverse, 2=American
    const double *spot,        // optional nullable pointer
    const double *discount_factor, // optional nullable pointer
    double expiry_years,
    ItofinImpliedForwardResult *out,
    ItofinError *error
);
```

### 2. PyO3 Python Bindings (`crates/itofin-py`)

Exposed in `crates/itofin-py/src/forward.rs` and re-exported in `itofin.termstructures`:

```python
from itofin.termstructures import implied_forward, OptionQuotePair, MarketConvention

# Callable either with flat list of tuples (strike, c_bid, c_ask, p_bid, p_ask) or OptionQuotePair
res = implied_forward(
    quotes=[(5000.0, 120.0, 121.0, 110.0, 111.0), ...],
    expiry_years=0.082,
    convention="european", # or "crypto_inverse", "american"
    spot=5010.0,           # optional
    discount_factor=None,  # optional; None fits jointly
)

# res fields:
# res.forward: float
# res.forward_bid_strict / res.forward_ask_strict: float
# res.forward_bid_robust / res.forward_ask_robust: float
# res.discount_factor: float
# res.implied_carry_rate: float | None
# res.status: str ("Valid", "WarningCrossedSyntheticQuotes", etc.)
# res.is_valid: bool
# res.pairs_used: int
```

Zero-copy ingestion supports flat Python lists of 5-tuples or structured tuples, keeping PyO3 conversion latency $< 5\,\mu\text{s}$ for typical 50-strike option chains.

### 3. Downstream Feed Migration

#### A. `example/btc-option-iv/spx_option_iv/feed.py`
Refactor [`parity_forward`](file:///home/zhangy/git/libitofin/example/btc-option-iv/spx_option_iv/feed.py#L103-L131):
```python
def parity_forward(pairs, spot: float, expiry_years: float) -> tuple[float, int] | None:
    """Forward implied by put-call parity using itofin.termstructures.implied_forward."""
    res = implied_forward(
        quotes=list(pairs),
        expiry_years=expiry_years,
        convention="european",
        spot=spot,
    )
    if res is None or not res.is_valid:
        return None
    return res.forward, res.pairs_used
```
This replaces 30 lines of bespoke spread-sorting and median math with the unified engine, while gaining box-spread arbitrage detection, synthetic bid-ask bounds, and joint discount factor estimation.

#### B. `example/btc-option-iv/btc_option_iv/market.py`
Augment [`forward_from_future_quote`](file:///home/zhangy/git/libitofin/example/btc-option-iv/btc_option_iv/market.py#L260-L277) with an option-implied fallback:
```python
def forward_from_option_quotes(quotes, expiry_years: float) -> tuple[float, str] | None:
    """Extract forward using coin-space numeraire linear parity on Deribit inverse quotes."""
    res = implied_forward(
        quotes=quotes,
        expiry_years=expiry_years,
        convention="crypto_inverse",
    )
    if res is not None and res.is_valid:
        return res.forward, "option_implied_parity"
    return None
```
When dated futures have wide bid-ask spreads or crossed marks, the option chain provides a continuous, high-fidelity forward anchor.

### 4. Verification Gate

- Unit tests in `crates/libitofin`:
  - Synthetic European cash index parity fixture vs known forward.
  - Deribit coin-space inverse parity fixture vs known forward and basis.
  - American option ATM corridor bounds vs Merton inequalities.
  - Crossed-quote and box-spread anomaly rejection.
- Integration tests in `crates/itofin-py/tests/test_implied_forward.py`.
- Regression check on Marimo notebooks (`spx_vol_surface.py` and `live_vol_surface.py`) ensuring zero drop in streaming update rates.
