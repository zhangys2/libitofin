---
title: Total Variance Cubic Smile FFI, Python, and Go Bindings
type: Task (AFK)
status: closed
blocked_by: []
claimed_by: null
map: ../maps/total-variance-cubic-smile.md
resolution: "Implemented TotalVarianceCubicSmileSection across Rust core, C-FFI, PyO3 Python bindings, and Go SDK. All Rust unit tests (6/6), C-FFI unit tests (1/1), Python pytest suite (5/5), and full workspace clippy checks passed with 0 warnings."
---

# Total Variance Cubic Smile FFI, Python, and Go Bindings

## Question

How should `TotalVarianceCubicSmileSection` be exposed through C-FFI (`crates/libitofin-ffi`), PyO3 Python bindings (`crates/itofin-py`), and Go SDK (`sdk/go`)?

## Resolution

1. **Rust Core:**
   - Implemented `TotalVarianceCubicSmileSection`, `RogerLeeWing`, `RogerLeeWingConfig`, `ButterflyArbitrageReport`, and `ArbitrageFallbackPolicy` in `crates/libitofin/src/termstructures/volatility/total_variance_cubic_smile.rs`.
   - Implemented `SmileSection` trait with `volatility_impl` and `variance`.
   - Verified with 6 unit tests covering Roger Lee wing clamping, Smoothstep $C^2$ bridge, least-squares QR fitting, butterfly arbitrage detection & adaptive ramp, and input validation.

2. **C-FFI (`crates/libitofin-ffi`):**
   - Added `itofin_total_variance_cubic_smile_new`, `itofin_total_variance_cubic_smile_query`, and `itofin_total_variance_cubic_smile_check_arbitrage` in `crates/libitofin-ffi/src/cubicsmile_api.rs`.
   - Updated C header definitions in `crates/libitofin-ffi/include/itofin.h`.
   - Verified with C-FFI integration tests in `cubicsmile_api::tests`.

3. **Python Bindings (`crates/itofin-py`):**
   - Implemented `PyTotalVarianceCubicSmileSection` and `PyButterflyArbitrageReport` in `crates/itofin-py/src/cubicsmile.rs`.
   - Exposed under `itofin.termstructures` in `crates/itofin-py/src/lib.rs`.
   - Verified with pytest suite `crates/itofin-py/tests/test_total_variance_cubic_smile.py` (5 passed).

4. **Go SDK (`sdk/go`):**
   - Implemented `TotalVarianceCubicSmileSection` and `ButterflyArbitrageReport` in `sdk/go/total_variance_cubic_smile.go`.
   - Added unit test in `sdk/go/total_variance_cubic_smile_test.go`.
