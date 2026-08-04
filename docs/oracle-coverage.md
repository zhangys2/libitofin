# QuantLib oracle coverage map

This document tracks how libitofin features map to QuantLib `test-suite/*.cpp`
oracles. A feature is **done** only when the matching numbers pass within the
documented tolerance.

## Scope note

Backlog priority is **rates + equity**. Credit and inflation remain large absolute
holes (see QuantLib-1 `ql/termstructures/{credit,inflation}` and matching
engines) but are demoted until promoted by desk need. Upstream epic #676 covers
credit.

## Covered / partially covered

| Domain | libitofin surface | QuantLib oracle(s) | Status |
|--------|-------------------|--------------------|--------|
| European vanilla | `AnalyticEuropeanEngine`, FDM/MC European | `europeanoption.cpp` | Done (Milestone 1) |
| Binomial (CRR) vanilla | `BinomialVanillaEngine`, `CoxRossRubinstein` | `europeanoption.cpp` (vs analytic) | European/American; converges to Black-Scholes; groundwork for convertibles |
| American vanilla | `FdmAmericanEngine`, `AmericanExercise` | `americanoption.cpp` `testFdValues` / Ju (1999) | Done @ 8e-2 |
| Bermudan vanilla | `FdmBermudanEngine`, `BermudanExercise` | `americanoption.cpp` (Bermudan FD path) | Discrete-exercise FD; identity-bounded by European/American |
| Heston | analytic + calibration | `hestonmodel.cpp` | Core done |
| Hull–White / short rate | calibration, tree swaption | `shortratemodels.cpp`, swaption suite | Core done |
| Swaps / OIS / swaptions / caps | instruments + engines | swap/swaption/capfloor suites | Core done |
| Float-float swap | `FloatFloatSwap` | `ql/instruments/floatfloatswap` | Two-Ibor-leg slice; identity-verified (identical legs, fair spread) |
| XCCY basis swap | `XccyBasisSwap` | `ql/instruments/` (cross-currency) | Float-float w/ notional exchange; identity-verified (degenerate, FX view, fair spread) |
| Fixed-rate bonds | `FixedRateBond` + discounting | `bonds.cpp` cached fixed | Done |
| Callable / puttable fixed bonds | `CallableFixedRateBond` + `TreeCallableFixedRateBondEngine` | `callablebonds.cpp` (HW tree) | Tree engine (Hull-White); identity-verified vs straight bond |
| Zero / floating bonds | `ZeroCouponBond`, `FloatingRateBond` | bonds suite (extend) | Smoke done; cached oracles follow-up |
| FRA | `ForwardRateAgreement` + `FraRateHelper` | `ratehelpers.cpp`, FRA examples | Instrument + helper |
| CMS / digital coupons | `CmsCoupon`, `DigitalIborCoupon` | CMS/digital suites | Raw-rate / cash-or-nothing slice |
| CMS swap | `CmsSwap` | `ql/instruments/*cms*` | Fixed-vs-CMS (raw rate); identity-verified (fair rate) |
| Asset swap | `AssetSwap` | `ql/instruments/assetswap` | Par asset swap; leg construction per `assetswap.cpp`; identity-verified |
| Barrier / Asian | `BarrierOption`, geometric Asian | `barrieroption.cpp`, `asianoptions.cpp` | First analytic slice |
| Money / FX rates | `Money`, `ExchangeRate` | `money.cpp`, `exchangerate.cpp` | Value types |
| FX forward | `FxForward` | money layer (covered interest parity) | Outright; parity-identity verified |
| C ABI | `libitofin-ffi` | n/a | Version + error stubs only |

## Not started (rates+equity desk)

| Domain | QuantLib location | Priority |
|--------|-------------------|----------|
| Convertible bonds, bond forward | `ql/instruments/bonds/`, `bondforward` | P1 |
| Bates / G2 / GSR / LMM | `ql/processes`, `ql/models/marketmodels` | P2 |
| Credit / CDS | `ql/termstructures/credit`, `ql/pricingengines/credit` | P2 (demoted) |
| Inflation | `ql/termstructures/inflation`, CPI/YoY instruments | P2 (demoted) |
| Full cbindgen C ABI | planned `libitofin-ffi` | P2 |

## How to extend this map

1. Pick a QuantLib `test-suite/*.cpp` case.
2. Port the instrument/engine slice with matching inputs.
3. Assert numbers within the C++ tolerance.
4. Move the row from “Not started” to “Covered” with the tolerance noted.
