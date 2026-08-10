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
| Fixed-rate bonds (cached) | `FixedRateBond` + `DiscountingBondEngine` | `bonds.cpp` `testCachedFixed` | bond1–3 @ 1e-6 (plain / varying coupons / next-to-last stub) |
| Fixed-rate bonds (given dates) | `FixedRateBond` + `Schedule::with_metadata` | `bonds.cpp` `testFixedBondWithGivenDates` | schedule ≡ date-vector copy @ 1e-6 (plain / varying / stub Actual360) |
| Callable / puttable fixed bonds | `CallableFixedRateBond` + `TreeCallableFixedRateBondEngine` | `callablebonds.cpp` (HW tree) | Tree engine (Hull-White); identity-verified vs straight bond |
| Convertible bonds (TF binomial) | `ConvertibleFixedCouponBond` / `ConvertibleZeroCouponBond` + `BinomialConvertibleEngine` (CRR) | `convertiblebonds.cpp` `testBond` | OTM ≈ credit-spread vanilla @ 1e-2 / 2e-2 (1001 steps); ATM exceeds straight bond |
| Bond forward | `BondForward` (spot-minus-income) | `ql/instruments/bondforward` | Fair-strike NPV≈0; income-free dirty fwd = spot/DF; clean = dirty − AI |
| Bates (log-normal jumps) | `BatesProcess` / `BatesModel` / `BatesEngine` (Gatheral + `addOnTerm`) | `batesmodel.cpp` `testAnalyticVsBlack` | Tiny λ/δ → Black @ 2e-7; λ→0 ≡ Heston Gatheral |
| G2++ (affine + swaption + process + dynamics) | `G2` / `G2Dynamics` + `G2SwaptionEngine` + `G2Process` | `g2`, `g2swaptionengine`, `g2process`, `twofactormodel` | Affine pins; payer⇔−receiver; `r=φ+x+y`; joint OU array cov |
| FD nine-point / mixed ∂² | `NinePointLinearOp`, `second_order_mixed_derivative_op` | `ninepointlinearop`, `secondordermixedderivativeop` | `f=xy` → 1 on uniform 2D grid; annihilates f(x), g(y) |
| FdmG2Op | `FdmG2Op` | `fdmg2op.{hpp,cpp}` | apply = dirs + mixed; ρ=0 kills mixed; splitting inverts; φ̄ discount |
| Fdm2Dim / FdmG2 solvers | `FdmSolverDesc`, `Fdm2DimSolver`, `FdmG2Solver` | `fdm2dimsolver`, `fdmg2solver` | Zero-op preserves payoff; G2 constant→discount-ish; zero payoff→0 |
| FdmSimpleProcess1dMesher | `fdm_simple_process_1d_mesher` | `fdmsimpleprocess1dmesher` | OU endpoints = quantile evolve; avg = mean of per-t grids; FdG2 layout smoke |
| FdmAffineModelTermStructure | `FdmAffineModelTermStructure` | `fdmaffinemodeltermstructure` | G2 origin≡curve; factors≡discountBond; setVariable notifies |
| FdmAffineModelSwapInnerValue (G2) | `FdmAffineModelSwapInnerValue` | `fdmaffinemodelswapinnervalue` | ATM≈0; deep ITM payer>0; avg=inner; setVariable reuse |
| FdG2SwaptionEngine | `FdG2SwaptionEngine` (Hundsdorfer default) | `fdg2swaptionengine` / `testCachedG2Values` | ITM European>0; ≈ analytic G2; Bermudan≥European; cached FDM @ 5e-3 |
| HundsdorferScheme | `HundsdorferScheme` + factories | `hundsdorferscheme` | BS replay; diagonal closed form; dual BC apply cycles |
| TreeLattice2D | `TwoFactorTree` / `TreeLattice2D` | `lattice2d.hpp` | size=product; ρ=0⇒independent; |ρ| HW term; neg ρ flips m; probs∑≈1; grid fails; flat rollback |
| G2 two-factor tree | `TwoFactorShortRateTree` / `G2::tree` | `twofactormodel` / `g2` | discount=exp(-(φ+x+y)dt); root φ-only; product size; builds under analytic φ |
| TreeG2SwaptionEngine | `TreeG2SwaptionEngine` + date snapping | `treeswaptionengine` / `testCachedG2Values` | cached tree Bermudan @ 5e-3; Bermudan≥European analytic |
| Tree HW Bermudan (cached) | `TreeSwaptionEngine` + `DiscretizedSwaption` snap | `bermudanswaption.cpp` `testCachedValues` | ITM/ATM/OTM @ 1e-4 (non-par coupons) |
| Zero-coupon bonds (cached) | `ZeroCouponBond` + `DiscountingBondEngine` | `bonds.cpp` `testCachedZero` | three maturities @ 1e-6 |
| Floating bonds (cached) | `FloatingRateBond` + `USDLibor` / `Libor` | `bonds.cpp` `testCachedFloating` | bond1–4 @ 1e-6 (plain / dual / spreads / fixing+ex-coupon) |
| Brazilian NTN-F (Andima) | `BondFunctions` yield clean/dirty + `Business252` | `bonds.cpp` `testBrazilianCached` | six maturities @ 1e-4 |
| Bond price/yield consistency | `BondFunctions` yield clean/dirty ↔ `yield_rate` | `bonds.cpp` `testYield` | clean/dirty round-trip @ 1e-7 |
| SA R2048 (date-vector schedule) | `Schedule::with_metadata` + yield dirty | `bonds.cpp` `testBondFromScheduleWithDateVector` | dirty 95.75706 @ 1e-5 |
| Bond price/ATM rate consistency | `BondFunctions::atm_rate` + `BondPrice` | `bonds.cpp` `testAtmRate` | clean/dirty → coupon @ 1e-7 |
| Bond theoretical price/yield | `DiscountingBondEngine` ↔ Continuous yield | `bonds.cpp` `testTheoretical` | engine ≡ yield price; yield recovery @ 1e-7 |
| Bond price/z-spread consistency | `BondFunctions` z-spread clean/dirty ↔ solve | `bonds.cpp` `testZspread` | clean/dirty round-trip @ 1e-7 |
| Bond price/yield (cached) | `FixedRateBond` + engine + `BondFunctions` | `bonds.cpp` `testCached` | bond1–3 price/yield @ 1e-6 (schedule vs bare ISMA) |
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
| GSR / LMM | `ql/models/shortrate`, `ql/models/marketmodels` | P2 |
| Credit / CDS | `ql/termstructures/credit`, `ql/pricingengines/credit` | P2 (demoted) |
| Inflation | `ql/termstructures/inflation`, CPI/YoY instruments | P2 (demoted) |
| Full cbindgen C ABI | planned `libitofin-ffi` | P2 |

## How to extend this map

1. Pick a QuantLib `test-suite/*.cpp` case.
2. Port the instrument/engine slice with matching inputs.
3. Assert numbers within the C++ tolerance.
4. Move the row from “Not started” to “Covered” with the tolerance noted.
