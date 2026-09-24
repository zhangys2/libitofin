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
| QMC European | `MakeMcEuropeanEngine::<LowDiscrepancy>` | `europeanoption.cpp` `testQmcEngines` | 108 call/put grid, 1 step, 4095 samples; abs(QMC−analytic)/spot ≤ 0.01 |
| European asset-or-nothing | `AssetOrNothingPayoff` + `BlackCalculator` visitor | `digitaloption.cpp` `testAssetOrNothingEuropeanValues` | Haug p.90 put 20.2069 @ 1e-4; call+put ≡ S e^{-qT} |
| European gap | `GapPayoff` + `BlackCalculator` visitor | `digitaloption.cpp` `testGapEuropeanValues` | Haug p.88 call −0.0053 @ 1e-4; gap ≡ vanilla ± cash-or-nothing |
| American digital at-hit | `AnalyticDigitalAmericanEngine` | `digitaloption.cpp` `testCashAtHitOrNothingAmericanValues` / `testAssetAtHitOrNothingAmericanValues` | Haug p.95 cash+asset at-hit @ 1e-4 (ITM @ 1e-16) |
| American digital at-expiry (knock-in) | `AnalyticDigitalAmericanEngine` + `AmericanPayoffAtExpiry` | `digitaloption.cpp` `testCashAtExpiryOrNothingAmericanValues` / `testAssetAtExpiryOrNothingAmericanValues` | Haug p.95 knock-in cash+asset at-expiry @ 1e-4 (ITM discounted @ 1e-12) |
| American digital at-expiry (knock-out) | `AnalyticDigitalAmericanKOEngine` | `digitaloption.cpp` `testCashAtExpiryOrNothingAmericanValues` / `testAssetAtExpiryOrNothingAmericanValues` | Haug p.95 KO cash 4.9081/3.0461 + asset 40.1574/17.2983 @ 1e-4; out-of-bonds 0; KI+KO ≡ prepaid |
| Analytic quanto vanilla | `QuantoEuropeanEngine` (`QuantoEngine<VanillaOption, AnalyticEuropeanEngine>`) | `quantooption.cpp` `testValues` | Haug call 5.3280/1.5, put 8.1636 @ 1e-4; NPV/greeks ≡ quanto-q Black |
| Analytic quanto greeks | `QuantoEuropeanEngine` | `quantooption.cpp` `testGreeks` | FD bump grid (δ/γ/θ/ρ/divRho/vega/qρ/qvega/qλ) @ 1e-5 relative to spot |
| Dedicated QuantoVanillaOption | `QuantoVanillaOption` (`QuantoOptionResults`) | `quantooption.cpp` `testValues` | Dedicated instrument with typed quanto greeks (`qvega`, `qrho`, `qlambda`) matching inner analytical relations @ 1e-12; Haug values @ 1e-4; expired zeroes @ 1e-12 |
| Analytic quanto barrier | `QuantoBarrierEngine` (`QuantoEngine<BarrierOption, AnalyticBarrierEngine>`) | `quantooption.cpp` `testBarrierValues` | Haug DownOut call 8.247 / put 2.274, DownIn put 2.85 @ tol 0.5; NPV ≡ quanto-q barrier |
| Analytic quanto forward | `QuantoForwardEuropeanEngine` + `ForwardVanillaOption` / `AnalyticForwardVanillaEngine` | `quantooption.cpp` `testForwardValues` | Haug reset=0 call 5.3280/1.5 / put 8.1636; FinCAD reset=0.25 call 2.0171 / put 6.7296 @ 1e-4; reset=0 ≡ quanto European |
| Analytic quanto forward greeks | `QuantoForwardEuropeanEngine` | `quantooption.cpp` `testForwardGreeks` | FD bump grid (δ/γ/θ/ρ/divRho/vega/qρ/qvega/qλ) @ 1e-5 relative to spot; reset 6/9mo |
| Analytic quanto forward performance | `QuantoForwardPerformanceEuropeanEngine` + `AnalyticForwardPerformanceVanillaEngine` | `quantooption.cpp` `testForwardPerformanceValues` | reset=0 call 5.3280/150 / put 0.0816; reset=0.25 call 0.0201 / put 0.0672 @ 1e-4; NPV ≈ plain forward / 100 |
| Analytic forward vanilla (non-quanto) | `AnalyticForwardVanillaEngine` | `forwardoption.cpp` `testValues` | Haug p.37 call 4.4064 / put 8.2971 @ 1e-4 |
| Analytic forward vanilla greeks | `AnalyticForwardVanillaEngine` | `forwardoption.cpp` `testGreeks` | FD bump grid (δ/γ/θ/ρ/divRho/vega) @ 1e-5 relative to spot; reset 6/9mo |
| Analytic forward vanilla greeks-init | `BinomialForwardVanillaEngine` CRR 300 | `forwardoption.cpp` `testGreeksInitialization` | Inner leaves ρ/divRho/vega Null (forward too); forward δ Null because `strikeSensitivity` is missing |
| Forward-start MC (BS) | `McForwardEuropeanBsEngine` PR 100×5000 seed 42 | `forwardoption.cpp` `testMCPrices` | Call moneyness 0.8..1.2 vs analytic, relativeError vs S=100 @ [0.002, 0.001, 0.0006, 5e-4, 5e-4] |
| Forward-start MC (flat Heston vs BS) | `McForwardEuropeanHestonEngine` LD 50×4095 seed 42 | `forwardoption.cpp` `testHestonMCPrices` Test 1 | Call/Put moneyness 0.8..1.2 vs analytic BS; tols [7e-4, 8e-4, 6e-4, 5e-4, 5e-4] / [6e-4, 5e-4, 6e-4, 1e-3, 1e-3] |
| Forward-start MC (smile Heston t=0 vs vanilla) | `McForwardEuropeanHestonEngine` LD 50×4095 seed 42 | `forwardoption.cpp` `testHestonMCPrices` Test 2 MC | Call/Put moneyness 0.8..1.2 vs `AnalyticHestonEngine(model, 96)` at reset=today; tols [9e-4, 9e-4, 6e-4, 5e-4, 5e-4] / [6e-4, 5e-4, 8e-4, 2e-3, 2e-3] |
| Forward-start analytic Heston (t=0 vs vanilla) | `AnalyticHestonForwardEuropeanEngine` T=0 P1/P2 | `forwardoption.cpp` `testHestonMCPrices` Test 2 analytic | Call/Put moneyness 0.8..1.2 vs `AnalyticHestonEngine(model, 96)` at reset=today @ 5e-4; `tReset>0` deferred |
| Analytic forward performance (non-quanto) | `AnalyticForwardPerformanceVanillaEngine` | `forwardoption.cpp` `testPerformanceValues` | Haug × e^{-q t_reset}/S @ 1e-4 |
| Analytic forward performance greeks | `AnalyticForwardPerformanceVanillaEngine` | `forwardoption.cpp` `testPerformanceGreeks` | FD bump grid (δ/γ/θ/ρ/divRho/vega) @ 1e-5 relative to spot; reset 6/9mo |
| Analytic quanto double-barrier | `QuantoDoubleBarrierEngine` + `AnalyticDoubleBarrierEngine` | `quantooption.cpp` `testDoubleBarrierValues` | KnockOut call 3.4623 / 0.5236, put 1.1320; KnockIn call 2.6313 / 1.9305 @ 1e-4; NPV ≡ quanto-q double barrier |
| FD vanilla escrowed dividends | `FdBlackScholesVanillaEngine` + `EscrowedDividendAdjustment` (European) | `dividendoption.cpp` `testFdEuropeanDegenerate` / `testEscrowedDividendModel` | empty/zero divs @ 1e-6; FD 200×400 vs Black on prepaid spot @ 1e-3 |
| FD American escrowed dividends | `FdmEscrowedLogInnerValueCalculator` + American/Bermudan on `FdBlackScholesVanillaEngine` | `americanoption.cpp` `testEscrowedVsSpotAmericanOption`; `dividendoption.cpp` `testFdAmericanDegenerate` | escrowed vs spot NPV/delta @ 1e-2 (vol × S/(S−D)); empty/zero divs @ 1e-6 |
| FD quanto helper | `FdmQuantoHelper` + `QuantoTermStructure` + mesher/op/engine | `quantooption.cpp` `testFDMQuantoHelper` / `testPDEOptionValues` | adj @ 1e-10; mesher bounds @ 1e-10; FD vs quanto-q Black NPV @ 2e-4, delta @ 1e-4 |
| FD American quanto | `FdBlackScholesVanillaEngine` American + cash div + quanto | `quantooption.cpp` `testAmericanQuantoOption` | cached 8.90611734 @ 1e-4 (BS and local vol); BS≡LV @ 1e-6 |
| FD Heston American quanto | `FdHestonVanillaEngine` American + cash div + quanto | `quantooption.cpp` `testAmericanQuantoOption` | near-Black Heston (`σ=1e-4`) cached 8.90611734 @ 1e-4 (Hundsdorfer undamped; 2-D implicit Euler is #636) |
| FD Heston-SLV American quanto | `FdHestonVanillaEngine` + constant leverage | `quantooption.cpp` `testAmericanQuantoOption` | `L=2`, `v0=θ=0.25 σ²` cached 8.90611734 @ 1e-4; equity mesher vol × path-averaged `L` |
| FD Heston-SLV variance mesher | `FdmHestonLocalVolatilityVarianceMesher` | `fdheston.cpp` `testFdmHestonVarianceMesher` | CIR locations @ 1e-6; const `L=2.5` scales vol exactly @ 1e-6; parable `L` path-average @ 1e-3 |
| Black-Scholes process (variance curve) | `GeneralizedBlackScholesProcess` + linear `BlackVarianceCurve` → `LocalVolCurve` | `ql/processes/blackscholesprocess.cpp` `localVolatility()` | strike-independent; `expectation`/`variance`/`evolve` exact vs `t σ_B^2(t)` increment |
| Process discretization | `EulerDiscretization` + `DiscretizedProcess` / `DiscretizedProcess1D` | identity (`ql/processes/eulerdiscretization.cpp`) | wrap-only adapters; source exact overrides kept; invalid dt/dims/nonfinite fail |
| Binomial (CRR) vanilla | `BinomialVanillaEngine`, `CoxRossRubinstein` | `europeanoption.cpp` (vs analytic) | European/American; converges to Black-Scholes; groundwork for convertibles |
| American vanilla | `FdmAmericanEngine`, `AmericanExercise` | `americanoption.cpp` `testFdValues` / Ju (1999) | Done @ 8e-2 |
| Barone-Adesi–Whaley American | `BaroneAdesiWhaleyApproximationEngine` | `americanoption.cpp` `testBaroneAdesiWhaleyValues` | Haug p.24 NPV @ 3e-3 (QL table tolerance); negative-rate reject |
| Bjerksund–Stensland American | `BjerksundStenslandApproximationEngine` | `americanoption.cpp` `testBjerksundStenslandValues` / `testBjerksundStenslandEuropeanGreeks` / `testSingleBjerksundStenslandGreeks` / `testBjerksundStenslandAmericanGreeks` | Haug/VBA/R 8-row table @ 5e-5 (QL tolerance); early-exercise European greeks equivalence (with Thirty360 European vol) @ 1000*EPSILON; single-greeks pin @ 1e6*EPSILON with exerciseType/thetaPerDay; Call/Put mixed-DC American greeks vs FD; full 2,880-point American-greeks FD grid deferred |
| Ju quadratic American | `JuQuadraticApproximationEngine` | `americanoption.cpp` `testJuValues` | Ju (1999) 47-row table @ 1e-3 (QL tolerance); delta & gamma vs FD @ 1e-3; zero-dividend American call matches European greeks @ 1e-12; negative-rate reject |
| Bermudan vanilla | `FdmBermudanEngine`, `BermudanExercise` | `americanoption.cpp` (Bermudan FD path) | Discrete-exercise FD; identity-bounded by European/American |
| Heston | analytic + calibration | `hestonmodel.cpp` | Core done |
| COS Heston | `CosHestonEngine` | `hestonmodel.cpp` COS cached + cumulants | 4 cached prices @ 1e-10; live spot/param; c1–c4 vs QL fixture |
| Exponential-fitting Heston | `ExponentialFittingHestonEngine` | `hestonmodel.cpp` extreme-moneyness / CV grid | 88 fitted-quadrature prices @ 1e-8; six control variates; AsymptoticChF α=−0.5 |
| Heston FD barrier cached | `FdHestonBarrierEngine` 200×400×100 | `hestonmodel.cpp` `testFdBarrierVsCached` | DownOut 9.0246 / DownIn 7.7627 @ 1e-3 |
| Heston FD vanilla cached | `FdHestonVanillaEngine` 100×200×100 | `hestonmodel.cpp` `testFdVanillaVsCached` | put 0.06325 @ 1e-4 |
| Heston FD vanilla + cash dividends | `FdHestonVanillaEngine` 200×400×100 | `hestonmodel.cpp` `testFdVanillaWithDividendsVsCached` | call 12.946 @ 5e-3 |
| Heston FD American vs BS FD | `FdHestonVanillaEngine` / `FdBlackScholesVanillaEngine` 200×400 | `hestonmodel.cpp` `testFdAmerican` | near-Black Heston put ≡ BS FD @ 1e-3 |
| Heston FD vs Black (Hundsdorfer) | `FdHestonVanillaEngine` 100×400×3 | `fdheston.cpp` `testFdmHestonBlackScholes` | near-Black puts S=8..12 ≡ analytic European @ 1e-4 (ExplicitEuler deferred) |
| Heston FD Haug barrier vs BS | `FdHestonBarrierEngine` 200×101×3 | `fdheston.cpp` `testFdmHestonBarrierVsBlackScholes` | QL `values[]` (Haug p.72 layout; 11 rows non-Haug q/r/t/v) vs `AnalyticBarrierEngine` @ 0.25% relative |
| Heston FD American NPV | `FdHestonVanillaEngine` 200×100×50 | `fdheston.cpp` `testFdmHestonAmerican` | put 5.66032 @ 1e-2 (δ/γ deferred) |
| Heston FD Ikonen–Toivanen | `FdHestonVanillaEngine` 100×400 | `fdheston.cpp` `testFdmHestonIkonenToivanen` | American puts S=8..12 table @ 1e-3 |
| Heston FD American + cash div | `FdHestonVanillaEngine` 50×100×50 | `fdheston.cpp` `testFdmHestonEuropeanWithDividends` | put 7.38216 @ 1e-2 (δ/γ deferred) |
| Heston FD ADI convergence | `FdHestonVanillaEngine` 60×101×51 | `fdheston.cpp` `testFdmHestonConvergence` | Hundsdorfer / MCS / mod-Hundsdorfer / Craig–Sneyd vs analytic @ 2% or 0.002 (TrBDF2 #636; CN deferred) |
| Heston FD Method of Lines | `FdHestonVanillaEngine` 10×21×7 / `FdHestonBarrierEngine` 100×31×11 | `fdheston.cpp` `testMethodOfLinesAndCN` | American put / DownOut barrier MOL ≡ Hundsdorfer @ 0.005 / 0.01; CN still #636 (true 2-D `CrankNicolsonScheme`, not Douglas θ=0.5) |
| Heston FD American call–put parity | `FdHestonVanillaEngine` 200×25, 50 steps/year | `fdheston.cpp` `testAmericanCallPutParity` | Battauz Heston symmetry, two specs, American call ≡ transformed put @ 0.025 |
| Heston FD spurious oscillations | `FdmHestonSolver::gamma_at` 6×200×13 | `fdheston.cpp` `testSpuriousOscillations` | CS/HS/mod-HS/Douglas max \|Δγ\| > 0.01 on S∈[99,101]; Implicit/TrBDF2/CN still #636 |
| Heston FD UpOut barrier NPV | `FdHestonBarrierEngine` 50×400×100 | `fdheston.cpp` `testFdmHestonBarrier` | UpOut call 9.1530 @ 1e-2 (δ/γ deferred) |
| Hull–White / short rate | calibration, tree swaption | `shortratemodels.cpp`, swaption suite | Core done |
| GSR process (constant a/σ) | `GsrProcess` + `ForwardMeasureProcess1D` | `gsr.cpp` `testGsrProcess` | E/V vs Hull–White forward (f≡0) @ 1e-8; flat≡stepwise-equal; piecewise core deferred |
| LMM simple covariance | `LmExponentialCorrelationModel` + `LmLinearExponentialVolatilityModel` + `LfmCovarianceProxy` | `libormarketmodel.cpp` `testSimpleCovarianceModels` | corr/covar reconstr. @ 1e-14; lin-exp vol formula; process/engines deferred |
| Abcd market-model vol | `AbcdFunction` + `AbcdSquared` + `AbcdMathFunction` | `marketmodel.cpp` `testAbcdDegenerateCases`, `testAbcdVolatilityIntegration` | covar @ 1e-14 (degenerate); analytical vs SegmentIntegral @ 1e-4; evolvers/products deferred |
| MF state process | `MfStateProcess` | `markovfunctional.cpp` `testMfStateProcess` | diffusion buckets + variance (a=0 / a=0.01) @ 1e-10; MarkovFunctional model/engines deferred |
| Black–Karasinski dynamics | `BlackKarasinski` + `BlackKarasinskiDynamics` | identity (no QL suite) | log transform round-trip @ 1e-15; OU α/σ pins; `tree()`/fit deferred |
| Hull–White forward process | `HullWhiteForwardProcess` | identity (hybrid suite is engines) | f≡0 E/V vs closed form; T-forward drift Δ; `a>0`/`a=0` `M_T`; notify on set T @ 1e-12; hybrid join deferred |
| Heston SLV process | `HestonSLVProcess` | identity (`testDiffusionAndDriftSlvProcess` needs LV+FD) | const-L scales spot diffusion/drift; mixing scales √v row; evolve finite; FDM/MC models deferred |
| FDM SABR operator | `FdmSabrOp` | identity (`fdsabr.cpp` `testFdmSabrOp` needs engine) | closed-form L[f²]/L[x²]/L[fx] interior pins (ν≠1); Shared yield snapshot; engine/NoArb deferred |
| Bachelier cap/floor | `BachelierCapFloorEngine` + CapHelper Normal | `capfloor.cpp` `testBachelierOptionLetsDelta` | parity / vega FD / optionletsPrice; CapHelper Normal ≡ ATM; analytic δ vs forward FD @ 1e-6 |
| Cap/floor Black implied vol | `CapFloor::implied_volatility` | `capfloor.cpp` `testImpliedVolatility` | Black ShiftedLognormal grid @ 1e-8; Normal arm reduced round-trip |
| Cap/floor optionlet | `CapFloor::optionlet` | `capfloor.cpp` `testConsistency` recomposition | collar ≡ cap−floor @ 1e-10; Σ optionlet NPV ≡ parent @ 1e-10 (un-nested); overnight `from_overnight` indexes by `coupon_count()` |
| Cap/floor optionlets vega | Black/`BachelierCapFloorEngine` `optionletsVega`/`StdDev` | `blackcapfloorengine.cpp:160-166` | Σ optionletsVega ≡ vega; StdDev for cap/floor only |
| Cap/floor deepUpdate | `CapFloor::deep_update` | `capfloor.cpp:271-276` | clears calculated after NPV; reprice recovers prior NPV @ 1e-12 (coupon deepUpdate deferred) |
| Swaption Black implied vol | `Swaption::implied_volatility` | `swaption.cpp` `testImpliedVolatility` / `testImpliedVolatilityOis` | Spot Physical Black + reduced IBOR/OIS Spot/Cash/Forward cartesian cells @ 1e-8 (incl. Cash×Forward); reduced Spot Physical Normal |
| Black cap/floor delta | `BlackCapFloorEngine` `optionletsDelta` | `capfloor.cpp` `testOptionLetsDelta` | analytic vs forward FD @ 1e-6; discount/ATM-forward results |
| Tree cap/floor | `TreeCapFloorEngine` + `DiscretizedCapFloor` | QL tree collar oracle + CapHelper Normal | collar/cap/floor @ 1e-8 (30/100 steps); TimeGrid ctor; past-start known-fixing intrinsic; CapHelper Normal market/model pin + tree σ calibrate; tree→analytic rel <1e-3 @600. MC/G1d deferred |
| Overnight index future | `OvernightIndexFuture` + `OvernightIndexFutureRateHelper` / `SofrFutureRateHelper` | `sofrfutures.cpp` + holiday-clipped daily accrual | Juneteenth/bootstrap prices + curve nodes @ 1e-9; Simple/Compound holiday clip + today-fixing @ 1e-9; convexity/lifecycle |
| Iterative bootstrap robustness | `IterativeBootstrap` options on yield piecewise | independent QL 1.43 `oracle.py` | sign-aware bound widening; fallback scan; cached recovery; 99-iter limit @ 1e-12 nodes; yield factories only |
| Custom pillars | FRA/swap/OIS/inflation helpers `Pillar::CustomDate` | independent QL helper dates | window-valid custom pillars; invalid bounds; last-valid date recovery |
| Joint Ibor-Ibor yield curves | `JointYieldCurves` + `IborIborBasisSwapRateHelper` | `piecewiseyieldcurve.cpp` multi-curve + independent FRA/swap reprice | 3M/6M coupled GlobalBootstrap; 1e-12 reprice; live quote/date/fixing |
| Overnight-Ibor basis helper | `OvernightIborBasisSwapRateHelper` | experimental `basisswapratehelpers.cpp` + overnight_basis fixture | omitted/explicit discount + coupled OIS/Ibor @ 1e-12 rel |
| Municipal BMA swap + helper | `BMASwap` + `BMASwapRateHelper` | `piecewiseyieldcurve.cpp` BMA / independent fixture | ten 1Y–30Y quotes reprice @ 1e-9 (local bootstrap @ 1e-6); independent QL prices/holidays |
| Optionlet stripper (Black + Normal) | `OptionletStripper1` + adapter | `optionletstripper.cpp` nonflat roundtrip | lognormal and Normal grids vs flat-vol engine @ 2.5e-8 |
| Overnight optionlet strip + cap | `OptionletStripper1::new_overnight` + `CapFloor::from_overnight` | `optionletstripper.cpp` overnight + independent oracle | stripped nodes @ 1e-9; compounded overnight cap @ 2.5e-8 |
| Optionlet Stripper2 ATM | `OptionletStripper2` + `CapFloorTermVolCurve` | `optionletstripper.cpp` ATM correction | smile nodes unchanged when ATM matches the surface |
| Swaption vol matrix | `SwaptionVolatilityMatrix` | `swaptionvolatilitymatrix.cpp` | five constructors; 120 nodes @ 1e-16; live-input observability + handle relink |
| SABR cube backward-flat | `SabrSwaptionVolatilityCube` | `swaptionvolatilitycube.cpp` sparse/dense | 72 QL 1.43 rows, both flags, @ 1e-6; live quote/date; ZABR deferred |
| Swaps / OIS / swaptions / caps | instruments + engines | swap/swaption/capfloor suites | Core done |
| Float-float swap | `FloatFloatSwap` | `ql/instruments/floatfloatswap` | Two-Ibor-leg slice; identity-verified (identical legs, fair spread) |
| XCCY basis swap | `XccyBasisSwap` | `ql/instruments/` (cross-currency) | Float-float w/ notional exchange; identity-verified (degenerate, FX view, fair spread) |
| Fixed-rate bonds (cached) | `FixedRateBond` + `DiscountingBondEngine` | `bonds.cpp` `testCachedFixed` | bond1–3 @ 1e-6 (plain / varying coupons / next-to-last stub) |
| Fixed-rate bonds (given dates) | `FixedRateBond` + `Schedule::with_metadata` | `bonds.cpp` `testFixedBondWithGivenDates` | schedule ≡ date-vector copy @ 1e-6 (plain / varying / stub Actual360) |
| Callable / puttable fixed bonds (cached) | `CallableFixedRateBond` + `TreeCallableFixedRateBondEngine` | `callablebonds.cpp` `testCached` | HW tree call/put/both @ 1e-8 |
| Callable / puttable fixed bonds (consistency / degenerate) | `CallableFixedRateBond` + HW tree | `callablebonds.cpp` `testConsistency` / `testDegenerate` | call < plain < put; empty/OTM ≡ straight @ 1e-4 |
| Callable zero (degenerate / observability) | `CallableZeroCouponBond` + HW tree | `callablebonds.cpp` `testDegenerate` / `testObservability` | empty/OTM ≡ straight @ 1e-4; quote move updates NPV |
| Callable zero (call/put interplay) | `CallableZeroCouponBond` + HW tree | `callablebonds.cpp` `testInterplay` | early ITM exercise blocks later opposite right @ 1e-2 |
| European callable (Black engine) | `BlackCallableFixedRateBondEngine` / zero alias | `callablebonds.cpp` `testBlackEngine` / `testBlackEngineDeepInTheMoney` | zero clean 74.54521578 @ 1e-4; deep-ITM → discounted strike @ 1e-8 |
| European callable (implied vol) | `CallableFixedRateBond::implied_volatility` | `callablebonds.cpp` `testImpliedVol` | dirty/clean 78.50 round-trip @ 1e-4 |
| Callable OAS (notional invariance) | `CallableFixedRateBond::oas` / `clean_price_oas` + HW tree spread | `callablebonds.cpp` `testCallableBondOasWithDifferentNotinals` | OAS & cleanPriceOAS identical for face 100 vs 25 |
| Callable OAS (effective duration / convexity) | `effective_duration` / `effective_convexity` | `callablebonds.cpp` `testEffectiveDurationAndConvexity` | dirty-price FD @ 1e-4%; ≠ clean-denominator |
| Callable snap-to-coupon | HW tree snap + OAS | `callablebonds.cpp` `testSnappingExerciseDate2ClosestCouponDate` | callable NPV ≡ truncated straight @ 1e-10; OAS falls with later call |
| Callable OAS (ex-coupon continuity) | `with_ex_coupon` + indenture accrued on call | `callablebonds.cpp` `testOasContinuityThroughExCouponWindow` | OAS range ≤ 50 bps through ex-coupon window |
| Callable fixed bonds (arbitrary schedule) | `CallableFixedRateBond` + date-vector schedule | `callablebonds.cpp` `testCallableFixedRateBondWithArbitrarySchedule` | HW tree clean price succeeds |
| Fixed-rate bonds (arbitrary schedule) | `FixedRateBond` + `Schedule::from_dates` | `bonds.cpp` `testFixedRateBondWithArbitrarySchedule` | `NoFrequency`; clean price prices without error |
| Convertible bonds (TF binomial) | `ConvertibleFixedCouponBond` / `ConvertibleZeroCouponBond` / `ConvertibleFloatingRateBond` + `BinomialConvertibleEngine` (CRR) | `convertiblebonds.cpp` `testBond` | OTM ≈ credit-spread vanilla @ 1e-2 (zero) / 2e-2 (fixed, float); 1001 steps; ATM exceeds straight bond |
| Convertible bonds (vs vanilla option) | `ConvertibleZeroCouponBond` + `BinomialConvertibleEngine` / `BinomialVanillaEngine` (CRR) | `convertiblebonds.cpp` `testOption` | zero, no credit spread ≡ discounted redemption + ratio × call @ 5e-2 (2001 steps) |
| Convertible dividends (vs settlement) | `DiscretizedConvertible` dividend filter | `convertiblebonds.cpp` `testDividendsSpanningSettlementDate` | pre-settlement dropped; post-settlement PV @ 1e-12 |
| Convertible bonds (INF regression) | `ConvertibleFixedCouponBond` + `BinomialConvertibleEngine` (CRR) | `convertiblebonds.cpp` `testRegression` | 2168% vol tree throws overflow rather than returning Inf |
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
| FdmHullWhiteOp | `FdmHullWhiteOp` | `fdmhullwhiteop.{hpp,cpp}` | apply = home dir; mixed=0; splitting inverts; φ̄ discount |
| FdmHullWhiteSolver | `FdmHullWhiteSolver` | `fdmhullwhitesolver` | constant→discount-ish; zero payoff→0 |
| FdmHullWhiteSwapInnerValue | `FdmHullWhiteSwapInnerValue` | `fdmaffinemodelswapinnervalue` HW spec | ATM≈0; deep ITM payer>0; getState = short rate |
| FdHullWhiteSwaptionEngine | `FdHullWhiteSwaptionEngine` (Douglas default) | `fdhullwhiteswaptionengine` / `testCachedValues` | ITM European>0; ≈ Jamshidian; Bermudan≥European; cached FDM @ 1e-4 (non-par) |
| Bermudan OIS (HW FDM) | OIS-underlying `Swaption` + overnight FDM rebuild | `bermudanswaption.cpp` `testBermudanOISSwaptionWithHW` (+ reduced averaging feature pin) | ITM/ATM/OTM >0, monotone; vs Vanilla @ 5% rel; Simple≠Compound ≥0.1% |
| Bermudan OIS (G2 FDM) | OIS-underlying `Swaption` + `FdG2SwaptionEngine` | `bermudanswaption.cpp` `testBermudanOISSwaptionWithG2` (+ reduced averaging feature pin) | ATM OIS >0; vs Vanilla @ 5% rel; Simple≠Compound ≥0.1% |
| VanillaSwap notifications | `VanillaSwap` observer chain | `swap.cpp` `testNotifications` | Flag raises on forecast-curve relink after NPV |
| VanillaSwap rate/spread dependency | `VanillaSwap` + `DiscountingSwapEngine` | `swap.cpp` `testRateDependency` / `testSpreadDependency` | Payer NPV weakly ↓ in fixed rate; weakly ↑ in floating spread |
| VanillaSwap ThirdWednesdayInclusive | `VanillaSwap` schedules | `swap.cpp` `testThirdWednesdayAdjustment` | floating start 16-Sep-2015 / end 21-Sep-2016 |
| MakeVanillaSwap with_rule | `MakeVanillaSwap::with_rule` | `makevanillaswap.cpp:238` + ThirdWednesdayInclusive | Inclusive snaps floating end 16-Sep-2016 → 21-Sep-2016 |
| MakeVanillaSwap stub dates | `with_*_leg_first_date` / `with_*_leg_next_to_last_date` | `makevanillaswap.cpp:139/146` + Schedule stub pins | float/fixed first + float/fixed ntl irregular; `floating_leg` stub parity; matches hand-built `Schedule` |
| Vanilla IRS in-arrears Hull NPV | `BlackIborCouponPricer` Black76 convexity + `IborLeg::in_arrears` | `swap.cpp` `testInArrears` | NPV −144813 ± 1 (Hull 4th ed. p.550; vol 0.22) |
| HundsdorferScheme | `HundsdorferScheme` + factories | `hundsdorferscheme` | BS replay; diagonal closed form; dual BC apply cycles |
| TreeLattice2D | `TwoFactorTree` / `TreeLattice2D` | `lattice2d.hpp` | size=product; ρ=0⇒independent; |ρ| HW term; neg ρ flips m; probs∑≈1; grid fails; flat rollback |
| G2 two-factor tree | `TwoFactorShortRateTree` / `G2::tree` | `twofactormodel` / `g2` | discount=exp(-(φ+x+y)dt); root φ-only; product size; builds under analytic φ |
| TreeG2SwaptionEngine | `TreeG2SwaptionEngine` + date snapping | `treeswaptionengine` / `testCachedG2Values` | cached tree Bermudan @ 5e-3; Bermudan≥European analytic |
| Tree HW Bermudan (cached) | `TreeSwaptionEngine` + `DiscretizedSwaption` snap | `bermudanswaption.cpp` `testCachedValues` | ITM/ATM/OTM @ 1e-4 (non-par coupons) |
| Zero-coupon bonds (cached) | `ZeroCouponBond` + `DiscountingBondEngine` | `bonds.cpp` `testCachedZero` | three maturities @ 1e-6 |
| Floating bonds (cached) | `FloatingRateBond` + `USDLibor` / `Libor` | `bonds.cpp` `testCachedFloating` | bond1–4 @ 1e-6 (plain / dual / spreads / fixing+ex-coupon) |
| Floating bonds (fixing convention) | `FloatingRateBond` + `AUDLibor` | `bonds.cpp` `testFixingConvention` | Preceding→Fri / Following→Mon for Sat accrual start |
| Brazilian NTN-F (Andima) | `BondFunctions` yield clean/dirty + `Business252` | `bonds.cpp` `testBrazilianCached` | six maturities @ 1e-4 |
| Bond price/yield consistency | `BondFunctions` yield clean/dirty ↔ `yield_rate` | `bonds.cpp` `testYield` | clean/dirty round-trip @ 1e-7 |
| SA R2048 (date-vector schedule) | `Schedule::with_metadata` + yield dirty | `bonds.cpp` `testBondFromScheduleWithDateVector` | dirty 95.75706 @ 1e-5 |
| Bond price/ATM rate consistency | `BondFunctions::atm_rate` + `BondPrice` | `bonds.cpp` `testAtmRate` | clean/dirty → coupon @ 1e-7 |
| Bond theoretical price/yield | `DiscountingBondEngine` ↔ Continuous yield | `bonds.cpp` `testTheoretical` | engine ≡ yield price; yield recovery @ 1e-7 |
| Bond price/z-spread consistency | `BondFunctions` z-spread clean/dirty ↔ solve | `bonds.cpp` `testZspread` | clean/dirty round-trip @ 1e-7 |
| Bond price/yield (cached) | `FixedRateBond` + engine + `BondFunctions` | `bonds.cpp` `testCached` | bond1–3 price/yield @ 1e-6 (schedule vs bare ISMA) |
| Ex-coupon UK gilt / Australian bond | `BondFunctions` + `CashFlows` yield/duration/convexity | `bonds.cpp` `testExCouponGilt` / `testExCouponAustralianBond` | Bloomberg tables @ 1e-6 / 1e-4–1e-3 |
| Thirty/360 bond (settle on 31st) | `BondFunctions` yield/Macaulay/convexity/accrued | `bonds.cpp` `testThirty360BondWithSettlementOn31st` | CUSIP 3130A0X70 @ 1e-4 / 1e-3 / 1e-6 |
| Bond basis-point value | `BondFunctions` / `CashFlows` BPV & YVBP | `bonds.cpp` `testBasisPointValue` | yield 0.041301; BPV/YVBP table @ 1e-6 |
| FRA | `ForwardRateAgreement` + `FraRateHelper` | `ratehelpers.cpp`, FRA examples | Instrument + helper |
| CMS / digital coupons | `CmsCoupon`, `DigitalIborCoupon` | CMS/digital suites | Raw-rate / cash-or-nothing slice |
| CMS swap | `CmsSwap` | `ql/instruments/*cms*` | Fixed-vs-CMS (raw rate); identity-verified (fair rate) |
| Asset swap | `AssetSwap` | `ql/instruments/assetswap` | Par asset swap; leg construction per `assetswap.cpp`; identity-verified |
| Barrier / Asian | `BarrierOption` + Haug `AnalyticBarrierEngine`, geometric Asian | `barrieroption.cpp`, `asianoptions.cpp` | Haug continuous with rebate; geometric Asian slice |
| Continuous geometric Asian | `AnalyticContinuousGeometricAveragePriceAsianEngine` | `asianoptions.cpp` `testAnalyticContinuousGeometricAveragePrice` / `testAnalyticContinuousGeometricAveragePriceGreeks` / `testContinuousSeasonedAsianOptions` | Haug put NPV 4.6922 @ 1e-4; δ/γ/θ/ρ/divρ/ν vs FD @ 1e-5 rel. to spot; seasoned continuous geometric rejects |
| Continuous geometric Asian (Heston) | `AnalyticContinuousGeometricAveragePriceAsianHestonEngine` | `asianoptions.cpp` `testAnalyticContinuousGeometricAveragePriceHeston` | Kim–Wee Tables 1/4 @ 1e-2; Kim–Kim–Kim–Wee continuous @ QL tols |
| Continuous arithmetic Asian (Levy) | `ContinuousArithmeticAsianLevyEngine` | `asianoptions.cpp` `testLevyEngine` / `testContinuousSeasonedAsianOptions` | Haug continuous arithmetic AP table @ 1e-4 (seasoned/unseasoned; option `startDate`); seasoned put monotonicity vs fresh / higher average |
| Continuous arithmetic Asian (Vecer) | `ContinuousArithmeticAsianVecerEngine` | `asianoptions.cpp` `testVecerEngine` | published 7-row call table @ QL tols (200×200 CN, z∈[-1,1]; unseasoned) |
| Discrete geometric Asian | `AnalyticDiscreteGeometricAveragePriceAsianEngine` / `AnalyticDiscreteGeometricAveragePriceAsianHestonEngine` / `MCDiscreteGeometricAveragePriceAsianEngine` / `MCDiscreteGeometricAveragePriceAsianHestonEngine` | `asianoptions.cpp` `testAnalyticDiscreteGeometricAveragePrice` / `testAnalyticDiscreteGeometricAveragePriceGreeks` / `testAnalyticDiscreteGeometricAveragePriceHeston` / `testDiscreteGeometricAveragePriceHestonPastFixings` / `testMCDiscreteGeometricAveragePrice` / `testMCDiscreteGeometricAveragePriceHeston` | Levy call NPV 5.3425606635 @ 1e-10; δ/γ/θ/ρ/divρ/ν vs FD @ 1e-5; Kim–Kim–Kim–Wee analytic Heston Tables 1–3 @ QL tols (3e-2–8e-2); seasoned geom Heston analytic vs MC @ QL tols (8191 samples, seed 43); MC vs analytic @ 4e-3 (8191 samples); Kim–Kim–Kim–Wee Heston MC tables @ QL tols (`LowDiscrepancy`, seed 43, 8191 samples) |
| Discrete arithmetic Asian | `MCDiscreteArithmeticAveragePriceAsianEngine` / `MCDiscreteArithmeticAveragePriceAsianHestonEngine` / `MCDiscreteArithmeticAverageStrikeAsianEngine` / `TurnbullWakemanAsianEngine` / `ChoiAsianEngine` | `asianoptions.cpp` `testMCDiscreteArithmeticAveragePrice` / `testMCDiscreteArithmeticAveragePriceHeston` / `testMCDiscreteArithmeticAverageStrike` / `testMCDiscreteArithmeticAverageStrikeExerciseDate` / `testPastFixings` / `testPastFixingsModelDependency` / `testTurnbullWakemanAsianEngine` / `testAllFixingsInThePast` / `testChoiAsianEngineVsMC` / `testChoiAsianEngineSpecialCases` | Levy (1997) 30-row put table @ 2e-2 (`LowDiscrepancy`, 2047 samples, geometric CV); Ballestra call @ 5e-2 (4095) / CV @ 3e-2 (48 steps); Albrecher–Zeng @ 9e-2 (16383; QL 8191 non-CV) / CV @ 3e-2 (8191, seed 42); Levy average-strike 30-row call table @ 2e-2 (`LowDiscrepancy`, seed 3456789, 1023 samples, BB on); #646 later exercise raises ASO NPV (r=q=0, seed 42, 8191 samples); past fixings change AP/AS/geom NPV; vector past-fixings ctor matches classic @ 1e-8; TW guaranteed-exercise call/put + δ/γ @ 1e-8; Haug Table 4-28 NPV @ 2.5e-3, δ/γ vs bump @ 1e-6; all-past fixings raise `all fixings are in the past` for AP/AS/geom MC and Choi fails; Choi vs MC @ 1e-2 (λ=20, max steps 4096, MC 32000 seed 43); Choi special cases @ 1000ε |
| Discrete geometric average-strike Asian | `AnalyticDiscreteGeometricAverageStrikeAsianEngine` | `asianoptions.cpp` `testAnalyticDiscreteGeometricAverageStrike` | call NPV 4.97109 @ 1e-5 |
| Continuous floating lookback | `AnalyticContinuousFloatingLookbackEngine` | `lookbackoptions.cpp` `testAnalyticContinuousFloatingLookback` | Haug + Broadie–Glasserman–Kou floating-strike table @ 1e-4 |
| Continuous fixed lookback | `AnalyticContinuousFixedLookbackEngine` | `lookbackoptions.cpp` `testAnalyticContinuousFixedLookback` | Haug fixed-strike 36-row table @ 1e-4 |
| Continuous partial floating lookback | `AnalyticContinuousPartialFloatingLookbackEngine` | `lookbackoptions.cpp` `testAnalyticContinuousPartialFloatingLookback` | Haug 2006 p.146 36-row table @ 1e-4 |
| Continuous partial fixed lookback | `AnalyticContinuousPartialFixedLookbackEngine` | `lookbackoptions.cpp` `testAnalyticContinuousPartialFixedLookback` | Haug 2006 p.148 36-row table @ 1e-4 |
| Continuous lookback MC vs analytic | `MCLookbackEngine` | `lookbackoptions.cpp` `testMonteCarloLookback` | partial fixed / fixed / partial floating / floating call+put vs analytic @ 0.1 (2000 steps, antithetic, seed 1) |
| Soft barrier Haug values | `AnalyticSoftBarrierEngine` | `softbarrieroption.cpp` `testSoftBarrierHaug` | Haug 2nd ed. p.166 DownOut call table @ 1e-4 (59 rows; one tight-barrier/high-vol case omitted as in QL) |
| Partial-time barrier EndB1 | `AnalyticPartialTimeBarrierOptionEngine` | `partialtimebarrieroption.cpp` `testAnalyticEngine` / `testAnalyticEnginePutOption` / `testPutCallSymmetry` | DownOut call + UpOut put 20-row tables @ 1e-4; 10 put-call symmetry pairs @ 1e-4 |
| Simple chooser Haug value | `AnalyticSimpleChooserEngine` | `chooseroption.cpp` `testAnalyticSimpleChooserEngine` | Haug 2nd ed. pp.39–40 @ 3e-5 |
| Complex chooser Haug value | `AnalyticComplexChooserEngine` | `chooseroption.cpp` `testAnalyticComplexChooserEngine` | Haug example @ 1e-4 |
| Margrabe exchange options (European & American) | `MargrabeOption`, `AnalyticEuropeanMargrabeEngine`, `AnalyticAmericanMargrabeEngine` | `margrabeoption.cpp` `testEuroExchangeTwoAssets` / `testAmericanExchangeTwoAssets` | 21-row European table for NPV and 6 greeks @ 1e-3; 18-row American table @ 1e-3; reduced moving-market FD check (delta1/2, gamma1/2, theta, rho); setup_expired zeros greeks; full testGreeks grid deferred |
| Compound option | `CompoundOption`, `AnalyticCompoundOptionEngine` | `compoundoption.cpp` `testValues` / `testPutCallParity` | 20-row Haug/sitmo/mathfinance table for NPV/delta/gamma/vega/theta @ 1e-3; QL `testPutCallParity` 10 unique daughter-market cases (11 iterations in QL) @ 1e-8; instrument `setup_expired` zeros greeks |
| Two-asset min/max basket (Stulz 1982) | `BasketOption`, `MinBasketPayoff`, `MaxBasketPayoff`, `StulzEngine` | `basketoption.cpp` `testEuroTwoValues` | 39-row Firth & Haug table for min/max Call & Put @ 1e-3 / 1e-4; rejects non-European exercise, non-min/max payoff, and |ρ|>1 |
| Two-asset spread option (Kirk 1995) | `BasketOption`, `SpreadBasketPayoff`, `KirkEngine` | `basketoption.cpp` `testEuroTwoValues` / `testStrangSplittingSpreadEngineVsMathematica` | 18-row Haug p.59–60 table for European spread call @ 1e-3 (completing full 57-row testEuroTwoValues); 15-row Mathematica reference table (Kirk NPV column of testStrangSplittingSpreadEngineVsMathematica) @ 100*EPSILON; independent q≠r call & put parity pin; rejects non-European exercise, non-spread payoff, and |ρ|>1 |
| Two-asset spread option (Bjerksund & Stensland 2014) | `BasketOption`, `SpreadBasketPayoff`, `BjerksundStenslandSpreadEngine` | `basketoption.cpp` `testBjerksundStenslandSpreadEngine` | PyFENG 0.2.6 reference Put price (17.850835947276213) & Call-Put parity @ 100*EPSILON; exchange option (K=0) vs Margrabe closed form @ 1e-12; independent q≠r parity pin; rejects non-European exercise, non-spread payoff, and |ρ|>1 |
| Two-asset spread option (Pearson 1995) | `BasketOption`, `SpreadBasketPayoff`, `PearsonSpreadEngine` | `basketoption.cpp` `testPearsonSpreadEngine` | Call-Put parity (C - P) / df = F1 - F2 - K @ 1e-10; exchange option (K=0) vs Bjerksund-Stensland @ 1e-6; distinct q1≠q2≠r dividend parity pin; zero-residual-vol (ρ=1, K=0, σ1=σ2) & zero-vol exact deterministic pins; negative-strike parity pin; rejects non-European exercise, non-spread payoff, and |ρ|>1 |
| Two-asset spread option (Operator Splitting Lo 2015) | `BasketOption`, `SpreadBasketPayoff`, `OperatorSplittingSpreadEngine` | `basketoption.cpp` `testOperatorSplittingSpreadEngine` / `testStrangSplittingSpreadEngineVsMathematica` / `testNoDivByZeroOperatorSplitting` | Chi-Fai Lo 15-row table for First (@ 1e-4) & Second (@ 5e-4) order across ρ ∈ [-0.9, 0.9]; 15-row Mathematica reference table (First and Second order columns of testStrangSplittingSpreadEngineVsMathematica) @ 100*EPSILON; testNoDivByZeroOperatorSplitting continuity across ρσ1=σ2 critical point @ 5e-8; distinct q1≠q2≠r dividend parity pin; rejects non-European exercise, non-spread payoff, and |ρ|>1 |
| Two-asset correlation Haug NPV | `AnalyticTwoAssetCorrelationEngine` | `twoassetcorrelationoption.cpp` `testAnalyticEngine` | Haug European call @ 1e-4; independent put and q≠0 call pins (QL has no further suite cases) |
| Two-asset barrier Haug NPV | `AnalyticTwoAssetBarrierEngine` | `twoassetbarrieroption.cpp` `testHaugValues` | Haug 4-row Out table @ 4e-3; independent q≠0 KO, distinct-asset KO, and q≠0 KI pins (QL has no further suite cases) |
| Writer-extensible Haug NPV | `AnalyticWriterExtensibleOptionEngine` | `extensibleoptions.cpp` `testAnalyticWriterExtensibleOptionEngine` | Haug writer call @ 1e-4; independent put and q≠0 call pins (QL has no further writer cases) |
| Holder-extensible Haug NPV | `AnalyticHolderExtensibleOptionEngine` | `extensibleoptions.cpp` `testAnalyticHolderExtensibleOptionEngine` | Haug holder call @ 1e-4; independent put and q≠0 call pins (QL suite is call-only) |
| Everest MC cached NPV | `MCEverestEngine` | `everestoption.cpp` `testCached` | PseudoRandom 1023 samples / 1 step/year / seed 86421 @ 1e-8; errorEstimate absolute-tolerance arm |
| Cliquet Haug value | `AnalyticCliquetEngine` | `cliquetoption.cpp` `testValues` | Haug p.37 call @ 1e-4 |
| Analytic performance cliquet | `AnalyticPerformanceEngine` | `cliquetoption.cpp` `testPerformanceGreeks` | δ=γ=0; NPV independent of spot; expired greeks 0; ρ/divρ/ν/θ vs FD @ 1e-5 relative to spot |
| Binary barrier Haug values | `AnalyticBinaryBarrierEngine` | `binaryoption.cpp` `testCashOrNothingHaugValues` / `testAssetOrNothingHaugValues` | Haug p.180 cash+asset book rows @ 1e-4; cash book-vba q≠0 and touched-barrier extras closed; double-binary deferred |
| Merton-76 jump diffusion | `Merton76Process` + `JumpDiffusionEngine` | `jumpdiffusion.cpp` `testMerton76` | Haug p.9 European call NPV subset @ 1e-2 (QL-corrected vs book); greeks deferred |
| Double-barrier Haug values | `AnalyticDoubleBarrierEngine` | `doublebarrieroption.cpp` `testEuropeanHaugValues` | Ikeda/Kunitomo 90-row table @ 1e-4 (KnockOut/In call+put) |
| Double-barrier Heston FD | `FdHestonDoubleBarrierEngine` 251×76×3 | `doublebarrieroption.cpp` `testEuropeanHaugValues` | KnockOut subset @ 0.025 (near-Black σ=0.001); KnockIn / leverage deferred |
| Double-barrier MC vs analytic | `MCDoubleBarrierEngine` | `doublebarrieroption.cpp` `testMonteCarloDoubleBarrierWithAnalytical` | KnockIn relative ≤ 1% @ 5000 steps/antithetic/seed 1; KnockOut absolute ≤ 0.01 @ seed 10 |
| Double-barrier Vanna/Volga FX | `VannaVolgaDoubleBarrierEngine` + `AnalyticDoubleBarrierEngine` | `doublebarrieroption.cpp` `testVannaVolgaDoubleBarrierValues` | 20 FX rows × KO/KI @ 5e-3 (analytic inner, adaptVanDelta) |
| Single-barrier Vanna/Volga FX | `VannaVolgaBarrierEngine` + `AnalyticBarrierEngine` | `barrieroption.cpp` `testVannaVolgaSimpleBarrierValues` | FX subset (UpOut/UpIn/DownOut/DownIn, T=1/2) @ 1e-4 (`adaptVanDelta`) |
| Variance swap (replicating) | `VarianceSwap` + `ReplicatingVarianceSwapEngine` | `varianceswap.cpp` `testReplicatingVarianceSwap` | Derman 1999 fair variance 0.04189 @ 1e-4; Long NPV 93.271669 / Short=−Long; `q=0.05` hybrid 0.0422989 (carry uses `2r`/`S/DF_r` as C++, options see `q`); MC / variance option deferred |
| FD BS swing | `VanillaSwingOption` + `FdSimpleBSSwingEngine` | `swingoption.cpp` `testFdBSSwingOption` | monthly Put swing vs Bermudan upper (+0.01) and remaining-European lower (−4e-2); ExtOU-jump deferred |
| Barrier knock-in/out parity | `AnalyticBarrierEngine` + `AnalyticEuropeanEngine` | `barrieroption.cpp` `testParity` | DownIn + DownOut ≡ European call @ 1e-7 (Actual360 and Business252 vol) |
| Barrier put-call symmetry | `AnalyticBarrierEngine` | `barrieroption.cpp` `testPutCallSymmetry` | inverted knock-out put ≡ scaled call @ 1e-4 (DownOut/UpOut pairs) |
| Barrier Haug values | `AnalyticBarrierEngine` | `barrieroption.cpp` `testHaugValues` | European table @ 1e-4 (rebate 3) |
| Barrier Haug FD | `FdBlackScholesBarrierEngine` 200×400 | `barrieroption.cpp` `testHaugValues` | European table @ 5e-3; rejects zero spot / triggered / American |
| Barrier Haug binomial | `BinomialBarrierEngine` CRR Boyle–Lau / Derman–Kani 400 | `barrieroption.cpp` `testHaugValues` | American + European Boyle–Lau @ 1.1e-2, Derman–Kani @ 4e-2; rejects zero spot / triggered |
| Barrier Babsiri / Beaglehole | `AnalyticBarrierEngine` / `MCBarrierEngine` Sobol | `barrieroption.cpp` `testBabsiriValues` / `testBeagleholeValues` | published calls analytic @ 1e-5 / 1e-3; MC LowDiscrepancy 131071 samples, 1 step/year, Brownian bridge, relative 2e-2 / 1e-2 |
| Barrier Heston FD (knock-out) | `FdHestonBarrierEngine` 100×400×50 Hundsdorfer | `barrieroption.cpp` `testLocalVolAndHestonComparison` | DownOut put NPV 111.5 @ 1% relative |
| Barrier Heston FD vanilla | `FdHestonVanillaEngine` 40×80×25 Hundsdorfer | vs `AnalyticHestonEngine` | no-div European call @ 1% relative |
| Barrier Heston FD (knock-in + discrete div) | `FdHestonVanillaEngine` + `FdHestonRebateEngine` + `FdHestonBarrierEngine` 50×101×3 Hundsdorfer | `barrieroption.cpp` `testDividendBarrierOption` | DownOut/UpOut/DownIn/UpIn @ 2e-4 |
| Barrier local-vol FD (knock-out) | `FdBlackScholesBarrierEngine` Dupire 100×400 Douglas, `illegalLocalVolOverwrite=0.35` | `barrieroption.cpp` `testLocalVolAndHestonComparison` | DownOut put NPV 132.8 @ 1% relative |
| Barrier local-vol FD (knock-in) | `FdBlackScholesVanillaEngine` + `FdBlackScholesRebateEngine` + barrier Dupire | `barrieroption.cpp` `testDividendBarrierOption` (constant vol) / `testLocalVolAndHestonComparison` | DownIn/UpIn Douglas @ 2e-4 vs 29.154 / 4.765; surface DownIn put NPV 465.0 @ 1% relative |
| Barrier low volatility | `AnalyticBarrierEngine` | `barrieroption.cpp` `testLowVolatility` | vol 1e-7 zero-vol limits, no NaN @ 0.5 |
| Barrier implied vol | `BarrierOption::implied_volatility` + analytic / FD | `barrieroption.cpp` `testImpliedVolatility` | no-div put targets @ 1e-5; discrete-div FD put targets @ 1e-5 |
| Barrier FD (discrete div) | `FdBlackScholesBarrierEngine` + vanilla/rebate FD | `barrieroption.cpp` `testDividendBarrierOption` | DownOut/UpOut/DownIn/UpIn Douglas/CN/Hundsdorfer/CraigSneyd/MCS/MethodOfLines/TrBDF2 @ 2e-4 |
| Barrier FD past-maturity div | `FdBlackScholesBarrierEngine` / `FdHestonBarrierEngine` | `barrieroption.cpp` `testDividendBarrierOptionWithDividendsPastMaturity` | +18M cash vs T=1Y identity @ 1e-12 (BS and Heston) |
| Money / FX rates | `Money`, `ExchangeRate` | `money.cpp`, `exchangerate.cpp` | Value types |
| FX forward | `FxForward` | money layer (covered interest parity) | Outright; parity-identity verified |
| C ABI | `libitofin-ffi` | n/a | Version + error stubs only |

## Gaps (rates + equity)

Still-missing surface/oracle gaps (rates + equity) live in
[`quantlib-gaps.md`](quantlib-gaps.md). This document tracks **covered** oracles
only.

## How to extend this map

1. Pick a QuantLib `test-suite/*.cpp` case.
2. Port the instrument/engine slice with matching inputs.
3. Assert numbers within the C++ tolerance.
4. Move the row from “Not started” to “Covered” with the tolerance noted.
