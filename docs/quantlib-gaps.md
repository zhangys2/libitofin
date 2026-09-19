# QuantLib gaps (rates + equity)

Canonical register of **still-missing** libitofin gaps versus C++ QuantLib.
Covered/done oracles live in [`oracle-coverage.md`](oracle-coverage.md) only.

Classification: [Gap classification scheme](../.wayfinder/tickets/gap-classification-scheme.md)
(`has_surface` × `has_matching_oracle` ∈ `none`|`partial`|`full`). A row belongs
here iff `has_surface = false` or `has_matching_oracle ≠ full`.

Row grain: one feature slice (same style as coverage rows). Cite suite files and
headers in the QL columns. Do not use this file as an execution backlog — promote
prioritized work to GitHub issues after the prioritization lens.

**QL pin for this inventory:** QuantLib tag `v1.43` (matches repo fixtures).
Inventories merged 2026-09-19 from Wayfinder research tickets
[oracle case inventory](../.wayfinder/tickets/oracle-case-inventory.md) and
[surface module inventory](../.wayfinder/tickets/surface-module-inventory.md).

| Domain | Feature | QL surface | QL oracle(s) | has_surface | has_matching_oracle | Notes |
|--------|---------|------------|--------------|-------------|---------------------|-------|
| Rates | GSR (Gaussian short-rate) | `ql/models/shortrate/onefactormodels/gsr.hpp`, `gsrprocess` | `gsr.cpp` | partial | partial | Constant a/σ `GsrProcess` + `ForwardMeasureProcess1D`; piecewise core / `Gsr` model / engines deferred |
| Rates | Libor market model | `ql/legacy/libormarketmodels/*` | `libormarketmodel.cpp`, `libormarketmodelprocess.cpp` | partial | partial | Exp corr + lin-exp vol + `LfmCovarianceProxy` (`testSimpleCovarianceModels`); process/cap/swaption/calibration deferred |
| Rates | Generic market model / SMM / CMS-MM | `ql/models/marketmodels/*` | `marketmodel.cpp`, `marketmodel_cms.cpp`, `marketmodel_smm*.cpp` | partial | partial | `AbcdFunction`/`AbcdSquared` + `AbcdMathFunction` (`testAbcdDegenerateCases`, `testAbcdVolatilityIntegration`); evolvers/products/CMS/SMM deferred |
| Rates | Markov functional / Gaussian 1D | `markovfunctional`, `gaussian1dmodel` | `markovfunctional.cpp`, `gaussian1dmodel.cpp` | partial | partial | `MfStateProcess` (`testMfStateProcess`); `MarkovFunctional` / Gaussian1d model + engines deferred |
| Rates | Black–Karasinski | `blackkarasinski.hpp` | (no QL suite) | partial | none | Model + `BlackKarasinskiDynamics` log transform (identity); `tree()` / Brent fit deferred |
| Rates | Hybrid Heston × Hull–White | `HybridHestonHullWhiteProcess`, hybrid vanilla engines | `hybridhestonhullwhiteprocess.cpp`, `hestonhullwhite.cpp` | partial | none | `HullWhiteForwardProcess` (α/B/M_T + forward drift); hybrid join / evolve / engines deferred |
| Rates | Heston SLV models | `HestonSLV*Model` | `hestonslvmodel.cpp` / `hestonslv*.cpp` | partial | none | `HestonSLVProcess` (drift/diffusion/apply/evolve + const-L identity); FDM/MC model calibration deferred |
| Rates | FD SABR / no-arb SABR model | `FdmSabr*`, `NoArbSabr*` | `fdsabr.cpp`, `noarbsabr.cpp` | false | none | SABR **cube** interpolation ≠ these models |
| Rates | Bachelier / normal cap–floor | `BachelierCapFloorEngine` | `capfloor.cpp` | false | none | Blocks normal vol in CapHelper / stripper |
| Rates | Tree / MC / Gaussian1d cap–floor | `TreeCapFloorEngine`, `MCHullWhiteEngine`, … | `capfloor.cpp` | false | none | Only Analytic + Black cap/floor today |
| Rates | Cap/floor Black extras | `CapFloor`, Black engine | `capfloor.cpp` (implied vol, Bachelier δ, ATM, parity) | true | partial | Cached NPV/vega pinned; extras open |
| Rates | Swaption Black extras | `Swaption`, `BlackSwaptionEngine` | `swaption.cpp` (implied vol*, Bachelier) | true | partial | Cached NPV/vega/cash-settled pinned |
| Rates | Gaussian1d / float–float / nonstandard swaption | `Gaussian1d*SwaptionEngine`, `FloatFloatSwaption`, `NonstandardSwaption` | `swaption.cpp`, `gaussian1dswaption.cpp` | false | none | HW/G2 engines only |
| Rates | Bermudan OIS swaption (HW/G2) | OIS-underlying `Swaption` | `bermudanswaption.cpp` OIS cases | true | none | IBOR Bermudan cached; OIS cases not |
| Rates | Vanilla IRS extras | `VanillaSwap` | `swap.cpp` (beyond cached/fair) | true | partial | Cached + fair pinned; in-arrears/stubs open |
| Rates | OIS bootstrap / cached NPV | `OvernightIndexedSwap`, `MakeOIS` | `overnightindexedswap.cpp` | true | partial | Compound bootstrap pinned; arithmetic/lookback siblings + type-level cached NPV deferred |
| Rates | Float–float basis swap | `FloatFloatSwap` | `floatfloatswap.cpp` | true | partial | Identity/fair-spread only |
| Rates | Const-notional XCCY swaps | `ConstNotionalCrossCurrency*` | `constnotionalcrosscurrency*.cpp` | false | none | `XccyBasisSwap` ≠ this family |
| Rates | CMS swap (cached/parity) | `CmsSwap`, CMS pricers | `cms.cpp` | true | partial | Fair-rate identity only |
| Rates | MakeCMS / CMS leg builders | `MakeCms` | `cms.cpp` | false | none | |
| Rates | CMS spread coupons | `CmsSpreadCoupon`, experimental pricers | `cmsspread.cpp` / `cmsspreadcoupon.cpp` | false | none | Plain CMS only |
| Rates | Capped/floored CMS coupon | CMS `CappedFlooredCoupon` | `capflooredcoupon.cpp` | false | none | Deferred in cashflows |
| Rates | Digital / capped Ibor coupons | `DigitalIborCoupon`, `CappedFlooredIborCoupon` | `digitalcoupon.cpp`, `capflooredcoupon.cpp` | true | partial | Raw-rate / cash-or-nothing slice; deep ITM/parity open |
| Rates | Asset swap (market ASW / Z-spread) | `AssetSwap` | `assetswap.cpp` | true | partial | Par construction identity only |
| Rates | Multiple-resets / averaged OIS legs | `MultipleResetsSwap` | `multipleresetsswap.cpp` | false | none | |
| Rates | Zero-coupon swap | `ZeroCouponSwap` | `zerocouponswap.cpp` | false | none | |
| Rates | BMA swap + curve helper | `BMASwap`, BMA helpers | `piecewiseyieldcurve.cpp` BMA cases | false | none | Helper deferred (#343) |
| Rates | Amortizing bonds (fixed/float/CMS) | `Amortizing*Bond` | `amortizingbond.cpp` | false | none | Only amortizing payment helper |
| Rates | CMS fixed-rate bond | `CmsRateBond` | `bonds.cpp` | false | none | |
| Rates | Fitted bond discount curve | Nelson–Siegel / Svensson-style helpers | `fittedbondcurve.cpp` | false | none | Piecewise bootstrap only |
| Rates | Tree discounting swap engine | `TreeSwapEngine` | `swap.cpp` | false | none | `DiscountingSwapEngine` only |
| Rates | Overnight index future | `OvernightIndexFuture` | `overnightindexfuture.cpp` | false | none | |
| Equity | Equity total return swap | `EquityTotalReturnSwap` | `equitytotalreturnswap.cpp` | false | none | |
| Rates | Caplet vol stripping (normal / shifted / ON) | `OptionletStripper*` | `optionletstripper.cpp` | true | partial | Flat Black strip pinned; normal/shifted/ON open |
| Rates | Swaption vol matrix observability | `SwaptionVolatilityMatrix` | `swaptionvolatilitymatrix.cpp` | true | partial | Coherence pinned; observability open |
| Rates | Swaption SABR/ZABR cube extras | `SabrSwaptionVolatilityCube` | `swaptionvolatilitycube.cpp` | true | partial | Some SABR fixtures; ZABR/smile/ATM grid open |
| Rates | Piecewise yield bootstrap extras | `PiecewiseYieldCurve` | `piecewiseyieldcurve.cpp` | true | partial | Many arms pinned; BMA / some globals skipped |
| Rates | Hull–White / affine short-rate suite | `HullWhite`, trees, helpers | `shortratemodels.cpp` | true | partial | Coverage “Core done”; full `testSwaps` breadth not mapped |
| Equity | Heston analytic/FD/MC full suite | `HestonModel`, engines | `hestonmodel.cpp` | true | partial | FD-cached ambition closed (`testFdBarrierVsCached` / `testFdVanillaVsCached` / dividends / `testFdAmerican`). Remaining: Lewis/Kahl–Jaeckel, COS/AP/integrals, piecewise TD, etc. |
| Equity | Heston FD scheme grid | `FdHeston*` | `fdheston.cpp` | true | partial | Variance mesher + Black limit + American/Ikonen/div/barrier NPV + ADI convergence (HS/MCS/mod-HS/CS) + Haug barrier-vs-BS table + MethodOfLines vs Hundsdorfer + American call–put parity + spurious oscillations (CS/HS/mod-HS/Douglas) closed. Still open: ExplicitEuler arm, 2-D CN/TrBDF2/ImplicitEuler (#636), intraday (`QL_HIGH_RESOLUTION_DATE`) |
| Equity | Forward-start vanilla (non-quanto) | `ForwardVanillaOption`, forward engines | `forwardoption.cpp` | true | partial | Haug NPV + analytic greeks (`testGreeks` / `testPerformanceGreeks`) + greeks-init closed. Still open: MC, Heston MC / analytic-vs-MC |
| Equity | Digitals beyond cash-or-nothing | Digital payoffs + American/MC engines | `digitaloption.cpp`, `binaryoption.cpp` | true | partial | One Haug cash-or-nothing; asset/gap/American/MC open |
| Equity | Asset-or-nothing / gap / super payoffs | `AssetOrNothing`, `Gap`, `Super*` | `binaryoption.cpp` | false | none | Documented follow-up in `payoffs.rs` |
| Equity | Binary / double-binary barrier | `AnalyticBinaryBarrierEngine`, `AnalyticDoubleBarrierBinaryEngine` | `binaryoption.cpp`, `doublebarrieroption.cpp` | false | none | |
| Equity | Compound option | `CompoundOption`, analytic engine | `compoundoption.cpp` | false | none | |
| Equity | Margrabe / exchange | `MargrabeOption` | `margrabeoption.cpp` | false | none | |
| Equity | Two-asset barrier | `TwoAssetBarrierOption` | `twoassetbarrieroption.cpp` / barrier suite | false | none | |
| Equity | Two-asset correlation | `TwoAssetCorrelationOption` | `twoassetcorrelationoption.cpp` | false | none | |
| Equity | Extensible options | Holder/writer extensible | `extensibleoptions.cpp` | false | none | |
| Equity | Alphabet baskets (Everest/Himalaya/Pagoda) | experimental exotic options + MC | `everestoption.cpp` (+ Himalaya/Pagoda often experimental-only) | false | none | QL v1.43: Everest has suite; Himalaya/Pagoda experimental |
| Equity | Basket beyond Choi / single-factor | Kirk/Stulz/Pearson/MC/FD-nD basket | `basketoption.cpp`, `spreadoption.cpp` | true | partial | Choi golden example pinned |
| Equity | FD Heston double-barrier | `FdHestonDoubleBarrierEngine` | `doublebarrieroption.cpp` | false | none | |
| Equity | Vanna–Volga single barrier | VV barrier (non-double) | `barrieroption.cpp` | false | none | Double-barrier VV only |
| Equity | Cliquet performance engines | `AnalyticPerformanceEngine`, `MCPerformanceEngine` | `cliquetoption.cpp` | false | none | Analytic cliquet Haug done |
| Equity | Variance swap / variance option | `VarianceSwap`, `VarianceOption` | `varianceswap.cpp`, `varianceoption.cpp` | false | none | |
| Equity | Swing option | `VanillaSwingOption`, FD engines | `swingoption.cpp` | false | none | |
| Equity | Sticky ratchet | `StickyRatchet` | `stickyratchet.cpp` | false | none | |
| Equity | Merton jump-diffusion engine | jump engines | `jumpdiffusion.cpp` | false | none | Bates separate |
| Equity | Variance gamma | `VarianceGamma*` | `variancegamma.cpp` | false | none | |
| Equity | GJR-GARCH | `GJRGARCHModel` + engines | `gjrgarch.cpp` | false | none | |
| Equity | Piecewise time-dependent Heston | `PiecewiseTimeDependentHestonModel` | `hestonmodel.cpp` | false | none | |
| Equity | Analytic American approximations | Barone-Adesi–Whaley, Bjerksund–Stensland, Ju, QD+ | `americanoption.cpp` | false | none | FD/binomial American present |
| Equity | FD CEV / CIR / SABR / Bates vanilla | `FdCev*`, `FdCir*`, `FdSabr*`, `FdBates*` | matching suites | false | none | |
| Equity | FD Black–Scholes Asian | `FdBlackScholesAsianEngine` | `asianoptions.cpp` | false | none | Analytic/MC Asian set present |
| Equity | LSMC American max option | `McLongstaffSchwartz` max | `mclongstaffschwartzengine.cpp` | true | partial | Vanilla LSMC pinned; max-option case not |
| Equity | Range accrual | range accrual coupons/bonds | `rangeaccrual.cpp` | false | none | |
| Equity | Dedicated `QuantoVanillaOption` type | `quantovanillaoption.hpp` | `quantooption.cpp` | false | none | Quanto engines on vanilla/barrier exist |
| Equity | Rough Heston | _(not in QL `v1.43` tree)_ | `roughhestonmodel.cpp` empty/absent on pin | false | none | Forward-compat only; promote when QL pin moves |

## How to update

1. Research tickets refresh rows from QL + libitofin using the classification scheme.
2. When a feature reaches `has_surface = true` and `has_matching_oracle = full`, remove it here and add/keep the done row in `oracle-coverage.md`.
3. Suite cases not yet on a coverage ambition stay Notes/fog until ambition is extended.

## Inventory method notes

- Excluded: credit, inflation (demoted), bindings, pure calendar/math, FX/money-only (except XCCY rates and equity–rates hybrids).
- `has_surface` from module presence + priceable path judgment, not a full API audit.
- Large suites collapsed to feature rows; this register is for prioritization decisions, not a BOOST case dump.
