# Default-probability jump oracle

`credit_jumps_oracle.cpp` runs against QuantLib 1.44. The small derived hazard
curve supplies the same 2% continuous intensity as `FlatHazardRate`; QuantLib's
base owns all jump behavior. Its stock flat-curve constructors omit jumps.

Contract: `ql/termstructures/defaulttermstructure.cpp:64-103` generates December
31 dates once, updates their times when the reference date changes, and multiplies
the factors of every jump whose time is strictly inside `(0, t)`, so a jump rolled
on or before the reference date no longer applies (v1.44; 1.43 stopped at the first
jump not before `t` and applied past jumps, which changed the 3 `moving_shifted` rows).
Applied quotes must be valid and in `(0, 1]`. Density and quoted hazard remain
unscaled (`ql/termstructures/credit/hazardratestructure.hpp:106-108`).

Regenerate from the repository root with the pinned QuantLib build:

```sh
QL_SRC=/path/to/QuantLib-src-v1.44 QL_BUILD=/path/to/QuantLib-build-v1.44
c++ -std=c++17 -I"$QL_BUILD" -I"$QL_SRC" -I/opt/homebrew/opt/boost/include \
  sdk/go/testdata/credit_jumps_oracle.cpp -L"$QL_BUILD/ql" -lQuantLib \
  -Wl,-rpath,"$QL_BUILD/ql" -o /tmp/credit-jumps-oracle
/tmp/credit-jumps-oracle > sdk/go/testdata/credit_jumps_oracle.csv
```

Adjust the Boost include path for the host. Rust tests consume all 12 rows at
`1e-15` absolute tolerance. Separate analytic tests cover errors, constructor
length/date validation, input ordering, quote ownership and cached CDS repricing.

`TestPiecewiseDefaultJumpsQuantLibOracle` replays the same rows on a bootstrapped
`PiecewiseDefaultCurve`, whose nodes are not a flat 2%: it checks the jump factor
`survival * hazard / density` of each row against the curve's survival over its
unjumped node integral at `1e-13`. That ratio reads the curve's own nodes, so it
cannot see a broken bootstrap: the test also requires `Data()` to differ from the
unjumped curve and to move when a jump quote moves. Rolled moving rows follow the
base contract; a C++ `PiecewiseDefaultCurve` keeps its construction-time jump times
(`ql/termstructures/credit/piecewisedefaultcurve.hpp:248-258`).
