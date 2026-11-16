# Callable bond pins from QuantLib v1.44 (#1414)

Generators for the QuantLib v1.44 values pinned by #1414:

| File | Produces | Pinned in |
|---|---|---|
| `c4_unexercisable.cpp` | Callable call/put clean prices with an exercise on the coupon date (row 8, `discretizedcallablefixedratebond.cpp:101-108`): 105.92283572483946, 117.40002838621254 | `crates/libitofin/tests/callable_bond_events.rs` |
| `c6_effdur.py` | `callablebonds.cpp:1113` `testEffectiveDurationAndConvexity` OAS, effective duration and convexity with the explicit 30 Nov 2023 settlement (row 21; see #1465 for the adapted port) | `crates/libitofin/tests/medium6_callable_sensitivities.rs` |
| `snap_oas.py` | `callablebonds.cpp:571` `testSnappingExerciseDate2ClosestCouponDate` OAS per call date, and settlement-normalized `cleanPriceOAS` at 4, 7 and 11 Feb 2022 with spreads 0 and 0.01 (#1466: the snapped-exercise discount ratio includes the spread, `discretizedcallablefixedratebond.cpp:76-90`) | `crates/libitofin/tests/callable_bond_snapping.rs` |

Regenerate:

```sh
# C++: QL_SRC = v1.44 source (git -C QuantLib archive v1.44 | tar -x -C "$QL_SRC"), QL_BUILD = its CMake build
c++ -std=c++17 -O2 -I"$QL_BUILD" -I"$QL_SRC" -I"$BOOST_INC" c4_unexercisable.cpp \
  -L"$QL_BUILD/ql" -lQuantLib -Wl,-rpath,"$QL_BUILD/ql" -o /tmp/c4 && /tmp/c4
# Python
uv run --with QuantLib==1.44 python c6_effdur.py
uv run --with QuantLib==1.44 python snap_oas.py
```
