# libitofin

A ground-up port of [QuantLib](https://github.com/lballabio/QuantLib) into idiomatic, FFI-agnostic Rust.

`libitofin` is the core quantitative-finance library: dates and calendars, day counters,
interpolation, integration, distributions, solvers, RNGs, quotes, term structures, stochastic
processes, and pricing engines. QuantLib is the correctness oracle - every ported number is matched
against its `test-suite/*.cpp` case within tolerance.

The import path is `libitofin`:

```rust
use libitofin::time::Date;
```

## Status

Early, pre-1.0, and under active development. Milestone 1 is complete: a European option prices
end-to-end (quote -> flat yield/vol curves -> Black-Scholes process -> analytic engine -> lazy
instrument greeks), matching QuantLib's `europeanoption.cpp` at double-rounding precision. Layers L0
through L4 and L10–L11 have core coverage; L5–L9 continue to expand.

The public API will change until 1.0. Language bindings live in sibling crates:

- [`itofin`](https://pypi.org/project/itofin/) — Python (PyO3 + maturin), published on PyPI
- `libitofin-ffi` — minimal C ABI stub today; a fuller `cbindgen` surface is planned

## License

[BSD-3-Clause](https://github.com/benbenbang/libitofin/blob/main/LICENSE) - the same license as QuantLib, the ported source.
