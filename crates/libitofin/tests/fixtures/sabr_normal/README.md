# Normal SABR formula oracle

`generate.py` prints QuantLib-Python 1.44 `sabrVolatility(..., ql.Normal)` at ten
(strike, forward, expiry, alpha, beta, nu, rho) points. The rows are transcribed
into `normal_volatility_matches_quantlib_python` in
`src/termstructures/volatility/sabr.rs`. Regenerate with
`uv run --with QuantLib==1.44 python -I generate.py`.
