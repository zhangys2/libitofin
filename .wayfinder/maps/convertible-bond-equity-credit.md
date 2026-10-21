# Convertible Bond Pricing (Equity + Credit)

## Destination

Design and implement, in Rust under `crates/libitofin`, a convertible-bond extension that keeps `BinomialConvertibleEngine` as the flat-spread QuantLib anchor and adds a one-dimensional finite-difference engine for the Tsiveriotis–Fernandes coupled system (credit-risky cash flows at \(r+s\), conversion rights at \(r\)), a stock-dependent hazard \(\lambda(S)=\lambda_0(S/S_0)^{-\alpha}\) that recovers a constant hazard at \(\alpha=0\) with an explicit meeting point to that spread, and grid Greeks plus a discrete cash-gamma hedge P&L identity. Flat-spread prices must agree with the existing binomial oracles; \(\lambda(S)\) and the gamma P&L must pass mesh self-convergence and any published numerical checks the inventory selects.

## Notes

- Domain: equity-credit convertibles already priced by the Cox–Ross–Rubinstein Tsiveriotis–Fernandes port (`BinomialConvertibleEngine`, `DiscretizedConvertible`). Credit input today is a constant spread quote. Finite-difference machinery exists for Black–Scholes, Heston, and Hull–White, not for convertibles. Convertibles are not exposed in `itofin-py`.
- Execution exception: a later session may implement this destination only after every decision ticket on this map is closed and the user confirms that gate. Until then this map is a decision record, not an execution backlog.
- Skills: `grilling` for HITL tickets, `wayfinder` for the map, `tdd` when implementation starts. Consult `domain-modeling` if the PDE ticket makes terms canonical.
- Standing preference: the binomial engine stays the flat-spread regression anchor. Do not replace it.

## Decisions so far

- Solver split (2026-10-19, grilling): keep the Cox–Ross–Rubinstein engine as the flat-spread anchor and add a one-dimensional finite-difference engine for the coupled system.
- Hazard relationship (2026-10-19, grilling): \(\lambda(S)=\lambda_0(S/S_0)^{-\alpha}\) sits beside the flat spread and must recover a constant hazard at \(\alpha=0\), with an explicit meeting point to the binomial spread quote.
- Gamma scalping (2026-10-19, grilling): grid Greeks and a discrete cash-gamma hedge P&L identity. Live order routing is out of scope.
- Language surface (2026-10-19, grilling): Rust core and in-crate tests only.
- Correctness bar (2026-10-19, grilling): flat-spread agreement with the existing QuantLib binomial oracles, plus mesh self-convergence and published numerical checks where they exist.
- Execution gate (2026-10-19, grilling): implementation waits until the decision tickets close and the user confirms that gate.
- [PDE contract features](../tickets/cb-pde-contract-features.md) — first finite-difference cut is voluntary conversion, issuer call, holder put, and fixed coupons.
- [Flat versus term-structure coefficients](../tickets/cb-flat-versus-term-structure.md) — freeze flat \(r\), \(q\), and \(\sigma\) at the conversion exercise date, matching the binomial engine.
- [Tsiveriotis–Fernandes coupled PDE statement](../tickets/cb-tf-coupled-pde-statement.md) — discretize equity \(E\) at \(r\) and cash \(C\) at \(r+s\); do not copy the blended tree discount. Flat-spread agreement is the \(s=0\) and never-converted oracles.
- [Published numerical checks](../tickets/cb-published-numerical-checks.md) — binomial tolerances are the flat-spread bar; the 1998 paper has no recoverable price; Andersen–Buffum Table 2 naïve column is the tabulated power-law price.
- [Cash-gamma scalping identity](../tickets/cb-cash-gamma-identity.md) — the caller supplies \((\Delta t, S')\); no-default P&L is grid revaluation minus \(\Delta\) shares, plus coupon and financing; credit sits inside grid theta.
- [Stock-dependent hazard and constant-spread limit](../tickets/cb-stock-hazard-spread-limit.md) — \(\lambda(S)=\lambda_0(S/S_0)^{-\alpha}\), constant at \(\alpha=0\). The spread map is \(s=\lambda_0(1-R)\).
- [Finite-difference layout for the coupled system](../tickets/cb-fdm-layout.md) — two successive one-dimensional solves on one log-spot mesh, then a joint call/put/coupon/conversion projection. \(C\) reuses the Black–Scholes operator at rates \((r+s, q+s)\).
- [Flat-spread acceptance bar](../tickets/cb-tf-acceptance-bar.md) — QuantLib binomial tolerances plus mesh self-convergence. No absolute published price.
- [Hazard recovery convention](../tickets/cb-hazard-recovery-convention.md) — partial default, no stock jump, recover \(R\) times the pre-default bond component, \(s=\lambda_0(1-R)\) at \(\alpha=0\).
- [Hazard-rate acceptance oracle](../tickets/cb-hazard-acceptance-oracle.md) — mesh self-convergence. Andersen–Buffum Table 2 does not match this recovery.
- [Default-jump term in the gamma identity](../tickets/cb-gamma-default-jump.md) — the function returns the no-default wealth change only.
- Implementation gate (2026-10-19): the last open choices are agreed, and the earlier request to implement stands. Implementation is in scope.
- Implementation (2026-10-19): `FdConvertibleEngine` and `ConvertibleCashGamma` in `crates/libitofin/src/pricingengines/bond/fdconvertibleengine.rs`. The seven new checks and the existing binomial convertible tests passed.

## Not yet specified

*(none)*

## Out of scope

- Python and C-FFI bindings for convertibles.
- Live hedge execution and order routing.
- Replacing or rewriting the binomial QuantLib port.
- Stochastic-volatility or hybrid-rate convertibles.
- Calibrating \(\lambda_0\) from a CDS curve.
- Soft call, discrete dividends, and floating-rate coupons on the finite-difference engine. The binomial engine already prices those; this cut does not.
- Face recovery, a stock jump to zero, and recovery of the whole convertible.
- Andersen–Buffum Table 2 as an acceptance oracle.
- A realized default-jump term in the cash-gamma identity.

## Frontier (open, unblocked, unclaimed)

*(none — decision record complete; implementation is in scope)*

## Claimed (not frontier)

*(none)*

## Blocked (not frontier)

*(none)*
