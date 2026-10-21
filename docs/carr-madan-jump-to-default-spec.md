# Carr–Madan Local Volatility with Jump to Default (`libitofin`) Specification

**Status:** Proposed implementation and test spec
**Base branch:** `feat/fd-convertible-tf` (commit `2732acb5`), because the work builds on `FdmBlackScholesOp::with_extra_discount`. Branch from it, e.g. `feat/carr-madan-jtd`.
**Language surface:** Rust core and in-crate tests only. No Python, C-FFI, Go or .NET bindings.
**Source:** P. Carr & D. B. Madan, "Local Volatility Enhanced by a Jump to Default", *SIAM Journal on Financial Mathematics* 1(1), 2–15 (2010). Equation numbers below refer to the 2007 working paper: <https://engineering.nyu.edu/sites/default/files/2019-01/CarrSIAMJournalJan2019-a.pdf>.

---

## 1. Summary

Add the Carr–Madan model: a local-volatility stock that can jump to zero (default) with intensity

$$\lambda(t, S) = f(t)\, S^{p}, \qquad p < 0 .$$

Deliver three things:

1. **Pricing:** a finite-difference engine for European calls and puts on the defaultable stock, given $\sigma(S,t)$, $f(t)$ and $p$.
2. **Calibration of $f(t)$** from a survival curve (CDS) and the market call surface (eq. 8).
3. **Calibration of $\sigma(K,T)$**, the default-adjusted local volatility, from the same inputs (eq. 10), as a `LocalVolTermStructure`.

The default-free density and truncated power prices come from the **market implied-vol surface** (nonparametric). The paper's VGSSD model is not used.

## 2. Decisions already made

| ID | Decision | Source |
| --- | --- | --- |
| DEC-1 | First cut: calibration (eqs. 8, 10) plus a vanilla FD engine. Convertibles are out of scope. | User, 2026-10-21 |
| DEC-2 | Density and power moments come from the implied-vol surface, not VGSSD. | User, 2026-10-21 |
| DEC-3 | Base branch `feat/fd-convertible-tf`. | User, 2026-10-21 |
| DEC-4 | Survival input is any `DefaultProbabilityTermStructure` (e.g. a CDS-bootstrapped curve). The paper's Weibull fit is not ported. | Spec default |
| DEC-5 | European exercise only. American puts need a source term (see §8). | Spec default |
| DEC-6 | Rust core and in-crate tests only. | Spec default; matches the convertible map |

The existing convertible engine (`FdConvertibleEngine::with_hazard`) uses a **different** credit convention: partial default, no stock jump, recovery $R$ times the bond component (`.wayfinder/tickets/cb-hazard-recovery-convention.md`). Do not change it. Carr–Madan is the "stock jumps to zero" convention that the convertible map ruled out for itself.

## 3. Model and conventions

Notation follows Carr–Madan. Deterministic $r(t)$, $q(t)$; $D(T) = e^{-\int_0^T r}$.

**Dynamics (eq. 1).** Before default,

$$dS = (r - q)\,S\,dt + \sigma(S,t)\,S\,dW - S\,[\,dN - \lambda(t,S)\,dt\,],$$

so the drift of the surviving stock is $r - q + \lambda$. At default, $S$ jumps to $0$ and stays there.

**Survival (eq. 2).** $V(t) = \mathbb{Q}(\tau > t) = \mathbb{E}\big[\exp(-\int_0^t f(u) S_u^{p}\,du)\big]$. Calibration takes $V$ from a credit curve. Write $h_V(t) = -\partial_t \log V(t)$; this is `DefaultProbabilityTermStructure::hazard_rate`.

**Default-free law.** $\tilde{\mathbb{Q}}$ is the law of $S_t$ conditional on survival, with density $\tilde p(s,t)$ integrating to 1.

**Market calls.** $C(K,T)$ is the price of a call on the defaultable stock, $K > 0$. It is worth $0$ after default. $C_{KK}(s,T) = D(T)\,V(T)\,\tilde p(s,T)$ is the *defective* density of the surviving stock (total mass $D\,V$).

**Survival-weighted truncated power price** (this spec's working quantity):

$$M_p(K,T) := \int_K^\infty s^{p}\, C_{KK}(s,T)\, ds = D(T)\, \mathbb{E}^{\mathbb{Q}}\!\left[S_T^{\,p}\,\mathbf{1}_{\{S_T > K,\ \tau > T\}}\right] = V(T)\,\zeta(K,T),$$

where $\zeta$ is the paper's truncated power price (eq. 9). Working with $M_p$ means $V$ cancels from the local-vol formula.

**Hazard scale (eq. 8).** $h_V(t) = f(t)\,\mathbb{E}^{\tilde{\mathbb{Q}}}[S_t^{\,p}]$, and $\mathbb{E}^{\tilde{\mathbb{Q}}}[S_t^{\,p}] = M_p(0,t) / (D(t)\,V(t))$, so

$$f(t) = \frac{h_V(t)\, D(t)\, V(t)}{M_p(0,t)} .$$

**Local volatility (eq. 10).**

$$\sigma^2(K,T) = \frac{2\left[C_T + qC + (r-q) K C_K - K f(T)\, M_p(K,T)\right]}{K^2 C_{KK}} = \sigma^2_{\text{Dupire}}(K,T) - \frac{2 f(T)\, M_p(K,T)}{K\, C_{KK}(K,T)} ,$$

where $\sigma^2_{\text{Dupire}}$ is the usual Dupire local variance of the **market** surface (as `LocalVolSurface` computes it).

> **Erratum.** The working paper's eq. 11 prints $K^2\sigma^2 + K f V \zeta / C_{KK}$ = Dupire dollar variance, without the factor 2. That is inconsistent with its eq. 10 and with a direct Tanaka–Meyer derivation. Implement eq. 10 as written above.

**Pricing PDE** for a European claim $U(S,t)$ on the defaultable stock, with post-default value $g(t)$:

$$U_t + (r - q + \lambda) S U_S + \tfrac12 \sigma^2 S^2 U_{SS} - (r + \lambda) U + \lambda\, g(t) = 0 .$$

- Call: $g = 0$, terminal $(S-K)^+$.
- European put: $g(t) = K\,D(T)/D(t)$ (the put pays $K$ at expiry once the stock is $0$). Because $g_t = r g$, the shift $U = P - g$ removes the source term: $U$ solves the homogeneous PDE with terminal $(K - S)^+ - K = -\min(S, K)$. Price puts this way, not by adding a source term.
- Put–call parity is unchanged: $C - P = S_0 e^{-\int q} - K D(T)$.

## 4. Repository facts the design relies on

- `FdmBlackScholesOp` (`methods/finitedifferences/operators/fdmblackscholesop.rs`) supports local vol and, on this branch, `with_extra_discount(Array)`: a per-node, **time-independent** addition to the discount.
- `FdConvertibleEngine` (`pricingengines/bond/fdconvertibleengine.rs`) shows how to drive `FdmBackwardSolver` with a custom `FdmBlackScholesOp`.
- `LocalVolSurface` (`termstructures/volatility/localvolsurface.rs`) computes Dupire in Gatheral's log-moneyness form from a `BlackVolTermStructure` (trait in `termstructures/volatility/mod.rs`) plus yield and spot handles.
- `FdBlackScholesVanillaEngine` (`pricingengines/vanilla/fdblackscholesvanillaengine.rs`) is the vanilla FD engine with a `local_vol` flag.
- Credit curves: `DefaultProbabilityTermStructure` (`termstructures/credit/defaulttermstructure.rs`), `FlatHazardRate`, `InterpolatedHazardRateCurve`.
- `black_formula` and the analytic European engine (`pricingengines/blackformula.rs`, `pricingengines/vanilla/mod.rs`) supply closed-form oracles.
- No jump-to-default process, operator or engine exists yet.

## 5. Design

Names are proposals; keep the repo's naming style if something fits better.

### 5.1 `PowerHazard` (data)

`crates/libitofin/src/models/equity/jumptodefault/powerhazard.rs`

```rust
pub struct PowerHazard {
    times: Vec<Time>,   // strictly increasing knot times, times[0] > 0
    scales: Vec<Real>,  // f on (times[i-1], times[i]], all >= 0; flat after the last knot
    exponent: Real,     // p, must be <= 0 and finite
}
impl PowerHazard {
    pub fn new(times: Vec<Time>, scales: Vec<Real>, exponent: Real) -> QlResult<Self>;
    pub fn constant(scale: Real, exponent: Real) -> QlResult<Self>;
    pub fn scale(&self, t: Time) -> Real;               // f(t), piecewise constant
    pub fn intensity(&self, t: Time, spot: Real) -> Real; // f(t) * spot^p; spot > 0
    pub fn exponent(&self) -> Real;
}
```

Validation in `new` rejects unequal lengths, empty input, non-increasing times, negative or non-finite scales, and $p > 0$. ($p = 0$ is allowed: it is the constant-hazard case used by the oracles.)

### 5.2 Operator: time-dependent jump-to-default intensity

Extend `FdmBlackScholesOp` with

```rust
pub fn with_jump_to_default(self, hazard: Shared<PowerHazard>) -> Self;
```

In `set_time(t1, t2)`, evaluate $\lambda_i = f(\bar t)\, S_i^{p}$ at each node, with $S_i = e^{x_i}$ and $\bar t$ the same time the operator already uses for the step's rates. Then:

- **add $\lambda_i$ to the drift** of the $x = \ln S$ equation (drift becomes $r - q + \lambda_i - \tfrac12\sigma^2$);
- **add $\lambda_i$ to the reaction term** (discount $r + \lambda_i$), on top of any `extra_discount`.

Leave the diffusion and the existing `extra_discount` behaviour untouched. With no hazard set, the operator must be bit-for-bit unchanged; existing tests guard this.

### 5.3 Engine: `FdJumpToDefaultVanillaEngine`

`crates/libitofin/src/pricingengines/vanilla/fdjumptodefaultvanillaengine.rs`

- Inputs: `Shared<GeneralizedBlackScholesProcess>` (rates, dividend, spot, and the vol used for meshing), a `Handle<dyn LocalVolTermStructure>` for $\sigma(S,t)$ (or the process's local vol), `Shared<PowerHazard>`, and grid sizes `(t_grid, x_grid)` plus `FdmSchemeDesc`, following `FdBlackScholesVanillaEngine`.
- European exercise only; reject other exercise types with a clear error.
- Calls: terminal $(S-K)^+$, homogeneous PDE.
- Puts: solve for $U$ with terminal $-\min(S,K)$, then add $K\,D(T)/D(t_0)$ (§3).
- Lower boundary: the mesh is in $\ln S$ and never reaches $S = 0$, so keep the same boundary treatment as the BS engine.
- Results: NPV, delta, gamma, theta at spot, as the BS engine reports them.
- Also expose a **survival probability** helper: the value of a claim paying 1 at $T$ if no default ($g = 0$, terminal 1) divided by $D(T)$. Tests use it to build $V(T)$ from the model itself.

### 5.4 Call surface seam

`crates/libitofin/src/models/equity/jumptodefault/callsurface.rs`

```rust
pub trait CallSurface {
    fn discount(&self, t: Time) -> QlResult<Real>;
    fn call(&self, strike: Real, t: Time) -> QlResult<Real>;
    fn call_dk(&self, strike: Real, t: Time) -> QlResult<Real>;   // C_K
    fn call_dkk(&self, strike: Real, t: Time) -> QlResult<Real>;  // C_KK
}
```

Two implementations, so the seam is real:

1. `BlackVolCallSurface`: over a `BlackVolTermStructure` plus risk-free, dividend and spot handles. $C_K$ and $C_{KK}$ are analytic in total variance $w(k,T)$ and its strike derivatives. $C_{KK}$ must be the Gatheral density form $D\, n(d_2)\, g(k) / (K \sqrt{w})$, so that it is consistent with `LocalVolSurface`'s Dupire, which is $\partial_T w / g$.
2. A closed-form test surface (tests only): $C = V(T) \cdot \text{Black}(\tilde F, K, \sigma^2 T)\cdot D(T)$ with $V(T) = e^{-hT}$ and $\tilde F = F e^{hT}$. This is exactly the market surface of the constant-hazard, $p = 0$ model.

### 5.5 Power moments

`crates/libitofin/src/models/equity/jumptodefault/powermoments.rs`

```rust
pub fn truncated_power_price(surface: &dyn CallSurface, p: Real, strike: Real, t: Time, grid: &MomentGrid) -> QlResult<Real>; // M_p(K,T)
pub fn power_price(surface: &dyn CallSurface, p: Real, t: Time, grid: &MomentGrid) -> QlResult<Real>; // M_p(0,T)
```

Compute $M_p$ by integration by parts, which needs only $C_K$ and is more stable than integrating $C_{KK}$:

$$M_p(K,T) = -K^{p}\, C_K(K,T) - p \int_K^{\infty} s^{p-1}\, C_K(s,T)\, ds .$$

Integrate in $y = \ln s$ on a finite range set by `MomentGrid`, e.g. $\ln F \pm n_{\text{sd}}\sqrt{w_{\text{ATM}}}$ with $n_{\text{sd}}$ default 10, using composite Gauss–Legendre or Simpson with a configurable point count. For $M_p(0,T)$ start at the grid's lower bound and add the boundary term there.

**Low-strike risk.** With $p < 0$, $M_p(0,T)$ is dominated by low strikes, where implied vols are extrapolated. The truncation and the surface's extrapolation must be explicit parameters, and tests must show convergence as the range widens (§6).

### 5.6 Hazard-scale calibration (eq. 8)

`crates/libitofin/src/models/equity/jumptodefault/hazardcalibration.rs`

```rust
pub fn calibrate_power_hazard(
    surface: &dyn CallSurface,
    survival: &dyn DefaultProbabilityTermStructure,
    exponent: Real,
    times: &[Time],
    grid: &MomentGrid,
) -> QlResult<PowerHazard>;
```

For each knot $t_i$: $f(t_i) = h_V(t_i)\, D(t_i)\, V(t_i) / M_p(0, t_i)$. Use the curve's `hazard_rate(t_i)` for $h_V$. Return a `PowerHazard` that is piecewise constant on the knots.

### 5.7 `CarrMadanLocalVolSurface` (eq. 10)

`crates/libitofin/src/termstructures/volatility/carrmadanlocalvol.rs`

Implements `LocalVolTermStructure`. Holds a `LocalVolSurface` built on the same market Black surface, a `BlackVolCallSurface`, a `Shared<PowerHazard>` and a `MomentGrid`.

$$\sigma^2_{\text{CM}}(K,T) = \sigma^2_{\text{Dupire}}(K,T) - \frac{2 f(T)\, M_p(K,T)}{K\, C_{KK}(K,T)} .$$

- If $\sigma^2_{\text{CM}} \le 0$ or the result is non-finite, return an error naming $K$ and $T$, matching `LocalVolSurface`'s failure style. Do not floor silently. The FD engine's existing `illegal_local_vol_overwrite` path can still apply at engine level.
- If $f \equiv 0$, the result must equal `LocalVolSurface` exactly.

## 6. Tests and acceptance criteria

Unit tests live beside each module; the end-to-end round trip goes in `crates/libitofin/tests/carr_madan_jump_to_default.rs`.

| # | Module | Test | Pass condition |
| --- | --- | --- | --- |
| 1 | `PowerHazard` | Validation rejects unequal lengths, non-increasing times, negative scale, $p > 0$; `scale`/`intensity` on and between knots, flat extrapolation | Exact |
| 2 | Operator | No hazard set: existing BS operator and engine tests unchanged | All existing tests pass |
| 3 | Engine | **Constant hazard, $p = 0$, flat $\sigma$:** call equals Black–Scholes with rate $r + \lambda$ and dividend $q$ (equivalently $e^{-\lambda T}\,\text{BS}(r, q - \lambda)$) | Relative error $\le 10^{-4}$ on a fine grid; strikes $0.5S_0$–$1.5S_0$, $T \in \{0.25, 1, 3\}$ |
| 4 | Engine | **$f = 0$:** equals `FdBlackScholesVanillaEngine` with the same local vol and grid | $\le 10^{-10}$ absolute |
| 5 | Engine | **Put–call parity** with $p = -1.5$, non-flat local vol | $|C - P - (S_0 e^{-\int q} - K D)| \le 10^{-6} S_0$ |
| 6 | Engine | **Mesh self-convergence** with $p = -1.5$: three grids, each refining by 2 | Successive differences shrink at least first order; report the observed order |
| 7 | Engine | **Monotonicity:** call price falls and put price rises as $f$ grows | Strict on a sample grid |
| 8 | Engine | Survival helper with constant $\lambda$, $p = 0$: $V(T) = e^{-\lambda T}$ | $\le 10^{-5}$ |
| 9 | Power moments | Closed-form test surface, $V \equiv 1$: $M_p(K,T) = D\,F^{p} e^{\frac12 p(p-1)\sigma^2 T} N(d_2 + p\sigma\sqrt T)$, $d_2 = (\ln(F/K) - \tfrac12\sigma^2 T)/(\sigma\sqrt T)$; also $K \to 0$ | Relative $\le 10^{-6}$ for $p \in \{0, -0.5, -1.5\}$ |
| 10 | Power moments | Truncation convergence: widen $n_{\text{sd}}$ 6 → 8 → 10 → 12 | Changes fall below $10^{-8}$ relative |
| 11 | Power moments | `BlackVolCallSurface` over `BlackConstantVol` agrees with the closed-form test surface | Relative $\le 10^{-8}$ |
| 12 | Hazard calibration | Constant-hazard test surface ($V = e^{-hT}$, $p = 0$) with `FlatHazardRate(h)`: recovers $f = h$ | Relative $\le 10^{-6}$ |
| 13 | Hazard calibration | $p = -1.5$ on the test surface: $f(t) = h / (\tilde F^{p} e^{\frac12 p(p-1)\sigma^2 t})$ | Relative $\le 10^{-6}$ |
| 14 | Local vol | $f \equiv 0$: equals `LocalVolSurface` | Exact (bitwise or $\le 10^{-14}$) |
| 15 | Local vol | Constant-hazard test surface, $p = 0$, $f = h$: returns flat $\sigma$ although market Dupire does not | $\le 10^{-6}$ absolute in vol, strikes $0.5F$–$2F$ |
| 16 | Local vol | Negative-variance input returns an error naming $K$, $T$ | Error, no panic |
| 17 | **Round trip** | $p = -1.5$, constant $f_0$, flat diffusion $\sigma_0$. (a) Price calls on a strike × maturity grid with the engine; compute $V(T)$ with the survival helper. (b) Invert to Black vols and build a `BlackVarianceSurface`; build an `InterpolatedHazardRateCurve` from $V$. (c) Calibrate $f$ and $\sigma$. | $f$ within 2% of $f_0$; $\sigma$ within 2% relative of $\sigma_0$ for $K \in [0.7F, 1.5F]$, $T \in [0.5, 2]$. If the tolerances can't be met, report the observed errors and the dominant source (FD error, interpolation, truncation) before loosening anything. |

**Done when:** tests 1–17 pass; clippy and rustfmt are clean on changed files; existing tests are no worse than the baseline (§7); every public item has a doc comment citing the Carr–Madan equation it implements.

## 7. Verification

From the repo root, core crate only (no Python needed):

- Focused: `cargo test -p libitofin --lib <module filter>`, e.g. `jumptodefault`, `carrmadan`.
- Integration: `cargo test -p libitofin --test carr_madan_jump_to_default`.
- Full suite at the end: `cargo test -p libitofin --no-fail-fast`. This takes about 6 minutes of test time on vps1 in debug. `--no-fail-fast` keeps one failure from hiding the rest.
- Lint: `cargo clippy -p libitofin --all-targets -- -D warnings`.
- Format: `rustfmt --check` on changed `.rs` files only. A whole-tree `cargo fmt --check` shows pre-existing drift that CI does not gate (see `AGENTS.md`).

**Baseline warning.** On 2026-10-21 a probe run of the lib tests at `2732acb5` showed two failures unrelated to this work:

- `methods::finitedifferences::meshers::concentrating1dmesher::tests::a_required_critical_point_lands_on_a_grid_node`
- `pricingengines::vanilla::fdblackscholesvanillaengine::test_fd_bermudan::binding_market_regressions_cover_european_american_and_quarterly_bermudan`

Before writing code, rerun both on a clean checkout of the base. If they fail there too, record them as pre-existing, don't fix them in this work, and report them. Test 4 compares against `FdBlackScholesVanillaEngine`; if the Bermudan failure turns out to touch the European path, stop and report.

## 8. Out of scope

- American and Bermudan options. A put's post-default value $K$ is not proportional to $D(t)$, so the shift in §3 fails and a source term $\lambda K$ is needed. That needs an affine operator or an operator-splitting step condition. Note it as follow-up work.
- Convertibles and any change to `FdConvertibleEngine`.
- The VGSSD model, the Weibull survival fit, and the paper's GM/Ford empirical study.
- Monte Carlo, stochastic volatility (LSV), stochastic rates.
- Bindings (Python, C-FFI, Go, .NET).

## 9. Suggested order

1. `PowerHazard` + operator extension + engine (tests 1–8). This is self-contained and gives the pricer the round trip needs.
2. `CallSurface` + power moments (tests 9–11).
3. Hazard calibration (tests 12–13).
4. `CarrMadanLocalVolSurface` (tests 14–16).
5. Round trip (test 17), then the full verification in §7.

Commit after each step, using the repo's `feat(scope): …` style, e.g. `feat(credit-equity): add a jump-to-default FD vanilla engine`. Don't push or open a PR unless asked.

## 10. Risks

- **Low-strike extrapolation** dominates $M_p(0,T)$ for $p < 0$ and drives $f(t)$. Treat test 10 seriously, and document the extrapolation that `BlackVolCallSurface` assumes.
- **Noisy $C_{KK}$** in the local-vol denominator can make $\sigma^2_{\text{CM}}$ negative. It surfaces as an error (test 16), never a silent floor.
- **Unmerged base.** This work depends on `feat/fd-convertible-tf`. If that branch is rebased, rebase on top of it; don't copy `with_extra_discount` onto `main`.
- **Eq. 11 erratum.** Anyone checking against the paper will see the missing factor 2. The comment in `carrmadanlocalvol.rs` should point to §3 of this spec.
