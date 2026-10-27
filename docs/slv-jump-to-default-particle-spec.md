# Stochastic Local Volatility with Jump to Default (`libitofin`) Specification

**Status:** Proposed implementation and test spec  
**Base branch:** `feat/carr-madan-jtd` (commit `c0eed408`), because this work builds directly on `PowerHazard`, `BlackVolCallSurface`, `CarrMadanLocalVolSurface`, `FdJumpToDefaultVanillaEngine`, and `HestonSLVProcess`. Branch from it, e.g. `feat/slv-jtd-particle`.  
**Language surface:** Rust core (`crates/libitofin`) and in-crate unit/integration tests only. No Python, C-FFI, Go, or .NET bindings in the initial slice.  
**Sources:**
1. P. Carr & D. B. Madan, "Local Volatility Enhanced by a Jump to Default", *SIAM Journal on Financial Mathematics* 1(1), 2–15 (2010).
2. J. Guyon & P. Henry-Labordère, *Nonlinear Option Pricing*, Chapman & Hall/CRC Financial Mathematics Series (2014), Chapter 11 (McKean–Vlasov particle calibration of SLV models).
3. L. Andersen & D. Buffum, "Calibration and Implementation of Convertible Bond Models", *Journal of Computational Finance* 7(2), 1–34 (2004).
4. Reference Python prototype and verification suite: `/home/zhangy/notes/slv_jd/`.

---

## 1. Summary

Extend `libitofin`'s Carr–Madan jump-to-default framework (`feat/carr-madan-jtd`) from pure local volatility (`CarrMadanLocalVolSurface`) to **Stochastic Local Volatility with Jump to Default (SLV-JD)** by coupling `HestonSLVProcess` with a state-dependent default intensity

$$\lambda(t, S) = f(t)\, \max(S, S_{\min})^{p}, \qquad p \le 0, \quad S_{\min} > 0 .$$

Deliver five components:

1. **`InterpolatedLocalVolSurface`** (`termstructures/volatility/interpolatedlocalvolsurface.rs`): a bilinear $(t, K)$-interpolated `LocalVolTermStructure` (QuantLib's `FixedLocalVolSurface` equivalent) holding the calibrated leverage grid $L(t_n, K_j)$.
2. **`SlvJumpToDefaultProcess`** (`processes/slvjumptodefaultprocess.rs`): couples `HestonSLVProcess` with a `PowerHazard` and spot floor $S_{\min}$, providing Andersen QE variance updates, default-compensated log-spot updates, and Rao–Blackwellized path survival weights.
3. **Guyon–Labordère McKean–Vlasov Particle Calibration** (`models/equity/jumptodefault/slvparticlemodel.rs`): forward particle calibration that jointly calibrates the piecewise-constant hazard scale $f(t_n)$ to a `DefaultProbabilityTermStructure` (CDS survival curve) and the leverage surface $L(t_n, K_j)$ to a `BlackVolTermStructure` (via `CarrMadanLocalVolSurface`).
4. **`McSlvJumpToDefaultVanillaEngine`** (`pricingengines/vanilla/mcslvjumptodefaultvanillaengine.rs`): a `PricingEngine` for European calls and puts on the defaultable stock, plus an Equity Default Swap (EDS) pricer.
5. **Diagnostics & Cross-Asset Greeks** (`models/equity/jumptodefault/slvdiagnostics.rs`): verification of martingale conservation, Put–Call Parity, CDS survival/par-spread recovery, Dupire-JTD skew decomposition, and Common-Random-Number (CRN) cross-asset Greeks (`delta`, `gamma`, `vega_1pct`, `cs01_10bps`, `jtd_pnl`).

---

## 2. Decisions and Design Principles

| ID | Decision | Rationale / Source |
| --- | --- | --- |
| DEC-1 | Reuse `CarrMadanLocalVolSurface` for the numerator $\sigma_{\text{CM}}^2(T, K)$ of the leverage ratio when possible, or evaluate the strike-space Dupire-JTD equation with the step-calibrated scale $f_n$. | Avoids duplicating `BlackVolCallSurface` and `truncated_power_price` (`M_p`). |
| DEC-2 | Use **Rao–Blackwellized Cox survival weights** $w_{t_n}^{(i)} = \exp(-\int_0^{t_n} \lambda(u, S_u^{(i)})\,du)$ instead of Bernoulli default coin-flips. | Keeps all $N$ particles alive across all strikes, eliminates jump-sampling variance, and makes payoffs smooth in parameters for CRN Greeks. |
| DEC-3 | Calibrate $f_n = f(t_{n+1})$ on the fly at each step $n$ via Newton–Raphson so $\frac{1}{N}\sum_{i=1}^N w_{t_n}^{(i)} \exp(-f_n \max(S_{t_n}^{(i)}, S_{\min})^p \Delta t_n) = V(t_{n+1})$. | Matches the input `DefaultProbabilityTermStructure` survival curve $V(t)$ to machine precision ($\le 10^{-13}$) at every time step regardless of stochastic volatility mixing. |
| DEC-4 | Enforce exact discrete-time martingale conservation $\frac{1}{N}\sum_{i=1}^N w_{t_{n+1}}^{(i)} S_{t_{n+1}}^{(i)} = F(t_{n+1}) = S_0 D_q(t_{n+1}) / D_r(t_{n+1})$ and second-moment-matched, orthogonalized antithetic normals $(z_1, z_2)$ at every step. | Eliminates sample-variance drift across time steps and guarantees Put–Call Parity to $\le 10^{-13} S_0$. |
| DEC-5 | Regularize Nadaraya–Watson KDE wing estimates of $\mathbb{E}^{\mathbb{Q}}[v_t \mid S_t = K, \tau > t]$ toward the survival-weighted affine regression prior $v_{\text{lin}}(\ln K)$ rather than the flat mean $\bar{v}_t$. | Preserves the negative spot-variance correlation slope ($\rho < 0$) in outer strike wings at early times where particle density is sparse. |
| DEC-6 | Rust core and in-crate tests only. | Matches `docs/carr-madan-jump-to-default-spec.md` (DEC-6). |

---

## 3. Mathematical Formulation and Conventions

### 3.1 Continuous-Time SDE System

Prior to default $\tau$, under the risk-neutral measure $\mathbb{Q}$:

$$\frac{dS_t}{S_{t-}} = \left(r(t) - q(t) + \lambda(t, S_{t-})\right) dt + L(t, S_{t-}) \sqrt{v_t}\, dW_t^S - dN_t$$

$$dv_t = \kappa (\theta - v_t)\, dt + \eta \sigma_v \sqrt{v_t}\, dW_t^v, \qquad d\langle W^S, W^v \rangle_t = \rho\, dt$$

where:
- $r(t)$ and $q(t)$ are instantaneous forward rates from `risk_free_rate: Handle<dyn YieldTermStructure>` and `dividend_yield: Handle<dyn YieldTermStructure>`, with discount factors $D_r(t) = e^{-\int_0^t r(u)du}$ and $D_q(t) = e^{-\int_0^t q(u)du}$.
- $v_0, \kappa, \theta, \sigma_v, \rho$ are the `HestonProcess` parameters, and $\eta > 0$ is the `HestonSLVProcess` `mixing_factor` (default `1.0`).
- $N_t = \mathbf{1}_{\{\tau \le t\}}$ is the default counting process with intensity $\lambda(t, S) = f(t)\, g(S)$, where $g(S) = \max(S, S_{\min})^p$ ($p \le 0$). At $t = \tau$, $S_\tau = 0$ and remains $0$ thereafter.

### 3.2 Gyöngy–Dupire Projection with Jump to Default

By the Tanaka–Meyer formula applied to $D_r(T)(S_T - K)^+ \mathbf{1}_{\{\tau > T\}}$, the model reproduces the market call surface $C(K, T)$ if and only if the effective local variance on surviving paths equals the Carr–Madan local variance:

$$\mathbb{E}^{\mathbb{Q}}\!\left[L^2(T, S_T)\, v_T \;\middle|\; S_T = K,\; \tau > T\right] = L^2(T, K)\, \mathbb{E}^{\mathbb{Q}}\!\left[v_T \;\middle|\; S_T = K,\; \tau > T\right] = \sigma_{\text{CM}}^2(K, T)$$

where (Carr–Madan eq. 10, as implemented in `CarrMadanLocalVolSurface`):

$$\sigma_{\text{CM}}^2(K, T) = \frac{2\left[C_T(K, T) + q(T) C(K, T) + (r(T) - q(T)) K C_K(K, T) - K f(T) M_p(K, T)\right]}{K^2 C_{KK}(K, T)} .$$

Hence the **leverage function** is given by:

$$L(T, K) = \frac{\sigma_{\text{CM}}(K, T)}{\sqrt{\mathbb{E}^{\mathbb{Q}}\!\left[v_T \;\middle|\; S_T = K,\; \tau > T\right]}} .$$

When $\eta \sigma_v \to 0$ and $v_0 = \theta = 1$, $\mathbb{E}^{\mathbb{Q}}[v_T \mid S_T = K, \tau > T] \equiv 1$ and $L(T, K) \equiv \sigma_{\text{CM}}(K, T)$ (recovering the 1D Carr–Madan PDE model).

### 3.3 Particle Discretization over $0 = t_0 < t_1 < \dots < t_M = T_{\max}$

For $N$ even ($N = 2 N_{\text{half}}$), initialize at $t_0 = 0$:

$$S_0^{(i)} = S_0, \qquad v_0^{(i)} = v_0, \qquad w_0^{(i)} = 1.0, \qquad i = 1, \dots, N .$$

At each step $n = 0, \dots, M - 1$ with $\Delta t = t_{n+1} - t_n$ and midpoint $\bar{t}_n = t_n + \tfrac12 \Delta t$:

1. **Step Hazard Scale $f_n$:**  
   Let $g_i = \max(S_{t_n}^{(i)}, S_{\min})^p$ and $V_{n+1} = V(t_{n+1})$. Solve for $f_n \ge 0$ via Newton–Raphson:
   $$\Phi(f_n) := \frac{1}{N} \sum_{i=1}^N w_{t_n}^{(i)} \exp\!\left(-f_n g_i \Delta t\right) - V_{n+1} = 0, \qquad \Phi'(f_n) = -\frac{\Delta t}{N} \sum_{i=1}^N w_{t_n}^{(i)} g_i \exp\!\left(-f_n g_i \Delta t\right) .$$
   Update survival weights:
   $$w_{t_{n+1}}^{(i)} = w_{t_n}^{(i)} \exp\!\left(-\lambda_n^{(i)} \Delta t\right), \qquad \lambda_n^{(i)} = f_n g_i .$$

2. **Conditional Variance Estimation $\hat{m}(K_j, t_n) \approx \mathbb{E}^{\mathbb{Q}}[v_{t_n} \mid S_{t_n} = K_j, \tau > t_n]$:**  
   - For $n = 0$: $\hat{m}(K_j, t_0) = v_0$.
   - For $n \ge 1$: Let $x_i = \ln(S_{t_n}^{(i)} / S_0)$ and $y_j = \ln(K_j / S_0)$ for strike knots $j = 1, \dots, G$. With Silverman bandwidth $h_n = \max\!\left(c_h \cdot \operatorname{std}(x) \cdot N^{-1/5},\; 0.015\right)$ and Gaussian kernel $K_h(u) = \exp(-\tfrac12 (u/h_n)^2)$:
     $$\hat{m}(K_j, t_n) = \frac{\sum_{i=1}^N w_{t_n}^{(i)} v_{t_n}^{(i)} K_h(y_j - x_i) + \varepsilon_{\text{reg}} W_n v_{\text{prior}}(y_j)}{\sum_{i=1}^N w_{t_n}^{(i)} K_h(y_j - x_i) + \varepsilon_{\text{reg}} W_n}$$
     where $W_n = \sum_{i=1}^N w_{t_n}^{(i)}$, $\varepsilon_{\text{reg}} = 10^{-4}$, and $v_{\text{prior}}(y_j)$ is the survival-weighted affine regression of $v_{t_n}$ on $x_{t_n}$ clamped to $[0.15 v_0, 3.50 v_0]$:
     $$v_{\text{prior}}(y_j) = \bar{v}_n + \frac{\operatorname{Cov}_w(v_{t_n}, x_{t_n})}{\max(\operatorname{Var}_w(x_{t_n}), 10^{-8})} (y_j - \bar{x}_n) .$$

3. **Leverage Slice Evaluation $L(\bar{t}_n, K_j)$:**  
   Evaluate $\sigma_{\text{CM}}(K_j, \bar{t}_n)$ using step hazard scale $f_n$ (clamped to $[\sigma_{\min}, \sigma_{\max}]$ when calibration-floor protection is enabled, or returning an error if strict mode is selected), and set:
   $$L(\bar{t}_n, K_j) = \operatorname{clamp}\!\left(\frac{\sigma_{\text{CM}}(K_j, \bar{t}_n)}{\sqrt{\hat{m}(K_j, t_n)}},\; L_{\min},\; L_{\max}\right) .$$

4. **Moment-Matched Orthogonalized Antithetic Draws:**  
   Draw $N_{\text{half}}$ pairs $(z_{1, k}, z_{2, k}) \sim \mathcal{N}(0, 1)$ via `Xoshiro256StarStarUniformRng` + `InverseCumulativeNormal`, form antithetic vectors $z_1 = [z_{1,\text{half}}, -z_{1,\text{half}}]$ and $z_2 = [z_{2,\text{half}}, -z_{2,\text{half}}]$, then standardize and Gram–Schmidt orthogonalize:
   $$z_1 \leftarrow \frac{z_1}{\sqrt{\frac{1}{N}\sum_{i=1}^N z_{1,i}^2}}, \qquad \tilde{z}_2 = z_2 - \left(\frac{1}{N}\sum_{i=1}^N z_{1,i} z_{2,i}\right) z_1, \qquad z_2 \leftarrow \frac{\tilde{z}_2}{\sqrt{\frac{1}{N}\sum_{i=1}^N \tilde{z}_{2,i}^2}} .$$

5. **Spot & Variance Evolution with Exact Martingale Normalization:**  
   Interpolate $L_i = L(\bar{t}_n, S_{t_n}^{(i)})$ linearly in strike, set $\sigma_{\text{eff}, i} = L_i \sqrt{\max(v_{t_n}^{(i)}, 0)}$, and evolve:
   $$\Delta W_{S, i} = \sqrt{\Delta t}\, z_{1, i}, \qquad \Delta W_{v, i} = \sqrt{\Delta t}\left(\rho z_{1, i} + \sqrt{1 - \rho^2}\, z_{2, i}\right)$$
   $$\tilde{S}_{t_{n+1}}^{(i)} = S_{t_n}^{(i)} \exp\!\left(\left(r_n - q_n + \lambda_n^{(i)} - \tfrac12 \sigma_{\text{eff}, i}^2\right)\Delta t + \sigma_{\text{eff}, i} \Delta W_{S, i}\right)$$
   $$S_{t_{n+1}}^{(i)} = \tilde{S}_{t_{n+1}}^{(i)} \cdot \frac{S_0 D_q(t_{n+1}) / D_r(t_{n+1})}{\frac{1}{N}\sum_{j=1}^N w_{t_{n+1}}^{(j)} \tilde{S}_{t_{n+1}}^{(j)}}$$
   $$v_{t_{n+1}}^{(i)} = \max\!\left(v_{t_n}^{(i)} + \kappa(\theta - v_{t_n}^{(i)})\Delta t + \eta \sigma_v \sqrt{\max(v_{t_n}^{(i)}, 0)}\, \Delta W_{v, i} + \tfrac14 \eta^2 \sigma_v^2 \left((\Delta W_{v, i})^2 - \Delta t\right),\; 10^{-4} \theta\right) .$$

### 3.4 Option and Hybrid Payoff Valuation on Defaultable Stock

At maturity $T$ (with discount factor $D_r(T)$ and ensemble default probability $P_{\text{def}}(T) = 1 - \frac{1}{N}\sum_{i=1}^N w_T^{(i)}$):
- **European Call ($K > 0$):** Pays $(S_T - K)^+$ on survival and $0$ on default:
  $$C(K, T) = D_r(T)\, \frac{1}{N}\sum_{i=1}^N \left(S_T^{(i)} - K\right)^+ w_T^{(i)} .$$
- **European Put ($K > 0$):** Pays $(K - S_T)^+$ on survival and $K$ on default ($S_\tau = 0$):
  $$P(K, T) = \underbrace{D_r(T)\, \frac{1}{N}\sum_{i=1}^N \left(K - S_T^{(i)}\right)^+ w_T^{(i)}}_{\text{Survival Put Component}} + \underbrace{D_r(T)\, K\, P_{\text{def}}(T)}_{\text{Credit Default Floor}} .$$
- **Put–Call Parity:**
  $$C(K, T) - P(K, T) = D_r(T)\, \frac{1}{N}\sum_{i=1}^N w_T^{(i)} S_T^{(i)} - D_r(T) K = S_0 D_q(T) - K D_r(T) .$$

---

## 4. Repository Facts the Design Relies On

- `models::equity::jumptodefault` (`crates/libitofin/src/models/equity/jumptodefault/mod.rs`) already exports `PowerHazard`, `CallSurface`, `BlackVolCallSurface`, `MomentGrid`, `power_price`, `truncated_power_price`, and `calibrate_power_hazard`.
- `termstructures::volatility::CarrMadanLocalVolSurface` (`carrmadanlocalvol.rs`) implements `LocalVolTermStructure` using `BlackVolCallSurface` and `PowerHazard`.
- `processes::HestonProcess` (`hestonprocess.rs`) and `processes::HestonSLVProcess` (`hestonslvprocess.rs`) exist; `HestonSLVProcess` takes `Shared<HestonProcess>`, `Shared<dyn LocalVolTermStructure>`, and `mixing_factor: Real`.
- `termstructures::credit::DefaultProbabilityTermStructure` (`defaulttermstructure.rs`) provides `survival_probability(t)` and `hazard_rate(t)`, with `FlatHazardRate` and `InterpolatedHazardRateCurve` implementations.
- `math::randomnumbers::Xoshiro256StarStarUniformRng` and `math::distributions::normal::InverseCumulativeNormal` provide fast, deterministic, zero-allocation random normal generation.
- `pricingengines::black_formula_implied_std_dev` and `black_formula` (`pricingengines/blackformula.rs`) provide implied volatility inversion and analytic Black-Scholes oracles.

---

## 5. Detailed Rust Module Design

### 5.1 `InterpolatedLocalVolSurface`

**File:** `crates/libitofin/src/termstructures/volatility/interpolatedlocalvolsurface.rs`  
Re-exported from `crates/libitofin/src/termstructures/volatility/mod.rs`.

```rust
pub struct InterpolatedLocalVolSurface {
    base: TermStructureBase,
    times: Vec<Time>,          // length M >= 2, strictly increasing, times[0] >= 0.0
    strikes: Vec<Real>,        // length G >= 2, strictly increasing, strikes[0] > 0.0
    local_vols: Matrix,        // rows = M (times), cols = G (strikes), all > 0 and finite
}

impl InterpolatedLocalVolSurface {
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        times: Vec<Time>,
        strikes: Vec<Real>,
        local_vols: Matrix,
    ) -> QlResult<Self>;

    pub fn times(&self) -> &[Time];
    pub fn strikes(&self) -> &[Real];
    pub fn matrix(&self) -> &Matrix;
}
```
- Implements `AsObservable`, `TermStructure`, `VolatilityTermStructure`, and `LocalVolTermStructure`.
- `local_vol_impl(t, strike)` performs bilinear interpolation on `(times, strikes)` with flat clamping outside `[times[0], times[M-1]]` and `[strikes[0], strikes[G-1]]`.

### 5.2 `SlvJumpToDefaultProcess`

**File:** `crates/libitofin/src/processes/slvjumptodefaultprocess.rs`  
Re-exported from `crates/libitofin/src/processes/mod.rs`.

```rust
pub struct SlvJumpToDefaultProcess {
    slv: Shared<HestonSLVProcess>,
    hazard: Shared<PowerHazard>,
    spot_floor: Real,
}

impl SlvJumpToDefaultProcess {
    pub fn new(
        slv: Shared<HestonSLVProcess>,
        hazard: Shared<PowerHazard>,
        spot_floor: Real,
    ) -> QlResult<Self>;

    pub fn slv(&self) -> &Shared<HestonSLVProcess>;
    pub fn hazard(&self) -> &Shared<PowerHazard>;
    pub fn spot_floor(&self) -> Real;

    /// Clamped intensity `f(t) * max(spot, spot_floor)^p`.
    pub fn intensity(&self, t: Time, spot: Real) -> QlResult<Real>;

    /// Single-particle step returning `(next_spot, next_var, survival_factor)`.
    pub fn evolve_with_survival(
        &self,
        t0: Time,
        spot0: Real,
        var0: Real,
        dt: Time,
        dw_spot: Real,
        dw_var: Real,
    ) -> QlResult<(Real, Real, Real)>;
}
```

### 5.3 Particle Calibration Engine (`slvparticlemodel.rs`)

**File:** `crates/libitofin/src/models/equity/jumptodefault/slvparticlemodel.rs`  
Re-exported from `crates/libitofin/src/models/equity/jumptodefault/mod.rs`.

```rust
#[derive(Clone, Debug)]
pub struct SlvJtdParticleConfig {
    pub n_particles: Size,            // default 20_000 (must be even and >= 100)
    pub time_steps_per_year: Size,    // default 100 (>= 10)
    pub n_strike_knots: Size,         // default 61 (>= 11)
    pub log_moneyness_low: Real,      // default -1.10
    pub log_moneyness_high: Real,     // default +0.90
    pub kde_bandwidth_scale: Real,    // default 0.85
    pub wing_regularization: Real,    // default 1.0e-4
    pub min_local_vol: Volatility,    // default 0.03
    pub max_local_vol: Volatility,    // default 1.50
    pub spot_floor_ratio: Real,       // default 0.05 (spot_floor = spot_floor_ratio * S_0)
    pub moment_grid: MomentGrid,      // default MomentGrid::standard()
    pub seed: u64,                    // default 42
}

pub struct SlvJtdParticleSnapshot {
    pub time: Time,
    pub spots: Vec<Real>,
    pub variances: Vec<Real>,
    pub survival_weights: Vec<Real>,
}

pub struct SlvJtdCalibrationResult {
    pub leverage_surface: Shared<InterpolatedLocalVolSurface>,
    pub calibrated_hazard: Shared<PowerHazard>,
    pub process: Shared<SlvJumpToDefaultProcess>,
    pub time_grid: Vec<Time>,
    pub strike_grid: Vec<Real>,
    pub leverage_grid: Matrix,
    pub dupire_jtd_grid: Matrix,
    pub dupire_no_default_grid: Matrix,
    pub cond_var_grid: Matrix,
    pub survival_probs_mc: Vec<Real>,
    pub survival_probs_target: Vec<Real>,
    pub snapshots: Vec<SlvJtdParticleSnapshot>,
}

pub fn calibrate_slv_jump_to_default(
    black_surface: Handle<dyn BlackVolTermStructure>,
    heston: Shared<HestonProcess>,
    survival_curve: Handle<dyn DefaultProbabilityTermStructure>,
    exponent: Real,
    mixing_factor: Real,
    max_maturity: Time,
    config: &SlvJtdParticleConfig,
) -> QlResult<SlvJtdCalibrationResult>;
```

### 5.4 Vanilla & Hybrid Pricing Engine (`mcslvjumptodefaultvanillaengine.rs`)

**File:** `crates/libitofin/src/pricingengines/vanilla/mcslvjumptodefaultvanillaengine.rs`  
Re-exported from `crates/libitofin/src/pricingengines/vanilla/mod.rs`.

```rust
pub struct McSlvJumpToDefaultVanillaEngine {
    calibration: Shared<SlvJtdCalibrationResult>,
}

impl McSlvJumpToDefaultVanillaEngine {
    pub fn new(calibration: Shared<SlvJtdCalibrationResult>) -> Self;

    /// Price European Call or Put at `expiry` and `strike`, returning
    /// `(npv, survival_component, default_floor_component)`.
    pub fn price_components(
        &self,
        option_type: OptionType,
        strike: Real,
        expiry: Time,
    ) -> QlResult<(Real, Real, Real)>;

    /// Price an Equity Default Swap (EDS) with protection barrier `barrier_ratio * S_0`
    /// and recovery `eds_recovery`, returning `(par_spread, total_trigger_prob, credit_default_prob)`.
    pub fn price_equity_default_swap(
        &self,
        maturity: Time,
        barrier_ratio: Real,
        eds_recovery: Real,
    ) -> QlResult<(Rate, Real, Real)>;
}
```
- Implements `PricingEngine` for `OneAssetOptionArguments` / `OneAssetOptionResults` (European exercise only; populates `value` and `additional_results` for `"survivalComponent"`, `"defaultFloor"`, and `"survivalProbability"`).

### 5.5 Verification Diagnostics & Cross-Asset Greeks (`slvdiagnostics.rs`)

**File:** `crates/libitofin/src/models/equity/jumptodefault/slvdiagnostics.rs`  
Re-exported from `crates/libitofin/src/models/equity/jumptodefault/mod.rs`.

```rust
#[derive(Clone, Debug)]
pub struct CrossAssetGreeks {
    pub price: Real,
    pub delta: Real,
    pub gamma: Real,
    pub vega_1pct: Real,
    pub cs01_10bps: Real,
    pub jtd_pnl: Real,
}

pub fn compute_cross_asset_greeks(
    calibration: &SlvJtdCalibrationResult,
    option_type: OptionType,
    strike: Real,
    expiry: Time,
) -> QlResult<CrossAssetGreeks>;
```
- Uses Common Random Numbers (CRN) with the fixed calibrated leverage surface `InterpolatedLocalVolSurface` under $\pm 1\%$ spot bumps, $+1\%$ parallel leverage bump, and $+10\text{ bps}$ hazard rate bump.

---

## 6. Tests and Acceptance Criteria

Unit tests live beside each module; the end-to-end integration test lives in `crates/libitofin/tests/slv_jump_to_default_particle.rs`.

| # | Module | Test | Pass condition |
| --- | --- | --- | --- |
| 1 | `InterpolatedLocalVolSurface` | Rejects non-increasing times/strikes, dimension mismatch, non-positive vols; exact on knots, bilinear between knots, flat outside bounds | Exact ($\le 10^{-14}$) |
| 2 | `SlvJumpToDefaultProcess` | Rejects `spot_floor <= 0`; with $f = 0$, `evolve_with_survival` returns `survival_factor == 1.0` and matches `HestonSLVProcess::evolve` | Exact ($\le 10^{-14}$) |
| 3 | Step Hazard Solver | Newton–Raphson step solver recovers constant hazard $f = h$ when $p = 0$ and matches arbitrary target survival steps $V(t_{n+1}) \in (0, V(t_n)]$ | $|V_{\text{MC}}(t_n) - V(t_n)| \le 10^{-13}$ |
| 4 | Martingale Conservation | Discounted survival-weighted equity spot $\frac{D_r(T)}{D_q(T)} \frac{1}{N}\sum_{i=1}^N w_T^{(i)} S_T^{(i)} = S_0$ across $T \in \{0.5, 1.0, 1.5, 2.0\}$ with $p = -1.25$ and $\sigma_v > 0$ | Relative error $\le 10^{-13}$ |
| 5 | Put–Call Parity | European Call and Put prices from `McSlvJumpToDefaultVanillaEngine` satisfy $C(K, T) - P(K, T) = S_0 D_q(T) - K D_r(T)$ across $K \in [0.65 S_0, 1.30 S_0]$ | Absolute error $\le 10^{-11} S_0$ |
| 6 | CDS Par Spread Recovery | Re-integrating the particle survival curve $V_{\text{MC}}(t_n)$ over protection and risky-annuity legs recovers input market CDS par spreads across all tenors | Spread error $\le 0.05\text{ bps}$ ($5 \times 10^{-6}$) |
| 7 | Zero-Vol-of-Vol Limit ($\eta \to 0$) | When `mixing_factor = 0.0` (or $\sigma_v \to 0$) and $v_0 = \theta = 1.0$, $\hat{m}(K, t) \equiv 1.0$ and $L(t, K)$ equals $\sigma_{\text{CM}}(K, t)$ | $\le 10^{-12}$ on grid knots |
| 8 | Zero-Hazard Limit ($f \equiv 0$) | With a zero-hazard credit curve, `dupire_jtd_grid` equals `dupire_no_default_grid`, $w_T^{(i)} \equiv 1$, and `put_default_floor == 0.0` | Exact ($\le 10^{-14}$) |
| 9 | Dupire-JTD Skew Reduction | With $h > 0$ and $p = -1.25$, $\sigma_{\text{CM}}(K, T) < \sigma_{\text{Dupire}}(K, T)$ across all strikes, and the reduction $\sigma_{\text{Dupire}} - \sigma_{\text{CM}}$ is strictly larger at $K = 0.75 S_0$ than at $K = 1.15 S_0$ | Strict inequality |
| 10 | Constant-Hazard Flat-Vol Oracle | Constant hazard $h = 0.03$, $p = 0$, flat Black diffusion $\sigma_0 = 0.25$: SLV-JD particle engine with $\sigma_v = 0.35$ matches `ConstantHazardCallSurface` prices | Implied vol RMSE $\le 20\text{ bps}$ for $K \in [0.8 F, 1.2 F]$, $T \in \{0.5, 1.0\}$ |
| 11 | Skewed Surface Joint Repricing | Skewed market `BlackVarianceSurface` + upward-sloping CDS curve (55–110 bps), $p = -1.25$, $\kappa = 2.2, \sigma_v = 0.45, \rho = -0.55$: particle engine reprices the market implied volatility surface across $T \in \{0.5, 1.0, 1.5\}$, $K/S_0 \in \{0.8, 0.9, 1.0, 1.1, 1.2\}$ | Surface IV RMSE $\le 25\text{ bps}$; ATM IV error $\le 10\text{ bps}$ |
| 12 | Agreement with 1D FD Engine | When $\eta = 0$ (no stochastic vol mixing), `McSlvJumpToDefaultVanillaEngine` agrees with `FdJumpToDefaultVanillaEngine` on European Calls and Puts | Price difference within $\le 25\text{ bps}$ of implied vol |
| 13 | Cross-Asset Greeks Consistency | Deep OTM Put ($K = 0.70 S_0$) exhibits negative `delta`, positive `gamma`, positive `vega_1pct`, **strictly positive `cs01_10bps`** (a $+10\text{ bps}$ credit spread widening increases put value), and positive `jtd_pnl` | Strict signs & finite values |
| 14 | Equity Default Swap (EDS) | `price_equity_default_swap` with $B = 0.30 S_0$ has `total_trigger_prob >= credit_default_prob`, and converges to the pure CDS par spread as `barrier_ratio -> 0` | Spread difference $\le 0.5\text{ bps}$ as $B \to 0$ |

**Done when:** tests 1–14 pass; `cargo clippy -p libitofin --all-targets -- -D warnings` is clean; `rustfmt --check` is clean on all added/modified `.rs` files; existing `carr_madan_jump_to_default` and `hestonslvprocess` tests pass without regression.

---

## 7. Verification Commands

From `/home/zhangy/git/libitofin` (core crate only, no Python needed):

```bash
# 1. Unit tests for new modules
cargo test -p libitofin --lib interpolatedlocalvolsurface
cargo test -p libitofin --lib slvjumptodefaultprocess
cargo test -p libitofin --lib slvparticlemodel
cargo test -p libitofin --lib mcslvjumptodefaultvanillaengine
cargo test -p libitofin --lib slvdiagnostics

# 2. Existing Carr-Madan & Heston-SLV regression tests
cargo test -p libitofin --lib jumptodefault
cargo test -p libitofin --lib carrmadanlocalvol
cargo test -p libitofin --lib hestonslvprocess
cargo test -p libitofin --test carr_madan_jump_to_default

# 3. End-to-end SLV-JD integration suite
cargo test -p libitofin --test slv_jump_to_default_particle

# 4. Lint & formatting gates on changed files
cargo clippy -p libitofin --all-targets -- -D warnings
rustfmt --check crates/libitofin/src/termstructures/volatility/interpolatedlocalvolsurface.rs \
  crates/libitofin/src/processes/slvjumptodefaultprocess.rs \
  crates/libitofin/src/models/equity/jumptodefault/slvparticlemodel.rs \
  crates/libitofin/src/models/equity/jumptodefault/slvdiagnostics.rs \
  crates/libitofin/src/pricingengines/vanilla/mcslvjumptodefaultvanillaengine.rs \
  crates/libitofin/tests/slv_jump_to_default_particle.rs
```

---

## 8. Suggested Implementation Order

1. **`InterpolatedLocalVolSurface`** (`interpolatedlocalvolsurface.rs` + test 1). Self-contained `LocalVolTermStructure` needed by `HestonSLVProcess`.
2. **`SlvJumpToDefaultProcess`** (`slvjumptodefaultprocess.rs` + test 2). Thin wrapper around `HestonSLVProcess` and `PowerHazard`.
3. **Particle Calibration Core** (`slvparticlemodel.rs` + tests 3, 4, 6, 7, 8, 9). Newton step hazard solver, affine-prior KDE conditional variance, and forward particle loop.
4. **`McSlvJumpToDefaultVanillaEngine`** (`mcslvjumptodefaultvanillaengine.rs` + tests 5, 10, 11, 12, 14). Vanilla option and EDS pricing from calibrated particle snapshots.
5. **Diagnostics & Cross-Asset Greeks** (`slvdiagnostics.rs` + test 13) and the integration test file `crates/libitofin/tests/slv_jump_to_default_particle.rs`.

Commit after each step using `feat(credit-equity): ...`, e.g. `feat(credit-equity): add McKean-Vlasov particle calibration for SLV-JD`.

---

## 9. Out of Scope (Follow-Up Slices)

- 2D Fokker–Planck PDE calibration (`HestonSLVFDMModel` with jump-to-default).
- American / Bermudan options and convertible bonds with stochastic volatility (requires Longstaff–Schwartz or Deep BSDE backward induction on top of the calibrated particle paths).
- Stochastic interest rates coupled with SLV-JD.
- Python (`crates/itofin-py`), C-FFI (`crates/libitofin-ffi`), Go, and .NET bindings.
