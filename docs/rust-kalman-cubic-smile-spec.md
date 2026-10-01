# Fixed-Knot Kalman Smile Filter in Rust (`libitofin`) Specification

**Status:** Proposed Architecture & Implementation Spec  
**Target Module:** `crates/libitofin/src/termstructures/volatility/kalman_smile.rs`  
**Related Specs:**
- [`docs/superpowers/specs/2026-09-30-kalman-cubic-smile-design.md`](superpowers/specs/2026-09-30-kalman-cubic-smile-design.md) (Python Prototype Specification)
- [`docs/superpowers/specs/2026-09-29-cubic-smile-standard-deviation.md`](superpowers/specs/2026-09-29-cubic-smile-standard-deviation.md) (Static Cubic Smile Specification)

---

## 1. Executive Summary & Expected Speedup

The Python prototype (`example/btc-option-iv/btc_option_iv/kalman.py`) validated the mathematical model for filtering live Black implied volatility surfaces:
- Natural cubic spline basis on 9 standardized moneyness knots.
- Joseph-stabilized covariance updates preserving symmetry and positive semi-definiteness.
- Monotonic coordinate transport under moving forwards, time-to-expiry, and ATM volatilities.
- Statistical 5-$\sigma$ innovation gating for outlier rejection.

Porting this filter to native Rust in `libitofin` provides significant latency and throughput improvements because the state dimensionality is fixed at compile time ($n = 9$).

### Microbenchmark & Speedup Projection

| Metric / Operation | Python Prototype (`numpy`/`scipy`) | Native Rust (`libitofin` stack/SIMD) | Expected Speedup |
| :--- | :--- | :--- | :--- |
| **Single-quote update ($m = 1$)** | **171.6 µs** (with `eigvalsh` PSD check)<br>*(42.0 µs unvalidated)* | **0.05 – 0.08 µs** (50 – 80 ns) | **~2,000×** *(~500× vs unvalidated)* |
| **Batch update ($m = 15$)** | **196.6 µs** | **1.0 – 2.5 µs** | **~80× – 200×** |
| **Peak throughput (single core)** | ~5,800 updates / sec | **> 12,000,000 updates / sec** | **~2,000×** |
| **Heap allocations per update** | Multiple array objects / slices | **0 bytes** (100% stack allocated) | **$\infty$** (allocation-free) |
| **State memory footprint** | Several KB (Python objects) | **720 bytes** ($9 + 81$ `f64`s) | **Fits entirely in L1 cache** |

#### Root Causes of the Speedup:
1. **Zero Allocations & Cache Locality:** A 9-element mean vector and a $9 \times 9$ covariance matrix require only 720 bytes total. All operations run directly on stack arrays (`[f64; 9]` and `[[f64; 9]; 9]`).
2. **Scalar Specialization ($m = 1$):** Live streaming ticks arrive contract-by-contract. When $m = 1$, the innovation covariance $S = h P h^T + R$ is a scalar float. Cholesky factorization and matrix inversion are completely bypassed.
3. **No LAPACK/C-Bridge Overhead:** Small-matrix LAPACK calls in Python (`cho_factor`/`cho_solve`) carry substantial dispatch and Python object marshalling overhead.
4. **Fast Basis Interpolation:** The 9 cardinal basis functions on fixed knots are piecewise cubics with precomputed analytical coefficients, evaluated in $< 10\text{ ns}$ via Horner's method.

---

## 2. Core Architecture & Data Types

The implementation lives in `crates/libitofin/src/termstructures/volatility/kalman_smile.rs` and implements the `SmileSection` trait.

```rust
use crate::types::{Rate, Real, Size, Time, Volatility};
use crate::errors::QlResult;

pub const N_KNOTS: Size = 9;
pub const KNOTS: [Real; N_KNOTS] = [-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0];
pub const MAX_OVERHANG: Real = 0.25;
pub const MAX_END_SLOPE: Real = 0.10;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterConfig {
    pub process_iv_rate: Real,      // default: 0.001 (0.1% / sqrt(sec))
    pub measurement_sd: Real,       // default: 0.005 (0.5% ref SD)
    pub spread_floor: Real,         // default: 0.001
    pub reference_spread: Real,     // default: 0.010 (1% ref spread)
    pub correlation_length: Real,   // default: 1.0
    pub independent_fraction: Real, // default: 0.10
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmileContext {
    pub forward: Rate,
    pub exercise_time: Time,
    pub atm_vol: Volatility,
}

impl SmileContext {
    #[inline]
    pub fn scale(&self) -> Real {
        self.atm_vol * self.exercise_time.sqrt()
    }

    #[inline]
    pub fn coordinate(&self, strike: Rate) -> Real {
        (strike.ln() - self.forward.ln()) / self.scale()
    }

    #[inline]
    pub fn strike(&self, x: Real) -> Rate {
        (self.forward.ln() + x * self.scale()).exp()
    }
}

#[derive(Clone, Debug)]
pub struct KalmanSmileSection {
    mean: [Volatility; N_KNOTS],
    covariance: [[Real; N_KNOTS]; N_KNOTS],
    context: SmileContext,
    config: FilterConfig,
    last_update_ms: u64,
}
```

---

## 3. Mathematical Specifications

### 3.1 Precomputed Cardinal Natural Spline Basis

For the fixed knot vector $z \in \{-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0\}$, the 9 cardinal basis splines $b_j(x)$ satisfy:
$$b_j(z_k) = \delta_{jk}, \quad s''(z_0) = s''(z_8) = 0$$

On each of the 8 intervals $[z_k, z_{k+1}]$, each $b_j(x)$ is a cubic polynomial:
$$b_j(x) = c_{0, k, j} + c_{1, k, j}(x - z_k) + c_{2, k, j}(x - z_k)^2 + c_{3, k, j}(x - z_k)^3$$

The coefficient tensor `BASIS_COEFFS: [[[f64; 4]; 9]; 8]` is evaluated at compile time (`const`) or lazily initialized once. Evaluating the vector $h(x) = [b_0(x), \dots, b_8(x)]$ requires only:
1. Finding interval $k$ ($O(1)$ branch/search).
2. Evaluating 9 cubics with Horner's rule (vectorizable with SIMD `fma`).

### 3.2 Temporal Prediction (Process Noise)

Given elapsed time $\Delta t \ge 0$ seconds:
$$v_{\text{pred}} = v$$
$$P_{\text{pred}} = P + Q(\Delta t)$$
$$Q(\Delta t) = q \cdot \Delta t \cdot \left((1 - f) C + f I\right)$$
where $q = (\text{process\_iv\_rate})^2$, $f = \text{independent\_fraction}$, and $C_{ij} = \exp(-|z_i - z_j| / \ell)$.
$C$ is constant for a given correlation length $\ell$ and is precomputed.

### 3.3 Coordinate Transport

When $(F, T, \sigma_{\text{atm}})$ changes:
1. Compute coordinates of new knots in the old coordinate system:
   $$u_j = \frac{\ln(F_{\text{new}}) - \ln(F_{\text{old}}) + z_j \cdot \text{scale}_{\text{new}}}{\text{scale}_{\text{old}}}$$
2. Calculate overhang distances $d_j = \max(0, -3 - u_j, u_j - 3)$.
3. If $\max(d_j) > 0.25$, return `Err(FilterError::OverhangExceeded)` to trigger cold bootstrap.
4. Construct Jacobian matrix $J \in \mathbb{R}^{9 \times 9}$:
   - Interior knots: $J_{j, :} = h(\text{clamp}(u_j, -3, 3))$.
   - Boundary overhang: add $d_j \cdot \text{clamp}(b'(z_{\text{end}}), -0.10, 0.10)$.
5. Transport state:
   $$v \leftarrow J v$$
   $$P \leftarrow J P J^T + \text{diag}\left(\left(0.01 \frac{d_j}{0.25}\right)^2\right)$$
   $$P \leftarrow \frac{1}{2}(P + P^T)$$

### 3.4 Measurement Updates

#### A. Scalar Fast Path ($m = 1$)
For an incoming single quote $(K, y, w)$ where $w = \text{ask\_iv} - \text{bid\_iv}$:
1. Compute coordinate $x = ( \ln K - \ln F ) / \text{scale}$. If $|x| > 3$, ignore quote.
2. Evaluate basis vector $h = [b_0(x), \dots, b_8(x)]$.
3. Compute variance $R = \sigma_{\text{ref}}^2 \cdot \max(w, w_{\text{floor}}) / w_{\text{ref}}$.
4. Compute intermediate vector $p_h = P h^T \in \mathbb{R}^9$.
5. Innovation scalar: $e = y - h v$.
6. Innovation variance: $S = h p_h + R$.
7. **5-$\sigma$ Innovation Gate:**
   $$\text{If } \frac{|e|}{\sqrt{S}} > 5.0 \implies \text{reject as outlier, record diagnostic}.$$
8. Kalman Gain:
   $$K = \frac{1}{S} p_h \in \mathbb{R}^9$$
9. Mean update:
   $$v \leftarrow v + K \cdot e$$
10. Joseph-form covariance update:
    $$A = I - K h$$
    $$P \leftarrow A P A^T + R \cdot (K K^T)$$
    $$P \leftarrow \frac{1}{2}(P + P^T)$$

#### B. Batch Path ($m > 1$)
For simultaneous quotes (e.g. order-book snapshots):
1. $H \in \mathbb{R}^{m \times 9}$, $R = \text{diag}(\sigma_1^2, \dots, \sigma_m^2)$.
2. Innovation vector: $e = y - H v$.
3. Innovation covariance: $S = H P H^T + R \in \mathbb{R}^{m \times m}$.
4. **Safe Cholesky Factorization:** Factorize $S = L L^T$. If factorization fails (non-PD), return `Err(FilterError::DegenerateCovariance)` without panicking or modifying the predicted state.
5. Solve $L D = H P$ and $L^T K^T = D$ via forward/back substitution to find gain $K \in \mathbb{R}^{9 \times m}$.
6. Update:
   $$v \leftarrow v + K e$$
   $$A = I - K H$$
   $$P \leftarrow A P A^T + K R K^T$$
   $$P \leftarrow \frac{1}{2}(P + P^T)$$

---

## 4. Error Handling & Diagnostics

Unlike the legacy C++ QuantLib / Fortran patterns, the Rust port strictly forbids panics on singular or ill-conditioned inputs:

```rust
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    #[error("coordinate overhang {0:.4} exceeds limit 0.25")]
    OverhangExceeded(Real),

    #[error("insufficient distinct strikes for bootstrap: required {required}, got {actual}")]
    InsufficientBootstrapData { required: usize, actual: usize },

    #[error("innovation covariance matrix is non-positive-definite")]
    DegenerateCovariance,

    #[error("observation coordinate {0:.4} lies outside knot boundary [-3, 3]")]
    OutOfDomain(Real),
}

#[derive(Clone, Copy, Debug)]
pub enum UpdateStatus {
    Accepted { innovation: Real, z_score: Real },
    GatedOutlier { innovation: Real, z_score: Real },
    OutOfDomain,
    StaleIgnored,
}
```

### Regime-Change Detection
Maintain a rolling ring-buffer of the last 10 gated observations. If $\ge 3$ distinct strikes breach the 5-$\sigma$ gate in the same direction within a 3.0-second window, the filter flags a `RegimeChangeDetected` diagnostic and triggers an automatic cold-bootstrap from fresh book quotes.

---

## 5. API Surface & Interoperability

### 5.1 Rust Core API (`libitofin`)

```rust
impl KalmanSmileSection {
    /// Initialize with at least 2 distinct in-range quotes using curvature regularization.
    pub fn bootstrap(
        strikes: &[Rate],
        mid_ivs: &[Volatility],
        spreads: &[Real],
        context: SmileContext,
        config: FilterConfig,
    ) -> QlResult<Self>;

    /// Advance process noise for elapsed seconds.
    pub fn predict(&mut self, dt: Real) -> QlResult<()>;

    /// Transport prior state across coordinate changes.
    pub fn transport(&mut self, new_context: SmileContext) -> QlResult<()>;

    /// Assimilate single ticker quote (zero allocation).
    pub fn update_scalar(
        &mut self,
        strike: Rate,
        mid_iv: Volatility,
        spread: Real,
    ) -> QlResult<UpdateStatus>;

    /// Convert state to static CubicSmileSection for evaluation / plotting.
    pub fn to_cubic_smile_section(&self) -> QlResult<CubicSmileSection>;

    /// Direct volatility query satisfying the SmileSection trait.
    pub fn volatility(&self, strike: Rate) -> QlResult<Volatility>;
}
```

### 5.2 Python Bindings (`crates/itofin-py`)

A PyO3 class `PyKalmanSmile` is exposed in `itofin.termstructures`:
- Exposes `predict(dt)`, `transport(...)`, `update_scalar(...)`, and `update_batch(...)`.
- Exposes `knot_ivs() -> list[float]` and `covariance() -> list[list[float]]`.
- Direct conversion to Python `CubicSmileSection`.
- Calling latency via PyO3: **~100 ns** total.

### 5.3 C / FFI & Go Bindings (`crates/libitofin-ffi` and `sdk/go`)

- `itofin_kalman_smile_create(...)`
- `itofin_kalman_smile_update_scalar(...)`
- `itofin_kalman_smile_volatility(...)`
- `itofin_kalman_smile_destroy(...)`

---

## 6. Implementation & Verification Plan

### Phase 1: Core Mathematical Primitives
- [ ] Compile-time cardinal basis evaluation on fixed knot coordinates.
- [ ] Fixed-size matrix algebra helpers for $9 \times 9$ Joseph updates (`no_std` compatible).
- [ ] Unit tests checking $H$ evaluation matches SciPy `CubicSpline(KNOTS, eye(9))` to $10^{-14}$.

### Phase 2: Filter Implementation & Unit Tests
- [ ] Implement `predict`, `transport`, `update_scalar`, and `update_batch`.
- [ ] Golden oracle tests: Verify numerical agreement with `btc_option_iv.kalman` Python implementation across 1,000 randomized synthetic trajectories (tolerance $\le 10^{-12}$).
- [ ] Numerical stability tests: Verify positive semi-definiteness ($P \succeq 0$) under 100,000 ill-conditioned quote cycles.

### Phase 3: Benchmarks & Python Bindings
- [ ] Add `benches/kalman_smile.rs` using Criterion.
- [ ] Implement PyO3 wrapper in `crates/itofin-py`.
- [ ] Integrate into `example/btc-option-iv` as a drop-in acceleration option.
