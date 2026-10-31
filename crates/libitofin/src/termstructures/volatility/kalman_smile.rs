//! Fixed-knot natural cubic Kalman smile in standardized log-moneyness.
//!
//! Provides recursive Bayesian updating of nine fixed-knot implied volatility
//! ordinates under streaming option quotes, coordinate transport under moving
//! forward/ATM volatility/time, and robust innovation handling (Huber
//! down-weighting plus a hard sigma gate).

#![allow(clippy::needless_range_loop)]

use std::sync::LazyLock;

use crate::errors::QlResult;
use crate::math::comparison::close_enough;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::cubic::{CubicDerivativeApprox, CubicInterpolation};
use crate::termstructures::volatility::VolatilityType;
use crate::termstructures::volatility::cubicsmile::{CubicSmileSection, DEFAULT_STD_DEV_POINTS};
use crate::termstructures::volatility::smilesection::{SmileSection, SmileSectionBase};
use crate::time::daycounters::actual365fixed::Actual365Fixed;
use crate::types::{Rate, Real, Size, Time, Volatility};
use crate::{fail, require};

/// Default hard innovation gate in sigmas.
pub const DEFAULT_GATE_SIGMA: Real = 5.0;

/// Default Huber threshold in sigmas.
pub const DEFAULT_ROBUST_SIGMA: Real = 3.0;

/// Per-update clamp on the batch-mean normalized innovation squared used for adaptation.
const NIS_ADAPT_CLAMP: (Real, Real) = (0.1, 10.0);

/// Number of fixed knots in the standardized coordinate system.
pub const N_KNOTS: Size = 9;

/// Default natural-cubic knot locations in signed standard-deviation units.
pub const KNOTS: [Real; N_KNOTS] = DEFAULT_STD_DEV_POINTS;

/// Maximum allowed coordinate overhang in standardized units before triggering reinitialization.
pub const MAX_OVERHANG: Real = 0.25;

/// Maximum allowed tail slope for boundary extrapolation during transport.
pub const MAX_END_SLOPE: Real = 0.10;

/// Knot vector of size 9.
pub type KnotVector = [Real; N_KNOTS];

/// 9x9 knot matrix.
pub type KnotMatrix = [[Real; N_KNOTS]; N_KNOTS];

/// Result of coordinate transport: (mean, covariance, jacobian).
pub type TransportResult = (KnotVector, KnotMatrix, KnotMatrix);

/// Static cardinal natural cubic basis interpolations on the fixed 9-knot grid.
static BASIS: LazyLock<Vec<CubicInterpolation>> = LazyLock::new(|| {
    let knots = KNOTS.to_vec();
    (0..N_KNOTS)
        .map(|col| {
            let mut ords = vec![0.0; N_KNOTS];
            ords[col] = 1.0;
            CubicInterpolation::new(knots.clone(), ords, CubicDerivativeApprox::Spline)
                .expect("valid natural cubic basis")
        })
        .collect()
});

/// Maximum boundary snapping tolerance for knot-domain standardized coordinate queries.
const KNOT_DOMAIN_TOLERANCE: Real = 16.0 * Real::EPSILON * 3.0;

#[inline]
fn snap_to_knot_domain(x: Real) -> Real {
    if x < -3.0 && -3.0 - x <= KNOT_DOMAIN_TOLERANCE {
        -3.0
    } else if x > 3.0 && x - 3.0 <= KNOT_DOMAIN_TOLERANCE {
        3.0
    } else {
        x
    }
}

/// Evaluates the 9 cardinal natural cubic basis splines at standardized coordinate `x`.
pub fn basis_vector(x: Real) -> QlResult<[Real; N_KNOTS]> {
    require!(x.is_finite(), "basis queries must be finite");
    let x = snap_to_knot_domain(x);
    require!(
        (-3.0..=3.0).contains(&x),
        "basis query ({x}) is outside the knot domain [-3.0, 3.0]"
    );
    let mut h = [0.0; N_KNOTS];
    for j in 0..N_KNOTS {
        h[j] = BASIS[j].value(x)?;
    }
    Ok(h)
}

/// Evaluates the first derivatives of the 9 cardinal natural cubic basis splines at `x`.
pub fn basis_derivative(x: Real) -> QlResult<[Real; N_KNOTS]> {
    require!(x.is_finite(), "basis queries must be finite");
    let x = snap_to_knot_domain(x);
    require!(
        (-3.0..=3.0).contains(&x),
        "basis query ({x}) is outside the knot domain [-3.0, 3.0]"
    );
    let mut d = [0.0; N_KNOTS];
    for j in 0..N_KNOTS {
        d[j] = BASIS[j].derivative(x)?;
    }
    Ok(d)
}

/// Experimental noise controls for the Kalman smile filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterConfig {
    pub process_iv_rate: Real,
    pub measurement_sd: Real,
    pub spread_floor: Real,
    pub reference_spread: Real,
    pub correlation_length: Real,
    pub independent_fraction: Real,
    /// Quotes whose prior innovation exceeds this many sigmas are rejected outright.
    pub gate_sigma: Real,
    /// Quotes beyond this many sigmas (but inside `gate_sigma`) are Huber
    /// down-weighted by inflating their noise variance by `|z| / robust_sigma`.
    pub robust_sigma: Real,
    /// Process-noise adaptation gain in `[0, 1]`; `0` (default) disables it. Each
    /// update feeds the mean of `z²` (prior-standardized innovations of the accepted
    /// quotes) into an exponential average with weight `adapt_rate`, and the
    /// process-noise multiplier is then scaled by `average^adapt_rate`. An average
    /// near 1 means the filter is calibrated; above 1 it is overconfident. The
    /// average is taken before the log-domain step because the log of a single
    /// `z²` is biased low (E[ln χ²₁] ≈ -1.27).
    pub adapt_rate: Real,
    /// Treat a coherent batch rejection (at least 3 quotes and half the batch gated, at
    /// least 90% with the same innovation sign) as a regime shift: accept those quotes
    /// (Huber-weighted) and reset the adaptive multipliers to 1.
    pub regime_override: bool,
    /// Lower bound of the adaptive process-noise multiplier, in `(0, 1]`.
    pub adapt_min_scale: Real,
    /// Upper bound of the adaptive process-noise multiplier, at least 1.
    pub adapt_max_scale: Real,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            process_iv_rate: 0.001,
            measurement_sd: 0.005,
            spread_floor: 0.001,
            reference_spread: 0.010,
            correlation_length: 1.0,
            independent_fraction: 0.10,
            gate_sigma: DEFAULT_GATE_SIGMA,
            robust_sigma: DEFAULT_ROBUST_SIGMA,
            adapt_rate: 0.0,
            regime_override: false,
            adapt_min_scale: 0.01,
            adapt_max_scale: 16.0,
        }
    }
}

impl FilterConfig {
    pub fn validate(&self) -> QlResult<()> {
        require!(
            self.process_iv_rate.is_finite() && self.process_iv_rate >= 0.0,
            "process_iv_rate must be finite and nonnegative"
        );
        require!(
            self.measurement_sd.is_finite() && self.measurement_sd > 0.0,
            "measurement_sd must be finite and positive"
        );
        require!(
            self.spread_floor.is_finite() && self.spread_floor > 0.0,
            "spread_floor must be finite and positive"
        );
        require!(
            self.reference_spread.is_finite() && self.reference_spread > 0.0,
            "reference_spread must be finite and positive"
        );
        require!(
            self.correlation_length.is_finite() && self.correlation_length > 0.0,
            "correlation_length must be finite and positive"
        );
        require!(
            self.independent_fraction.is_finite()
                && self.independent_fraction > 0.0
                && self.independent_fraction <= 1.0,
            "independent_fraction must be in (0, 1]"
        );
        require!(
            self.robust_sigma.is_finite() && self.robust_sigma > 0.0,
            "robust_sigma must be finite and positive"
        );
        require!(
            self.gate_sigma.is_finite() && self.gate_sigma >= self.robust_sigma,
            "gate_sigma must be finite and at least robust_sigma"
        );
        require!(
            self.adapt_rate.is_finite() && (0.0..=1.0).contains(&self.adapt_rate),
            "adapt_rate must be in [0, 1]"
        );
        require!(
            self.adapt_min_scale.is_finite()
                && self.adapt_min_scale > 0.0
                && self.adapt_min_scale <= 1.0
                && self.adapt_max_scale.is_finite()
                && self.adapt_max_scale >= 1.0,
            "adaptive scale bounds must satisfy 0 < min <= 1 <= max"
        );
        Ok(())
    }

    /// Huber weight in `(0, 1]` for a standardized innovation `z`.
    #[inline]
    pub fn robust_weight(&self, z: Real) -> Real {
        let a = z.abs();
        if a <= self.robust_sigma {
            1.0
        } else {
            self.robust_sigma / a
        }
    }

    pub fn measurement_variance(&self, spread: Real) -> QlResult<Real> {
        require!(
            spread.is_finite() && spread >= 0.0,
            "IV spread must be finite and nonnegative"
        );
        let variance = self.measurement_sd * self.measurement_sd * spread.max(self.spread_floor)
            / self.reference_spread;
        require!(
            variance.is_finite() && variance > 0.0,
            "measurement variance must be finite and positive"
        );
        Ok(variance)
    }
}

/// Normalization context mapping strikes to standardized moneyness coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmileContext {
    pub forward: Rate,
    pub exercise_time: Time,
    pub atm_vol: Volatility,
}

impl SmileContext {
    pub fn new(forward: Rate, exercise_time: Time, atm_vol: Volatility) -> QlResult<Self> {
        require!(
            forward.is_finite() && forward > 0.0,
            "forward must be finite and positive"
        );
        require!(
            exercise_time.is_finite() && exercise_time > 0.0,
            "exercise time must be finite and positive"
        );
        require!(
            atm_vol.is_finite() && atm_vol > 0.0,
            "ATM volatility must be finite and positive"
        );
        let ctx = Self {
            forward,
            exercise_time,
            atm_vol,
        };
        let scale = ctx.scale();
        require!(scale.is_finite() && scale > 0.0, "invalid ATM scale");
        let min_strike = ctx.strike(KNOTS[0]);
        let max_strike = ctx.strike(KNOTS[N_KNOTS - 1]);
        require!(
            min_strike.is_finite()
                && min_strike > 0.0
                && max_strike.is_finite()
                && max_strike > min_strike,
            "knot strikes must be finite and positive"
        );
        Ok(ctx)
    }

    #[inline]
    pub fn scale(&self) -> Real {
        self.atm_vol * self.exercise_time.sqrt()
    }

    pub fn coordinate(&self, strike: Rate) -> QlResult<Real> {
        require!(
            strike.is_finite() && strike > 0.0,
            "strike must be finite and positive"
        );
        let mut x = (strike.ln() - self.forward.ln()) / self.scale();
        let min_strike = self.strike(KNOTS[0]);
        let max_strike = self.strike(KNOTS[N_KNOTS - 1]);
        if close_enough(strike, min_strike) {
            x = KNOTS[0];
        } else if close_enough(strike, max_strike) {
            x = KNOTS[N_KNOTS - 1];
        }
        Ok(x)
    }

    #[inline]
    pub fn strike(&self, x: Real) -> Rate {
        (self.forward.ln() + x * self.scale()).exp()
    }
}

/// Validates symmetry, finiteness, and positive semi-definiteness of the filter state.
///
/// Positive semi-definiteness is verified in stack-only operations using a
/// regularized Cholesky factorization ($P + \epsilon I = L L^T$), avoiding heap
/// allocations and Jacobi sweep panics on streaming hot paths.
pub fn validate_state(
    mean: &[Real; N_KNOTS],
    covariance: &[[Real; N_KNOTS]; N_KNOTS],
) -> QlResult<()> {
    require!(
        mean.iter().all(|v| v.is_finite()),
        "non-finite filter state"
    );
    require!(
        covariance.iter().flatten().all(|v| v.is_finite()),
        "non-finite filter state"
    );
    let max_abs = covariance
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0_f64, Real::max);
    let tolerance = 1e-12 * max_abs.max(1.0);
    for i in 0..N_KNOTS {
        for j in 0..N_KNOTS {
            require!(
                (covariance[i][j] - covariance[j][i]).abs() <= tolerance,
                "asymmetric covariance"
            );
        }
    }
    let mut l = [[0.0; N_KNOTS]; N_KNOTS];
    for i in 0..N_KNOTS {
        for j in i..N_KNOTS {
            let mut sum = covariance[i][j];
            if i == j {
                sum += tolerance;
            }
            for k in 0..i {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum < 0.0 || !sum.is_finite() {
                    fail!("covariance is not positive semidefinite");
                }
                l[i][i] = sum.sqrt();
            } else if l[i][i] > 0.0 {
                l[j][i] = sum / l[i][i];
            } else {
                l[j][i] = 0.0;
            }
        }
    }
    Ok(())
}

/// Generates the process covariance matrix Q(dt) for elapsed time dt.
pub fn process_covariance(config: &FilterConfig, dt: Real) -> QlResult<[[Real; N_KNOTS]; N_KNOTS]> {
    require!(
        dt.is_finite() && dt >= 0.0,
        "elapsed time must be finite and nonnegative"
    );
    config.validate()?;
    let mut q = [[0.0; N_KNOTS]; N_KNOTS];
    let q_scale = config.process_iv_rate * config.process_iv_rate * dt;
    let corr_len = config.correlation_length;
    let indep = config.independent_fraction;
    for i in 0..N_KNOTS {
        for j in 0..N_KNOTS {
            let dist = (KNOTS[i] - KNOTS[j]).abs();
            let corr = (-dist / corr_len).exp();
            let factor = (1.0 - indep) * corr + if i == j { indep } else { 0.0 };
            let val = q_scale * factor;
            require!(val.is_finite(), "non-finite process covariance");
            q[i][j] = val;
        }
    }
    Ok(q)
}

/// Result of a Kalman measurement update.
#[derive(Clone, Debug, PartialEq)]
pub struct UpdateResult<T> {
    pub mean: [Real; N_KNOTS],
    pub covariance: [[Real; N_KNOTS]; N_KNOTS],
    pub innovation: T,
    pub normalized_innovation: T,
}

/// Innovation `y - h·m`, covariance-projected gain numerator `P·h`, and
/// innovation variance `S = hᵀPh + R` for one scalar measurement.
///
/// Callers must have validated `mean`/`covariance`; this only checks `S`.
fn innovation_stats(
    mean: &[Real; N_KNOTS],
    covariance: &[[Real; N_KNOTS]; N_KNOTS],
    h: &[Real; N_KNOTS],
    y: Real,
    variance: Real,
) -> QlResult<(Real, [Real; N_KNOTS], Real)> {
    let mut y_pred = 0.0;
    for j in 0..N_KNOTS {
        y_pred += h[j] * mean[j];
    }
    let innovation = y - y_pred;

    let mut ph = [0.0; N_KNOTS];
    for i in 0..N_KNOTS {
        for j in 0..N_KNOTS {
            ph[i] += covariance[i][j] * h[j];
        }
    }

    let mut s = variance;
    for i in 0..N_KNOTS {
        s += h[i] * ph[i];
    }
    require!(s > 0.0 && s.is_finite(), "degenerate innovation variance");
    Ok((innovation, ph, s))
}

/// Fast allocation-free scalar Kalman update (m = 1) using Joseph stabilized covariance.
pub fn kalman_update_scalar(
    mean: &[Real; N_KNOTS],
    covariance: &[[Real; N_KNOTS]; N_KNOTS],
    h: &[Real; N_KNOTS],
    y: Real,
    variance: Real,
) -> QlResult<UpdateResult<Real>> {
    validate_state(mean, covariance)?;
    require!(
        h.iter().all(|v| v.is_finite()) && y.is_finite() && variance.is_finite() && variance > 0.0,
        "invalid measurement values or variances"
    );

    let (innovation, ph, s) = innovation_stats(mean, covariance, h, y, variance)?;

    let mut gain = [0.0; N_KNOTS];
    for i in 0..N_KNOTS {
        gain[i] = ph[i] / s;
    }

    let mut updated_mean = [0.0; N_KNOTS];
    for i in 0..N_KNOTS {
        updated_mean[i] = mean[i] + gain[i] * innovation;
    }

    let mut a = [[0.0; N_KNOTS]; N_KNOTS];
    for i in 0..N_KNOTS {
        for j in 0..N_KNOTS {
            let delta = if i == j { 1.0 } else { 0.0 };
            a[i][j] = delta - gain[i] * h[j];
        }
    }

    let mut a_cov = [[0.0; N_KNOTS]; N_KNOTS];
    for i in 0..N_KNOTS {
        for j in 0..N_KNOTS {
            let mut sum = 0.0;
            for k in 0..N_KNOTS {
                sum += a[i][k] * covariance[k][j];
            }
            a_cov[i][j] = sum;
        }
    }

    let mut updated_cov = [[0.0; N_KNOTS]; N_KNOTS];
    for i in 0..N_KNOTS {
        for j in 0..N_KNOTS {
            let mut sum = 0.0;
            for k in 0..N_KNOTS {
                sum += a_cov[i][k] * a[j][k];
            }
            sum += variance * gain[i] * gain[j];
            updated_cov[i][j] = sum;
        }
    }

    for i in 0..N_KNOTS {
        for j in i + 1..N_KNOTS {
            let sym = (updated_cov[i][j] + updated_cov[j][i]) * 0.5;
            updated_cov[i][j] = sym;
            updated_cov[j][i] = sym;
        }
    }

    validate_state(&updated_mean, &updated_cov)?;
    let normalized = innovation / s.sqrt();
    Ok(UpdateResult {
        mean: updated_mean,
        covariance: updated_cov,
        innovation,
        normalized_innovation: normalized,
    })
}

/// Batch Kalman measurement update (m >= 1) for independent measurement noise.
///
/// With a diagonal noise covariance, assimilating the quotes one at a time
/// (each with a Joseph-form scalar update) is mathematically identical to the
/// joint update, independent of order, and needs no `m x m` factorization.
/// Reported `innovation` and `normalized_innovation` are against the *prior*
/// state, matching the joint-update convention.
pub fn kalman_update_batch(
    mean: &[Real; N_KNOTS],
    covariance: &[[Real; N_KNOTS]; N_KNOTS],
    h: &[[Real; N_KNOTS]],
    y: &[Real],
    variances: &[Real],
) -> QlResult<UpdateResult<Vec<Real>>> {
    validate_state(mean, covariance)?;
    let m = y.len();
    require!(
        m > 0 && h.len() == m && variances.len() == m,
        "invalid measurement dimensions"
    );
    require!(
        y.iter().all(|v| v.is_finite())
            && variances.iter().all(|v| v.is_finite() && *v > 0.0)
            && h.iter().all(|row| row.iter().all(|v| v.is_finite())),
        "invalid measurement values or variances"
    );

    let mut innovations = Vec::with_capacity(m);
    let mut normalized = Vec::with_capacity(m);
    for i in 0..m {
        let (innovation, _, s) = innovation_stats(mean, covariance, &h[i], y[i], variances[i])?;
        innovations.push(innovation);
        normalized.push(innovation / s.sqrt());
    }

    let mut cur_mean = *mean;
    let mut cur_cov = *covariance;
    for i in 0..m {
        let res = kalman_update_scalar(&cur_mean, &cur_cov, &h[i], y[i], variances[i])?;
        cur_mean = res.mean;
        cur_cov = res.covariance;
    }

    Ok(UpdateResult {
        mean: cur_mean,
        covariance: cur_cov,
        innovation: innovations,
        normalized_innovation: normalized,
    })
}

/// Transports the strike-space curve across moving forward/ATM/expiry coordinates.
pub fn transport(
    mean: &[Real; N_KNOTS],
    covariance: &[[Real; N_KNOTS]; N_KNOTS],
    old_ctx: &SmileContext,
    new_ctx: &SmileContext,
) -> QlResult<TransportResult> {
    validate_state(mean, covariance)?;
    let old_ctx = SmileContext::new(old_ctx.forward, old_ctx.exercise_time, old_ctx.atm_vol)?;
    let new_ctx = SmileContext::new(new_ctx.forward, new_ctx.exercise_time, new_ctx.atm_vol)?;
    if old_ctx == new_ctx {
        let mut id = [[0.0; N_KNOTS]; N_KNOTS];
        for i in 0..N_KNOTS {
            id[i][i] = 1.0;
        }
        return Ok((*mean, *covariance, id));
    }

    let log_ratio = new_ctx.forward.ln() - old_ctx.forward.ln();
    let old_scale = old_ctx.scale();
    let new_scale = new_ctx.scale();

    let mut u = [0.0; N_KNOTS];
    let mut distances = [0.0; N_KNOTS];
    let mut max_dist = 0.0_f64;

    for j in 0..N_KNOTS {
        let u_val = (log_ratio + KNOTS[j] * new_scale) / old_scale;
        require!(u_val.is_finite(), "non-finite coordinate transport");
        u[j] = u_val;
        let dist = (-3.0 - u_val).max(u_val - 3.0).max(0.0);
        distances[j] = dist;
        if dist > max_dist {
            max_dist = dist;
        }
    }

    require!(
        max_dist <= MAX_OVERHANG + 1e-12,
        "coordinate overhang exceeds 0.25 standardized units"
    );

    let mut j_mat = [[0.0; N_KNOTS]; N_KNOTS];
    for i in 0..N_KNOTS {
        let u_clipped = u[i].clamp(-3.0, 3.0);
        j_mat[i] = basis_vector(u_clipped)?;
    }

    let mut values = [0.0; N_KNOTS];
    for i in 0..N_KNOTS {
        for k in 0..N_KNOTS {
            values[i] += j_mat[i][k] * mean[k];
        }
    }

    for i in 0..N_KNOTS {
        if distances[i] > 0.0 {
            let endpoint = if u[i] < -3.0 { -3.0 } else { 3.0 };
            let delta = u[i] - endpoint;
            let derivative = basis_derivative(endpoint)?;
            let mut slope = 0.0;
            for k in 0..N_KNOTS {
                slope += derivative[k] * mean[k];
            }
            let clamped_slope = slope.clamp(-MAX_END_SLOPE, MAX_END_SLOPE);
            values[i] += clamped_slope * delta;
            if slope.abs() <= MAX_END_SLOPE + 1e-14 {
                for k in 0..N_KNOTS {
                    j_mat[i][k] += delta * derivative[k];
                }
            }
        }
    }

    let mut j_cov = [[0.0; N_KNOTS]; N_KNOTS];
    for i in 0..N_KNOTS {
        for j in 0..N_KNOTS {
            let mut sum = 0.0;
            for k in 0..N_KNOTS {
                sum += j_mat[i][k] * covariance[k][j];
            }
            j_cov[i][j] = sum;
        }
    }

    let mut p = [[0.0; N_KNOTS]; N_KNOTS];
    for i in 0..N_KNOTS {
        for j in 0..N_KNOTS {
            let mut sum = 0.0;
            for k in 0..N_KNOTS {
                sum += j_cov[i][k] * j_mat[j][k];
            }
            if i == j {
                let ext_sd = 0.01 * distances[i] / MAX_OVERHANG;
                sum += ext_sd * ext_sd;
            }
            p[i][j] = sum;
        }
    }

    for i in 0..N_KNOTS {
        for j in i + 1..N_KNOTS {
            let sym = (p[i][j] + p[j][i]) * 0.5;
            p[i][j] = sym;
            p[j][i] = sym;
        }
    }

    validate_state(&values, &p)?;
    Ok((values, p, j_mat))
}

/// Status of an incoming quote update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UpdateStatus {
    /// Assimilated; `weight` is the Huber weight (1.0 = full weight).
    Accepted {
        innovation: Real,
        z_score: Real,
        weight: Real,
    },
    /// Rejected: |z| exceeded the hard gate.
    GatedOutlier { innovation: Real, z_score: Real },
    /// Strike lies outside the knot domain.
    OutOfDomain,
    /// Non-finite or non-positive strike or IV, or negative/non-finite spread.
    InvalidQuote,
}

/// A quote mapped to its basis row and noise variance, or a terminal status.
enum Prepared {
    Ready { h: [Real; N_KNOTS], variance: Real },
    Skip(UpdateStatus),
}

/// A Kalman-filtered smile section maintaining recursive Bayesian belief over fixed knots.
pub struct KalmanSmileSection {
    base: SmileSectionBase,
    mean: [Volatility; N_KNOTS],
    covariance: [[Real; N_KNOTS]; N_KNOTS],
    context: SmileContext,
    config: FilterConfig,
    process_scale: Real,
    nis_average: Real,
    min_strike: Rate,
    max_strike: Rate,
}

impl KalmanSmileSection {
    /// Cold-initializes the filter from market observations via regularized static fit.
    pub fn bootstrap(
        strikes: &[Rate],
        mid_ivs: &[Volatility],
        context: SmileContext,
        config: FilterConfig,
    ) -> QlResult<Self> {
        config.validate()?;
        let ref_smile = CubicSmileSection::new(
            strikes.to_vec(),
            mid_ivs.to_vec(),
            context.forward,
            context.exercise_time,
            context.atm_vol,
        )?;
        let knot_ivs = ref_smile.node_mid_ivs();
        require!(
            knot_ivs.len() == N_KNOTS,
            "bootstrap fit did not return 9 knots"
        );
        let mut mean = [0.0; N_KNOTS];
        mean.copy_from_slice(&knot_ivs[..N_KNOTS]);

        let mut covariance = [[0.0; N_KNOTS]; N_KNOTS];
        let init_var = 0.05 * 0.05;
        for i in 0..N_KNOTS {
            covariance[i][i] = init_var;
        }

        Self::from_state(mean, covariance, context, config)
    }

    /// Creates a filter from an existing mean and covariance state.
    pub fn from_state(
        mean: [Volatility; N_KNOTS],
        covariance: [[Real; N_KNOTS]; N_KNOTS],
        context: SmileContext,
        config: FilterConfig,
    ) -> QlResult<Self> {
        config.validate()?;
        validate_state(&mean, &covariance)?;
        let context = SmileContext::new(context.forward, context.exercise_time, context.atm_vol)?;

        let min_strike = context.strike(KNOTS[0]);
        let max_strike = context.strike(KNOTS[N_KNOTS - 1]);
        let base = SmileSectionBase::with_exercise_time(
            context.exercise_time,
            Actual365Fixed::new(),
            VolatilityType::ShiftedLognormal,
            0.0,
        )?;

        Ok(Self {
            base,
            mean,
            covariance,
            context,
            config,
            process_scale: 1.0,
            nis_average: 1.0,
            min_strike,
            max_strike,
        })
    }

    /// Current adaptive process-noise multiplier.
    pub fn process_scale(&self) -> Real {
        self.process_scale
    }

    /// Smoothed normalized innovation squared (≈ 1 when the filter is calibrated).
    pub fn nis_average(&self) -> Real {
        self.nis_average
    }

    fn adapt_process_scale(&mut self, mean_nis: Real) {
        let rate = self.config.adapt_rate;
        if rate > 0.0 && mean_nis.is_finite() {
            let nis = mean_nis.clamp(NIS_ADAPT_CLAMP.0, NIS_ADAPT_CLAMP.1);
            self.nis_average = (1.0 - rate) * self.nis_average + rate * nis;
            self.process_scale = (self.process_scale * self.nis_average.powf(rate))
                .clamp(self.config.adapt_min_scale, self.config.adapt_max_scale);
        }
    }

    /// Advances the process noise across elapsed time dt (in seconds), scaled by
    /// the adaptive multiplier (1.0 unless `adapt_rate > 0`).
    pub fn predict(&mut self, dt: Real) -> QlResult<()> {
        let q = process_covariance(&self.config, dt)?;
        let scale = self.process_scale;
        for i in 0..N_KNOTS {
            for j in 0..N_KNOTS {
                self.covariance[i][j] += scale * q[i][j];
            }
        }
        for i in 0..N_KNOTS {
            for j in i + 1..N_KNOTS {
                let sym = (self.covariance[i][j] + self.covariance[j][i]) * 0.5;
                self.covariance[i][j] = sym;
                self.covariance[j][i] = sym;
            }
        }
        validate_state(&self.mean, &self.covariance)?;
        Ok(())
    }

    /// Transports filter state across moving coordinate context.
    pub fn transport(&mut self, new_context: SmileContext) -> QlResult<()> {
        let (values, p, _) = transport(&self.mean, &self.covariance, &self.context, &new_context)?;
        self.mean = values;
        self.covariance = p;
        self.context = new_context;
        self.min_strike = new_context.strike(KNOTS[0]);
        self.max_strike = new_context.strike(KNOTS[N_KNOTS - 1]);
        self.base = SmileSectionBase::with_exercise_time(
            new_context.exercise_time,
            Actual365Fixed::new(),
            VolatilityType::ShiftedLognormal,
            0.0,
        )?;
        Ok(())
    }

    /// Validates a raw quote and maps it to its basis row and noise variance.
    fn prepare_quote(&self, strike: Rate, mid_iv: Volatility, spread: Real) -> QlResult<Prepared> {
        let valid = strike.is_finite()
            && strike > 0.0
            && mid_iv.is_finite()
            && mid_iv > 0.0
            && spread.is_finite()
            && spread >= 0.0;
        if !valid {
            return Ok(Prepared::Skip(UpdateStatus::InvalidQuote));
        }
        let x = snap_to_knot_domain(self.context.coordinate(strike)?);
        if !(-3.0..=3.0).contains(&x) {
            return Ok(Prepared::Skip(UpdateStatus::OutOfDomain));
        }
        let variance = self.config.measurement_variance(spread)?;
        let h = basis_vector(x)?;
        Ok(Prepared::Ready { h, variance })
    }

    /// Assimilates an incoming quote.
    ///
    /// Invalid inputs (non-finite or non-positive strike or IV, negative spread)
    /// return [`UpdateStatus::InvalidQuote`] and out-of-domain strikes return
    /// [`UpdateStatus::OutOfDomain`]; neither touches the state. Otherwise the
    /// prior innovation is gated at `gate_sigma` and Huber down-weighted beyond
    /// `robust_sigma`.
    pub fn update_scalar(
        &mut self,
        strike: Rate,
        mid_iv: Volatility,
        spread: Real,
    ) -> QlResult<UpdateStatus> {
        let (h, variance) = match self.prepare_quote(strike, mid_iv, spread)? {
            Prepared::Skip(status) => return Ok(status),
            Prepared::Ready { h, variance } => (h, variance),
        };

        let (innovation, _, s) =
            innovation_stats(&self.mean, &self.covariance, &h, mid_iv, variance)?;
        let z_score = innovation / s.sqrt();
        if z_score.abs() > self.config.gate_sigma {
            return Ok(UpdateStatus::GatedOutlier {
                innovation,
                z_score,
            });
        }

        let weight = self.config.robust_weight(z_score);
        let res =
            kalman_update_scalar(&self.mean, &self.covariance, &h, mid_iv, variance / weight)?;
        self.mean = res.mean;
        self.covariance = res.covariance;
        self.adapt_process_scale(z_score * z_score);
        Ok(UpdateStatus::Accepted {
            innovation,
            z_score,
            weight,
        })
    }

    /// Assimilates a coherent batch of incoming quotes.
    ///
    /// Each quote is first hard-gated against the *prior* (order independent, so a
    /// bad quote cannot poison the gate for the others). Survivors are then
    /// assimilated sequentially, tightest spread first, with Huber down-weighting
    /// against the running posterior; this equals the joint update for independent
    /// noise. Invalid quotes get [`UpdateStatus::InvalidQuote`] without aborting the
    /// batch. Reported `innovation`/`z_score` are those used by the update. The
    /// state changes only if the whole batch succeeds.
    pub fn update_batch(
        &mut self,
        strikes: &[Rate],
        mid_ivs: &[Volatility],
        spreads: &[Real],
    ) -> QlResult<Vec<UpdateStatus>> {
        let n = strikes.len();
        require!(
            n == mid_ivs.len() && n == spreads.len(),
            "strikes, mid_ivs, and spreads must have equal length"
        );
        let mut statuses = Vec::with_capacity(n);
        let mut candidates: Vec<(usize, [Real; N_KNOTS], Real)> = Vec::new();
        let mut prior_nis: Vec<Real> = Vec::new();
        let mut gated_list: Vec<(usize, [Real; N_KNOTS], Real, Real)> = Vec::new();

        for i in 0..n {
            let (h, variance) = match self.prepare_quote(strikes[i], mid_ivs[i], spreads[i])? {
                Prepared::Skip(status) => {
                    statuses.push(status);
                    continue;
                }
                Prepared::Ready { h, variance } => (h, variance),
            };
            let (innovation, _, s) =
                innovation_stats(&self.mean, &self.covariance, &h, mid_ivs[i], variance)?;
            let z_score = innovation / s.sqrt();
            if z_score.abs() > self.config.gate_sigma {
                gated_list.push((i, h, variance, innovation));
                statuses.push(UpdateStatus::GatedOutlier {
                    innovation,
                    z_score,
                });
            } else {
                // Placeholder, overwritten with the sequential statistics below.
                statuses.push(UpdateStatus::Accepted {
                    innovation,
                    z_score,
                    weight: 1.0,
                });
                candidates.push((i, h, variance));
                prior_nis.push(z_score * z_score);
            }
        }

        let mut regime_shift = false;
        if self.config.regime_override {
            let g = gated_list.len();
            let positive = gated_list.iter().filter(|q| q.3 > 0.0).count();
            if g >= 3
                && 2 * g >= g + candidates.len()
                && (10 * positive >= 9 * g || 10 * (g - positive) >= 9 * g)
            {
                regime_shift = true;
                for (i, h, variance, innovation) in gated_list {
                    statuses[i] = UpdateStatus::Accepted {
                        innovation,
                        z_score: 0.0,
                        weight: 1.0,
                    };
                    candidates.push((i, h, variance));
                }
                self.process_scale = 1.0;
                self.nis_average = 1.0;
            }
        }
        candidates.sort_by(|a, b| a.2.total_cmp(&b.2));
        let mut mean = self.mean;
        let mut covariance = self.covariance;
        for (idx, h, variance) in candidates {
            let (innovation, _, s) =
                innovation_stats(&mean, &covariance, &h, mid_ivs[idx], variance)?;
            let z_score = innovation / s.sqrt();
            let weight = self.config.robust_weight(z_score);
            let res =
                kalman_update_scalar(&mean, &covariance, &h, mid_ivs[idx], variance / weight)?;
            mean = res.mean;
            covariance = res.covariance;
            statuses[idx] = UpdateStatus::Accepted {
                innovation,
                z_score,
                weight,
            };
        }
        self.mean = mean;
        self.covariance = covariance;
        if !regime_shift && !prior_nis.is_empty() {
            self.adapt_process_scale(prior_nis.iter().sum::<Real>() / prior_nis.len() as Real);
        }

        Ok(statuses)
    }

    /// Evaluates the volatility at standardized coordinate point.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `point` is non-finite or outside the smile domain `[-3.0, 3.0]`.
    pub fn volatility_at_std_dev(&self, point: Real) -> QlResult<Volatility> {
        require!(point.is_finite(), "standard-deviation point must be finite");
        let point = snap_to_knot_domain(point);
        require!(
            (-3.0..=3.0).contains(&point),
            "standard-deviation point ({point}) is outside the smile domain [-3.0, 3.0]"
        );
        let h = basis_vector(point)?;
        let mut vol = 0.0;
        for j in 0..N_KNOTS {
            vol += h[j] * self.mean[j];
        }
        Ok(vol.max(0.0))
    }

    /// Current filtered knot IV ordinates.
    pub fn mean(&self) -> &[Volatility; N_KNOTS] {
        &self.mean
    }

    /// Current filter state covariance.
    pub fn covariance(&self) -> &[[Real; N_KNOTS]; N_KNOTS] {
        &self.covariance
    }

    /// Current smile normalization context.
    pub fn context(&self) -> &SmileContext {
        &self.context
    }

    /// Current filter configuration.
    pub fn config(&self) -> &FilterConfig {
        &self.config
    }

    /// Converts the current filtered knot IVs into a static CubicSmileSection.
    pub fn to_cubic_smile_section(&self) -> QlResult<CubicSmileSection> {
        let mut strikes = Vec::with_capacity(N_KNOTS);
        let mut vols = Vec::with_capacity(N_KNOTS);
        for i in 0..N_KNOTS {
            strikes.push(self.context.strike(KNOTS[i]));
            vols.push(self.mean[i].max(0.0));
        }
        CubicSmileSection::with_smoothing(
            strikes,
            vols,
            self.context.forward,
            self.context.exercise_time,
            self.context.atm_vol,
            KNOTS.to_vec(),
            0.0,
        )
    }
}

impl SmileSection for KalmanSmileSection {
    fn base(&self) -> &SmileSectionBase {
        &self.base
    }

    fn volatility_impl(&self, strike: Rate) -> QlResult<Volatility> {
        require!(
            strike.is_finite() && strike > 0.0,
            "strike must be finite and positive"
        );
        let point = self.context.coordinate(strike)?;
        self.volatility_at_std_dev(point)
    }

    fn min_strike(&self) -> Rate {
        self.min_strike
    }

    fn max_strike(&self) -> Rate {
        self.max_strike
    }

    fn atm_level(&self) -> Option<Rate> {
        Some(self.context.forward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORWARD: Rate = 100_000.0;
    const EXPIRY: Time = 30.0 / 365.0;
    const ATM_VOL: Volatility = 0.60;

    fn test_context() -> SmileContext {
        SmileContext::new(FORWARD, EXPIRY, ATM_VOL).unwrap()
    }

    #[test]
    fn basis_functions_partition_of_unity_and_interpolation() {
        for (k, &knot) in KNOTS.iter().enumerate() {
            let h = basis_vector(knot).unwrap();
            for (j, &val) in h.iter().enumerate() {
                if j == k {
                    assert!((val - 1.0).abs() < 1e-12, "knot {k} indicator not 1");
                } else {
                    assert!(val.abs() < 1e-12, "knot {k} off-target not 0");
                }
            }
        }

        for x in [-2.5, -1.2, -0.1, 0.0, 0.4, 1.2, 2.8] {
            let h = basis_vector(x).unwrap();
            let sum: Real = h.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "partition of unity failed at {x}"
            );
        }
    }

    #[test]
    fn process_covariance_symmetry_and_scaling() {
        let config = FilterConfig::default();
        let q0 = process_covariance(&config, 0.0).unwrap();
        for row in q0 {
            for val in row {
                assert_eq!(val, 0.0);
            }
        }

        let dt = 2.5;
        let q = process_covariance(&config, dt).unwrap();
        for i in 0..N_KNOTS {
            for j in 0..N_KNOTS {
                assert_eq!(q[i][j], q[j][i], "process covariance asymmetric");
                assert!(q[i][j] > 0.0, "positive covariance expected");
            }
        }
    }

    #[test]
    fn batch_of_one_matches_scalar_update() {
        let mean = [0.60; N_KNOTS];
        let mut cov = [[0.0; N_KNOTS]; N_KNOTS];
        for i in 0..N_KNOTS {
            cov[i][i] = 0.05 * 0.05;
        }

        let x = 0.3;
        let h = basis_vector(x).unwrap();
        let y = 0.65;
        let variance = 0.005 * 0.005;

        let res_scalar = kalman_update_scalar(&mean, &cov, &h, y, variance).unwrap();
        let res_batch = kalman_update_batch(&mean, &cov, &[h], &[y], &[variance]).unwrap();

        for i in 0..N_KNOTS {
            assert!(
                (res_scalar.mean[i] - res_batch.mean[i]).abs() < 1e-14,
                "mean mismatch at {i}"
            );
            for j in 0..N_KNOTS {
                assert!(
                    (res_scalar.covariance[i][j] - res_batch.covariance[i][j]).abs() < 1e-14,
                    "cov mismatch at ({i}, {j})"
                );
            }
        }
        assert!((res_scalar.innovation - res_batch.innovation[0]).abs() < 1e-14);
        assert!(
            (res_scalar.normalized_innovation - res_batch.normalized_innovation[0]).abs() < 1e-14
        );
    }

    fn diag_cov(sd: Real) -> KnotMatrix {
        let mut cov = [[0.0; N_KNOTS]; N_KNOTS];
        for i in 0..N_KNOTS {
            cov[i][i] = sd * sd;
        }
        cov
    }

    #[test]
    fn batch_update_matches_joint_python_oracle() {
        // Golden values from the Python prototype's joint Cholesky update
        // (example/btc-option-iv/btc_option_iv/kalman.py::kalman_update).
        let mean = [0.60; N_KNOTS];
        let cov = diag_cov(0.05);
        let xs = [-0.7, 0.25, 1.1];
        let h: Vec<_> = xs.iter().map(|&x| basis_vector(x).unwrap()).collect();
        let y = [0.63, 0.66, 0.58];
        let var = [
            0.005_f64.powi(2),
            0.005_f64.powi(2) * 1.5,
            0.005_f64.powi(2) * 0.8,
        ];

        let res = kalman_update_batch(&mean, &cov, &h, &y, &var).unwrap();
        let expected_mean = [
            0.600093752645,
            0.597374925944,
            0.618401639155,
            0.632464246002,
            0.649689236061,
            0.645289125591,
            0.584913685369,
            0.601986350721,
            0.599928790693,
        ];
        for i in 0..N_KNOTS {
            assert!(
                (res.mean[i] - expected_mean[i]).abs() < 1e-10,
                "batch mean mismatch at {i}: {} vs {}",
                res.mean[i],
                expected_mean[i]
            );
        }
        assert!((res.covariance[0][0] - 0.002499993373157105).abs() < 1e-12);
        assert!((res.covariance[4][4] - 0.001013157788842248).abs() < 1e-12);
        assert!((res.covariance[2][6] - -3.132019674145694e-05).abs() < 1e-12);
        let expected_z = [0.684398373634, 1.282198645125, -0.395454482694];
        for i in 0..3 {
            assert!((res.normalized_innovation[i] - expected_z[i]).abs() < 1e-9);
        }
        assert!((res.innovation[1] - 0.06).abs() < 1e-12);
    }

    #[test]
    fn batch_update_is_order_independent() {
        let mean = [0.60; N_KNOTS];
        let cov = diag_cov(0.05);
        let xs = [-0.7, 0.25, 1.1, 0.3];
        let h: Vec<_> = xs.iter().map(|&x| basis_vector(x).unwrap()).collect();
        let y = [0.63, 0.66, 0.58, 0.61];
        let var = [2.5e-5, 3.75e-5, 2.0e-5, 3.0e-5];

        let fwd = kalman_update_batch(&mean, &cov, &h, &y, &var).unwrap();
        let order = [3, 1, 0, 2];
        let h2: Vec<_> = order.iter().map(|&i| h[i]).collect();
        let y2: Vec<_> = order.iter().map(|&i| y[i]).collect();
        let v2: Vec<_> = order.iter().map(|&i| var[i]).collect();
        let perm = kalman_update_batch(&mean, &cov, &h2, &y2, &v2).unwrap();
        for i in 0..N_KNOTS {
            assert!((fwd.mean[i] - perm.mean[i]).abs() < 1e-12);
            for j in 0..N_KNOTS {
                assert!((fwd.covariance[i][j] - perm.covariance[i][j]).abs() < 1e-12);
            }
        }
    }

    fn flat_filter(config: FilterConfig) -> KalmanSmileSection {
        KalmanSmileSection::from_state([0.60; N_KNOTS], diag_cov(0.05), test_context(), config)
            .unwrap()
    }

    #[test]
    fn section_batch_equals_sequential_scalar_updates() {
        let ctx = test_context();
        let strikes = [ctx.strike(-0.7), ctx.strike(0.25), ctx.strike(1.1)];
        let ivs = [0.63, 0.66, 0.58];
        let spreads = [0.01, 0.01, 0.01];

        let mut batch = flat_filter(FilterConfig::default());
        let statuses = batch.update_batch(&strikes, &ivs, &spreads).unwrap();
        assert!(
            statuses
                .iter()
                .all(|s| matches!(s, UpdateStatus::Accepted { .. }))
        );

        let mut seq = flat_filter(FilterConfig::default());
        for i in 0..3 {
            seq.update_scalar(strikes[i], ivs[i], spreads[i]).unwrap();
        }
        for i in 0..N_KNOTS {
            assert!((batch.mean()[i] - seq.mean()[i]).abs() < 1e-12);
            for j in 0..N_KNOTS {
                assert!((batch.covariance()[i][j] - seq.covariance()[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn invalid_quotes_are_reported_not_gated_or_fatal() {
        let ctx = test_context();
        let k = ctx.strike(0.2);
        let mut f = flat_filter(FilterConfig::default());
        let before = (*f.mean(), *f.covariance());

        for (strike, iv, spread) in [
            (k, f64::INFINITY, 0.01),
            (k, f64::NEG_INFINITY, 0.01),
            (k, f64::NAN, 0.01),
            (k, 0.0, 0.01),
            (k, -0.3, 0.01),
            (k, 0.6, f64::NAN),
            (k, 0.6, -0.01),
            (f64::NAN, 0.6, 0.01),
            (-1.0, 0.6, 0.01),
        ] {
            assert_eq!(
                f.update_scalar(strike, iv, spread).unwrap(),
                UpdateStatus::InvalidQuote,
                "({strike}, {iv}, {spread})"
            );
        }
        assert_eq!((*f.mean(), *f.covariance()), before);

        // One bad quote must not discard the valid ones in a batch.
        let statuses = f
            .update_batch(&[k, k, k], &[f64::NAN, 0.62, f64::INFINITY], &[0.01; 3])
            .unwrap();
        assert_eq!(statuses[0], UpdateStatus::InvalidQuote);
        assert!(matches!(statuses[1], UpdateStatus::Accepted { .. }));
        assert_eq!(statuses[2], UpdateStatus::InvalidQuote);
        assert_ne!(*f.mean(), before.0);
    }

    #[test]
    fn huber_downweights_between_robust_and_gate_sigma() {
        let ctx = test_context();
        let mut f = flat_filter(FilterConfig::default());
        // At the ATM knot h = e_4, so S = 0.05^2 + R and z = innovation / sqrt(S).
        let r = 0.005_f64.powi(2); // spread == reference_spread -> sd^2
        let s = 0.05_f64.powi(2) + r;
        let target_z = 4.0;
        let iv = 0.60 + target_z * s.sqrt();

        let status = f.update_scalar(ctx.strike(0.0), iv, 0.010).unwrap();
        let weight = match status {
            UpdateStatus::Accepted {
                z_score, weight, ..
            } => {
                assert!((z_score - target_z).abs() < 1e-9);
                weight
            }
            other => panic!("expected accepted, got {other:?}"),
        };
        assert!((weight - 3.0 / 4.0).abs() < 1e-9);
        let gain = 0.05_f64.powi(2) / (0.05_f64.powi(2) + r / weight);
        assert!((f.mean()[4] - (0.60 + gain * (iv - 0.60))).abs() < 1e-9);

        // Beyond the hard gate nothing moves.
        let mut g = flat_filter(FilterConfig::default());
        let far = 0.60 + 6.0 * s.sqrt();
        let before = *g.mean();
        assert!(matches!(
            g.update_scalar(ctx.strike(0.0), far, 0.010).unwrap(),
            UpdateStatus::GatedOutlier { .. }
        ));
        assert_eq!(*g.mean(), before);
    }

    /// Deterministic standard-normal stream (LCG + Box-Muller).
    struct Gauss(u64);
    impl Gauss {
        fn uniform(&mut self) -> Real {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as Real + 0.5) / (1u64 << 53) as Real
        }
        fn next(&mut self) -> Real {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    /// Runs a level random-walk truth observed at the ATM knot; returns the final
    /// process multiplier.
    fn simulate_scale(true_rate: Real, adapt_rate: Real) -> Real {
        let ctx = test_context();
        let config = FilterConfig {
            adapt_rate,
            ..FilterConfig::default()
        };
        let mut f = flat_filter(config);
        let mut rng = Gauss(12345);
        let mut level = 0.60;
        let strike = ctx.strike(0.0);
        for _ in 0..1500 {
            level += true_rate * rng.next();
            let iv = level + config.measurement_sd * rng.next();
            f.predict(1.0).unwrap();
            f.update_scalar(strike, iv, config.reference_spread)
                .unwrap();
        }
        f.process_scale()
    }

    #[test]
    fn adaptive_process_noise_tracks_true_volatility_of_level() {
        // Configured rate is 0.001/sqrt(s).
        assert_eq!(
            simulate_scale(0.004, 0.0),
            1.0,
            "adaptation is off by default"
        );
        let up = simulate_scale(0.004, 0.02);
        let down = simulate_scale(0.00025, 0.02);
        assert!(up > 3.0, "overconfident filter should inflate Q, got {up}");
        assert!(
            down < 0.6,
            "underconfident filter should shrink Q, got {down}"
        );
    }

    #[test]
    fn adaptation_config_is_validated() {
        let ok = FilterConfig::default();
        assert!(
            FilterConfig {
                adapt_rate: -0.1,
                ..ok
            }
            .validate()
            .is_err()
        );
        assert!(
            FilterConfig {
                adapt_rate: 1.5,
                ..ok
            }
            .validate()
            .is_err()
        );
        assert!(
            FilterConfig {
                adapt_min_scale: 0.0,
                ..ok
            }
            .validate()
            .is_err()
        );
        assert!(
            FilterConfig {
                adapt_min_scale: 2.0,
                ..ok
            }
            .validate()
            .is_err()
        );
        assert!(
            FilterConfig {
                adapt_max_scale: 0.5,
                ..ok
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn regime_override_accepts_coherent_rejections_only() {
        let ctx = test_context();
        let xs = [-1.0, -0.4, 0.0, 0.4, 1.0, 1.4];
        let strikes: Vec<_> = xs.iter().map(|&x| ctx.strike(x)).collect();
        let spreads = [0.01; 6];
        // +0.40 on a 0.05-sd prior is far beyond the 5σ gate for every quote.
        let up = [1.00; 6];
        let mixed = [1.0, 0.2, 1.0, 0.2, 1.0, 0.2];

        let off = FilterConfig::default();
        let on = FilterConfig {
            regime_override: true,
            ..off
        };

        let mut f = flat_filter(off);
        let st = f.update_batch(&strikes, &up, &spreads).unwrap();
        assert!(
            st.iter()
                .all(|s| matches!(s, UpdateStatus::GatedOutlier { .. }))
        );

        let mut f = flat_filter(on);
        let st = f.update_batch(&strikes, &up, &spreads).unwrap();
        assert!(
            st.iter()
                .all(|s| matches!(s, UpdateStatus::Accepted { .. }))
        );
        assert!(f.mean()[4] > 0.8, "state should follow the shift");

        // Opposite-signed rejections are not a regime shift.
        let mut f = flat_filter(on);
        let st = f.update_batch(&strikes, &mixed, &spreads).unwrap();
        assert!(
            st.iter()
                .all(|s| matches!(s, UpdateStatus::GatedOutlier { .. }))
        );
        assert_eq!(*f.mean(), [0.60; N_KNOTS]);

        // Too few rejected quotes (two) are not a regime shift either.
        let mut f = flat_filter(on);
        let mut few = [0.60; 6];
        few[0] = 1.0;
        few[1] = 1.0;
        let st = f.update_batch(&strikes, &few, &spreads).unwrap();
        assert!(matches!(st[0], UpdateStatus::GatedOutlier { .. }));
        assert!(matches!(st[1], UpdateStatus::GatedOutlier { .. }));
    }

    #[test]
    fn regime_override_resets_adaptive_scale() {
        let ctx = test_context();
        let cfg = FilterConfig {
            adapt_rate: 0.5,
            regime_override: true,
            ..FilterConfig::default()
        };
        let mut f = flat_filter(cfg);
        // Calm quotes at the prior mean drive the multiplier down.
        for _ in 0..20 {
            f.predict(1.0).unwrap();
            f.update_scalar(ctx.strike(0.0), 0.60, 0.01).unwrap();
        }
        assert!(f.process_scale() < 0.5);

        let xs = [-1.0, -0.4, 0.0, 0.4, 1.0];
        let strikes: Vec<_> = xs.iter().map(|&x| ctx.strike(x)).collect();
        f.update_batch(&strikes, &[0.95; 5], &[0.01; 5]).unwrap();
        assert_eq!(f.process_scale(), 1.0);
        assert_eq!(f.nis_average(), 1.0);
    }

    #[test]
    fn robust_and_gate_sigma_are_validated() {
        let ok = FilterConfig::default();
        assert!(ok.validate().is_ok());
        assert!(
            FilterConfig {
                robust_sigma: 0.0,
                ..ok
            }
            .validate()
            .is_err()
        );
        assert!(
            FilterConfig {
                gate_sigma: 2.0,
                ..ok
            }
            .validate()
            .is_err()
        );
        assert!(
            FilterConfig {
                gate_sigma: f64::NAN,
                ..ok
            }
            .validate()
            .is_err()
        );
        assert_eq!(ok.robust_weight(1.0), 1.0);
        assert_eq!(ok.robust_weight(-6.0), 0.5);
    }

    #[test]
    fn coordinate_transport_overhang_and_tails() {
        let mean = [0.60; N_KNOTS];
        let mut cov = [[0.0; N_KNOTS]; N_KNOTS];
        for i in 0..N_KNOTS {
            cov[i][i] = 0.05 * 0.05;
        }

        let ctx1 = test_context();
        let ctx2 = SmileContext::new(FORWARD * 1.01, EXPIRY, ATM_VOL).unwrap();

        let (m_trans, cov_trans, _) = transport(&mean, &cov, &ctx1, &ctx2).unwrap();
        assert_eq!(m_trans.len(), N_KNOTS);
        validate_state(&m_trans, &cov_trans).unwrap();

        // Excess overhang (> 0.25) must fail
        let ctx_far = SmileContext::new(FORWARD * 1.5, EXPIRY, ATM_VOL).unwrap();
        assert!(transport(&mean, &cov, &ctx1, &ctx_far).is_err());
    }

    #[test]
    fn filter_bootstrap_predict_update_and_gating() {
        let ctx = test_context();
        let config = FilterConfig::default();

        let points: [Real; 7] = [-1.4, -1.0, -0.6, 0.0, 0.6, 1.0, 1.4];
        let strikes: Vec<_> = points.iter().map(|&x| ctx.strike(x)).collect();
        let vols: Vec<_> = points.iter().map(|&x| 0.55 + 0.05 * x * x).collect();

        let mut filter = KalmanSmileSection::bootstrap(&strikes, &vols, ctx, config).unwrap();
        assert_eq!(filter.mean().len(), N_KNOTS);

        // Predict 1 second
        filter.predict(1.0).unwrap();

        // Update with in-range quote
        let test_strike = ctx.strike(0.2);
        let status = filter.update_scalar(test_strike, 0.56, 0.01).unwrap();
        match status {
            UpdateStatus::Accepted { z_score, .. } => {
                assert!(z_score.abs() < 5.0);
            }
            _ => panic!("expected accepted update, got {:?}", status),
        }

        // Outlier quote (e.g. 15-sigma spike) must be gated
        let outlier_status = filter.update_scalar(test_strike, 1.50, 0.01).unwrap();
        match outlier_status {
            UpdateStatus::GatedOutlier { z_score, .. } => {
                assert!(z_score.abs() > 5.0);
            }
            _ => panic!("expected gated outlier, got {:?}", outlier_status),
        }

        // Out of domain quote (|x| > 3.0)
        let ood_strike = ctx.strike(4.0);
        let ood_status = filter.update_scalar(ood_strike, 0.60, 0.01).unwrap();
        assert_eq!(ood_status, UpdateStatus::OutOfDomain);

        // SmileSection trait query
        let query_vol = filter.volatility(test_strike).unwrap();
        assert!(query_vol > 0.0);

        // Boundary queries at min_strike and max_strike must succeed
        assert!(filter.volatility(filter.min_strike()).is_ok());
        assert!(filter.volatility(filter.max_strike()).is_ok());

        // Out of domain queries must fail consistently
        assert!(filter.volatility(ood_strike).is_err());
        assert!(filter.volatility(ctx.strike(-3.5)).is_err());
        assert!(filter.volatility_at_std_dev(3.5).is_err());
        assert!(filter.volatility_at_std_dev(-3.5).is_err());
        assert!(filter.volatility_at_std_dev(f64::NAN).is_err());

        // Conversion to static CubicSmileSection
        let cubic = filter.to_cubic_smile_section().unwrap();
        assert_eq!(cubic.node_mid_ivs().len(), N_KNOTS);
    }

    #[test]
    fn filter_update_batch_gates_outliers_and_updates_accepted() {
        let ctx = test_context();
        let config = FilterConfig::default();

        let bootstrap_points: [Real; 7] = [-1.4, -1.0, -0.6, 0.0, 0.6, 1.0, 1.4];
        let strikes: Vec<_> = bootstrap_points.iter().map(|&x| ctx.strike(x)).collect();
        let vols: Vec<_> = bootstrap_points
            .iter()
            .map(|&x| 0.55 + 0.05 * x * x)
            .collect();

        let mut filter = KalmanSmileSection::bootstrap(&strikes, &vols, ctx, config).unwrap();

        let batch_strikes = vec![
            ctx.strike(-0.5),
            ctx.strike(0.2),
            ctx.strike(0.2), // outlier
            ctx.strike(4.5), // out of domain
        ];
        let batch_vols = vec![0.56, 0.55, 1.80, 0.60];
        let batch_spreads = vec![0.01, 0.01, 0.01, 0.01];

        let statuses = filter
            .update_batch(&batch_strikes, &batch_vols, &batch_spreads)
            .unwrap();
        assert_eq!(statuses.len(), 4);
        assert!(matches!(statuses[0], UpdateStatus::Accepted { .. }));
        assert!(matches!(statuses[1], UpdateStatus::Accepted { .. }));
        assert!(matches!(statuses[2], UpdateStatus::GatedOutlier { .. }));
        assert_eq!(statuses[3], UpdateStatus::OutOfDomain);
    }

    #[test]
    fn golden_oracle_comparison_with_python_prototype() {
        let cfg = FilterConfig::default();
        let ctx1 = SmileContext::new(100000.0, 30.0 / 365.0, 0.60).unwrap();
        let ctx2 = SmileContext::new(102000.0, 29.5 / 365.0, 0.61).unwrap();

        // 1. Basis at x = 0.42
        let h = basis_vector(0.42).unwrap();
        let expected_h = [
            0.0001725968,
            -0.0048327104,
            0.0323295379,
            -0.0903436371,
            0.3130791946,
            0.9290986706,
            -0.2097358467,
            0.031351905,
            -0.0011197109,
        ];
        for i in 0..N_KNOTS {
            assert!(
                (h[i] - expected_h[i]).abs() < 1e-9,
                "basis mismatch at {i}: got {}, expected {}",
                h[i],
                expected_h[i]
            );
        }

        // 2. Process cov at dt = 1.5
        let q = process_covariance(&cfg, 1.5).unwrap();
        assert!((q[0][0] - 1.5e-6).abs() < 1e-12);
        assert!((q[0][4] - 6.7213e-8).abs() < 1e-12);

        // 3. Kalman update
        let mean = [0.60; N_KNOTS];
        let mut cov = [[0.0; N_KNOTS]; N_KNOTS];
        for i in 0..N_KNOTS {
            cov[i][i] = 0.05 * 0.05;
        }
        let y = 0.64;
        let var = 0.005 * 0.005;
        let h_025 = basis_vector(0.25).unwrap();
        let res = kalman_update_scalar(&mean, &cov, &h_025, y, var).unwrap();

        let expected_m_up = [
            0.6000136398,
            0.5996180861,
            0.6025549015,
            0.592860427,
            0.6317247863,
            0.6261489888,
            0.5918315121,
            0.6012210486,
            0.5999563911,
        ];
        for i in 0..N_KNOTS {
            assert!(
                (res.mean[i] - expected_m_up[i]).abs() < 1e-9,
                "update mean mismatch at {i}: got {}, expected {}",
                res.mean[i],
                expected_m_up[i]
            );
        }
        assert!((res.covariance[0][0] - 0.0024999997).abs() < 1e-9);
        assert!((res.innovation - 0.04).abs() < 1e-9);
        assert!((res.normalized_innovation - 0.8572493809).abs() < 1e-9);

        // 4. Transport
        let (m_tr, c_tr, _) = transport(&res.mean, &res.covariance, &ctx1, &ctx2).unwrap();
        let expected_m_tr = [
            0.59924061,
            0.6013916607,
            0.5992492955,
            0.5962454698,
            0.6370491174,
            0.6151259506,
            0.5891472892,
            0.6057359848,
            0.5969476751,
        ];
        for i in 0..N_KNOTS {
            assert!(
                (m_tr[i] - expected_m_tr[i]).abs() < 1e-8,
                "transport mean mismatch at {i}: got {}, expected {}",
                m_tr[i],
                expected_m_tr[i]
            );
        }
        assert!((c_tr[0][0] - 0.0022143486).abs() < 1e-9);
    }

    #[test]
    fn from_state_rejects_unvalidated_context() {
        let mean = [0.5; N_KNOTS];
        let mut cov = [[0.0; N_KNOTS]; N_KNOTS];
        for i in 0..N_KNOTS {
            cov[i][i] = 0.01;
        }
        let config = FilterConfig::default();

        // Invalid forward (0.0 / negative / NaN)
        let bad_ctx = SmileContext {
            forward: 0.0,
            exercise_time: 0.25,
            atm_vol: 0.5,
        };
        assert!(KalmanSmileSection::from_state(mean, cov, bad_ctx, config).is_err());

        // Invalid exercise time
        let bad_ctx2 = SmileContext {
            forward: 100.0,
            exercise_time: -0.1,
            atm_vol: 0.5,
        };
        assert!(KalmanSmileSection::from_state(mean, cov, bad_ctx2, config).is_err());
    }

    #[test]
    fn transport_rejects_unvalidated_contexts() {
        let mean = [0.5; N_KNOTS];
        let mut cov = [[0.0; N_KNOTS]; N_KNOTS];
        for i in 0..N_KNOTS {
            cov[i][i] = 0.01;
        }

        // Equal invalid contexts must not bypass validation via identity check
        let bad_ctx = SmileContext {
            forward: 0.0,
            exercise_time: 0.25,
            atm_vol: 0.5,
        };
        assert!(transport(&mean, &cov, &bad_ctx, &bad_ctx).is_err());

        // Zero expiry or ATM vol must fail
        let valid_ctx = SmileContext::new(100_000.0, 30.0 / 365.0, 0.6).unwrap();
        let zero_expiry_ctx = SmileContext {
            forward: 100_000.0,
            exercise_time: 0.0,
            atm_vol: 0.6,
        };
        assert!(transport(&mean, &cov, &valid_ctx, &zero_expiry_ctx).is_err());
    }

    #[test]
    fn validate_state_rejects_non_psd_fallibly() {
        let mean = [0.5; N_KNOTS];
        let mut cov = [[0.0; N_KNOTS]; N_KNOTS];
        cov[0][0] = -0.1; // Negative variance
        assert!(validate_state(&mean, &cov).is_err());

        // Valid diagonal but negative eigenvalue off-diagonal
        let mut cov2 = [[0.0; N_KNOTS]; N_KNOTS];
        for i in 0..N_KNOTS {
            cov2[i][i] = 1.0;
        }
        cov2[0][1] = 2.0;
        cov2[1][0] = 2.0;
        assert!(validate_state(&mean, &cov2).is_err());
    }
}
