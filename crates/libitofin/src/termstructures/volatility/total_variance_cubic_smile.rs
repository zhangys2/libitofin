//! Total implied variance cubic smile section with Roger Lee no-arbitrage wing asymptotics.
//!
//! Fits a natural cubic spline in total variance space $w(k) = \sigma^2(k) T$ against
//! log-moneyness $k = \ln(K / F)$. Asymptotic wing slopes are bounded by Roger Lee's (2004)
//! extreme-strike moment formula ($\beta_R \in [0, 2]$, $\beta_L \in [-2, 0]$) and spliced
//! via an exterior Smoothstep $C^2$ Transition Bridge without distorting interior fit.
//! Butterfly arbitrage ($g(k) \ge 0$, Durrleman's condition) is diagnosed and regularized
//! via an adaptive $\lambda$-ramp.

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::cubic::{CubicDerivativeApprox, CubicInterpolation};
use crate::math::matrix::Matrix;
use crate::math::matrixutilities::{qr_decomposition, qr_solve};
use crate::termstructures::volatility::VolatilityType;
use crate::termstructures::volatility::smilesection::{SmileSection, SmileSectionBase};
use crate::time::daycounters::actual365fixed::Actual365Fixed;
use crate::types::{Rate, Real, Time, Volatility};
use crate::{fail, require};

/// Mode for extrapolating total variance beyond the natural cubic spline knots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WingExtrapolationMode {
    /// Smooth C2 transition bridge to Roger Lee asymptotic linear wing.
    #[default]
    AutoSmooth,
    /// Pure linear extrapolation clamped directly at boundary knots.
    ClampedLinear,
    /// Flat total variance extrapolation.
    Flat,
}

/// Configuration parameters for Roger Lee wing splicing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RogerLeeWingConfig {
    pub epsilon: Real,
    pub mode: WingExtrapolationMode,
    pub transition_width: Option<Real>,
}

impl RogerLeeWingConfig {
    pub fn new(
        epsilon: Real,
        mode: WingExtrapolationMode,
        transition_width: Option<Real>,
    ) -> QlResult<Self> {
        let config = Self {
            epsilon,
            mode,
            transition_width,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> QlResult<()> {
        require!(
            self.epsilon.is_finite() && self.epsilon > 0.0 && self.epsilon < 2.0,
            "Roger Lee epsilon must be finite and in (0, 2), got {}",
            self.epsilon
        );
        if let Some(w) = self.transition_width {
            require!(
                w.is_finite() && w > 0.0,
                "Roger Lee transition width must be positive and finite, got {w}"
            );
        }
        Ok(())
    }
}

impl Default for RogerLeeWingConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            mode: WingExtrapolationMode::AutoSmooth,
            transition_width: None,
        }
    }
}

/// Precomputed wing extrapolation model for O(1) evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RogerLeeWing {
    boundary_k: Real,
    boundary_w: Real,
    boundary_slope: Real,
    asymptotic_slope: Real,
    transition_width: Real,
    is_right_wing: bool,
}

impl RogerLeeWing {
    pub fn new_right(
        k_max: Real,
        w_max: Real,
        w_prime_max: Real,
        config: &RogerLeeWingConfig,
        default_delta_k: Real,
    ) -> Self {
        let max_slope = if config.epsilon.is_finite() && config.epsilon < 2.0 {
            (2.0 - config.epsilon).max(0.0)
        } else {
            0.0
        };
        let beta = match config.mode {
            WingExtrapolationMode::Flat => 0.0,
            WingExtrapolationMode::AutoSmooth | WingExtrapolationMode::ClampedLinear => {
                if max_slope > 0.0 {
                    w_prime_max.clamp(0.0, max_slope)
                } else {
                    0.0
                }
            }
        };
        let needs_bridge =
            config.mode == WingExtrapolationMode::AutoSmooth && (w_prime_max - beta).abs() > 1e-12;
        let transition_width = if needs_bridge {
            let mut width = config.transition_width.unwrap_or(default_delta_k).max(1e-4);
            if w_prime_max < 0.0 && w_max > 0.0 {
                let max_safe_width = 1.8 * w_max / (-w_prime_max);
                width = width.min(max_safe_width).max(1e-4);
            }
            width
        } else {
            0.0
        };

        Self {
            boundary_k: k_max,
            boundary_w: w_max.max(0.0),
            boundary_slope: w_prime_max,
            asymptotic_slope: beta,
            transition_width,
            is_right_wing: true,
        }
    }

    pub fn new_left(
        k_min: Real,
        w_min: Real,
        w_prime_min: Real,
        config: &RogerLeeWingConfig,
        default_delta_k: Real,
    ) -> Self {
        let min_slope = if config.epsilon.is_finite() && config.epsilon < 2.0 {
            (-2.0 + config.epsilon).min(0.0)
        } else {
            0.0
        };
        let beta = match config.mode {
            WingExtrapolationMode::Flat => 0.0,
            WingExtrapolationMode::AutoSmooth | WingExtrapolationMode::ClampedLinear => {
                if min_slope < 0.0 {
                    w_prime_min.clamp(min_slope, 0.0)
                } else {
                    0.0
                }
            }
        };
        let needs_bridge =
            config.mode == WingExtrapolationMode::AutoSmooth && (w_prime_min - beta).abs() > 1e-12;
        let transition_width = if needs_bridge {
            let mut width = config.transition_width.unwrap_or(default_delta_k).max(1e-4);
            if w_prime_min > 0.0 && w_min > 0.0 {
                let max_safe_width = 1.8 * w_min / w_prime_min;
                width = width.min(max_safe_width).max(1e-4);
            }
            width
        } else {
            0.0
        };

        Self {
            boundary_k: k_min,
            boundary_w: w_min.max(0.0),
            boundary_slope: w_prime_min,
            asymptotic_slope: beta,
            transition_width,
            is_right_wing: false,
        }
    }

    pub fn total_variance(&self, k: Real) -> Real {
        let dk = self.transition_width;
        let wb = self.boundary_w;
        let wpb = self.boundary_slope;
        let beta = self.asymptotic_slope;

        let w = if dk <= 0.0 {
            wb + beta * (k - self.boundary_k)
        } else {
            let (dist, sign) = if self.is_right_wing {
                (k - self.boundary_k, 1.0)
            } else {
                (self.boundary_k - k, -1.0)
            };

            let u = dist / dk;
            if u <= 1.0 {
                let poly = wpb * u + (beta - wpb) * (u * u * u - 0.5 * u * u * u * u);
                wb + sign * dk * poly
            } else {
                let w1 = wb + sign * dk * 0.5 * (wpb + beta);
                let k1 = self.boundary_k + sign * dk;
                w1 + beta * (k - k1)
            }
        };
        w.max(0.0)
    }

    pub fn derivative(&self, k: Real) -> Real {
        let dk = self.transition_width;
        let wpb = self.boundary_slope;
        let beta = self.asymptotic_slope;

        if dk <= 0.0 {
            return beta;
        }

        let dist = if self.is_right_wing {
            k - self.boundary_k
        } else {
            self.boundary_k - k
        };

        let u = dist / dk;
        if u <= 1.0 {
            wpb + (beta - wpb) * (3.0 * u * u - 2.0 * u * u * u)
        } else {
            beta
        }
    }

    pub fn second_derivative(&self, k: Real) -> Real {
        let dk = self.transition_width;
        let wpb = self.boundary_slope;
        let beta = self.asymptotic_slope;

        if dk <= 0.0 {
            return 0.0;
        }

        let dist = if self.is_right_wing {
            k - self.boundary_k
        } else {
            self.boundary_k - k
        };

        let u = dist / dk;
        if u <= 1.0 {
            let d_wp_du = (beta - wpb) * (6.0 * u - 6.0 * u * u);
            if self.is_right_wing {
                d_wp_du / dk
            } else {
                -d_wp_du / dk
            }
        } else {
            0.0
        }
    }

    pub fn durrleman_density(&self, k: Real) -> Real {
        let w = self.total_variance(k);
        if w <= 0.0 {
            return -1.0;
        }
        let wp = self.derivative(k);
        let wpp = self.second_derivative(k);

        let term1 = 1.0 - k * wp / (2.0 * w);
        let term2 = 0.25 * wp * wp * (1.0 / w + 0.25);
        let term3 = 0.5 * wpp;

        term1 * term1 - term2 + term3
    }

    pub fn boundary_k(&self) -> Real {
        self.boundary_k
    }

    pub fn boundary_w(&self) -> Real {
        self.boundary_w
    }

    pub fn boundary_slope(&self) -> Real {
        self.boundary_slope
    }

    pub fn asymptotic_slope(&self) -> Real {
        self.asymptotic_slope
    }

    pub fn transition_width(&self) -> Real {
        self.transition_width
    }

    pub fn is_right_wing(&self) -> bool {
        self.is_right_wing
    }
}

/// Default natural-cubic knot locations in signed standard-deviation units.
pub const DEFAULT_STD_DEV_POINTS: [Real; 9] = [-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0];

/// Default weight on integrated squared curvature in total variance coordinates.
pub const DEFAULT_SMILE_SMOOTHING: Real = 0.01;

/// Default numerical tolerance for butterfly arbitrage detection.
pub const DEFAULT_BUTTERFLY_TOLERANCE: Real = 1e-8;

/// Diagnostic report for butterfly arbitrage verification across a smile section.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ButterflyArbitrageReport {
    pub min_density: Real,
    pub argmin_k: Real,
    pub has_arbitrage: bool,
    pub tolerance: Real,
    pub points_checked: usize,
    pub ramp_iterations: usize,
    pub final_smoothing: Real,
}

/// Policy for handling detected butterfly arbitrage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ArbitrageFallbackPolicy {
    #[default]
    AllowWithReport,
    FailOnError,
    AffineFallback,
}

/// One-expiry Black volatility smile fitted to mid-IV observations in total variance space.
pub struct TotalVarianceCubicSmileSection {
    base: SmileSectionBase,
    interpolation: CubicInterpolation,
    forward: Rate,
    exercise_time: Time,
    atm_vol: Volatility,
    requested_smoothing: Real,
    smoothing: Real,
    std_dev_points: Vec<Real>,
    knots_k: Vec<Real>,
    fitted_w: Vec<Real>,
    observed_strikes: Vec<Rate>,
    observed_k: Vec<Real>,
    observed_mid_ivs: Vec<Volatility>,
    min_strike: Rate,
    max_strike: Rate,
    left_wing: RogerLeeWing,
    right_wing: RogerLeeWing,
    wing_config: RogerLeeWingConfig,
    butterfly_report: ButterflyArbitrageReport,
}

impl TotalVarianceCubicSmileSection {
    /// Fit a total variance cubic smile using default 9 standard-deviation knots and smoothing 0.01.
    pub fn new(
        strikes: Vec<Rate>,
        mid_ivs: Vec<Volatility>,
        forward: Rate,
        exercise_time: Time,
        atm_vol: Volatility,
    ) -> QlResult<Self> {
        Self::with_options(
            strikes,
            mid_ivs,
            forward,
            exercise_time,
            atm_vol,
            DEFAULT_STD_DEV_POINTS.to_vec(),
            DEFAULT_SMILE_SMOOTHING,
            RogerLeeWingConfig::default(),
            true,
            ArbitrageFallbackPolicy::default(),
        )
    }

    /// Fit with custom smoothing parameter.
    pub fn with_smoothing(
        strikes: Vec<Rate>,
        mid_ivs: Vec<Volatility>,
        forward: Rate,
        exercise_time: Time,
        atm_vol: Volatility,
        std_dev_points: Vec<Real>,
        smoothing: Real,
    ) -> QlResult<Self> {
        Self::with_options(
            strikes,
            mid_ivs,
            forward,
            exercise_time,
            atm_vol,
            std_dev_points,
            smoothing,
            RogerLeeWingConfig::default(),
            true,
            ArbitrageFallbackPolicy::default(),
        )
    }

    /// Fit with all customizable options including Roger Lee wing configuration and arbitrage repair.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        strikes: Vec<Rate>,
        mid_ivs: Vec<Volatility>,
        forward: Rate,
        exercise_time: Time,
        atm_vol: Volatility,
        std_dev_points: Vec<Real>,
        smoothing: Real,
        wing_config: RogerLeeWingConfig,
        arbitrage_repair: bool,
        fallback_policy: ArbitrageFallbackPolicy,
    ) -> QlResult<Self> {
        wing_config.validate()?;
        require!(
            strikes.len() == mid_ivs.len(),
            "strikes and mid_ivs must have equal length ({} vs {})",
            strikes.len(),
            mid_ivs.len()
        );
        require!(
            strikes.len() >= 2,
            "total variance cubic smile needs at least 2 observations, got {}",
            strikes.len()
        );
        require!(
            forward.is_finite() && forward > 0.0,
            "forward must be finite and positive"
        );
        require!(
            exercise_time.is_finite() && exercise_time > 0.0,
            "exercise_time must be finite and positive"
        );
        require!(
            atm_vol.is_finite() && atm_vol > 0.0,
            "atm_vol must be finite and positive"
        );
        require!(
            smoothing.is_finite() && smoothing >= 0.0,
            "smoothing parameter must be finite and nonnegative"
        );
        require!(
            std_dev_points.len() >= 2,
            "total variance cubic smile needs at least 2 knot locations, got {}",
            std_dev_points.len()
        );
        for &point in &std_dev_points {
            require!(point.is_finite(), "knot locations must be finite");
        }
        let mut std_dev_points = std_dev_points;
        std_dev_points.sort_by(Real::total_cmp);
        require!(
            std_dev_points.windows(2).all(|pair| pair[0] < pair[1]),
            "knot locations must be distinct"
        );

        let scale = atm_vol * exercise_time.sqrt();
        require!(
            scale.is_finite() && scale > 0.0,
            "ATM standard deviation must be finite and positive"
        );

        // Map knots to log-moneyness: k_j = x_j * scale
        let knots_k: Vec<Real> = std_dev_points.iter().map(|&x| x * scale).collect();
        let k_min = knots_k[0];
        let k_max = knots_k[knots_k.len() - 1];

        let min_strike = forward * k_min.exp();
        let max_strike = forward * k_max.exp();
        require!(
            min_strike.is_finite()
                && min_strike > 0.0
                && max_strike.is_finite()
                && max_strike > min_strike,
            "knot locations must map to distinct finite positive boundary strikes"
        );

        // Process observations
        let mut observations = Vec::with_capacity(strikes.len());
        for (strike, mid_iv) in strikes.into_iter().zip(mid_ivs) {
            require!(
                strike.is_finite() && strike > 0.0,
                "strikes must be finite and positive, got {strike}"
            );
            require!(
                mid_iv.is_finite() && mid_iv > 0.0,
                "mid_ivs must be finite and positive, got {mid_iv}"
            );
            require!(
                2.0 * mid_iv * exercise_time > 1e-12,
                "mid_iv {mid_iv} with exercise_time {exercise_time} is too small"
            );
            let k = (strike / forward).ln();
            require!(k.is_finite(), "log-moneyness coordinate must be finite");
            observations.push((k, strike, mid_iv));
        }
        observations.sort_by(|a, b| a.0.total_cmp(&b.0));
        require!(
            observations.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "duplicate source coordinates are not allowed"
        );

        let observed_k: Vec<Real> = observations.iter().map(|o| o.0).collect();
        let observed_strikes: Vec<Rate> = observations.iter().map(|o| o.1).collect();
        let observed_mid_ivs: Vec<Volatility> = observations.iter().map(|o| o.2).collect();

        // Filter in-range observations
        let tolerance = 16.0 * Real::EPSILON * k_min.abs().max(k_max.abs()).max(1.0);
        let in_range: Vec<(Real, Volatility)> = observed_k
            .iter()
            .copied()
            .zip(observed_mid_ivs.iter().copied())
            .filter_map(|(k, iv)| {
                if k < k_min - tolerance || k > k_max + tolerance {
                    None
                } else {
                    Some((k.clamp(k_min, k_max), iv))
                }
            })
            .collect();

        let minimum_count = if smoothing > 0.0 { 2 } else { knots_k.len() };
        require!(
            in_range.len() >= minimum_count,
            "fixed-knot total variance cubic fit needs at least {minimum_count} in-range observations, got {}",
            in_range.len()
        );

        let in_range_k: Vec<Real> = in_range.iter().map(|p| p.0).collect();
        let in_range_ivs: Vec<Volatility> = in_range.iter().map(|p| p.1).collect();

        let default_delta_k = (knots_k[knots_k.len() - 1] - knots_k[knots_k.len() - 2])
            .max(knots_k[1] - knots_k[0])
            .max(scale);

        // Fit loop with optional adaptive regularizer ramp
        let mut current_smoothing = smoothing;
        let mut ramp_iterations = 0;
        let max_ramp = if arbitrage_repair { 5 } else { 0 };

        loop {
            // Effective curvature penalty: lambda_eff = lambda * atm_vol / (4 * sqrt(T))
            let lambda_eff = current_smoothing * atm_vol / (4.0 * exercise_time.sqrt());

            let mut fitted_w = fit_knot_ordinates_total_variance(
                &knots_k,
                &in_range_k,
                &in_range_ivs,
                exercise_time,
                lambda_eff,
            )?;

            let mut interpolation = CubicInterpolation::new(
                knots_k.clone(),
                fitted_w.clone(),
                CubicDerivativeApprox::Spline,
            )?;

            let w_prime_min = interpolation.derivative(k_min)?;
            let w_prime_max = interpolation.derivative(k_max)?;

            let mut left_wing = RogerLeeWing::new_left(
                k_min,
                fitted_w[0],
                w_prime_min,
                &wing_config,
                default_delta_k,
            );
            let mut right_wing = RogerLeeWing::new_right(
                k_max,
                fitted_w[fitted_w.len() - 1],
                w_prime_max,
                &wing_config,
                default_delta_k,
            );

            let mut report = check_butterfly_arbitrage_internal(
                &interpolation,
                &knots_k,
                &left_wing,
                &right_wing,
                DEFAULT_BUTTERFLY_TOLERANCE,
            );
            report.ramp_iterations = ramp_iterations;
            report.final_smoothing = current_smoothing;

            if !report.has_arbitrage || ramp_iterations >= max_ramp {
                if report.has_arbitrage {
                    match fallback_policy {
                        ArbitrageFallbackPolicy::AllowWithReport => {}
                        ArbitrageFallbackPolicy::FailOnError => {
                            fail!(
                                "butterfly arbitrage detected after {ramp_iterations} ramp iterations: min_density = {:.6e} at k = {:.4}",
                                report.min_density,
                                report.argmin_k
                            );
                        }
                        ArbitrageFallbackPolicy::AffineFallback => {
                            // Least-squares fit w(k) = w0 + beta * k on in-range observations
                            let n = in_range_k.len() as Real;
                            let in_range_w: Vec<Real> = in_range_ivs
                                .iter()
                                .map(|&iv| iv * iv * exercise_time)
                                .collect();
                            let mean_k = in_range_k.iter().sum::<Real>() / n;
                            let mean_w = in_range_w.iter().sum::<Real>() / n;
                            let mut cov_kw = 0.0;
                            let mut var_k = 0.0;
                            for (&ki, &wi) in in_range_k.iter().zip(&in_range_w) {
                                cov_kw += (ki - mean_k) * (wi - mean_w);
                                var_k += (ki - mean_k) * (ki - mean_k);
                            }
                            let raw_beta = if var_k > 1e-14 { cov_kw / var_k } else { 0.0 };
                            let beta_min = -2.0 + wing_config.epsilon;
                            let beta_max = 2.0 - wing_config.epsilon;
                            let beta = raw_beta.clamp(beta_min, beta_max);
                            let mut w0 = mean_w - beta * mean_k;
                            for &kj in &knots_k {
                                let val = w0 + beta * kj;
                                if val < 0.0 {
                                    w0 -= val;
                                }
                            }
                            fitted_w = knots_k
                                .iter()
                                .map(|&kj| (w0 + beta * kj).max(0.0))
                                .collect();
                            interpolation = CubicInterpolation::new(
                                knots_k.clone(),
                                fitted_w.clone(),
                                CubicDerivativeApprox::Spline,
                            )?;
                            left_wing = RogerLeeWing::new_left(
                                k_min,
                                fitted_w[0],
                                beta,
                                &wing_config,
                                default_delta_k,
                            );
                            right_wing = RogerLeeWing::new_right(
                                k_max,
                                fitted_w[fitted_w.len() - 1],
                                beta,
                                &wing_config,
                                default_delta_k,
                            );
                            report = check_butterfly_arbitrage_internal(
                                &interpolation,
                                &knots_k,
                                &left_wing,
                                &right_wing,
                                DEFAULT_BUTTERFLY_TOLERANCE,
                            );
                            report.ramp_iterations = ramp_iterations;
                            report.final_smoothing = current_smoothing;
                        }
                    }
                }

                let base = SmileSectionBase::with_exercise_time(
                    exercise_time,
                    Actual365Fixed::new(),
                    VolatilityType::ShiftedLognormal,
                    0.0,
                )?;

                return Ok(Self {
                    base,
                    interpolation,
                    forward,
                    exercise_time,
                    atm_vol,
                    requested_smoothing: smoothing,
                    smoothing: current_smoothing,
                    std_dev_points,
                    knots_k,
                    fitted_w,
                    observed_strikes,
                    observed_k,
                    observed_mid_ivs,
                    min_strike,
                    max_strike,
                    left_wing,
                    right_wing,
                    wing_config,
                    butterfly_report: report,
                });
            }

            // Ramp lambda by 2.0
            current_smoothing = if current_smoothing == 0.0 {
                0.01
            } else {
                current_smoothing * 2.0
            };
            ramp_iterations += 1;
        }
    }

    pub fn total_variance_at_log_moneyness(&self, k: Real) -> QlResult<Real> {
        require!(k.is_finite(), "log-moneyness must be finite, got {k}");
        let k_min = self.knots_k[0];
        let k_max = self.knots_k[self.knots_k.len() - 1];
        let w = if k < k_min {
            self.left_wing.total_variance(k)
        } else if k > k_max {
            self.right_wing.total_variance(k)
        } else {
            self.interpolation.value(k)?
        };
        Ok(w.max(0.0))
    }

    pub fn total_variance_derivative(&self, k: Real) -> QlResult<Real> {
        require!(k.is_finite(), "log-moneyness must be finite, got {k}");
        let k_min = self.knots_k[0];
        let k_max = self.knots_k[self.knots_k.len() - 1];
        if k < k_min {
            Ok(self.left_wing.derivative(k))
        } else if k > k_max {
            Ok(self.right_wing.derivative(k))
        } else {
            self.interpolation.derivative(k)
        }
    }

    pub fn total_variance_second_derivative(&self, k: Real) -> QlResult<Real> {
        require!(k.is_finite(), "log-moneyness must be finite, got {k}");
        let k_min = self.knots_k[0];
        let k_max = self.knots_k[self.knots_k.len() - 1];
        if k < k_min {
            Ok(self.left_wing.second_derivative(k))
        } else if k > k_max {
            Ok(self.right_wing.second_derivative(k))
        } else {
            self.interpolation.second_derivative(k)
        }
    }

    pub fn durrleman_density(&self, k: Real) -> QlResult<Real> {
        require!(k.is_finite(), "log-moneyness must be finite, got {k}");
        let w = self.total_variance_at_log_moneyness(k)?;
        if w <= 0.0 {
            return Ok(-1.0);
        }
        let wp = self.total_variance_derivative(k)?;
        let wpp = self.total_variance_second_derivative(k)?;
        let term1 = 1.0 - k * wp / (2.0 * w);
        let term2 = 0.25 * wp * wp * (1.0 / w + 0.25);
        let term3 = 0.5 * wpp;
        Ok(term1 * term1 - term2 + term3)
    }

    pub fn butterfly_report(&self) -> &ButterflyArbitrageReport {
        &self.butterfly_report
    }

    pub fn left_wing(&self) -> &RogerLeeWing {
        &self.left_wing
    }

    pub fn right_wing(&self) -> &RogerLeeWing {
        &self.right_wing
    }

    pub fn knots_k(&self) -> &[Real] {
        &self.knots_k
    }

    pub fn fitted_total_variances(&self) -> &[Real] {
        &self.fitted_w
    }

    pub fn forward(&self) -> Rate {
        self.forward
    }

    pub fn atm_vol(&self) -> Volatility {
        self.atm_vol
    }

    pub fn smoothing(&self) -> Real {
        self.smoothing
    }

    pub fn requested_smoothing(&self) -> Real {
        self.requested_smoothing
    }

    pub fn std_dev_points(&self) -> &[Real] {
        &self.std_dev_points
    }

    pub fn observed_strikes(&self) -> &[Rate] {
        &self.observed_strikes
    }

    pub fn observed_log_moneyness(&self) -> &[Real] {
        &self.observed_k
    }

    pub fn observed_mid_ivs(&self) -> &[Volatility] {
        &self.observed_mid_ivs
    }

    pub fn wing_config(&self) -> &RogerLeeWingConfig {
        &self.wing_config
    }
}

impl SmileSection for TotalVarianceCubicSmileSection {
    fn base(&self) -> &SmileSectionBase {
        &self.base
    }

    fn min_strike(&self) -> Rate {
        self.min_strike
    }

    fn max_strike(&self) -> Rate {
        self.max_strike
    }

    fn atm_level(&self) -> Option<Rate> {
        Some(self.forward)
    }

    fn volatility_impl(&self, strike: Rate) -> QlResult<Volatility> {
        require!(
            strike.is_finite() && strike > 0.0,
            "strike must be positive and finite, got {strike}"
        );
        let k = (strike / self.forward).ln();
        let w = self.total_variance_at_log_moneyness(k)?;
        Ok((w.max(0.0) / self.exercise_time).sqrt())
    }

    fn variance(&self, strike: Rate) -> QlResult<Real> {
        require!(
            strike.is_finite() && strike > 0.0,
            "strike must be positive and finite, got {strike}"
        );
        let k = (strike / self.forward).ln();
        self.total_variance_at_log_moneyness(k)
    }
}

fn fit_knot_ordinates_total_variance(
    knots_k: &[Real],
    in_range_k: &[Real],
    in_range_ivs: &[Volatility],
    exercise_time: Time,
    effective_smoothing: Real,
) -> QlResult<Vec<Real>> {
    let knot_count = knots_k.len();
    let minimum_count = if effective_smoothing > 0.0 {
        2
    } else {
        knot_count
    };
    require!(
        in_range_k.len() >= minimum_count,
        "fixed-knot total variance cubic fit needs at least {minimum_count} in-range observations, got {}",
        in_range_k.len()
    );

    let mut basis = Vec::with_capacity(knot_count);
    for column in 0..knot_count {
        let mut ordinates = vec![0.0; knot_count];
        ordinates[column] = 1.0;
        basis.push(CubicInterpolation::new(
            knots_k.to_vec(),
            ordinates,
            CubicDerivativeApprox::Spline,
        )?);
    }

    let penalty_rows = if effective_smoothing > 0.0 {
        2 * (knot_count - 1)
    } else {
        0
    };
    let row_count = in_range_k.len() + penalty_rows;
    let mut design = Matrix::with_size(row_count, knot_count);
    let mut target = Array::with_size(row_count);
    let data_weight = 1.0 / (in_range_k.len() as Real).sqrt();

    // Data rows: IV-normalized linear residual: (w(k_i) - w_i) / (2 * sigma_i * T)
    for (row, (&k, &iv)) in in_range_k.iter().zip(in_range_ivs.iter()).enumerate() {
        let norm_factor = 2.0 * iv * exercise_time;
        target[row] = (iv / 2.0) * data_weight;
        for (column, interp) in basis.iter().enumerate() {
            design[(row, column)] = (interp.value(k)? / norm_factor) * data_weight;
        }
    }

    // Penalty rows: Gauss-Legendre 2-point quadrature
    if effective_smoothing > 0.0 {
        let gaussian_offset = 1.0 / 3.0_f64.sqrt();
        for (segment, pair) in knots_k.windows(2).enumerate() {
            let half_width = (pair[1] - pair[0]) / 2.0;
            let midpoint = pair[0] + half_width;
            let weight = (effective_smoothing * half_width).sqrt();
            for (q, sign) in [-1.0, 1.0].iter().enumerate() {
                let x = midpoint + sign * half_width * gaussian_offset;
                let row = in_range_k.len() + 2 * segment + q;
                for (column, interp) in basis.iter().enumerate() {
                    let value = weight * interp.second_derivative(x)?;
                    require!(value.is_finite(), "non-finite curvature penalty");
                    design[(row, column)] = value;
                }
            }
        }
    }

    let (_, r, _) = qr_decomposition(&design, true);
    let max_diagonal = (0..knot_count)
        .map(|i| r[(i, i)].abs())
        .fold(0.0_f64, Real::max);
    require!(
        max_diagonal.is_finite() && max_diagonal > 0.0,
        "fixed-knot total variance cubic fit is rank deficient"
    );
    let rank_tolerance = 128.0 * Real::EPSILON * max_diagonal;
    require!(
        (0..knot_count).all(|i| r[(i, i)].abs() > rank_tolerance),
        "fixed-knot total variance cubic fit is rank deficient; use positive smoothing and distinct in-range observations"
    );

    let fitted: Array = qr_solve(&design, &target, true, None);
    require!(
        fitted.iter().all(|ordinate| ordinate.is_finite()),
        "fixed-knot total variance cubic fit produced non-finite ordinates"
    );

    Ok(fitted.to_vec())
}

fn check_butterfly_arbitrage_internal(
    interpolation: &CubicInterpolation,
    knots_k: &[Real],
    left_wing: &RogerLeeWing,
    right_wing: &RogerLeeWing,
    tolerance: Real,
) -> ButterflyArbitrageReport {
    let mut min_density = Real::INFINITY;
    let mut argmin_k = 0.0;
    let mut points_checked = 0;

    let eval_point = |k: Real, min_d: &mut Real, arg_k: &mut Real, pts: &mut usize| {
        *pts += 1;
        let k_min = knots_k[0];
        let k_max = knots_k[knots_k.len() - 1];
        let g = if k < k_min {
            left_wing.durrleman_density(k)
        } else if k > k_max {
            right_wing.durrleman_density(k)
        } else {
            let w = interpolation.value(k).unwrap_or(0.0);
            if w <= 0.0 {
                -1.0
            } else {
                let wp = interpolation.derivative(k).unwrap_or(0.0);
                let wpp = interpolation.second_derivative(k).unwrap_or(0.0);
                let term1 = 1.0 - k * wp / (2.0 * w);
                let term2 = 0.25 * wp * wp * (1.0 / w + 0.25);
                let term3 = 0.5 * wpp;
                term1 * term1 - term2 + term3
            }
        };
        if g < *min_d {
            *min_d = g;
            *arg_k = k;
        }
    };

    // 1. Check all knots
    for &k in knots_k {
        eval_point(k, &mut min_density, &mut argmin_k, &mut points_checked);
    }

    // 2. 5-point Gauss interior points per segment
    // Gauss-Legendre 5-point roots on [-1, 1]
    let xi_points = [
        0.0,
        0.538_469_310_105_683_1,
        -0.538_469_310_105_683_1,
        0.906_179_845_938_664,
        -0.906_179_845_938_664,
    ];
    for pair in knots_k.windows(2) {
        let half_w = (pair[1] - pair[0]) / 2.0;
        let mid = pair[0] + half_w;
        for &xi in &xi_points {
            let k = mid + half_w * xi;
            eval_point(k, &mut min_density, &mut argmin_k, &mut points_checked);
        }
    }

    // 3. Wing evaluation points
    let k_min = knots_k[0];
    let k_max = knots_k[knots_k.len() - 1];
    let test_wing_offsets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1e5];
    for &offset in &test_wing_offsets {
        eval_point(
            k_max + offset,
            &mut min_density,
            &mut argmin_k,
            &mut points_checked,
        );
        eval_point(
            k_min - offset,
            &mut min_density,
            &mut argmin_k,
            &mut points_checked,
        );
    }

    let has_arbitrage = min_density < -tolerance;

    ButterflyArbitrageReport {
        min_density,
        argmin_k,
        has_arbitrage,
        tolerance,
        points_checked,
        ramp_iterations: 0,
        final_smoothing: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roger_lee_right_wing_clamping_and_c2_bridge() {
        let config = RogerLeeWingConfig::default();
        let k_max = 0.5;
        let w_max = 0.05;
        // Slope exceeds 2.0 (e.g. 2.5), should clamp to 2.0 - eps = 1.9999
        let w_prime_max = 2.5;
        let delta_k = 0.2;

        let wing = RogerLeeWing::new_right(k_max, w_max, w_prime_max, &config, delta_k);

        // 1. Boundary matching at k = k_max (u = 0)
        let w_at_b = wing.total_variance(k_max);
        let wp_at_b = wing.derivative(k_max);
        let wpp_at_b = wing.second_derivative(k_max);

        assert!((w_at_b - w_max).abs() < 1e-12, "w(k_max) mismatch");
        assert!(
            (wp_at_b - w_prime_max).abs() < 1e-12,
            "w'(k_max) mismatch: expected {w_prime_max}, got {wp_at_b}"
        );
        assert!(
            wpp_at_b.abs() < 1e-12,
            "w''(k_max) must be 0 for natural spline matching"
        );

        // 2. Far wing matching at k = k_max + delta_k (u = 1)
        let k_trans = k_max + delta_k;
        let wp_at_trans = wing.derivative(k_trans);
        let wpp_at_trans = wing.second_derivative(k_trans);
        let expected_beta = 2.0 - config.epsilon;

        assert!(
            (wp_at_trans - expected_beta).abs() < 1e-12,
            "w'(k_trans) mismatch"
        );
        assert!(
            wpp_at_trans.abs() < 1e-12,
            "w''(k_trans) must be 0 for linear wing matching"
        );

        // 3. Far linear wing
        let k_far = 10.0;
        let w_far = wing.total_variance(k_far);
        let wp_far = wing.derivative(k_far);
        let wpp_far = wing.second_derivative(k_far);

        assert!((wp_far - expected_beta).abs() < 1e-12);
        assert_eq!(wpp_far, 0.0);
        assert!(w_far > w_max);

        // 4. Asymptotic Durrleman density limit as k -> infinity
        let k_asymp = 1e7;
        let g_asymp = wing.durrleman_density(k_asymp);
        let expected_g_limit = (4.0 - expected_beta * expected_beta) / 16.0;
        assert!(
            (g_asymp - expected_g_limit).abs() < 1e-4,
            "Expected g(k_asymp) to approach (4 - beta^2)/16"
        );
        assert!(
            g_asymp >= 0.0,
            "Extreme strike density must be non-negative"
        );
    }

    #[test]
    fn test_roger_lee_left_wing_clamping_and_c2_bridge() {
        let config = RogerLeeWingConfig::default();
        let k_min = -0.5;
        let w_min = 0.05;
        // Slope below -2.0 (e.g. -2.5), should clamp to -2.0 + eps = -1.9999
        let w_prime_min = -2.5;
        let delta_k = 0.2;

        let wing = RogerLeeWing::new_left(k_min, w_min, w_prime_min, &config, delta_k);

        // Boundary matching at k = k_min (u = 0)
        let w_at_b = wing.total_variance(k_min);
        let wp_at_b = wing.derivative(k_min);
        let wpp_at_b = wing.second_derivative(k_min);

        assert!((w_at_b - w_min).abs() < 1e-12);
        assert!((wp_at_b - w_prime_min).abs() < 1e-12);
        assert!(wpp_at_b.abs() < 1e-12);

        // Far wing matching at k = k_min - delta_k
        let k_trans = k_min - delta_k;
        let wp_at_trans = wing.derivative(k_trans);
        let wpp_at_trans = wing.second_derivative(k_trans);
        let expected_beta = -2.0 + config.epsilon;

        assert!((wp_at_trans - expected_beta).abs() < 1e-12);
        assert!(wpp_at_trans.abs() < 1e-12);

        // Far linear wing (k << k_trans)
        let k_far = -10.0;
        let wp_far = wing.derivative(k_far);
        assert!((wp_far - expected_beta).abs() < 1e-12);

        let k_asymp = -1e7;
        let g_asymp = wing.durrleman_density(k_asymp);
        let expected_g_limit = (4.0 - expected_beta * expected_beta) / 16.0;
        assert!((g_asymp - expected_g_limit).abs() < 1e-4);
        assert!(g_asymp >= 0.0);
    }

    #[test]
    fn test_total_variance_cubic_smile_fit() {
        let forward = 100.0;
        let expiry = 0.5;
        let atm_vol = 0.20;
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let mid_ivs = vec![0.24, 0.21, 0.20, 0.21, 0.23];

        let section = TotalVarianceCubicSmileSection::new(
            strikes.clone(),
            mid_ivs.clone(),
            forward,
            expiry,
            atm_vol,
        )
        .expect("Fit should succeed");

        // ATM volatility should closely match 0.20
        let atm_iv = section.volatility(100.0).expect("volatility at 100.0");
        assert!(
            (atm_iv - 0.20).abs() < 0.005,
            "ATM vol {atm_iv} should be close to 0.20"
        );

        // Total variance at ATM k=0 should match atm_vol^2 * T = 0.02
        let tv_atm = section
            .total_variance_at_log_moneyness(0.0)
            .expect("total variance at k=0");
        assert!((tv_atm - 0.02).abs() < 0.001);

        // Wing queries beyond knot range
        let vol_left = section.volatility(40.0).expect("left wing volatility");
        let vol_right = section.volatility(250.0).expect("right wing volatility");
        assert!(vol_left > 0.0);
        assert!(vol_right > 0.0);

        // Roger Lee asymptotic slopes
        assert!(section.right_wing().asymptotic_slope >= 0.0);
        assert!(section.right_wing().asymptotic_slope <= 2.0);
        assert!(section.left_wing().asymptotic_slope <= 0.0);
        assert!(section.left_wing().asymptotic_slope >= -2.0);
    }

    #[test]
    fn test_butterfly_arbitrage_detection_and_adaptive_ramp() {
        let forward = 100.0;
        let expiry = 0.25;
        let atm_vol = 0.30;
        // Construct quotes with steep, sharp kink that introduces butterfly arbitrage at low smoothing
        let strikes = vec![85.0, 92.0, 96.0, 100.0, 104.0, 108.0, 115.0];
        let mid_ivs = vec![0.55, 0.42, 0.28, 0.29, 0.32, 0.45, 0.58];

        // 1. Fit without repair and tiny smoothing -> should detect arbitrage
        let section_no_repair = TotalVarianceCubicSmileSection::with_options(
            strikes.clone(),
            mid_ivs.clone(),
            forward,
            expiry,
            atm_vol,
            DEFAULT_STD_DEV_POINTS.to_vec(),
            1e-5,
            RogerLeeWingConfig::default(),
            false,
            ArbitrageFallbackPolicy::AllowWithReport,
        )
        .expect("Fit without repair should succeed");

        assert!(
            section_no_repair.butterfly_report().has_arbitrage,
            "Sharp unregularized smile must trigger butterfly arbitrage detection"
        );
        assert!(section_no_repair.butterfly_report().min_density < 0.0);

        // 2. Fit with automated repair enabled -> should ramp smoothing to extinguish or reduce arbitrage
        let section_repaired = TotalVarianceCubicSmileSection::with_options(
            strikes.clone(),
            mid_ivs.clone(),
            forward,
            expiry,
            atm_vol,
            DEFAULT_STD_DEV_POINTS.to_vec(),
            1e-5,
            RogerLeeWingConfig::default(),
            true,
            ArbitrageFallbackPolicy::AllowWithReport,
        )
        .expect("Fit with repair should succeed");

        assert!(
            section_repaired.butterfly_report().ramp_iterations > 0,
            "Should have performed regularizer ramp iterations"
        );
        assert!(
            section_repaired.butterfly_report().final_smoothing > 1e-5,
            "Final smoothing must be greater than initial smoothing"
        );
        assert!(
            section_repaired.butterfly_report().min_density
                > section_no_repair.butterfly_report().min_density,
            "Adaptive ramp must strictly increase minimum Durrleman density"
        );
    }

    #[test]
    fn test_total_variance_input_validation() {
        let forward = 100.0;
        let expiry = 1.0;
        let atm_vol = 0.20;

        // Mismatched lengths
        assert!(
            TotalVarianceCubicSmileSection::new(
                vec![90.0, 100.0],
                vec![0.2],
                forward,
                expiry,
                atm_vol
            )
            .is_err()
        );

        // Fewer than 2 observations
        assert!(
            TotalVarianceCubicSmileSection::new(vec![100.0], vec![0.2], forward, expiry, atm_vol)
                .is_err()
        );

        // Non-positive forward
        assert!(
            TotalVarianceCubicSmileSection::new(
                vec![90.0, 100.0],
                vec![0.2, 0.2],
                0.0,
                expiry,
                atm_vol
            )
            .is_err()
        );

        // Non-positive expiry
        assert!(
            TotalVarianceCubicSmileSection::new(
                vec![90.0, 100.0],
                vec![0.2, 0.2],
                forward,
                0.0,
                atm_vol
            )
            .is_err()
        );

        // Duplicate knots
        assert!(
            TotalVarianceCubicSmileSection::with_smoothing(
                vec![90.0, 100.0],
                vec![0.2, 0.2],
                forward,
                expiry,
                atm_vol,
                vec![-1.0, -1.0],
                0.01,
            )
            .is_err()
        );
    }

    #[test]
    fn test_roger_lee_flat_mode() {
        let config = RogerLeeWingConfig {
            mode: WingExtrapolationMode::Flat,
            ..Default::default()
        };
        let wing = RogerLeeWing::new_right(0.5, 0.04, 1.2, &config, 0.1);
        assert_eq!(wing.asymptotic_slope, 0.0);
        assert_eq!(wing.derivative(10.0), 0.0);
        assert_eq!(wing.second_derivative(10.0), 0.0);
    }

    #[test]
    fn test_call_wing_only_quotes_variance_non_negative_and_option_price_no_nan() {
        use crate::option::OptionType;
        let forward: Real = 100.0;
        let expiry: Real = 0.5;
        let atm_vol: Real = 0.20;
        let scale: Real = atm_vol * expiry.sqrt();

        // Call-wing-only quotes
        let x_points: [Real; 5] = [0.0, 0.6, 1.0, 1.5, 2.5];
        let mid_ivs = vec![0.20, 0.24, 0.28, 0.34, 0.45];
        let strikes: Vec<Real> = x_points
            .iter()
            .map(|&x| forward * (x * scale).exp())
            .collect();

        let section =
            TotalVarianceCubicSmileSection::new(strikes, mid_ivs, forward, expiry, atm_vol)
                .expect("Call-wing smile should fit successfully");

        // Check deep OTM put strikes where w dipped negative previously
        for strike in [10.0, 25.0, 50.0, 75.0, 90.0, 100.0, 120.0, 150.0, 200.0] {
            let var = section.variance(strike).expect("Variance should succeed");
            assert!(
                var >= 0.0,
                "Variance at {strike} must be nonnegative, got {var}"
            );
            let vol = section
                .volatility(strike)
                .expect("Volatility should succeed");
            assert!(
                vol >= 0.0 && vol.is_finite(),
                "Volatility at {strike} must be finite, got {vol}"
            );
            let call_price = section
                .option_price(strike, OptionType::Call, 1.0)
                .expect("Call price should succeed");
            assert!(
                call_price.is_finite() && !call_price.is_nan(),
                "Call price must not be NaN at {strike}"
            );
            let put_price = section
                .option_price(strike, OptionType::Put, 1.0)
                .expect("Put price should succeed");
            assert!(
                put_price.is_finite() && !put_price.is_nan(),
                "Put price must not be NaN at {strike}"
            );
        }
    }

    #[test]
    fn test_affine_fallback_eliminates_arbitrage() {
        let forward = 100.0;
        let expiry = 1.0;
        let atm_vol = 0.30;
        let strikes = vec![85.0, 92.0, 96.0, 100.0, 104.0, 108.0, 115.0];
        let ivs = vec![0.55, 0.42, 0.28, 0.29, 0.32, 0.45, 0.58];

        let section = TotalVarianceCubicSmileSection::with_options(
            strikes,
            ivs,
            forward,
            expiry,
            atm_vol,
            DEFAULT_STD_DEV_POINTS.to_vec(),
            1e-5,
            RogerLeeWingConfig::default(),
            true,
            ArbitrageFallbackPolicy::AffineFallback,
        )
        .expect("AffineFallback fit should succeed");

        assert!(
            !section.butterfly_report().has_arbitrage,
            "AffineFallback must eliminate butterfly arbitrage, report: {:?}",
            section.butterfly_report()
        );
        let d2 = section
            .total_variance_second_derivative(0.0)
            .expect("Second derivative");
        assert!(
            d2.abs() < 1e-6,
            "Affine line should have zero second derivative, got {d2}"
        );
    }

    #[test]
    fn test_strike_and_log_moneyness_domain_checks() {
        let forward = 100.0;
        let expiry = 1.0;
        let atm_vol = 0.20;
        let strikes = vec![90.0, 100.0, 110.0];
        let ivs = vec![0.22, 0.20, 0.22];

        let section = TotalVarianceCubicSmileSection::new(strikes, ivs, forward, expiry, atm_vol)
            .expect("Fit should succeed");

        assert!(section.volatility(0.0).is_err());
        assert!(section.volatility(-10.0).is_err());
        assert!(section.volatility(f64::NAN).is_err());
        assert!(section.volatility(f64::INFINITY).is_err());
        assert!(section.variance(0.0).is_err());
        assert!(section.variance(-5.0).is_err());
        assert!(section.variance(f64::NAN).is_err());
        assert!(section.total_variance_at_log_moneyness(f64::NAN).is_err());
        assert!(section.total_variance_derivative(f64::NAN).is_err());
        assert!(
            section
                .total_variance_second_derivative(f64::INFINITY)
                .is_err()
        );
        assert!(section.durrleman_density(f64::NAN).is_err());
    }

    #[test]
    fn test_zero_and_tiny_mid_iv_rejected() {
        let forward = 100.0;
        let expiry = 1.0;
        let atm_vol = 0.20;

        let err_zero = TotalVarianceCubicSmileSection::new(
            vec![90.0, 100.0, 110.0],
            vec![0.0, 0.20, 0.22],
            forward,
            expiry,
            atm_vol,
        );
        match err_zero {
            Err(e) => assert!(
                e.to_string().contains("positive"),
                "Expected error about positive mid_iv"
            ),
            Ok(_) => panic!("Expected error for zero mid_iv"),
        }

        let err_tiny = TotalVarianceCubicSmileSection::new(
            vec![90.0, 100.0, 110.0],
            vec![1e-16, 0.20, 0.22],
            forward,
            expiry,
            atm_vol,
        );
        match err_tiny {
            Err(e) => assert!(
                e.to_string().contains("too small"),
                "Expected error about too small mid_iv"
            ),
            Ok(_) => panic!("Expected error for tiny mid_iv"),
        }
    }

    #[test]
    fn test_repaired_section_smoothing_matches_report() {
        let forward = 100.0;
        let expiry = 1.0;
        let atm_vol = 0.30;
        let strikes = vec![85.0, 92.0, 96.0, 100.0, 104.0, 108.0, 115.0];
        let ivs = vec![0.55, 0.42, 0.28, 0.29, 0.32, 0.45, 0.58];

        let section = TotalVarianceCubicSmileSection::with_options(
            strikes,
            ivs,
            forward,
            expiry,
            atm_vol,
            DEFAULT_STD_DEV_POINTS.to_vec(),
            1e-5,
            RogerLeeWingConfig::default(),
            true,
            ArbitrageFallbackPolicy::AllowWithReport,
        )
        .expect("Fit should succeed");

        assert_eq!(
            section.smoothing(),
            section.butterfly_report().final_smoothing
        );
        assert_eq!(section.requested_smoothing(), 1e-5);
    }

    #[test]
    fn test_bad_roger_lee_config_rejected() {
        assert!(RogerLeeWingConfig::new(3.0, WingExtrapolationMode::AutoSmooth, None).is_err());
        assert!(RogerLeeWingConfig::new(-0.1, WingExtrapolationMode::AutoSmooth, None).is_err());
        assert!(RogerLeeWingConfig::new(0.0, WingExtrapolationMode::AutoSmooth, None).is_err());
        assert!(
            RogerLeeWingConfig::new(f64::NAN, WingExtrapolationMode::AutoSmooth, None).is_err()
        );
        assert!(
            RogerLeeWingConfig::new(1e-4, WingExtrapolationMode::AutoSmooth, Some(-0.5)).is_err()
        );
        assert!(
            RogerLeeWingConfig::new(1e-4, WingExtrapolationMode::AutoSmooth, Some(0.0)).is_err()
        );

        // Defensive check: direct RogerLeeWing construction with out-of-range epsilon does NOT panic
        let bad_config = RogerLeeWingConfig {
            epsilon: 3.0,
            mode: WingExtrapolationMode::AutoSmooth,
            transition_width: None,
        };
        let right_wing = RogerLeeWing::new_right(0.5, 0.05, 2.5, &bad_config, 0.2);
        assert_eq!(right_wing.asymptotic_slope, 0.0);
        let left_wing = RogerLeeWing::new_left(-0.5, 0.05, -2.5, &bad_config, 0.2);
        assert_eq!(left_wing.asymptotic_slope, 0.0);
    }
}
