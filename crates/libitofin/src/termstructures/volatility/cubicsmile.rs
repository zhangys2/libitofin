//! Fixed-knot natural cubic least-squares smile in standardized log-moneyness.
//!
//! Source mid-IV observations are mapped using one fixed ATM volatility:
//! `x = (ln(strike) - ln(forward)) / (atm_vol * sqrt(exercise_time))`. The
//! configured standard-deviation points (nine by default) are the only spline
//! knots. Their IV ordinates minimize mean squared quote residual plus a
//! curvature penalty inside the knot range; outside observations are not fitted.
//!
//! This is a one-expiry fit only and does not enforce static arbitrage.

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::comparison::close_enough;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::cubic::{CubicDerivativeApprox, CubicInterpolation};
use crate::math::matrix::Matrix;
use crate::math::matrixutilities::{qr_decomposition, qr_solve};
use crate::termstructures::volatility::VolatilityType;
use crate::termstructures::volatility::smilesection::{SmileSection, SmileSectionBase};
use crate::time::daycounters::actual365fixed::Actual365Fixed;
use crate::types::{Rate, Real, Time, Volatility};
use crate::{fail, require};

/// Default natural-cubic knot locations in signed standard-deviation units.
pub const DEFAULT_STD_DEV_POINTS: [Real; 9] = [-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0];

/// Default weight on integrated squared curvature in standardized coordinates.
pub const DEFAULT_SMILE_SMOOTHING: Real = 0.01;

/// One-expiry Black volatility smile fitted to mid-IV observations on fixed knots.
///
/// The natural cubic spline uses exactly the configured standard-deviation knot
/// locations; source observations determine the knot IVs by curvature-regularized
/// least squares. [`SmileSection::volatility`] accepts strikes. Observations outside
/// the knot range are retained for inspection but excluded from fitting and have
/// `None` residuals.
pub struct CubicSmileSection {
    base: SmileSectionBase,
    interpolation: CubicInterpolation,
    forward: Rate,
    atm_vol: Volatility,
    smoothing: Real,
    std_dev_points: Vec<Real>,
    node_std_dev_points: Vec<Real>,
    node_mid_ivs: Vec<Volatility>,
    observed_strikes: Vec<Rate>,
    observed_std_dev_points: Vec<Real>,
    observed_mid_ivs: Vec<Volatility>,
    min_strike: Rate,
    max_strike: Rate,
}

impl CubicSmileSection {
    /// Fit a natural cubic smile using the default nine knot locations.
    ///
    /// `mid_ivs` are annualized decimal Black implied volatilities, paired with
    /// `strikes`. The knot ordinates are fitted by curvature-regularized least
    /// squares using in-range observations. Call/put observations at the same strike
    /// must be consolidated by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal input lengths, fewer than two observations,
    /// non-finite or non-positive forward/time/ATM volatility/strikes, negative
    /// or non-finite mid-IVs, invalid/duplicate knots, duplicate source points,
    /// fewer than two in-range observations, or a rank-deficient fit.
    pub fn new(
        strikes: Vec<Rate>,
        mid_ivs: Vec<Volatility>,
        forward: Rate,
        exercise_time: Time,
        atm_vol: Volatility,
    ) -> QlResult<Self> {
        Self::with_std_dev_points(
            strikes,
            mid_ivs,
            forward,
            exercise_time,
            atm_vol,
            DEFAULT_STD_DEV_POINTS.to_vec(),
        )
    }

    /// Fit a natural cubic smile using caller-specified knot locations.
    ///
    /// Knots are sorted into increasing order. The fit uses only observations
    /// within the knot range; direct queries beyond the knot range fail unless
    /// extrapolation is enabled on the returned section.
    pub fn with_std_dev_points(
        strikes: Vec<Rate>,
        mid_ivs: Vec<Volatility>,
        forward: Rate,
        exercise_time: Time,
        atm_vol: Volatility,
        std_dev_points: Vec<Real>,
    ) -> QlResult<Self> {
        Self::with_smoothing(
            strikes,
            mid_ivs,
            forward,
            exercise_time,
            atm_vol,
            std_dev_points,
            DEFAULT_SMILE_SMOOTHING,
        )
    }

    /// Fit with an explicit nonnegative curvature penalty weight.
    ///
    /// Minimizes `mean((s(x_i) - iv_i)^2) + smoothing * integral(s''(x)^2 dx)`
    /// over the fixed knot domain, then floors negative knot IVs at zero.
    /// Positive smoothing supports as few as two distinct in-range observations;
    /// zero requests the unregularized fit and requires full observation rank.
    /// Unsampled wings are model-dependent, even when inside the knot domain.
    pub fn with_smoothing(
        strikes: Vec<Rate>,
        mid_ivs: Vec<Volatility>,
        forward: Rate,
        exercise_time: Time,
        atm_vol: Volatility,
        std_dev_points: Vec<Real>,
        smoothing: Real,
    ) -> QlResult<Self> {
        require!(
            smoothing.is_finite() && smoothing >= 0.0,
            "smoothing must be finite and nonnegative"
        );
        require!(
            strikes.len() == mid_ivs.len(),
            "strikes and mid_ivs must have equal length ({} vs {})",
            strikes.len(),
            mid_ivs.len()
        );
        require!(
            strikes.len() >= 2,
            "cubic smile needs at least 2 observations, got {}",
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
            std_dev_points.len() >= 2,
            "cubic smile needs at least 2 knot locations, got {}",
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

        let mut observations = Vec::with_capacity(strikes.len());
        for (strike, mid_iv) in strikes.into_iter().zip(mid_ivs) {
            require!(
                strike.is_finite() && strike > 0.0,
                "strikes must be finite and positive"
            );
            require!(
                mid_iv.is_finite() && mid_iv >= 0.0,
                "mid_ivs must be finite and nonnegative"
            );
            let x = (strike.ln() - forward.ln()) / scale;
            require!(
                x.is_finite(),
                "standardized strike coordinate must be finite"
            );
            observations.push((x, strike, mid_iv));
        }
        observations.sort_by(|left, right| left.0.total_cmp(&right.0));
        require!(
            observations.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "duplicate standardized source coordinates are not allowed"
        );

        let observed_std_dev_points: Vec<Real> = observations
            .iter()
            .map(|observation| observation.0)
            .collect();
        let observed_strikes: Vec<Rate> = observations
            .iter()
            .map(|observation| observation.1)
            .collect();
        let observed_mid_ivs: Vec<Volatility> = observations
            .iter()
            .map(|observation| observation.2)
            .collect();
        let node_std_dev_points = std_dev_points.clone();
        let x_min = node_std_dev_points[0];
        let x_max = node_std_dev_points[node_std_dev_points.len() - 1];
        let min_strike = (forward.ln() + x_min * scale).exp();
        let max_strike = (forward.ln() + x_max * scale).exp();
        require!(
            min_strike.is_finite()
                && min_strike > 0.0
                && max_strike.is_finite()
                && max_strike > min_strike,
            "knot locations must map to distinct finite positive boundary strikes"
        );
        // Use the same strike-space endpoint snapping as queries. At small ATM
        // scales, a boundary strike's log round trip can move x beyond the
        // x-space tolerance even though the strike is at the boundary.
        let fit_coordinates: Vec<Real> = observations
            .iter()
            .map(|&(x, strike, _)| {
                if x <= x_min && close_enough(strike, min_strike) {
                    x_min
                } else if x >= x_max && close_enough(strike, max_strike) {
                    x_max
                } else {
                    x
                }
            })
            .collect();
        let node_mid_ivs = fit_knot_ordinates(
            &std_dev_points,
            &fit_coordinates,
            &observed_mid_ivs,
            smoothing,
        )?;
        let interpolation = CubicInterpolation::new(
            node_std_dev_points.clone(),
            node_mid_ivs.clone(),
            CubicDerivativeApprox::Spline,
        )?;
        let base = SmileSectionBase::with_exercise_time(
            exercise_time,
            Actual365Fixed::new(),
            VolatilityType::ShiftedLognormal,
            0.0,
        )?;

        Ok(Self {
            base,
            interpolation,
            forward,
            atm_vol,
            smoothing,
            std_dev_points,
            node_std_dev_points,
            node_mid_ivs,
            observed_strikes,
            observed_std_dev_points,
            observed_mid_ivs,
            min_strike,
            max_strike,
        })
    }

    /// Set whether queries beyond the fixed knot range extend the end cubic
    /// segments. Extrapolation is disabled by default.
    pub fn with_extrapolation(mut self, allow: bool) -> Self {
        self.interpolation = self.interpolation.with_extrapolation(allow);
        self
    }

    /// Evaluate the fitted implied volatility at a standard-deviation point.
    ///
    /// Negative cubic overshoot is floored at zero. Out-of-range queries fail
    /// unless extrapolation has been enabled.
    pub fn volatility_at_std_dev(&self, point: Real) -> QlResult<Volatility> {
        require!(point.is_finite(), "standard-deviation point must be finite");
        let point = self.snap_to_domain_boundary(point);
        let vol = self.interpolation.value(point)?;
        require!(vol.is_finite(), "interpolated mid-IV must be finite");
        Ok(vol.max(0.0))
    }

    /// Convert a standard-deviation point back to its strike.
    pub fn strike_at_std_dev(&self, point: Real) -> QlResult<Rate> {
        require!(point.is_finite(), "standard-deviation point must be finite");
        let strike = (self.forward.ln() + point * self.scale()).exp();
        if !strike.is_finite() || strike <= 0.0 {
            fail!("standard-deviation point maps to a non-finite or non-positive strike");
        }
        Ok(strike)
    }

    /// Return the fitted IV at each configured knot location.
    pub fn sampled_mid_ivs(&self) -> QlResult<Vec<Option<Volatility>>> {
        self.std_dev_points
            .iter()
            .map(|&point| {
                if !self.interpolation.allows_extrapolation() && !self.is_in_or_near_range(point) {
                    Ok(None)
                } else {
                    self.volatility_at_std_dev(point).map(Some)
                }
            })
            .collect()
    }

    /// Configured spline knots, sorted in increasing order.
    pub fn std_dev_points(&self) -> &[Real] {
        &self.std_dev_points
    }

    /// Fixed knot coordinates used by the fitted spline.
    pub fn node_std_dev_points(&self) -> &[Real] {
        &self.node_std_dev_points
    }

    /// Fitted IV ordinates paired with [`node_std_dev_points`](Self::node_std_dev_points).
    pub fn node_mid_ivs(&self) -> &[Volatility] {
        &self.node_mid_ivs
    }

    /// Source strikes, sorted by standardized coordinate.
    pub fn observed_strikes(&self) -> &[Rate] {
        &self.observed_strikes
    }

    /// Standardized coordinates of the source market observations.
    pub fn observed_std_dev_points(&self) -> &[Real] {
        &self.observed_std_dev_points
    }

    /// Source mid-IVs paired with [`observed_strikes`](Self::observed_strikes).
    pub fn observed_mid_ivs(&self) -> &[Volatility] {
        &self.observed_mid_ivs
    }

    /// Fitted-minus-observed IV at each source observation.
    ///
    /// Returns `None` for observations outside the fixed knot range because the
    /// fit deliberately does not extrapolate to them.
    pub fn observation_residuals(&self) -> QlResult<Vec<Option<Real>>> {
        self.observed_std_dev_points
            .iter()
            .zip(&self.observed_mid_ivs)
            .map(|(&point, &observed)| {
                if self.is_in_or_near_range(point) {
                    Ok(Some(self.volatility_at_std_dev(point)? - observed))
                } else {
                    Ok(None)
                }
            })
            .collect()
    }

    /// Piecewise polynomial `[a, b, c]` coefficients for each adjacent knot pair.
    ///
    /// Entry `i` applies on `[x_i, x_{i+1}]` using
    /// `sigma(x) = sigma_i + a * (x - x_i) + b * (x - x_i)^2 + c * (x - x_i)^3`.
    pub fn segment_coefficients(&self) -> Vec<[Real; 3]> {
        self.interpolation.segment_coefficients()
    }

    /// Fitted-minus-ordinate residuals at the fixed knot locations.
    ///
    /// These are zero up to floating-point rounding. Use
    /// [`observation_residuals`](Self::observation_residuals) to inspect fit errors
    /// against market observations.
    pub fn node_residuals(&self) -> QlResult<Vec<Real>> {
        self.node_std_dev_points
            .iter()
            .zip(&self.node_mid_ivs)
            .map(|(&point, &ordinate)| Ok(self.volatility_at_std_dev(point)? - ordinate))
            .collect()
    }

    /// Forward used as the smile's ATM level.
    pub fn forward(&self) -> Rate {
        self.forward
    }

    /// Fixed ATM volatility used to normalize strike coordinates.
    pub fn atm_vol(&self) -> Volatility {
        self.atm_vol
    }

    /// Weight on integrated squared curvature; zero disables regularization.
    pub fn smoothing(&self) -> Real {
        self.smoothing
    }

    /// Lower strike at the fitted knot-domain boundary.
    pub fn min_strike(&self) -> Rate {
        self.min_strike
    }

    /// Upper strike at the fitted knot-domain boundary.
    pub fn max_strike(&self) -> Rate {
        self.max_strike
    }

    fn scale(&self) -> Real {
        self.atm_vol * SmileSection::exercise_time(self).sqrt()
    }

    fn endpoint_tolerance(&self) -> Real {
        16.0 * Real::EPSILON
            * self
                .interpolation
                .x_min()
                .abs()
                .max(self.interpolation.x_max().abs())
                .max(1.0)
    }

    /// Snap a query that round-trips to a knot-domain boundary strike.
    ///
    /// `K = F * exp(x * scale)` then `x' = ln(K/F) / scale` is not exact. At a
    /// small ATM volatility the residual can exceed a few ulps in x-space, so
    /// the comparison is made on the strikes.
    fn snapped_endpoint(&self, point: Real) -> Option<Real> {
        let strike = (self.forward.ln() + point * self.scale()).exp();
        if !strike.is_finite() || strike <= 0.0 {
            return None;
        }
        let x_min = self.interpolation.x_min();
        let x_max = self.interpolation.x_max();
        if point <= x_min && close_enough(strike, self.min_strike) {
            Some(x_min)
        } else if point >= x_max && close_enough(strike, self.max_strike) {
            Some(x_max)
        } else {
            None
        }
    }

    fn is_in_or_near_range(&self, point: Real) -> bool {
        if self.snapped_endpoint(point).is_some() {
            return true;
        }
        let tolerance = self.endpoint_tolerance();
        point >= self.interpolation.x_min() - tolerance
            && point <= self.interpolation.x_max() + tolerance
    }

    fn snap_to_domain_boundary(&self, point: Real) -> Real {
        if let Some(endpoint) = self.snapped_endpoint(point) {
            return endpoint;
        }
        let tolerance = self.endpoint_tolerance();
        if point < self.interpolation.x_min() && self.interpolation.x_min() - point <= tolerance {
            self.interpolation.x_min()
        } else if point > self.interpolation.x_max()
            && point - self.interpolation.x_max() <= tolerance
        {
            self.interpolation.x_max()
        } else {
            point
        }
    }
}

/// Fit a fixed-knot natural cubic with a mean-square data term and integrated
/// squared-curvature penalty. Observations outside the knots are excluded.
fn fit_knot_ordinates(
    knots: &[Real],
    observed_x: &[Real],
    observed_y: &[Volatility],
    smoothing: Real,
) -> QlResult<Vec<Volatility>> {
    require!(
        observed_x.len() == observed_y.len(),
        "observation coordinates and IVs must have equal length"
    );
    let knot_count = knots.len();
    let x_min = knots[0];
    let x_max = knots[knot_count - 1];
    let tolerance = 16.0 * Real::EPSILON * x_min.abs().max(x_max.abs()).max(1.0);
    let in_range: Vec<(Real, Volatility)> = observed_x
        .iter()
        .copied()
        .zip(observed_y.iter().copied())
        .filter_map(|(x, y)| {
            if x < x_min - tolerance || x > x_max + tolerance {
                None
            } else {
                Some((x.clamp(x_min, x_max), y))
            }
        })
        .collect();
    let minimum_count = if smoothing > 0.0 { 2 } else { knot_count };
    require!(
        in_range.len() >= minimum_count,
        "fixed-knot cubic fit needs at least {minimum_count} in-range observations, got {}",
        in_range.len()
    );

    // Natural cubic interpolation is linear in its knot ordinates. Interpolating
    // each unit ordinate therefore gives the design matrix for the least-squares
    // fit without introducing any additional spline knots.
    let mut basis = Vec::with_capacity(knot_count);
    for column in 0..knot_count {
        let mut ordinates = vec![0.0; knot_count];
        ordinates[column] = 1.0;
        basis.push(CubicInterpolation::new(
            knots.to_vec(),
            ordinates,
            CubicDerivativeApprox::Spline,
        )?);
    }

    let penalty_rows = if smoothing > 0.0 {
        2 * (knot_count - 1)
    } else {
        0
    };
    let row_count = in_range.len() + penalty_rows;
    let mut design = Matrix::with_size(row_count, knot_count);
    let mut target = Array::with_size(row_count);
    let data_weight = 1.0 / (in_range.len() as Real).sqrt();
    for (row, (x, observed_iv)) in in_range.iter().copied().enumerate() {
        target[row] = observed_iv * data_weight;
        for (column, interpolation) in basis.iter().enumerate() {
            design[(row, column)] = interpolation.value(x)? * data_weight;
        }
    }
    if smoothing > 0.0 {
        // s'' is linear on each cubic segment. Two-point Gauss-Legendre
        // quadrature integrates its square exactly, so these zero-target rows
        // encode lambda * integral(s''^2), including unequal knot spacing.
        // The penalty's nullspace is affine smiles: two distinct quotes fix
        // their level and slope, without fabricated quotes or ATM anchoring.
        let gaussian_offset = 1.0 / 3.0_f64.sqrt();
        for (segment, pair) in knots.windows(2).enumerate() {
            let half_width = (pair[1] - pair[0]) / 2.0;
            let midpoint = pair[0] + half_width;
            let weight = (smoothing * half_width).sqrt();
            for (q, sign) in [-1.0, 1.0].iter().enumerate() {
                let x = midpoint + sign * half_width * gaussian_offset;
                let row = in_range.len() + 2 * segment + q;
                for (column, interpolation) in basis.iter().enumerate() {
                    let value = weight * interpolation.second_derivative(x)?;
                    require!(value.is_finite(), "non-finite curvature penalty");
                    design[(row, column)] = value;
                }
            }
        }
    }

    // Reject a rank-deficient fit rather than returning unstable wing
    // coefficients. Pivoted QR avoids squaring the design matrix's condition
    // number as normal equations would.
    let (_, r, _) = qr_decomposition(&design, true);
    let max_diagonal = (0..knot_count)
        .map(|i| r[(i, i)].abs())
        .fold(0.0_f64, Real::max);
    require!(
        max_diagonal.is_finite() && max_diagonal > 0.0,
        "fixed-knot cubic fit is rank deficient"
    );
    let rank_tolerance = 128.0 * Real::EPSILON * max_diagonal;
    require!(
        (0..knot_count).all(|i| r[(i, i)].abs() > rank_tolerance),
        "fixed-knot cubic fit is rank deficient; use positive smoothing and distinct in-range observations"
    );

    let fitted: Array = qr_solve(&design, &target, true, None);
    require!(
        fitted.iter().all(|ordinate| ordinate.is_finite()),
        "fixed-knot cubic fit produced non-finite ordinates"
    );
    // Implied volatility cannot be negative; floor fitted knot values as well as
    // any between-knot cubic overshoot.
    Ok(fitted.iter().map(|ordinate| ordinate.max(0.0)).collect())
}

impl SmileSection for CubicSmileSection {
    fn base(&self) -> &SmileSectionBase {
        &self.base
    }

    fn volatility_impl(&self, strike: Rate) -> QlResult<Volatility> {
        require!(
            strike.is_finite() && strike > 0.0,
            "strike must be finite and positive"
        );
        let point = (strike.ln() - self.forward.ln()) / self.scale();
        self.volatility_at_std_dev(point)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORWARD: Rate = 100_000.0;
    const EXPIRY: Time = 30.0 / 365.0;
    const ATM_VOL: Volatility = 0.60;

    fn scale() -> Real {
        ATM_VOL * EXPIRY.sqrt()
    }

    fn strike_at(point: Real) -> Rate {
        FORWARD * (point * scale()).exp()
    }

    fn smile(points: &[Real], vols: &[Volatility]) -> CubicSmileSection {
        let mut strikes: Vec<_> = points.iter().copied().map(strike_at).collect();
        let mut mid_ivs = vols.to_vec();
        strikes.reverse();
        mid_ivs.reverse();
        CubicSmileSection::with_smoothing(
            strikes,
            mid_ivs,
            FORWARD,
            EXPIRY,
            ATM_VOL,
            points.to_vec(),
            0.0,
        )
        .unwrap()
    }

    #[test]
    fn sparse_quotes_fit_all_nine_knots_without_fabricating_observations() {
        let points: [Real; 7] = [-1.4, -1.0, -0.6, 0.0, 0.6, 1.0, 1.4];
        let strikes = points.iter().copied().map(strike_at).collect();
        let vols = points.iter().map(|x| 0.3 + 0.02 * x * x).collect();
        let curve = CubicSmileSection::new(strikes, vols, FORWARD, EXPIRY, ATM_VOL).unwrap();
        assert_eq!(curve.node_std_dev_points(), &DEFAULT_STD_DEV_POINTS);
        assert_eq!(curve.segment_coefficients().len(), 8);
        assert_eq!(curve.observed_mid_ivs().len(), 7);
        assert!(
            curve
                .sampled_mid_ivs()
                .unwrap()
                .iter()
                .all(|v| v.unwrap().is_finite())
        );
        assert!(
            curve
                .observation_residuals()
                .unwrap()
                .iter()
                .all(Option::is_some)
        );
    }

    #[test]
    fn two_in_range_quotes_determine_a_linear_nine_knot_smile() {
        let points = [-0.6, 0.6];
        let vols = [0.288, 0.312];
        let curve = CubicSmileSection::new(
            points.iter().copied().map(strike_at).collect(),
            vols.to_vec(),
            FORWARD,
            EXPIRY,
            ATM_VOL,
        )
        .unwrap();
        for &x in &DEFAULT_STD_DEV_POINTS {
            assert!((curve.volatility_at_std_dev(x).unwrap() - (0.3 + 0.02 * x)).abs() < 1e-11);
        }
    }

    #[test]
    fn curvature_penalty_matches_closed_form_three_knot_solution() {
        let knots = [-1.0, 0.0, 1.0];
        let observed = [0.3, 0.5, 0.3];
        let lambda = 0.1;
        let fitted = fit_knot_ordinates(&knots, &knots, &observed, lambda).unwrap();
        // For unit-spaced three-knot natural cubics,
        // integral(s''^2) = 1.5 * (v0 - 2*v1 + v2)^2.
        // Solve (I + 4.5*lambda*c*c^T)v = y independently.
        let contrast = [1.0, -2.0, 1.0];
        let dot = observed
            .iter()
            .zip(contrast)
            .map(|(y, c)| y * c)
            .sum::<Real>();
        for i in 0..3 {
            let expected = observed[i] - 4.5 * lambda / (1.0 + 27.0 * lambda) * contrast[i] * dot;
            assert!((fitted[i] - expected).abs() < 1e-12);
        }
        assert!(fitted[1] < observed[1]);
    }

    #[test]
    fn stronger_smoothing_reduces_curvature_on_uneven_knots() {
        let knots = [-3.0, -1.0, -0.6, 0.0, 1.5, 3.0];
        let observed = [0.5, 0.3, 0.37, 0.29, 0.4, 0.5];
        let energy = |lambda| {
            let values = fit_knot_ordinates(&knots, &knots, &observed, lambda).unwrap();
            let interpolation =
                CubicInterpolation::new(knots.to_vec(), values, CubicDerivativeApprox::Spline)
                    .unwrap();
            interpolation
                .segment_coefficients()
                .iter()
                .zip(knots.windows(2))
                .map(|([_, b, c], pair)| {
                    let h = pair[1] - pair[0];
                    4.0 * b * b * h + 12.0 * b * c * h * h + 12.0 * c * c * h * h * h
                })
                .sum::<Real>()
        };
        assert!(energy(0.1) < energy(0.001));
        assert!(energy(0.001) < energy(0.0));
    }

    #[test]
    fn sparse_fit_requires_two_in_range_quotes_and_valid_smoothing() {
        let strikes = vec![strike_at(-0.6), strike_at(4.0)];
        assert!(CubicSmileSection::new(strikes, vec![0.3; 2], FORWARD, EXPIRY, ATM_VOL).is_err());
        for lambda in [-1.0, Real::NAN, Real::INFINITY, 0.0] {
            assert!(
                CubicSmileSection::with_smoothing(
                    vec![strike_at(-0.6), strike_at(0.6)],
                    vec![0.3; 2],
                    FORWARD,
                    EXPIRY,
                    ATM_VOL,
                    DEFAULT_STD_DEV_POINTS.to_vec(),
                    lambda,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn default_grid_is_the_nine_knot_locations() {
        let points = DEFAULT_STD_DEV_POINTS;
        let vols: Vec<_> = points.iter().map(|x| 0.40 + 0.02 * x).collect();
        let strikes: Vec<_> = points.iter().map(|&point| strike_at(point)).collect();
        let curve =
            CubicSmileSection::new(strikes, vols.clone(), FORWARD, EXPIRY, ATM_VOL).unwrap();

        assert_eq!(curve.std_dev_points(), &DEFAULT_STD_DEV_POINTS);
        assert_eq!(curve.node_std_dev_points(), &DEFAULT_STD_DEV_POINTS);
        for (&fitted, &observed) in curve.node_mid_ivs().iter().zip(&vols) {
            assert!((fitted - observed).abs() < 1e-12);
        }
        let sampled = curve.sampled_mid_ivs().unwrap();
        assert_eq!(sampled.len(), DEFAULT_STD_DEV_POINTS.len());
        assert!(sampled.iter().all(Option::is_some));
        assert!((sampled[0].unwrap() - vols[0]).abs() < 1e-12);
        assert!((sampled[8].unwrap() - vols[8]).abs() < 1e-12);
    }

    #[test]
    fn low_atm_vol_recovers_default_grid_wings() {
        let atm_vol = 0.05;
        let scale = atm_vol * EXPIRY.sqrt();
        let points = DEFAULT_STD_DEV_POINTS;
        let vols: Vec<_> = points.iter().map(|x| 0.40 + 0.02 * x).collect();
        let strikes: Vec<_> = points.iter().map(|x| FORWARD * (x * scale).exp()).collect();
        let curve = CubicSmileSection::new(strikes.clone(), vols.clone(), FORWARD, EXPIRY, atm_vol)
            .unwrap();
        let sampled = curve.sampled_mid_ivs().unwrap();
        assert!((sampled.first().unwrap().unwrap() - vols[0]).abs() < 1e-12);
        assert!((sampled.last().unwrap().unwrap() - vols[8]).abs() < 1e-12);
        assert!((curve.volatility_at_std_dev(-3.0).unwrap() - vols[0]).abs() < 1e-12);
        assert!((curve.volatility_at_std_dev(3.0).unwrap() - vols[8]).abs() < 1e-12);
        assert!((curve.volatility(strikes[0]).unwrap() - vols[0]).abs() < 1e-12);
    }

    #[test]
    fn sorts_source_observations_and_recovers_mid_ivs_at_knots() {
        let points = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let vols = [0.24, 0.22, 0.20, 0.22, 0.24];
        let curve = smile(&points, &vols);

        assert!(curve.node_std_dev_points().windows(2).all(|w| w[0] < w[1]));
        assert!(
            curve
                .observed_std_dev_points()
                .windows(2)
                .all(|w| w[0] < w[1])
        );
        for (&point, &vol) in points.iter().zip(&vols) {
            assert!((curve.volatility_at_std_dev(point).unwrap() - vol).abs() < 1e-13);
            assert!((curve.volatility(strike_at(point)).unwrap() - vol).abs() < 1e-12);
        }
    }

    #[test]
    fn natural_cubic_fit_reproduces_linear_volatility_and_strike_mapping() {
        let points = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let vols: Vec<_> = points.iter().map(|x| 0.30 + 0.02 * x).collect();
        let curve = smile(&points, &vols);
        let query = 0.4;
        assert!(
            (curve.volatility_at_std_dev(query).unwrap() - (0.30 + 0.02 * query)).abs() < 1e-13
        );
        assert!((curve.strike_at_std_dev(query).unwrap() - strike_at(query)).abs() < 1e-10);
        assert_eq!(curve.atm_level(), Some(FORWARD));
    }

    #[test]
    fn exposes_segments_and_distinguishes_knot_from_observation_residuals() {
        let knots = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let source_x = [-3.0, -2.0, -1.5, -1.0, -0.2, 0.4, 1.0, 1.5, 2.0, 3.0];
        let source_y: Vec<_> = source_x
            .iter()
            .enumerate()
            .map(|(i, x)| 0.22 + 0.01 * x + 0.004 * x * x + if i % 2 == 0 { 0.002 } else { -0.001 })
            .collect();
        let strikes: Vec<_> = source_x.iter().copied().map(strike_at).collect();
        let curve = CubicSmileSection::with_std_dev_points(
            strikes,
            source_y,
            FORWARD,
            EXPIRY,
            ATM_VOL,
            knots.to_vec(),
        )
        .unwrap();
        let xs = curve.node_std_dev_points();
        let ys = curve.node_mid_ivs();
        let coefficients = curve.segment_coefficients();

        assert_eq!(coefficients.len(), xs.len() - 1);
        for (i, [a, b, c]) in coefficients.iter().copied().enumerate() {
            let dx = (xs[i + 1] - xs[i]) * 0.37;
            let x = xs[i] + dx;
            let reconstructed = ys[i] + a * dx + b * dx * dx + c * dx * dx * dx;
            assert!((curve.volatility_at_std_dev(x).unwrap() - reconstructed).abs() < 1e-13);
        }
        assert!(
            curve
                .node_residuals()
                .unwrap()
                .iter()
                .all(|error| error.abs() < 1e-13)
        );
        let observation_residuals = curve.observation_residuals().unwrap();
        assert_eq!(observation_residuals.first(), Some(&None));
        assert_eq!(observation_residuals.last(), Some(&None));
        assert!(
            observation_residuals[1..9]
                .iter()
                .flatten()
                .any(|error| error.abs() > 1e-7)
        );
        assert_eq!(curve.observed_strikes().len(), source_x.len());
    }

    #[test]
    fn custom_knots_are_sorted_and_are_the_only_sample_points() {
        let curve = CubicSmileSection::with_std_dev_points(
            vec![strike_at(2.0), strike_at(0.0), strike_at(-2.0)],
            vec![0.24, 0.20, 0.24],
            FORWARD,
            EXPIRY,
            ATM_VOL,
            vec![2.0, 0.0, -2.0],
        )
        .unwrap();
        assert_eq!(curve.std_dev_points(), &[-2.0, 0.0, 2.0]);
        assert_eq!(curve.node_std_dev_points(), &[-2.0, 0.0, 2.0]);
        assert!(curve.sampled_mid_ivs().unwrap().iter().all(Option::is_some));
    }

    #[test]
    fn extrapolation_is_opt_in_and_extends_end_cubics() {
        let curve = smile(&[-1.0, 0.0, 1.0], &[0.20, 0.21, 0.22]);
        assert!(curve.volatility_at_std_dev(1.5).is_err());
        let extended = curve.with_extrapolation(true);
        assert!((extended.volatility_at_std_dev(1.5).unwrap() - 0.225).abs() < 1e-13);
    }

    #[test]
    fn negative_spline_overshoot_is_floored_at_zero() {
        let curve = smile(&[0.0, 1.0], &[0.20, 0.10]).with_extrapolation(true);
        assert_eq!(curve.volatility_at_std_dev(10.0).unwrap(), 0.0);
    }

    #[test]
    fn least_squares_matches_linear_regression_and_ignores_outside_quotes() {
        let knots = [-1.0, 1.0];
        let xs = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let ys = [100.0, 0.28, 0.31, 0.29, 0.34, 0.33, 200.0];
        let fitted = fit_knot_ordinates(&knots, &xs, &ys, DEFAULT_SMILE_SMOOTHING).unwrap();
        // Independent ordinary linear regression: intercept = mean(y),
        // slope = sum(x*y)/sum(x*x), since mean(x) = 0.
        let mean = ys[1..6].iter().sum::<Real>() / 5.0;
        let slope = xs[1..6]
            .iter()
            .zip(&ys[1..6])
            .map(|(x, y)| x * y)
            .sum::<Real>()
            / xs[1..6].iter().map(|x| x * x).sum::<Real>();
        assert!((fitted[0] - (mean - slope)).abs() < 1e-13);
        assert!((fitted[1] - (mean + slope)).abs() < 1e-13);
    }

    #[test]
    fn rejects_rank_deficient_fit_even_with_enough_observations() {
        let xs: Vec<Real> = (0..20).map(|i| -0.5 + i as Real * 0.05).collect();
        let error = fit_knot_ordinates(&DEFAULT_STD_DEV_POINTS, &xs, &vec![0.3; xs.len()], 0.0)
            .unwrap_err();
        assert!(error.to_string().contains("rank deficient"));
    }

    #[test]
    fn tiny_atm_scale_keeps_round_tripped_boundary_observations_in_fit() {
        let atm_vol = 0.005;
        let strikes: Vec<_> = DEFAULT_STD_DEV_POINTS
            .iter()
            .map(|x| FORWARD * (x * atm_vol * EXPIRY.sqrt()).exp())
            .collect();
        let curve =
            CubicSmileSection::new(strikes, vec![0.4; 9], FORWARD, EXPIRY, atm_vol).unwrap();
        assert!(
            curve
                .observation_residuals()
                .unwrap()
                .iter()
                .all(Option::is_some)
        );
    }

    #[test]
    fn rejects_bad_lengths_nodes_and_market_inputs() {
        assert!(CubicSmileSection::new(vec![1.0], vec![0.2], FORWARD, EXPIRY, ATM_VOL).is_err());
        assert!(
            CubicSmileSection::new(vec![1.0, 2.0], vec![0.2], FORWARD, EXPIRY, ATM_VOL).is_err()
        );
        assert!(
            CubicSmileSection::new(vec![1.0, 1.0], vec![0.2, 0.3], FORWARD, EXPIRY, ATM_VOL)
                .is_err()
        );
        assert!(
            CubicSmileSection::new(vec![0.0, 1.0], vec![0.2, 0.3], FORWARD, EXPIRY, ATM_VOL)
                .is_err()
        );
        assert!(
            CubicSmileSection::new(
                vec![1.0, 2.0],
                vec![0.2, f64::NAN],
                FORWARD,
                EXPIRY,
                ATM_VOL
            )
            .is_err()
        );
        assert!(
            CubicSmileSection::new(vec![1.0, 2.0], vec![0.2, -0.1], FORWARD, EXPIRY, ATM_VOL)
                .is_err()
        );
        assert!(
            CubicSmileSection::new(vec![1.0, 2.0], vec![0.2, 0.3], 0.0, EXPIRY, ATM_VOL).is_err()
        );
        assert!(
            CubicSmileSection::new(vec![1.0, 2.0], vec![0.2, 0.3], FORWARD, 0.0, ATM_VOL).is_err()
        );
        assert!(
            CubicSmileSection::new(vec![1.0, 2.0], vec![0.2, 0.3], FORWARD, EXPIRY, 0.0).is_err()
        );
        assert!(
            CubicSmileSection::with_std_dev_points(
                vec![1.0, 2.0],
                vec![0.2, 0.3],
                FORWARD,
                EXPIRY,
                ATM_VOL,
                vec![]
            )
            .is_err()
        );
        assert!(
            CubicSmileSection::with_smoothing(
                vec![strike_at(-1.0), strike_at(1.0)],
                vec![0.2, 0.3],
                FORWARD,
                EXPIRY,
                ATM_VOL,
                vec![-1.0, 0.0, 1.0],
                0.0,
            )
            .is_err()
        );
        assert!(
            CubicSmileSection::with_std_dev_points(
                vec![strike_at(-1.0), strike_at(1.0)],
                vec![0.2, 0.3],
                FORWARD,
                EXPIRY,
                ATM_VOL,
                vec![-1.0, -1.0]
            )
            .is_err()
        );
    }
}
