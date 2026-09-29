//! A natural cubic smile interpolated in standardized log-moneyness.
//!
//! The source nodes are market mid implied volatilities at strikes. Each strike
//! is mapped to signed standard-deviation units using one fixed ATM volatility:
//! `x = (ln(strike) - ln(forward)) / (atm_vol * sqrt(exercise_time))`. The
//! natural cubic spline passes through every source node exactly. Its default
//! sample grid is intended for reporting/plotting and is not the knot grid.
//!
//! This is a one-expiry interpolation only. It does not smooth noisy quotes,
//! consolidate duplicate strikes, or enforce static arbitrage.

use crate::errors::QlResult;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::cubic::{CubicDerivativeApprox, CubicInterpolation};
use crate::termstructures::volatility::VolatilityType;
use crate::termstructures::volatility::smilesection::{SmileSection, SmileSectionBase};
use crate::time::daycounters::actual365fixed::Actual365Fixed;
use crate::types::{Rate, Real, Time, Volatility};
use crate::{fail, require};

/// Default standard-deviation sample points, from the put wing through ATM to
/// the call wing. These are evaluation points, not spline nodes.
pub const DEFAULT_STD_DEV_POINTS: [Real; 9] = [-3.0, -1.5, -1.0, -0.6, 0.0, 0.6, 1.0, 1.5, 3.0];

/// One-expiry Black volatility smile fitted exactly through mid-IV observations.
///
/// The natural cubic spline is constructed in standardized log-moneyness, while
/// [`SmileSection::volatility`] accepts strikes. [`sampled_mid_ivs`](Self::sampled_mid_ivs)
/// evaluates the configured report grid; points outside the observed domain are
/// `None` unless extrapolation is enabled.
pub struct CubicSmileSection {
    base: SmileSectionBase,
    interpolation: CubicInterpolation,
    forward: Rate,
    atm_vol: Volatility,
    std_dev_points: Vec<Real>,
    node_std_dev_points: Vec<Real>,
    node_mid_ivs: Vec<Volatility>,
    min_strike: Rate,
    max_strike: Rate,
}

impl CubicSmileSection {
    /// Construct a natural cubic smile with the default report grid.
    ///
    /// `mid_ivs` are annualized decimal Black implied volatilities, paired with
    /// `strikes`. The input observations are sorted by their standardized
    /// coordinate before the spline is built. Call/put observations at the same
    /// strike must be consolidated by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal input lengths, fewer than two observations,
    /// non-finite or non-positive forward/time/ATM volatility/strikes, negative
    /// or non-finite mid-IVs, invalid sample points, or duplicate standardized
    /// source coordinates.
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

    /// Construct a natural cubic smile with a caller-specified sample grid.
    ///
    /// Sample points are preserved in their input order and may lie outside the
    /// observed source-node range. Such points produce `None` in
    /// [`sampled_mid_ivs`](Self::sampled_mid_ivs) unless extrapolation is
    /// enabled on the returned section.
    pub fn with_std_dev_points(
        strikes: Vec<Rate>,
        mid_ivs: Vec<Volatility>,
        forward: Rate,
        exercise_time: Time,
        atm_vol: Volatility,
        std_dev_points: Vec<Real>,
    ) -> QlResult<Self> {
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
            !std_dev_points.is_empty(),
            "std_dev_points must not be empty"
        );
        for &point in &std_dev_points {
            require!(
                point.is_finite(),
                "sample standard-deviation points must be finite"
            );
        }

        let scale = atm_vol * exercise_time.sqrt();
        require!(
            scale.is_finite() && scale > 0.0,
            "ATM standard deviation must be finite and positive"
        );

        let mut nodes = Vec::with_capacity(strikes.len());
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
            nodes.push((x, strike, mid_iv));
        }
        nodes.sort_by(|left, right| left.0.total_cmp(&right.0));

        let node_std_dev_points: Vec<Real> = nodes.iter().map(|node| node.0).collect();
        let node_mid_ivs: Vec<Volatility> = nodes.iter().map(|node| node.2).collect();
        let min_strike = nodes[0].1;
        let max_strike = nodes[nodes.len() - 1].1;
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
            std_dev_points,
            node_std_dev_points,
            node_mid_ivs,
            min_strike,
            max_strike,
        })
    }

    /// Set whether queries beyond the observed standard-deviation range extend
    /// the end cubic segments. Extrapolation is disabled by default.
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

    /// Evaluate the configured sample grid. Values outside the observed domain
    /// are `None` when extrapolation is disabled.
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

    /// Configured sample/evaluation points, in caller order.
    pub fn std_dev_points(&self) -> &[Real] {
        &self.std_dev_points
    }

    /// Sorted standardized coordinates of the observed strike/IV nodes.
    pub fn node_std_dev_points(&self) -> &[Real] {
        &self.node_std_dev_points
    }

    /// Mid-IV values paired with [`node_std_dev_points`](Self::node_std_dev_points).
    pub fn node_mid_ivs(&self) -> &[Volatility] {
        &self.node_mid_ivs
    }

    /// Forward used as the smile's ATM level.
    pub fn forward(&self) -> Rate {
        self.forward
    }

    /// Fixed ATM volatility used to normalize strike coordinates.
    pub fn atm_vol(&self) -> Volatility {
        self.atm_vol
    }

    /// Lower observed strike.
    pub fn min_strike(&self) -> Rate {
        self.min_strike
    }

    /// Upper observed strike.
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

    fn is_in_or_near_range(&self, point: Real) -> bool {
        let tolerance = self.endpoint_tolerance();
        point >= self.interpolation.x_min() - tolerance
            && point <= self.interpolation.x_max() + tolerance
    }

    fn snap_to_domain_boundary(&self, point: Real) -> Real {
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
        CubicSmileSection::new(strikes, mid_ivs, FORWARD, EXPIRY, ATM_VOL).unwrap()
    }

    #[test]
    fn default_sample_grid_matches_requested_standard_deviation_points() {
        let curve = smile(&[-3.0, -2.0, 0.0, 2.0, 3.0], &[0.8, 0.6, 0.5, 0.6, 0.8]);
        assert_eq!(curve.std_dev_points(), &DEFAULT_STD_DEV_POINTS);
        assert_eq!(
            curve.sampled_mid_ivs().unwrap().len(),
            DEFAULT_STD_DEV_POINTS.len()
        );
    }

    #[test]
    fn sorts_source_nodes_and_recovers_mid_ivs_at_observed_strikes() {
        let points = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let vols = [0.24, 0.22, 0.20, 0.22, 0.24];
        let curve = smile(&points, &vols);

        assert!(curve.node_std_dev_points().windows(2).all(|w| w[0] < w[1]));
        for (&point, &vol) in points.iter().zip(&vols) {
            assert!((curve.volatility_at_std_dev(point).unwrap() - vol).abs() < 1e-14);
            assert!((curve.volatility(strike_at(point)).unwrap() - vol).abs() < 1e-13);
        }
    }

    #[test]
    fn natural_cubic_spline_reproduces_linear_volatility_and_strike_mapping() {
        let points = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let vols: Vec<_> = points.iter().map(|x| 0.30 + 0.02 * x).collect();
        let curve = smile(&points, &vols);
        let query = 0.4;
        assert!(
            (curve.volatility_at_std_dev(query).unwrap() - (0.30 + 0.02 * query)).abs() < 1e-14
        );
        assert!((curve.strike_at_std_dev(query).unwrap() - strike_at(query)).abs() < 1e-10);
        assert_eq!(curve.atm_level(), Some(FORWARD));
    }

    #[test]
    fn custom_sample_points_preserve_order_and_mark_unavailable_wings() {
        let curve = CubicSmileSection::with_std_dev_points(
            vec![strike_at(1.0), strike_at(-1.0), strike_at(0.0)],
            vec![0.22, 0.22, 0.20],
            FORWARD,
            EXPIRY,
            ATM_VOL,
            vec![2.0, 0.0, -2.0],
        )
        .unwrap();
        assert_eq!(curve.std_dev_points(), &[2.0, 0.0, -2.0]);
        assert_eq!(
            curve.sampled_mid_ivs().unwrap(),
            vec![None, Some(0.20), None]
        );
    }

    #[test]
    fn extrapolation_is_opt_in_and_extends_end_cubics() {
        let points = [-1.0, 0.0, 1.0];
        let vols = [0.20, 0.21, 0.22];
        let curve = smile(&points, &vols);
        assert!(curve.volatility_at_std_dev(1.5).is_err());
        let extended = curve.with_extrapolation(true);
        assert!((extended.volatility_at_std_dev(1.5).unwrap() - 0.225).abs() < 1e-14);
        assert!(
            extended
                .sampled_mid_ivs()
                .unwrap()
                .iter()
                .all(Option::is_some)
        );
    }

    #[test]
    fn negative_spline_overshoot_is_floored_at_zero() {
        let curve = smile(&[0.0, 1.0], &[0.20, 0.10]).with_extrapolation(true);
        assert_eq!(curve.volatility_at_std_dev(10.0).unwrap(), 0.0);
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
    }
}
