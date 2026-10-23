//! Bilinear local-volatility grid.
//!
//! QuantLib's `FixedLocalVolSurface`: a matrix of local volatilities on a
//! strictly increasing time/strike grid, interpolated bilinearly and held
//! flat outside the grid. The SLV-JD particle calibration stores the
//! leverage function `L(t, K)` here.

use crate::errors::QlResult;
use crate::patterns::observable::{AsObservable, Observable};
use crate::require;
use crate::termstructures::volatility::VolatilityTermStructure;
use crate::termstructures::{TermStructure, TermStructureBase};
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{Rate, Real, Time, Volatility};

use super::LocalVolTermStructure;

/// Local volatility tabulated on `(time, strike)` and interpolated bilinearly.
pub struct InterpolatedLocalVolSurface {
    base: TermStructureBase,
    times: Vec<Time>,
    strikes: Vec<Real>,
    local_vols: crate::math::matrix::Matrix,
}

impl InterpolatedLocalVolSurface {
    /// Grid with `local_vols` rows aligned to `times` and columns to `strikes`.
    ///
    /// # Errors
    ///
    /// Returns an error when a grid axis has fewer than two nodes, times are
    /// not strictly increasing or start negative, strikes are not strictly
    /// increasing or start non-positive, the matrix shape does not match, or
    /// a local volatility is non-positive or non-finite.
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        times: Vec<Time>,
        strikes: Vec<Real>,
        local_vols: crate::math::matrix::Matrix,
    ) -> QlResult<Self> {
        require!(
            times.len() >= 2,
            "interpolated local vol needs at least two times, got {}",
            times.len()
        );
        require!(
            strikes.len() >= 2,
            "interpolated local vol needs at least two strikes, got {}",
            strikes.len()
        );
        require!(
            local_vols.rows() == times.len() && local_vols.columns() == strikes.len(),
            "local-vol matrix is {}x{}, expected {}x{}",
            local_vols.rows(),
            local_vols.columns(),
            times.len(),
            strikes.len()
        );
        require!(
            times.iter().all(|time| time.is_finite()) && times[0] >= 0.0,
            "local-vol times must be finite and start at or after 0"
        );
        require!(
            times.windows(2).all(|pair| pair[1] > pair[0]),
            "local-vol times must be strictly increasing"
        );
        require!(
            strikes.iter().all(|strike| strike.is_finite()) && strikes[0] > 0.0,
            "local-vol strikes must be finite and positive"
        );
        require!(
            strikes.windows(2).all(|pair| pair[1] > pair[0]),
            "local-vol strikes must be strictly increasing"
        );
        for row in 0..times.len() {
            for column in 0..strikes.len() {
                let vol = local_vols[(row, column)];
                require!(
                    vol.is_finite() && vol > 0.0,
                    "local vol at time {} and strike {} must be positive and finite, got {vol}",
                    times[row],
                    strikes[column]
                );
            }
        }
        Ok(Self {
            base: TermStructureBase::with_reference_date(reference_date, None, Some(day_counter)),
            times,
            strikes,
            local_vols,
        })
    }

    /// Time nodes.
    pub fn times(&self) -> &[Time] {
        &self.times
    }

    /// Strike nodes.
    pub fn strikes(&self) -> &[Real] {
        &self.strikes
    }

    /// Local-volatility matrix, one row per time and one column per strike.
    pub fn matrix(&self) -> &crate::math::matrix::Matrix {
        &self.local_vols
    }

    /// Bracket `x` on `knots` and return `(left_index, weight of the right node)`.
    ///
    /// Outside the knots the weight sits on the nearest boundary node, which
    /// is the flat extrapolation.
    fn bracket(knots: &[Real], x: Real) -> (usize, Real) {
        if x <= knots[0] {
            return (0, 0.0);
        }
        let last = knots.len() - 1;
        if x >= knots[last] {
            return (last - 1, 1.0);
        }
        let right = knots.partition_point(|knot| *knot < x);
        let left = right - 1;
        let width = knots[right] - knots[left];
        (left, (x - knots[left]) / width)
    }
}

impl AsObservable for InterpolatedLocalVolSurface {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl TermStructure for InterpolatedLocalVolSurface {
    fn base(&self) -> &TermStructureBase {
        &self.base
    }

    fn max_date(&self) -> Date {
        Date::max_date()
    }

    /// Last tabulated time.
    ///
    /// The grid ends here. [`local_vol_impl`](LocalVolTermStructure::local_vol_impl)
    /// still holds the boundary knot flat past this time, but only once the
    /// caller allows extrapolation (`local_vol(..., true)` or
    /// [`TermStructure::allows_extrapolation`](crate::termstructures::TermStructure::allows_extrapolation)).
    /// A query that does not extrapolate stops at this node. That is the same
    /// split as QuantLib's `FixedLocalVolSurface`: `maxDate` is the last knot,
    /// and flat values beyond it are extrapolation.
    fn max_time(&self) -> QlResult<Time> {
        Ok(self.times[self.times.len() - 1])
    }
}

impl VolatilityTermStructure for InterpolatedLocalVolSurface {
    fn business_day_convention(&self) -> BusinessDayConvention {
        BusinessDayConvention::Following
    }

    fn min_strike(&self) -> Rate {
        self.strikes[0]
    }

    fn max_strike(&self) -> Rate {
        self.strikes[self.strikes.len() - 1]
    }
}

impl LocalVolTermStructure for InterpolatedLocalVolSurface {
    /// Bilinear value on the grid. Times and strikes outside the knots are
    /// clamped to the boundary, so this hook answers them. The term-structure
    /// range check still rejects those queries unless extrapolation is on;
    /// see [`TermStructure::max_time`](crate::termstructures::TermStructure::max_time).
    fn local_vol_impl(&self, t: Time, strike: Real) -> QlResult<Volatility> {
        let (time_left, time_weight) = Self::bracket(&self.times, t);
        let (strike_left, strike_weight) = Self::bracket(&self.strikes, strike);
        let v00 = self.local_vols[(time_left, strike_left)];
        let v01 = self.local_vols[(time_left, strike_left + 1)];
        let v10 = self.local_vols[(time_left + 1, strike_left)];
        let v11 = self.local_vols[(time_left + 1, strike_left + 1)];
        let along_low = (1.0 - strike_weight) * v00 + strike_weight * v01;
        let along_high = (1.0 - strike_weight) * v10 + strike_weight * v11;
        Ok((1.0 - time_weight) * along_low + time_weight * along_high)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::math::matrix::Matrix;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;

    #[test]
    fn rejects_bad_grids_and_non_positive_vols() {
        let day_counter = Actual365Fixed::new();
        let today = Date::new(15, Month::June, 2026);
        let ok = Matrix::from([[0.2, 0.3], [0.4, 0.5]]);
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                day_counter,
                vec![1.0],
                vec![100.0, 120.0],
                Matrix::from([[0.2, 0.3]]),
            )
            .is_err()
        );
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                Actual365Fixed::new(),
                vec![0.0, 1.0],
                vec![100.0],
                Matrix::from([[0.2], [0.3]]),
            )
            .is_err()
        );
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                Actual365Fixed::new(),
                vec![1.0, 0.5],
                vec![100.0, 120.0],
                ok.clone(),
            )
            .is_err()
        );
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                Actual365Fixed::new(),
                vec![-0.1, 1.0],
                vec![100.0, 120.0],
                ok.clone(),
            )
            .is_err()
        );
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                Actual365Fixed::new(),
                vec![0.0, 1.0],
                vec![120.0, 100.0],
                ok.clone(),
            )
            .is_err()
        );
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                Actual365Fixed::new(),
                vec![0.0, 1.0],
                vec![0.0, 100.0],
                ok.clone(),
            )
            .is_err()
        );
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                Actual365Fixed::new(),
                vec![0.0, 1.0],
                vec![100.0, 120.0],
                Matrix::from([[0.2, 0.3]]),
            )
            .is_err()
        );
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                Actual365Fixed::new(),
                vec![0.0, 1.0],
                vec![100.0, 120.0],
                Matrix::from([[0.2, 0.0], [0.3, 0.4]]),
            )
            .is_err()
        );
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                Actual365Fixed::new(),
                vec![0.0, 1.0],
                vec![100.0, 120.0],
                Matrix::from([[0.2, -0.1], [0.3, 0.4]]),
            )
            .is_err()
        );
        assert!(
            InterpolatedLocalVolSurface::new(
                today,
                Actual365Fixed::new(),
                vec![0.0, 1.0],
                vec![100.0, 120.0],
                Matrix::from([[0.2, Real::NAN], [0.3, 0.4]]),
            )
            .is_err()
        );
    }

    #[test]
    fn bilinear_on_knots_between_knots_and_flat_outside() {
        let curve = InterpolatedLocalVolSurface::new(
            Date::new(15, Month::June, 2026),
            Actual365Fixed::new(),
            vec![0.5, 2.0],
            vec![100.0, 200.0],
            Matrix::from([[0.10, 0.30], [0.20, 0.60]]),
        )
        .unwrap();
        let at = |t, k| curve.local_vol(t, k, true).unwrap();

        assert!((at(0.5, 100.0) - 0.10).abs() <= 1e-14);
        assert!((at(0.5, 200.0) - 0.30).abs() <= 1e-14);
        assert!((at(2.0, 100.0) - 0.20).abs() <= 1e-14);
        assert!((at(2.0, 200.0) - 0.60).abs() <= 1e-14);
        // Midway in time and strike: equal weight on all four knots.
        assert!((at(1.25, 150.0) - 0.30).abs() <= 1e-14);
        assert!((at(0.0, 50.0) - 0.10).abs() <= 1e-14);
        assert!((curve.local_vol_impl(-1.0, 50.0).unwrap() - 0.10).abs() <= 1e-14);
        assert!((at(10.0, 1_000.0) - 0.60).abs() <= 1e-14);
        assert!((curve.max_time().unwrap() - 2.0).abs() <= 1e-14);
        assert!(curve.local_vol(1.0, 50.0, false).is_err());
        assert!(curve.local_vol(3.0, 150.0, false).is_err());
    }
}
