//! Discrete cash-dividend step condition on an equity finite-difference grid.
//!
//! Port of `ql/methods/finitedifferences/utilities/fdmdividendhandler.{hpp,cpp}`.
//! At each dividend time the option values are linearly interpolated from
//! `S` onto `S − D` (clamped at the lowest grid spot), which is the jump
//! `S ↦ S − D` written on a log-spot mesh.

use crate::cashflows::Dividend;
use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::linear::LinearInterpolation;
use crate::methods::finitedifferences::StepCondition;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{Real, Size, Time};

/// Applies discrete cash dividends as a step condition along one equity
/// direction (`fdmdividendhandler.hpp:37`).
pub struct FdmDividendHandler {
    /// Grid equity values in physical units (`x_`, `hpp:52`).
    x: Vec<Real>,
    dividend_times: Vec<Time>,
    dividend_dates: Vec<Date>,
    dividends: Vec<Real>,
    mesher: Shared<dyn FdmMesher>,
    equity_direction: Size,
}

impl FdmDividendHandler {
    /// `FdmDividendHandler(schedule, mesher, referenceDate, dayCounter, equityDirection)`
    /// (`cpp:30-54`).
    ///
    /// # Errors
    ///
    /// Returns an error if any dividend's [`amount`](crate::cashflow::CashFlow::amount)
    /// fails.
    pub fn new(
        schedule: &[Shared<dyn Dividend>],
        mesher: Shared<dyn FdmMesher>,
        reference_date: Date,
        day_counter: DayCounter,
        equity_direction: Size,
    ) -> QlResult<Self> {
        let n = mesher.layout().dim()[equity_direction];
        let mut dividend_times = Vec::with_capacity(schedule.len());
        let mut dividend_dates = Vec::with_capacity(schedule.len());
        let mut dividends = Vec::with_capacity(schedule.len());
        for dividend in schedule {
            dividends.push(dividend.amount()?);
            dividend_dates.push(dividend.date());
            dividend_times.push(day_counter.year_fraction(reference_date, dividend.date()));
        }

        let tmp = mesher.locations(equity_direction);
        let spacing = mesher.layout().spacing()[equity_direction];
        let mut x = vec![0.0; n];
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = tmp[i * spacing].exp();
        }

        Ok(Self {
            x,
            dividend_times,
            dividend_dates,
            dividends,
            mesher,
            equity_direction,
        })
    }

    /// The dividend times (`cpp:56-58`).
    pub fn dividend_times(&self) -> &[Time] {
        &self.dividend_times
    }

    /// The dividend dates (`cpp:60-62`).
    pub fn dividend_dates(&self) -> &[Date] {
        &self.dividend_dates
    }

    /// The cash amounts (`cpp:64-66`).
    pub fn dividends(&self) -> &[Real] {
        &self.dividends
    }

    fn interpolate_shift(&self, y: &[Real]) -> LinearInterpolation {
        LinearInterpolation::new(self.x.clone(), y.to_vec())
            .expect("dividend interpolation needs a populated equity grid")
            .with_extrapolation(true)
    }

    fn shifted_value(&self, interp: &LinearInterpolation, k: Size, dividend: Real) -> Real {
        let query = self.x[0].max(self.x[k] - dividend);
        interp
            .value(query)
            .expect("clamped query is inside the extrapolated equity grid")
    }
}

impl StepCondition for FdmDividendHandler {
    /// `cpp:68-106`: on an exact dividend time, rewrite each equity slice as
    /// the linearly interpolated value at `max(S_min, S − D)`.
    #[allow(clippy::float_cmp)]
    fn apply_to(&self, a: &mut Array, t: Time) {
        let Some(pos) = self.dividend_times.iter().position(|&ti| ti == t) else {
            return;
        };
        let dividend = self.dividends[pos];
        let a_copy = a.clone();

        if self.mesher.layout().dim().len() == 1 {
            let interp = self.interpolate_shift(&a_copy);
            for k in 0..self.x.len() {
                a[k] = self.shifted_value(&interp, k, dividend);
            }
            return;
        }

        let x_spacing = self.mesher.layout().spacing()[self.equity_direction];
        for (i, &y_spacing) in self.mesher.layout().spacing().iter().enumerate() {
            if i == self.equity_direction {
                continue;
            }
            for j in 0..self.mesher.layout().dim()[i] {
                let mut tmp = vec![0.0; self.x.len()];
                for k in 0..self.x.len() {
                    tmp[k] = a_copy[j * y_spacing + k * x_spacing];
                }
                let interp = self.interpolate_shift(&tmp);
                for k in 0..self.x.len() {
                    a[j * y_spacing + k * x_spacing] = self.shifted_value(&interp, k, dividend);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflows::FixedDividend;
    use crate::methods::finitedifferences::meshers::{FdmMesher, UniformGridMesher};
    use crate::methods::finitedifferences::operators::FdmLinearOpLayout;
    use crate::shared::shared;
    use crate::time::date::Month;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;

    fn log_mesher(spots: &[Real]) -> Shared<dyn FdmMesher> {
        let logs: Vec<(Real, Real)> = {
            let lo = spots[0].ln();
            let hi = spots[spots.len() - 1].ln();
            vec![(lo, hi)]
        };
        let layout = shared(FdmLinearOpLayout::new(vec![spots.len()]));
        shared(UniformGridMesher::new(layout, &logs).unwrap())
    }

    #[test]
    fn times_and_amounts_follow_the_schedule() {
        let today = Date::new(11, Month::February, 2018);
        let dc = Actual365Fixed::new();
        let mesher = log_mesher(&[80.0, 100.0, 120.0]);
        let div_date = Date::new(11, Month::August, 2018);
        let schedule: Vec<Shared<dyn Dividend>> = vec![shared(FixedDividend::new(30.0, div_date))];
        let handler = FdmDividendHandler::new(&schedule, mesher, today, dc.clone(), 0).unwrap();

        assert_eq!(handler.dividends(), &[30.0]);
        assert_eq!(handler.dividend_dates(), &[div_date]);
        assert!((handler.dividend_times()[0] - dc.year_fraction(today, div_date)).abs() < 1e-14);
    }

    #[test]
    fn a_cash_dividend_rewrites_values_at_s_minus_d() {
        let today = Date::new(11, Month::February, 2018);
        let dc = Actual365Fixed::new();
        let mesher = log_mesher(&[80.0, 100.0, 120.0]);
        let handler = FdmDividendHandler::new(
            &[
                shared(FixedDividend::new(20.0, Date::new(11, Month::August, 2018)))
                    as Shared<dyn Dividend>,
            ],
            Shared::clone(&mesher),
            today,
            dc,
            0,
        )
        .unwrap();

        // Identity payoff V(S) = S. After a cash drop D the grid is rewritten
        // as V(max(S_min, S − D)), which for V = S is that same argument.
        let spots: Vec<Real> = mesher.locations(0).iter().map(|x| x.exp()).collect();
        let mut values = Array::from(spots.clone());
        handler.apply_to(&mut values, handler.dividend_times()[0]);

        for (k, &spot) in spots.iter().enumerate() {
            let expected = spots[0].max(spot - 20.0);
            assert!(
                (values[k] - expected).abs() < 1e-12,
                "S={spot}: {} vs {expected}",
                values[k]
            );
        }
    }

    #[test]
    fn a_time_that_is_not_a_dividend_date_is_a_no_op() {
        let today = Date::new(11, Month::February, 2018);
        let dc = Actual365Fixed::new();
        let mesher = log_mesher(&[80.0, 100.0, 120.0]);
        let handler = FdmDividendHandler::new(
            &[
                shared(FixedDividend::new(20.0, Date::new(11, Month::August, 2018)))
                    as Shared<dyn Dividend>,
            ],
            mesher,
            today,
            dc,
            0,
        )
        .unwrap();

        let mut values = Array::from([1.0, 2.0, 3.0]);
        handler.apply_to(&mut values, 0.0);
        assert_eq!(values, Array::from([1.0, 2.0, 3.0]));
    }
}
