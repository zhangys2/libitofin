//! Implied Black vol term structure at a future reference date.
//!
//! Port of `ql/termstructures/volatility/equityfx/impliedvoltermstructure.hpp`:
//! re-bases another [`BlackVolTermStructure`] so that time `t` is measured from
//! a future reference date, with variance taken as the original forward
//! variance between the time shift and `time_shift + t`.

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::patterns::observable::{AsObservable, Observable, Observer, ResetThenNotify};
use crate::shared::{Shared, SharedMut, shared};
use crate::termstructures::volatility::{BlackVolTermStructure, VolatilityTermStructure};
use crate::termstructures::{TermStructure, TermStructureBase};
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendar::Calendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{Natural, Rate, Real, Time};

/// Implied vol term structure at a given date in the future.
///
/// The given date is the implied reference date. Day counter, calendar,
/// settlement days, strike domain and maximum date come from the underlying.
pub struct ImpliedVolTermStructure {
    base: Shared<TermStructureBase>,
    original: Handle<dyn BlackVolTermStructure>,
    _listener: SharedMut<ResetThenNotify>,
}

impl ImpliedVolTermStructure {
    /// Re-bases `original` to `reference_date`, registering with the handle
    /// and adopting the underlying curve's extrapolation setting.
    pub fn new(
        original: Handle<dyn BlackVolTermStructure>,
        reference_date: Date,
    ) -> ImpliedVolTermStructure {
        let base = shared(TermStructureBase::with_reference_date(
            reference_date,
            None,
            None,
        ));
        sync_extrapolation(&base, &original);
        let listener = ResetThenNotify::delivering(base.updater(), {
            let base = Shared::clone(&base);
            let original = original.clone();
            move || {
                sync_extrapolation(&base, &original);
            }
        });
        original.register_observer(&(listener.clone() as SharedMut<dyn Observer>));
        ImpliedVolTermStructure {
            base,
            original,
            _listener: listener,
        }
    }
}

fn sync_extrapolation(base: &TermStructureBase, original: &Handle<dyn BlackVolTermStructure>) {
    if let Ok(original) = original.current_link() {
        if original.allows_extrapolation() {
            base.enable_extrapolation();
        } else {
            base.disable_extrapolation();
        }
    }
}

impl AsObservable for ImpliedVolTermStructure {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl TermStructure for ImpliedVolTermStructure {
    fn base(&self) -> &TermStructureBase {
        &self.base
    }

    fn max_date(&self) -> Date {
        self.original
            .current_link()
            .map(|curve| curve.max_date())
            .unwrap_or_else(|_| Date::null())
    }

    fn day_counter(&self) -> Option<DayCounter> {
        self.original
            .current_link()
            .ok()
            .and_then(|curve| curve.day_counter())
    }

    fn calendar(&self) -> Option<Calendar> {
        self.original
            .current_link()
            .ok()
            .and_then(|curve| curve.calendar())
    }

    fn settlement_days(&self) -> QlResult<Natural> {
        self.original.current_link()?.settlement_days()
    }
}

impl VolatilityTermStructure for ImpliedVolTermStructure {
    fn business_day_convention(&self) -> BusinessDayConvention {
        self.original
            .current_link()
            .map(|c| c.business_day_convention())
            .unwrap_or(BusinessDayConvention::Following)
    }

    fn min_strike(&self) -> Rate {
        self.original
            .current_link()
            .map(|c| c.min_strike())
            .unwrap_or(Rate::MIN)
    }

    fn max_strike(&self) -> Rate {
        self.original
            .current_link()
            .map(|c| c.max_strike())
            .unwrap_or(Rate::MAX)
    }
}

impl BlackVolTermStructure for ImpliedVolTermStructure {
    fn black_vol_impl(&self, t: Time, strike: Real) -> QlResult<Real> {
        let var = self.black_variance_impl(t, strike)?;
        let t = if t < 1.0e-5 { 1.0e-5 } else { t };
        Ok((var / t).sqrt())
    }

    fn black_variance_impl(&self, t: Time, strike: Real) -> QlResult<Real> {
        let original = self.original.current_link()?;
        let day_counter = self.require_day_counter()?;
        let time_shift =
            day_counter.year_fraction(original.reference_date()?, self.reference_date()?);
        // `t` is relative to the implied reference; convert to original time.
        original.black_forward_variance(time_shift, time_shift + t, strike, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::shared;
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    #[test]
    fn implied_vol_matches_forward_variance_of_underlying() {
        let vol = 0.20;
        let original =
            Handle::new(
                shared(BlackConstantVol::new(today(), None, vol, Actual360::new()))
                    as Shared<dyn BlackVolTermStructure>,
            );
        let reset = today() + 90;
        let implied = ImpliedVolTermStructure::new(original.clone(), reset);
        let t = 0.25;
        let strike = 100.0;
        let calculated = implied.black_variance(t, strike, true).unwrap();
        let orig = original.current_link().unwrap();
        let dc = orig.day_counter().unwrap();
        let shift = dc.year_fraction(today(), reset);
        let expected = orig
            .black_forward_variance(shift, shift + t, strike, true)
            .unwrap();
        assert!(
            (calculated - expected).abs() < 1e-14,
            "implied var {calculated} vs forward {expected}"
        );
        let calc_vol = implied.black_vol(t, strike, true).unwrap();
        assert!((calc_vol - vol).abs() < 1e-12);
    }
}
