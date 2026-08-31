//! Implied Black vol term structure at a future reference date.
//!
//! Port of `ql/termstructures/volatility/equityfx/impliedvoltermstructure.hpp`:
//! re-bases a [`BlackVolTermStructure`] to a future reference date via forward
//! variance, remaining linked to the original structure.

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::patterns::observable::{AsObservable, Observable, Observer, ResetThenNotify};
use crate::shared::{Shared, SharedMut, shared};
use crate::termstructures::TermStructure;
use crate::termstructures::TermStructureBase;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{Rate, Real, Time, Volatility};

use super::BlackVolTermStructure;
use super::VolatilityTermStructure;

fn sync_extrapolation(base: &TermStructureBase, original: &Handle<dyn BlackVolTermStructure>) {
    if let Ok(original) = original.current_link() {
        if original.allows_extrapolation() {
            base.enable_extrapolation();
        } else {
            base.disable_extrapolation();
        }
    }
}

/// Implied vol term structure at a given date in the future.
pub struct ImpliedVolTermStructure {
    base: Shared<TermStructureBase>,
    original: Handle<dyn BlackVolTermStructure>,
    _listener: SharedMut<ResetThenNotify>,
}

impl ImpliedVolTermStructure {
    /// Re-bases `original` to `reference_date`, registering with the handle.
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
            move || sync_extrapolation(&base, &original)
        });
        original.register_observer(&(listener.clone() as SharedMut<dyn Observer>));
        ImpliedVolTermStructure {
            base,
            original,
            _listener: listener,
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
            .map(|ts| ts.max_date())
            .unwrap_or_else(|_| Date::null())
    }

    fn day_counter(&self) -> Option<DayCounter> {
        self.original
            .current_link()
            .ok()
            .and_then(|ts| ts.day_counter())
    }
}

impl VolatilityTermStructure for ImpliedVolTermStructure {
    fn business_day_convention(&self) -> BusinessDayConvention {
        self.original
            .current_link()
            .map(|ts| ts.business_day_convention())
            .unwrap_or(BusinessDayConvention::Following)
    }

    fn min_strike(&self) -> Rate {
        self.original
            .current_link()
            .map(|ts| ts.min_strike())
            .unwrap_or(Rate::MIN)
    }

    fn max_strike(&self) -> Rate {
        self.original
            .current_link()
            .map(|ts| ts.max_strike())
            .unwrap_or(Rate::MAX)
    }
}

impl BlackVolTermStructure for ImpliedVolTermStructure {
    fn black_vol_impl(&self, t: Time, strike: Real) -> QlResult<Volatility> {
        let non_zero = if t == 0.0 { 0.00001 } else { t };
        let variance = self.black_variance_impl(non_zero, strike)?;
        Ok((variance / non_zero).sqrt())
    }

    fn black_variance_impl(&self, t: Time, strike: Real) -> QlResult<Real> {
        let original = self.original.current_link()?;
        let day_counter = self.require_day_counter()?;
        let time_shift =
            day_counter.year_fraction(original.reference_date()?, self.reference_date()?);
        original.black_forward_variance(time_shift, time_shift + t, strike, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    #[test]
    fn implied_forward_variance_matches_the_underlying_curve() {
        let flat = BlackConstantVol::new(today(), None, 0.2, Actual360::new());
        let original = Handle::new(shared(flat) as Shared<dyn BlackVolTermStructure>);
        let reset = today() + 90;
        let maturity = today() + 180;
        let implied = ImpliedVolTermStructure::new(original.clone(), reset);

        let strike = 100.0;
        let time_shift = Actual360::new().year_fraction(today(), reset);
        let t = Actual360::new().year_fraction(reset, maturity);
        let expected = original
            .current_link()
            .unwrap()
            .black_forward_variance(time_shift, time_shift + t, strike, false)
            .unwrap();
        let actual = implied.black_variance(t, strike, false).unwrap();
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "implied {actual} vs underlying forward {expected}"
        );
    }
}
