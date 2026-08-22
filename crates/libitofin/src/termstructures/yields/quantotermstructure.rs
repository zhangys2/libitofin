//! Quanto-adjusted dividend yield curve.
//!
//! Port of `ql/termstructures/yield/quantotermstructure.hpp`: the continuous
//! zero yield is
//!
//! ```text
//! q(t) + r_d(t) − r_f(t) + ρ · σ_eq(t, K) · σ_fx(t, ATM)
//! ```
//!
//! so an equity FD mesher that rolls the forward on this curve sees the
//! quanto-adjusted drift. The structure stays linked to the five input
//! handles.

use super::ZeroYieldStructure;
use crate::errors::QlResult;
use crate::handle::Handle;
use crate::interestrate::Compounding;
use crate::patterns::observable::{AsObservable, Observable, Observer, ResetThenNotify};
use crate::shared::{Shared, SharedMut, shared};
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::termstructures::{TermStructure, TermStructureBase};
use crate::time::calendar::Calendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::frequency::Frequency;
use crate::types::{DiscountFactor, Natural, Rate, Real, Time};

/// Quanto dividend curve (`quantotermstructure.hpp:47`).
pub struct QuantoTermStructure {
    base: Shared<TermStructureBase>,
    underlying_dividend_ts: Handle<dyn YieldTermStructure>,
    risk_free_ts: Handle<dyn YieldTermStructure>,
    foreign_risk_free_ts: Handle<dyn YieldTermStructure>,
    underlying_black_vol_ts: Handle<dyn BlackVolTermStructure>,
    exch_rate_black_vol_ts: Handle<dyn BlackVolTermStructure>,
    strike: Real,
    exch_rate_atm_level: Real,
    underlying_exch_rate_correlation: Real,
    _listener: SharedMut<ResetThenNotify>,
}

impl QuantoTermStructure {
    /// `QuantoTermStructure(underlyingDividendTS, riskFreeTS, foreignRiskFreeTS,
    /// underlyingBlackVolTS, strike, exchRateBlackVolTS, exchRateATMlevel,
    /// underlyingExchRateCorrelation)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        underlying_dividend_ts: Handle<dyn YieldTermStructure>,
        risk_free_ts: Handle<dyn YieldTermStructure>,
        foreign_risk_free_ts: Handle<dyn YieldTermStructure>,
        underlying_black_vol_ts: Handle<dyn BlackVolTermStructure>,
        strike: Real,
        exch_rate_black_vol_ts: Handle<dyn BlackVolTermStructure>,
        exch_rate_atm_level: Real,
        underlying_exch_rate_correlation: Real,
    ) -> QuantoTermStructure {
        let day_counter = underlying_dividend_ts
            .current_link()
            .ok()
            .and_then(|c| c.day_counter());
        let base = shared(TermStructureBase::new(day_counter));
        let listener = ResetThenNotify::delivering(base.updater(), || {});
        let observer = listener.clone() as SharedMut<dyn Observer>;
        underlying_dividend_ts.register_observer(&observer);
        risk_free_ts.register_observer(&observer);
        foreign_risk_free_ts.register_observer(&observer);
        underlying_black_vol_ts.register_observer(&observer);
        exch_rate_black_vol_ts.register_observer(&observer);
        QuantoTermStructure {
            base,
            underlying_dividend_ts,
            risk_free_ts,
            foreign_risk_free_ts,
            underlying_black_vol_ts,
            exch_rate_black_vol_ts,
            strike,
            exch_rate_atm_level,
            underlying_exch_rate_correlation,
            _listener: listener,
        }
    }
}

impl AsObservable for QuantoTermStructure {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl TermStructure for QuantoTermStructure {
    fn base(&self) -> &TermStructureBase {
        &self.base
    }

    fn day_counter(&self) -> Option<DayCounter> {
        self.underlying_dividend_ts
            .current_link()
            .ok()
            .and_then(|c| c.day_counter())
    }

    fn calendar(&self) -> Option<Calendar> {
        self.underlying_dividend_ts
            .current_link()
            .ok()
            .and_then(|c| c.calendar())
    }

    fn settlement_days(&self) -> QlResult<Natural> {
        self.underlying_dividend_ts
            .current_link()?
            .settlement_days()
    }

    fn reference_date(&self) -> QlResult<Date> {
        self.underlying_dividend_ts.current_link()?.reference_date()
    }

    fn max_date(&self) -> Date {
        let dates = [
            self.underlying_dividend_ts
                .current_link()
                .map(|c| c.max_date()),
            self.risk_free_ts.current_link().map(|c| c.max_date()),
            self.foreign_risk_free_ts
                .current_link()
                .map(|c| c.max_date()),
            self.underlying_black_vol_ts
                .current_link()
                .map(|c| c.max_date()),
            self.exch_rate_black_vol_ts
                .current_link()
                .map(|c| c.max_date()),
        ];
        dates.into_iter().flatten().min().unwrap_or_else(Date::null)
    }
}

impl ZeroYieldStructure for QuantoTermStructure {
    fn zero_yield_impl(&self, t: Time) -> QlResult<Rate> {
        let q = self
            .underlying_dividend_ts
            .current_link()?
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, true)?
            .rate();
        let r_d = self
            .risk_free_ts
            .current_link()?
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, true)?
            .rate();
        let r_f = self
            .foreign_risk_free_ts
            .current_link()?
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, true)?
            .rate();
        let eq_vol =
            self.underlying_black_vol_ts
                .current_link()?
                .black_vol(t, self.strike, true)?;
        let fx_vol = self.exch_rate_black_vol_ts.current_link()?.black_vol(
            t,
            self.exch_rate_atm_level,
            true,
        )?;
        Ok(q + r_d - r_f + self.underlying_exch_rate_correlation * eq_vol * fx_vol)
    }
}

impl YieldTermStructure for QuantoTermStructure {
    fn discount_impl(&self, t: Time) -> QlResult<DiscountFactor> {
        self.discount_from_zero_yield(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::shared;
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;

    fn today() -> Date {
        Date::new(22, Month::April, 2019)
    }

    fn flat(rate: Rate) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(vol: Real) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(
            shared(BlackConstantVol::new(today(), None, vol, Actual360::new()))
                as Shared<dyn BlackVolTermStructure>,
        )
    }

    #[test]
    fn zero_yield_is_q_plus_rate_diff_plus_corr_vols() {
        let q = 0.3;
        let r_d = 0.1;
        let r_f = 0.2;
        let eq_vol = 0.3;
        let fx_vol = 0.2;
        let rho = -0.75;
        let curve = QuantoTermStructure::new(
            flat(q),
            flat(r_d),
            flat(r_f),
            flat_vol(eq_vol),
            100.0,
            flat_vol(fx_vol),
            1.0,
            rho,
        );
        let expected = q + r_d - r_f + rho * eq_vol * fx_vol;
        let got = curve
            .zero_rate(1.0, Compounding::Continuous, Frequency::NoFrequency, false)
            .unwrap()
            .rate();
        assert!(
            (got - expected).abs() < 1e-12,
            "got={got} expected={expected}"
        );
    }
}
