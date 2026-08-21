//! Present-value cash-dividend adjustment of the escrowed-dividend model.
//!
//! Port of `ql/methods/finitedifferences/utilities/escroweddividendadjustment.{hpp,cpp}`:
//! the remaining cash dividends in `[t, maturity]` are pulled out of the spot
//! as
//! `-Σ Dᵢ · P(t, tᵢ) · Q(t) / Q(tᵢ)`,
//! which is the C++ `dividendAdjustment(t)`. A finite-difference engine that
//! prices on this prepaid process then evaluates at `S₀ + dividendAdjustment(0)`
//! and does not apply discrete dividend jumps.

use crate::cashflows::Dividend;
use crate::errors::QlResult;
use crate::handle::Handle;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::types::{Real, Time};

/// Escrowed-dividend spot adjustment (`escroweddividendadjustment.hpp:36`).
pub struct EscrowedDividendAdjustment {
    dividend_schedule: Vec<Shared<dyn Dividend>>,
    r_ts: Handle<dyn YieldTermStructure>,
    q_ts: Handle<dyn YieldTermStructure>,
    to_time: Box<dyn Fn(Date) -> QlResult<Time>>,
    maturity: Time,
}

impl EscrowedDividendAdjustment {
    /// `EscrowedDividendAdjustment(schedule, rTS, qTS, toTime, maturity)`.
    pub fn new(
        dividend_schedule: Vec<Shared<dyn Dividend>>,
        r_ts: Handle<dyn YieldTermStructure>,
        q_ts: Handle<dyn YieldTermStructure>,
        to_time: impl Fn(Date) -> QlResult<Time> + 'static,
        maturity: Time,
    ) -> Self {
        Self {
            dividend_schedule,
            r_ts,
            q_ts,
            to_time: Box::new(to_time),
            maturity,
        }
    }

    /// Remaining-dividend PV to subtract from the spot at time `t`
    /// (`cpp:40-53`). Dividends strictly before `t` or after `maturity` are
    /// skipped.
    pub fn dividend_adjustment(&self, t: Time) -> QlResult<Real> {
        let r_ts = self.r_ts.current_link()?;
        let q_ts = self.q_ts.current_link()?;
        let mut div_adj = 0.0;
        for dividend in &self.dividend_schedule {
            let div_time = (self.to_time)(dividend.date())?;
            if div_time >= t && div_time <= self.maturity {
                div_adj -= dividend.amount()? * r_ts.discount(div_time, false)?
                    / r_ts.discount(t, false)?
                    * q_ts.discount(t, false)?
                    / q_ts.discount(div_time, false)?;
            }
        }
        Ok(div_adj)
    }

    /// The risk-free curve the adjustment discounts on.
    pub fn risk_free_rate(&self) -> Handle<dyn YieldTermStructure> {
        self.r_ts.clone()
    }

    /// The dividend-yield curve that grows the escrowed cash.
    pub fn dividend_yield(&self) -> Handle<dyn YieldTermStructure> {
        self.q_ts.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflows::FixedDividend;
    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::shared::{Shared, shared};
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(11, Month::November, 2025)
    }

    fn flat(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn to_time(r_ts: Handle<dyn YieldTermStructure>) -> impl Fn(Date) -> QlResult<Time> {
        move |d| r_ts.current_link()?.time_from_reference(d)
    }

    #[test]
    fn one_cash_dividend_is_the_prepaid_forward_drop() {
        let r = 0.025;
        let q = 0.05;
        let r_ts = flat(r);
        let q_ts = flat(q);
        let div_date = today() + 182;
        let amount = 5.0;
        let schedule =
            vec![shared(FixedDividend::new(amount, div_date))
                as Shared<dyn crate::cashflows::Dividend>];
        let maturity = Actual365Fixed::new().year_fraction(today(), today() + 365);
        let adj = EscrowedDividendAdjustment::new(
            schedule,
            r_ts.clone(),
            q_ts.clone(),
            to_time(r_ts.clone()),
            maturity,
        );

        let t_div = r_ts
            .current_link()
            .unwrap()
            .time_from_reference(div_date)
            .unwrap();
        let expected = -amount * (-r * t_div).exp() / (-q * t_div).exp();
        let calculated = adj.dividend_adjustment(0.0).unwrap();
        assert!(
            (calculated - expected).abs() < 1e-12,
            "adj={calculated} expected={expected}"
        );
        assert!(calculated < 0.0);
    }

    #[test]
    fn dividends_after_maturity_or_before_t_are_skipped() {
        let r_ts = flat(0.03);
        let q_ts = flat(0.01);
        let maturity = 1.0;
        let past = today() - 10;
        let future = today() + 400;
        let inside = today() + 180;
        let schedule = vec![
            shared(FixedDividend::new(5.0, past)) as Shared<dyn crate::cashflows::Dividend>,
            shared(FixedDividend::new(5.0, inside)) as Shared<dyn crate::cashflows::Dividend>,
            shared(FixedDividend::new(5.0, future)) as Shared<dyn crate::cashflows::Dividend>,
        ];
        let adj = EscrowedDividendAdjustment::new(
            schedule,
            r_ts.clone(),
            q_ts,
            to_time(r_ts.clone()),
            maturity,
        );
        let only_inside = EscrowedDividendAdjustment::new(
            vec![shared(FixedDividend::new(5.0, inside)) as Shared<dyn crate::cashflows::Dividend>],
            adj.risk_free_rate(),
            adj.dividend_yield(),
            to_time(r_ts),
            maturity,
        );
        assert!(
            (adj.dividend_adjustment(0.0).unwrap() - only_inside.dividend_adjustment(0.0).unwrap())
                .abs()
                < 1e-12
        );
        let t_inside = adj
            .risk_free_rate()
            .current_link()
            .unwrap()
            .time_from_reference(inside)
            .unwrap();
        assert!(
            (adj.dividend_adjustment(t_inside + 1e-8).unwrap()).abs() < 1e-12,
            "dividends strictly before t must drop out"
        );
    }
}
