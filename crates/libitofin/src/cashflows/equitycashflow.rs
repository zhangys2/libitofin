//! Equity cash flow.
//!
//! Port of `ql/cashflows/equitycashflow.{hpp,cpp}`: [`EquityCashFlow`] is an
//! [`IndexedCashFlow`] over an [`EquityIndex`].

use super::indexedcashflow::IndexedCashFlow;
use crate::cashflow::CashFlow;
use crate::cashflows::Coupon;
use crate::errors::QlResult;
use crate::event::Event;
use crate::indexes::EquityIndex;
use crate::patterns::observable::{AsObservable, Observable};
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Cash flow dependent on an equity index ratio (`EquityCashFlow`).
pub struct EquityCashFlow {
    base: IndexedCashFlow<EquityIndex>,
}

impl EquityCashFlow {
    /// Builds an equity cash flow paying `notional` scaled by the equity growth
    /// (or total ratio) between `base_date` and `fixing_date`.
    pub fn new(
        notional: Real,
        index: Shared<EquityIndex>,
        base_date: Date,
        fixing_date: Date,
        payment_date: Date,
        growth_only: bool,
    ) -> Self {
        EquityCashFlow {
            base: IndexedCashFlow::new(
                notional,
                index,
                base_date,
                fixing_date,
                payment_date,
                growth_only,
            ),
        }
    }

    /// The notional the ratio scales.
    pub fn notional(&self) -> Real {
        self.base.notional()
    }

    /// The equity index the ratio is taken on.
    pub fn index(&self) -> &Shared<EquityIndex> {
        self.base.index()
    }

    /// The date the base fixing is read at.
    pub fn base_date(&self) -> Date {
        self.base.base_date()
    }

    /// The date the index fixing is read at.
    pub fn fixing_date(&self) -> Date {
        self.base.fixing_date()
    }

    /// The date the cash flow is paid.
    pub fn payment_date(&self) -> Date {
        self.base.date()
    }

    /// Whether the flow pays growth $I_1 / I_0 - 1$ instead of $I_1 / I_0$.
    pub fn growth_only(&self) -> bool {
        self.base.growth_only()
    }

    /// The fixing at base date.
    pub fn base_fixing(&self) -> QlResult<Real> {
        self.base.base_fixing()
    }

    /// The fixing at fixing date.
    pub fn index_fixing(&self) -> QlResult<Real> {
        self.base.index_fixing()
    }
}

impl AsObservable for EquityCashFlow {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl Event for EquityCashFlow {
    fn date(&self) -> Date {
        self.base.date()
    }

    fn has_occurred(
        &self,
        settings: &Settings<Date>,
        ref_date: Option<Date>,
        include_ref_date: Option<bool>,
    ) -> QlResult<bool> {
        self.base.has_occurred(settings, ref_date, include_ref_date)
    }
}

impl CashFlow for EquityCashFlow {
    fn amount(&self) -> QlResult<Real> {
        self.base.amount()
    }

    fn ex_coupon_date(&self) -> Option<Date> {
        None
    }

    fn as_coupon(&self) -> Option<&dyn Coupon> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::currency::Currency;
    use crate::handle::Handle;
    use crate::indexes::index::Index;
    use crate::shared::shared;
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::date::Month;

    #[test]
    fn test_equity_cash_flow_growth_only() {
        let settings = shared(Settings::<Date>::new());
        let today = Date::new(4, Month::July, 2023);
        settings.set_evaluation_date(today);

        let index = shared(EquityIndex::new(
            "EQ_TEST",
            NullCalendar::new(),
            Currency::usd(),
            Handle::empty(),
            Handle::empty(),
            Handle::empty(),
            Shared::clone(&settings),
        ));

        let d0 = Date::new(1, Month::January, 2023);
        let d1 = Date::new(1, Month::July, 2023);
        let pay_date = Date::new(3, Month::July, 2023);

        index.add_fixing(d0, 100.0).unwrap();
        index.add_fixing(d1, 120.0).unwrap();

        let notional = 10_000.0;
        let flow = EquityCashFlow::new(
            notional,
            Shared::clone(&index),
            d0,
            d1,
            pay_date,
            true, // growth only
        );

        assert_eq!(flow.notional(), notional);
        assert_eq!(flow.base_date(), d0);
        assert_eq!(flow.fixing_date(), d1);
        assert_eq!(flow.payment_date(), pay_date);
        assert!(flow.growth_only());
        assert_eq!(flow.base_fixing().unwrap(), 100.0);
        assert_eq!(flow.index_fixing().unwrap(), 120.0);

        // Expected amount = 10,000 * (120/100 - 1) = 2,000.0
        let amount = flow.amount().unwrap();
        assert!((amount - 2000.0).abs() < 1e-10);
    }

    #[test]
    fn test_equity_cash_flow_full_ratio() {
        let settings = shared(Settings::<Date>::new());
        let today = Date::new(4, Month::July, 2023);
        settings.set_evaluation_date(today);

        let index = shared(EquityIndex::new(
            "EQ_TEST2",
            NullCalendar::new(),
            Currency::usd(),
            Handle::empty(),
            Handle::empty(),
            Handle::empty(),
            Shared::clone(&settings),
        ));

        let d0 = Date::new(1, Month::January, 2023);
        let d1 = Date::new(1, Month::July, 2023);
        let pay_date = Date::new(3, Month::July, 2023);

        index.add_fixing(d0, 100.0).unwrap();
        index.add_fixing(d1, 120.0).unwrap();

        let notional = 10_000.0;
        let flow = EquityCashFlow::new(
            notional,
            Shared::clone(&index),
            d0,
            d1,
            pay_date,
            false, // full ratio
        );

        // Expected amount = 10,000 * (120/100) = 12,000.0
        let amount = flow.amount().unwrap();
        assert!((amount - 12000.0).abs() < 1e-10);
    }
}
