//! Digital (cash-or-nothing) Ibor coupon.
//!
//! Pays a fixed cash amount when the Ibor fixing is above (call) or below (put)
//! a strike. Full QuantLib `DigitalIborCoupon` replication/collar coverage is
//! follow-up; this slice covers the cash-or-nothing rate decision.

use crate::cashflows::IborCoupon;
use crate::errors::QlResult;
use crate::indexes::IborIndex;
use crate::indexes::index::Index;
use crate::shared::Shared;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{Natural, Rate, Real, Spread};

/// A cash-or-nothing digital coupon on an Ibor fixing.
pub struct DigitalIborCoupon {
    underlying: IborCoupon,
    call_strike: Option<Rate>,
    put_strike: Option<Rate>,
    cash_rate: Rate,
}

impl DigitalIborCoupon {
    /// Builds a digital coupon that pays `cash_rate` when the fixing is ITM
    /// relative to the supplied call and/or put strike.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        payment_date: Date,
        nominal: Real,
        start_date: Date,
        end_date: Date,
        fixing_days: Natural,
        index: Shared<IborIndex>,
        gearing: Real,
        spread: Spread,
        day_counter: DayCounter,
        call_strike: Option<Rate>,
        put_strike: Option<Rate>,
        cash_rate: Rate,
    ) -> QlResult<Self> {
        let underlying = IborCoupon::new(
            payment_date,
            nominal,
            start_date,
            end_date,
            Some(fixing_days),
            index,
            gearing,
            spread,
            Some(start_date),
            Some(end_date),
            Some(day_counter),
            false,
            None,
            BusinessDayConvention::Preceding,
        )?;
        Ok(Self {
            underlying,
            call_strike,
            put_strike,
            cash_rate,
        })
    }

    /// Digital rate: `cash_rate` if ITM, else 0.
    pub fn digital_rate(&self) -> QlResult<Rate> {
        let fixing = Index::fixing(
            &**self.underlying.ibor_index(),
            self.underlying.fixing_date(),
            false,
        )?;
        let call_itm = self.call_strike.map(|k| fixing > k).unwrap_or(false);
        let put_itm = self.put_strike.map(|k| fixing < k).unwrap_or(false);
        Ok(if call_itm || put_itm {
            self.cash_rate
        } else {
            0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::indexes::ibor::Euribor;
    use crate::interestrate::Compounding;
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    #[test]
    fn digital_call_pays_cash_when_fixing_exceeds_strike() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let curve = Handle::new(shared(FlatForward::with_rate(
            today,
            0.05,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let index = shared(
            Euribor::new(
                Period::new(6, TimeUnit::Months),
                curve,
                Shared::clone(&settings),
            )
            .unwrap(),
        );
        let start = Date::new(15, Month::September, 2026);
        let end = Date::new(15, Month::March, 2027);
        let coupon = DigitalIborCoupon::new(
            end,
            1_000_000.0,
            start,
            end,
            2,
            index,
            1.0,
            0.0,
            Actual360::new(),
            Some(0.01),
            None,
            0.02,
        )
        .unwrap();
        assert_eq!(coupon.digital_rate().unwrap(), 0.02);
    }
}
