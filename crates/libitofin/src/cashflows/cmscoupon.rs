//! CMS coupon.
//!
//! Initial port of the CMS coupon surface: a floating coupon whose index is a
//! [`SwapIndex`]. Convexity-adjusted CMS pricers (Hagan, etc.) are follow-up;
//! the rate here is the raw swap-index fixing times gearing plus spread when no
//! pricer is attached.

use crate::cashflows::{Coupon, CouponBase, FloatingRateCoupon};
use crate::errors::QlResult;
use crate::indexes::SwapIndex;
use crate::indexes::index::Index;
use crate::patterns::observable::{AsObservable, Observable};
use crate::shared::Shared;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{Natural, Rate, Real, Spread};

/// A coupon paying a constant-maturity swap rate.
pub struct CmsCoupon {
    base: FloatingRateCoupon,
    swap_index: Shared<SwapIndex>,
}

impl CmsCoupon {
    /// Builds a CMS coupon on `swap_index`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        payment_date: Date,
        nominal: Real,
        start_date: Date,
        end_date: Date,
        fixing_days: Natural,
        swap_index: Shared<SwapIndex>,
        gearing: Real,
        spread: Spread,
        ref_period_start: Date,
        ref_period_end: Date,
        day_counter: DayCounter,
        is_in_arrears: bool,
    ) -> QlResult<Self> {
        let base = FloatingRateCoupon::new(
            payment_date,
            nominal,
            start_date,
            end_date,
            Some(fixing_days),
            Shared::clone(&swap_index),
            gearing,
            spread,
            Some(ref_period_start),
            Some(ref_period_end),
            Some(day_counter),
            is_in_arrears,
            None,
            BusinessDayConvention::Preceding,
        )?;
        Ok(Self { base, swap_index })
    }

    /// The underlying swap index.
    pub fn swap_index(&self) -> &Shared<SwapIndex> {
        &self.swap_index
    }

    /// Raw CMS rate without a convexity-adjusted pricer: gearing × fixing + spread.
    pub fn raw_rate(&self) -> QlResult<Rate> {
        let fixing = Index::fixing(&*self.swap_index, self.base.fixing_date(), false)?;
        Ok(self.base.gearing() * fixing + self.base.spread())
    }
}

impl AsObservable for CmsCoupon {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl Coupon for CmsCoupon {
    fn coupon_base(&self) -> &CouponBase {
        self.base.coupon_base()
    }

    fn amount(&self) -> QlResult<Real> {
        let rate = self.rate()?;
        Ok(self.nominal() * rate * self.accrual_period())
    }

    fn rate(&self) -> QlResult<Rate> {
        match self.base.rate() {
            Ok(rate) => Ok(rate),
            Err(_) => self.raw_rate(),
        }
    }

    fn day_counter(&self) -> DayCounter {
        self.base.day_counter()
    }

    fn accrued_amount(&self, date: Date) -> QlResult<Real> {
        self.base.accrued_amount(date)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::currency::Currency;
    use crate::handle::Handle;
    use crate::indexes::ibor::Euribor;
    use crate::interestrate::Compounding;
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    #[test]
    fn cms_raw_rate_reads_the_swap_index_fixing() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let curve = Handle::new(shared(FlatForward::with_rate(
            today,
            0.03,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let ibor = shared(Euribor::six_months(curve.clone(), Shared::clone(&settings)));
        let swap_index = shared(SwapIndex::new(
            "EuriborSwap".into(),
            Period::new(10, TimeUnit::Years),
            2,
            Currency::eur(),
            Target::new(),
            Period::new(1, TimeUnit::Years),
            BusinessDayConvention::Unadjusted,
            Thirty360::with_convention(Convention::BondBasis),
            ibor,
            Shared::clone(&settings),
        ));
        let start = Date::new(15, Month::September, 2026);
        let end = Date::new(15, Month::December, 2026);
        let coupon = CmsCoupon::new(
            end,
            100.0,
            start,
            end,
            2,
            Shared::clone(&swap_index),
            1.0,
            0.0,
            start,
            end,
            Thirty360::with_convention(Convention::BondBasis),
            false,
        )
        .unwrap();
        let rate = coupon.raw_rate().unwrap();
        assert!(rate.is_finite());
        assert!((rate - 0.03).abs() < 5e-3, "rate={rate}");
    }
}
