//! Zero-coupon bond.
//!
//! Port of `ql/instruments/bonds/zerocouponbond.{hpp,cpp}`.

use super::super::bond::Bond;
use crate::errors::QlResult;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendar::Calendar;
use crate::time::date::Date;
use crate::types::{Natural, Real};

/// A bond paying a single redemption and no coupons.
pub struct ZeroCouponBond {
    bond: Bond,
}

impl ZeroCouponBond {
    /// Builds a zero-coupon bond redeeming `face_amount * redemption / 100` on
    /// the calendar-adjusted maturity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        settlement_days: Natural,
        calendar: Calendar,
        face_amount: Real,
        maturity_date: Date,
        payment_convention: BusinessDayConvention,
        redemption: Real,
        issue_date: Option<Date>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let redemption_date = calendar.adjust(maturity_date, payment_convention);
        let mut bond = Bond::new(settlement_days, calendar, issue_date, Vec::new(), settings)?;
        bond.set_maturity_date(maturity_date);
        bond.set_single_redemption(face_amount, redemption, redemption_date)?;
        Ok(Self { bond })
    }

    /// The underlying [`Bond`] base.
    pub fn bond(&self) -> &Bond {
        &self.bond
    }

    /// Mutable access to the underlying [`Bond`] base.
    pub fn bond_mut(&mut self) -> &mut Bond {
        &mut self.bond
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::interestrate::Compounding;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::DiscountingBondEngine;
    use crate::shared::{shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    #[test]
    fn zero_coupon_bond_prices_below_par_on_a_positive_curve() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let maturity = Date::new(15, Month::June, 2031);
        let mut zcb = ZeroCouponBond::new(
            2,
            Target::new(),
            100.0,
            maturity,
            BusinessDayConvention::Following,
            100.0,
            Some(today),
            Shared::clone(&settings),
        )
        .unwrap();
        let curve = Handle::new(shared(FlatForward::with_rate(
            today,
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let engine = shared_mut(DiscountingBondEngine::new(
            curve,
            None,
            Shared::clone(&settings),
        )) as crate::shared::SharedMut<dyn PricingEngine>;
        zcb.bond_mut().base_mut().set_pricing_engine(engine);
        let npv = zcb.bond_mut().npv().unwrap();
        assert!(npv > 0.0 && npv < 100.0, "NPV={npv}");
    }
}
