//! Forward contract on a bond, valued by the spot-minus-income identity.
//!
//! A [`BondForward`] is an agreement to buy (long) or sell (short) an
//! underlying bond on `delivery_date` for an agreed `strike` cash amount. It is
//! priced analytically off a discount curve, mirroring the value-type approach
//! of [`FxForward`](crate::fxforward::FxForward) rather than going through a
//! pricing engine.
//!
//! With `V` the underlying bond's spot dirty value (the present value of all of
//! its remaining cash flows), `I` the present value of the coupons/redemptions
//! it pays strictly before delivery (which accrue to the current holder, not the
//! forward buyer), `P` the discount factor to delivery and `K` the strike, the
//! fair forward price is `(V - I) / P` and the present value of a long position
//! is `(V - I) - K * P`. QuantLib groups this with `ql/instruments/bondforward`;
//! there is no single cached `test-suite` oracle here, so the behaviour is
//! pinned by the forward-price identities in the tests (cross-checked against
//! the `DiscountingBondEngine` spot value).

use crate::cashflow::Leg;
use crate::cashflows::CashFlows;
use crate::errors::QlResult;
use crate::handle::Handle;
use crate::instruments::{Bond, Position};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::types::Real;

/// A forward contract on a bond.
pub struct BondForward {
    cashflows: Leg,
    discount_curve: Handle<dyn YieldTermStructure>,
    settings: Shared<Settings<Date>>,
    delivery_date: Date,
    strike: Real,
    position: Position,
}

impl BondForward {
    /// Builds a bond forward on `bond`.
    ///
    /// `discount_curve` prices the bond's cash flows; `strike` is the agreed
    /// cash amount exchanged for the bond on `delivery_date`; `position` is the
    /// side that receives the bond and pays the strike (long) at delivery.
    ///
    /// # Errors
    ///
    /// Fails when `strike` is not finite or not positive.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn new(
        bond: &Bond,
        discount_curve: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
        delivery_date: Date,
        strike: Real,
        position: Position,
    ) -> QlResult<Self> {
        require!(strike.is_finite(), "strike must be finite");
        require!(strike > 0.0, "strike must be positive");
        Ok(Self {
            cashflows: bond.cashflows().clone(),
            discount_curve,
            settings,
            delivery_date,
            strike,
            position,
        })
    }

    /// The bond spot dirty value `V`, the pre-delivery income `I` and the
    /// discount factor `P` to delivery.
    fn components(&self) -> QlResult<(Real, Real, Real)> {
        let curve = self.discount_curve.current_link()?;
        let valuation_date = curve.reference_date()?;
        let include_ref = self.settings.include_reference_date_events();

        let spot_value = CashFlows::npv(
            &self.cashflows,
            &*curve,
            &self.settings,
            Some(include_ref),
            Some(valuation_date),
            Some(valuation_date),
        )?;

        // Coupons/redemptions paid strictly before (and up to) delivery accrue
        // to the current holder, so their present value is subtracted from the
        // spot value when forming the forward.
        let mut income = 0.0;
        for flow in &self.cashflows {
            let date = flow.date();
            if date > valuation_date && date <= self.delivery_date {
                income += flow.amount()? * curve.discount_date(date, true)?;
            }
        }

        let discount = curve.discount_date(self.delivery_date, true)?;
        Ok((spot_value, income, discount))
    }

    /// The fair forward price `(V - I) / P` (cash amount at delivery).
    pub fn fair_forward_price(&self) -> QlResult<Real> {
        let (spot_value, income, discount) = self.components()?;
        Ok((spot_value - income) / discount)
    }

    /// The present value of the position.
    ///
    /// A long forward is worth `(V - I) - K * P`; the short side is its
    /// negative.
    pub fn npv(&self) -> QlResult<Real> {
        let (spot_value, income, discount) = self.components()?;
        let long_value = (spot_value - income) - self.strike * discount;
        Ok(match self.position {
            Position::Long => long_value,
            Position::Short => -long_value,
        })
    }

    /// The agreed delivery cash amount.
    pub fn strike(&self) -> Real {
        self.strike
    }

    /// The delivery (settlement) date.
    pub fn delivery_date(&self) -> Date {
        self.delivery_date
    }

    /// The side of the trade.
    pub fn position(&self) -> Position {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instrument::Instrument;
    use crate::instruments::FixedRateBond;
    use crate::interestrate::Compounding;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::bond::DiscountingBondEngine;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::time::schedule::MakeSchedule;

    const RATE: Real = 0.03;

    fn today() -> Date {
        Date::new(15, Month::January, 2020)
    }

    fn settings() -> Shared<Settings<Date>> {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        settings
    }

    fn curve() -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            RATE,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn priced_bond(settings: &Shared<Settings<Date>>) -> FixedRateBond {
        let schedule = MakeSchedule::new()
            .from(today())
            .to(Date::new(15, Month::January, 2025))
            .with_frequency(Frequency::Annual)
            .with_calendar(NullCalendar::new())
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .backwards()
            .build();
        let mut bond = FixedRateBond::new(
            0,
            100.0,
            schedule,
            vec![0.05],
            Actual360::new(),
            BusinessDayConvention::Unadjusted,
            100.0,
            Some(today()),
            None,
            None,
            NullCalendar::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(settings),
        )
        .unwrap();
        let engine = shared_mut(DiscountingBondEngine::new(
            curve(),
            None,
            Shared::clone(settings),
        ));
        bond.bond_mut()
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        bond
    }

    #[test]
    fn striking_at_the_fair_price_gives_zero_value() {
        let settings = settings();
        let bond = priced_bond(&settings);
        let delivery = Date::new(15, Month::July, 2022);

        let probe = BondForward::new(
            bond.bond(),
            curve(),
            Shared::clone(&settings),
            delivery,
            100.0,
            Position::Long,
        )
        .unwrap();
        let fair = probe.fair_forward_price().unwrap();

        let at_fair = BondForward::new(
            bond.bond(),
            curve(),
            Shared::clone(&settings),
            delivery,
            fair,
            Position::Long,
        )
        .unwrap();
        assert!(
            at_fair.npv().unwrap().abs() < 1e-9,
            "value at the fair price should vanish, got {}",
            at_fair.npv().unwrap()
        );
    }

    #[test]
    fn income_free_forward_equals_spot_over_discount() {
        // Delivery before the first coupon (2021-01-15): no intermediate income,
        // so fair * P must equal the bond's spot value from the engine.
        let settings = settings();
        let mut bond = priced_bond(&settings);
        let spot_value = bond.bond_mut().npv().unwrap();

        let delivery = Date::new(15, Month::July, 2020);
        let fwd = BondForward::new(
            bond.bond(),
            curve(),
            Shared::clone(&settings),
            delivery,
            100.0,
            Position::Long,
        )
        .unwrap();

        let discount = curve()
            .current_link()
            .unwrap()
            .discount_date(delivery, true)
            .unwrap();
        assert!(
            (fwd.fair_forward_price().unwrap() * discount - spot_value).abs() < 1e-9,
            "fair {} * P {discount} vs spot {spot_value}",
            fwd.fair_forward_price().unwrap()
        );
    }

    #[test]
    fn long_and_short_values_are_opposite() {
        let settings = settings();
        let bond = priced_bond(&settings);
        let delivery = Date::new(15, Month::July, 2022);

        let long = BondForward::new(
            bond.bond(),
            curve(),
            Shared::clone(&settings),
            delivery,
            90.0,
            Position::Long,
        )
        .unwrap();
        let short = BondForward::new(
            bond.bond(),
            curve(),
            Shared::clone(&settings),
            delivery,
            90.0,
            Position::Short,
        )
        .unwrap();

        let long_npv = long.npv().unwrap();
        assert!((long_npv + short.npv().unwrap()).abs() < 1e-12);
        // A strike below the fair price is favourable to the long.
        assert!(long.fair_forward_price().unwrap() > 90.0);
        assert!(
            long_npv > 0.0,
            "long value should be positive here: {long_npv}"
        );
    }

    #[test]
    fn a_non_positive_strike_is_rejected() {
        let settings = settings();
        let bond = priced_bond(&settings);
        let result = BondForward::new(
            bond.bond(),
            curve(),
            Shared::clone(&settings),
            Date::new(15, Month::July, 2022),
            0.0,
            Position::Long,
        );
        let Err(err) = result else {
            panic!("a non-positive strike must be rejected");
        };
        assert!(err.message().contains("strike must be positive"));
    }
}
