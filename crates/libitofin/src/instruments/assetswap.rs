//! Par asset swap: a bond's coupons exchanged for a floating (Ibor) leg.
//!
//! An [`AssetSwap`] packages a bond with an interest-rate swap: the bond leg
//! carries the bond's coupons plus its redemption, and the floating leg pays an
//! Ibor index plus an asset-swap spread on par notional, with an upfront
//! (`dirtyPrice − 100`) at the start and a par backpayment at maturity. Leg and
//! upfront construction follow QuantLib's `ql/instruments/assetswap.cpp`
//! (par-swap path); it reuses the generic [`Swap`] base and prices through a
//! swap engine.
//!
//! This slice implements the **par** asset swap (`parSwap = true`) with unit
//! gearing and a par (100) redemption, over a caller-supplied floating
//! schedule running to the bond's maturity. Non-par / market asset swaps,
//! partial deal maturities and overnight legs are follow-ups. There is no
//! cached `test-suite` oracle wired here, so the behaviour is pinned by the
//! identities in the tests (the fair spread zeroes the value; a bond quoted at
//! its on-curve forward price asset-swaps at ~zero spread).

use crate::cashflow::{CashFlow, Leg};
use crate::cashflows::{IborLeg, SimpleCashFlow};
use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::indexes::IborIndex;
use crate::instrument::{Instrument, InstrumentBase};
use crate::instruments::{Bond, Swap};
use crate::pricingengine::{Arguments, Results};
use crate::settings::Settings;
use crate::shared::{Shared, shared};
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::schedule::Schedule;
use crate::types::{Real, Spread};

const BASIS_POINT: Real = 1.0e-4;

/// A par asset swap. Leg 0 is the bond leg, leg 1 the floating leg.
pub struct AssetSwap {
    swap: Swap,
    spread: Spread,
}

impl AssetSwap {
    /// Builds a par asset swap on `bond` quoted at `bond_clean_price` (per 100,
    /// the forward clean price at the floating schedule's start).
    ///
    /// `pay_bond_coupon` chooses the side: `true` pays the bond coupons and
    /// receives the floating leg. The floating leg runs over `float_schedule`
    /// paying `ibor_index` + `spread` on the bond's notional.
    ///
    /// # Errors
    ///
    /// Propagates leg-building and [`Swap`] construction errors, and any bond
    /// accessor error.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pay_bond_coupon: bool,
        bond: &Bond,
        bond_clean_price: Real,
        ibor_index: Shared<IborIndex>,
        spread: Spread,
        float_schedule: Schedule,
        floating_day_counter: DayCounter,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<AssetSwap> {
        let payment_adjustment = BusinessDayConvention::Following;
        let deal_maturity = float_schedule.end_date();
        let final_date = float_schedule
            .calendar()
            .adjust(deal_maturity, payment_adjustment);
        let upfront_date = float_schedule.start_date();

        let dirty_price = bond_clean_price + bond.accrued_amount(Some(upfront_date))?;
        let notional = bond.notional(Some(upfront_date))?;

        // Bond leg: the bond's coupons that fall after the upfront date and on
        // or before the deal maturity, then a par redemption at the final date
        // (`assetswap.cpp:97-135`, par path with the default redemption).
        let bond_cashflows = bond.cashflows();
        let count = bond_cashflows.len();
        let mut bond_leg: Leg = Vec::new();
        for flow in &bond_cashflows[..count.saturating_sub(1)] {
            let date = flow.date();
            if date <= deal_maturity
                && !event_has_occurred(date, &settings, Some(upfront_date), Some(false))?
            {
                bond_leg.push(Shared::clone(flow));
            }
        }
        let redemption_amount = bond_cashflows[count - 1].amount()?;
        bond_leg.push(
            shared(SimpleCashFlow::new(redemption_amount, final_date)?) as Shared<dyn CashFlow>
        );

        // Floating leg: Ibor + spread on `notional`, bracketed by the upfront
        // and the par backpayment (`assetswap.cpp:137-173`, par path).
        let mut float_leg = IborLeg::new(float_schedule, ibor_index)
            .with_notional(notional)
            .with_payment_adjustment(payment_adjustment)
            .with_spreads(vec![spread])
            .with_payment_day_counter(floating_day_counter)
            .build()?;
        let upfront = (dirty_price - 100.0) / 100.0 * notional;
        float_leg.insert(
            0,
            shared(SimpleCashFlow::new(upfront, upfront_date)?) as Shared<dyn CashFlow>,
        );
        float_leg.push(shared(SimpleCashFlow::new(notional, final_date)?) as Shared<dyn CashFlow>);

        // payBondCoupon = pay the bond leg (-1), receive the floating leg (+1).
        let payer = vec![pay_bond_coupon, !pay_bond_coupon];
        let swap = Swap::new(vec![bond_leg, float_leg], payer, settings)?;
        Ok(AssetSwap { swap, spread })
    }

    /// The generic swap this wraps.
    pub fn swap(&self) -> &Swap {
        &self.swap
    }

    /// The bond leg's NPV.
    pub fn bond_leg_npv(&mut self) -> QlResult<Real> {
        self.swap.leg_npv(0)
    }

    /// The floating leg's NPV.
    pub fn floating_leg_npv(&mut self) -> QlResult<Real> {
        self.swap.leg_npv(1)
    }

    /// The floating leg's BPS.
    pub fn floating_leg_bps(&mut self) -> QlResult<Real> {
        self.swap.leg_bps(1)
    }

    /// The asset-swap spread that zeroes the swap NPV
    /// (`assetswap.cpp:236-247`: `spread − NPV / legBPS[1] · 1bp`).
    ///
    /// # Errors
    ///
    /// The engine must have priced the swap and provided the floating leg BPS.
    pub fn fair_spread(&mut self) -> QlResult<Spread> {
        self.calculate()?;
        let npv = self.npv()?;
        let bps = self.swap.leg_bps(1)?;
        Ok(self.spread - npv / bps * BASIS_POINT)
    }
}

impl Instrument for AssetSwap {
    fn base(&self) -> &InstrumentBase {
        self.swap.base()
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        self.swap.base_mut()
    }

    fn is_expired(&self) -> QlResult<bool> {
        self.swap.is_expired()
    }

    fn setup_expired(&mut self) {
        self.swap.setup_expired();
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        self.swap.setup_arguments(arguments)
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        self.swap.fetch_results(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::indexes::ibor::Euribor;
    use crate::instruments::FixedRateBond;
    use crate::interestrate::Compounding;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::bond::DiscountingBondEngine;
    use crate::pricingengines::swap::DiscountingSwapEngine;
    use crate::shared::{SharedMut, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;
    use crate::time::schedule::MakeSchedule;
    use crate::time::timeunit::TimeUnit;

    const FACE: Real = 100.0;
    const COUPON: Real = 0.04;
    const RATE: Real = 0.03;

    fn today() -> Date {
        Date::new(15, Month::January, 2020)
    }

    fn settings() -> Shared<Settings<Date>> {
        let s = shared(Settings::new());
        s.set_evaluation_date(today());
        s
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

    fn settlement() -> Date {
        Target::new().advance(
            today(),
            2,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        )
    }

    fn bond_schedule() -> Schedule {
        MakeSchedule::new()
            .from(settlement())
            .to(Target::new().advance(
                settlement(),
                6,
                TimeUnit::Years,
                BusinessDayConvention::Following,
                false,
            ))
            .with_frequency(Frequency::Annual)
            .with_calendar(Target::new())
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .backwards()
            .build()
    }

    fn float_schedule() -> Schedule {
        MakeSchedule::new()
            .from(settlement())
            .to(Target::new().advance(
                settlement(),
                6,
                TimeUnit::Years,
                BusinessDayConvention::Following,
                false,
            ))
            .with_frequency(Frequency::Semiannual)
            .with_calendar(Target::new())
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .backwards()
            .build()
    }

    fn plain_bond(s: &Shared<Settings<Date>>) -> FixedRateBond {
        FixedRateBond::new(
            2,
            FACE,
            bond_schedule(),
            vec![COUPON],
            Thirty360::with_convention(Convention::BondBasis),
            BusinessDayConvention::Unadjusted,
            100.0,
            Some(settlement()),
            None,
            None,
            Target::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(s),
        )
        .unwrap()
    }

    fn asset_swap(
        bond: &Bond,
        clean_price: Real,
        spread: Spread,
        s: &Shared<Settings<Date>>,
    ) -> AssetSwap {
        let index = shared(Euribor::six_months(curve(), Shared::clone(s)));
        let mut asw = AssetSwap::new(
            true,
            bond,
            clean_price,
            index,
            spread,
            float_schedule(),
            Actual360::new(),
            Shared::clone(s),
        )
        .unwrap();
        let engine = shared_mut(DiscountingSwapEngine::new(
            curve(),
            None,
            None,
            None,
            Shared::clone(s),
        ));
        asw.base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        asw
    }

    #[test]
    fn the_fair_spread_zeroes_the_npv() {
        let s = settings();
        let bond = plain_bond(&s);
        let mut off_market = asset_swap(bond.bond(), 98.5, 0.0, &s);
        let fair = off_market.fair_spread().unwrap();
        let mut repriced = asset_swap(bond.bond(), 98.5, fair, &s);
        assert!(
            repriced.npv().unwrap().abs() < 1e-6,
            "npv at the fair spread should vanish, got {}",
            repriced.npv().unwrap()
        );
    }

    #[test]
    fn a_higher_spread_raises_the_floating_receiver_value() {
        let s = settings();
        let bond = plain_bond(&s);
        let low = asset_swap(bond.bond(), 100.0, 0.0, &s).npv().unwrap();
        let high = asset_swap(bond.bond(), 100.0, 0.002, &s).npv().unwrap();
        assert!(
            high > low,
            "receiving a higher spread should raise NPV: {high} !> {low}"
        );
    }

    #[test]
    fn a_bond_at_its_on_curve_forward_price_asset_swaps_near_zero() {
        // Price the bond on the swap curve, take its forward clean price at the
        // swap start, and asset-swap it: the par spread must be ~0 (the QuantLib
        // par-asset-swap convention). Any gross sign/upfront error blows this up.
        let s = settings();
        let mut bond = plain_bond(&s);
        let disc_engine = shared_mut(DiscountingBondEngine::new(curve(), None, Shared::clone(&s)));
        bond.bond_mut()
            .base_mut()
            .set_pricing_engine(disc_engine as SharedMut<dyn PricingEngine>);
        let spot_npv = bond.bond_mut().npv().unwrap(); // PV of all bond flows to today

        let upfront = settlement();
        let df_upfront = curve()
            .current_link()
            .unwrap()
            .discount_date(upfront, true)
            .unwrap();
        let accrued = bond.bond().accrued_amount(Some(upfront)).unwrap();
        // forward dirty price at upfront = spot PV grossed up to the upfront date
        let forward_dirty = spot_npv / df_upfront;
        let forward_clean = forward_dirty - accrued;

        let mut asw = asset_swap(bond.bond(), forward_clean, 0.0, &s);
        let fair = asw.fair_spread().unwrap();
        assert!(
            fair.abs() < 5e-4,
            "a bond at its forward price should asset-swap near zero spread, got {fair}"
        );
    }
}
