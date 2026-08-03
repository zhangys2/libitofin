//! A swap exchanging two floating (Ibor) legs.
//!
//! `FloatFloatSwap` pays one Ibor leg and receives another, each with its own
//! schedule, index, spread and day count. It reuses the generic [`Swap`] base
//! and prices through any swap engine (e.g. `DiscountingSwapEngine`), mirroring
//! how [`VanillaSwap`](crate::instruments::VanillaSwap) wraps the swap
//! machinery. QuantLib's `FloatFloatSwap` adds caps/floors and per-leg
//! gearing/notional structuring; this first slice covers the plain
//! two-floating-leg swap.
//!
//! Leg 0 is paid and leg 1 is received, so the swap NPV is
//! `PV(leg 1) − PV(leg 0)`.

use crate::cashflows::IborLeg;
use crate::errors::QlResult;
use crate::indexes::IborIndex;
use crate::instrument::{Instrument, InstrumentBase};
use crate::instruments::Swap;
use crate::pricingengine::{Arguments, Results};
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::schedule::Schedule;
use crate::types::{Real, Spread};

const BASIS_POINT: Real = 1.0e-4;

/// A two-floating-leg swap (pay leg 0, receive leg 1).
pub struct FloatFloatSwap {
    swap: Swap,
    spread1: Spread,
}

impl FloatFloatSwap {
    /// Builds a float-float swap on a single `nominal`.
    ///
    /// Leg 1 (`schedule1`/`index1`/`spread1`/`day_count1`) is paid and leg 2
    /// (`schedule2`/`index2`/`spread2`/`day_count2`) is received. A `None`
    /// `payment_convention` leaves each leg builder's default.
    ///
    /// # Errors
    ///
    /// Propagates any [`IborLeg`] build error (e.g. an empty schedule or a
    /// zero gearing) and any [`Swap`] construction error.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nominal: Real,
        schedule1: Schedule,
        index1: Shared<IborIndex>,
        spread1: Spread,
        day_count1: DayCounter,
        schedule2: Schedule,
        index2: Shared<IborIndex>,
        spread2: Spread,
        day_count2: DayCounter,
        payment_convention: Option<BusinessDayConvention>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<FloatFloatSwap> {
        let leg1 = Self::build_leg(
            nominal,
            schedule1,
            index1,
            spread1,
            day_count1,
            payment_convention,
        )?;
        let leg2 = Self::build_leg(
            nominal,
            schedule2,
            index2,
            spread2,
            day_count2,
            payment_convention,
        )?;
        let swap = Swap::two_leg(leg1, leg2, settings);
        Ok(FloatFloatSwap { swap, spread1 })
    }

    fn build_leg(
        nominal: Real,
        schedule: Schedule,
        index: Shared<IborIndex>,
        spread: Spread,
        day_count: DayCounter,
        payment_convention: Option<BusinessDayConvention>,
    ) -> QlResult<crate::cashflow::Leg> {
        let mut builder = IborLeg::new(schedule, index)
            .with_notionals(vec![nominal])
            .with_payment_day_counter(day_count)
            .with_spreads(vec![spread]);
        if let Some(convention) = payment_convention {
            builder = builder.with_payment_adjustment(convention);
        }
        builder.build()
    }

    /// The generic swap this wraps.
    pub fn swap(&self) -> &Swap {
        &self.swap
    }

    /// The paid leg's NPV.
    pub fn pay_leg_npv(&mut self) -> QlResult<Real> {
        self.swap.leg_npv(0)
    }

    /// The received leg's NPV.
    pub fn receive_leg_npv(&mut self) -> QlResult<Real> {
        self.swap.leg_npv(1)
    }

    /// The paid leg's BPS.
    pub fn pay_leg_bps(&mut self) -> QlResult<Real> {
        self.swap.leg_bps(0)
    }

    /// The received leg's BPS.
    pub fn receive_leg_bps(&mut self) -> QlResult<Real> {
        self.swap.leg_bps(1)
    }

    /// The spread on the paid leg that zeroes the swap NPV.
    ///
    /// Uses the same first-order relation the discounting swap engine's
    /// `FixedVsFloatingSwap` fallback does: `spread - NPV / (legBPS / 1bp)`.
    ///
    /// # Errors
    ///
    /// The engine must have priced the swap and provided the paid leg's BPS.
    pub fn fair_pay_spread(&mut self) -> QlResult<Spread> {
        self.calculate()?;
        let npv = self.npv()?;
        let bps = self.swap.leg_bps(0)?;
        Ok(self.spread1 - npv / (bps / BASIS_POINT))
    }
}

impl Instrument for FloatFloatSwap {
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
    use crate::interestrate::Compounding;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::swap::DiscountingSwapEngine;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::time::schedule::MakeSchedule;
    use crate::time::timeunit::TimeUnit;

    const NOMINAL: Real = 1_000_000.0;

    fn today() -> Date {
        Date::new(15, Month::January, 2020)
    }

    fn settings() -> Shared<Settings<Date>> {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        settings
    }

    fn curve(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn schedule() -> Schedule {
        let calendar = Target::new();
        // Anchor at the T+2 settlement so the first floating fixing is a
        // forecast (after the evaluation date), not a required historical one.
        let settlement = calendar.advance(
            today(),
            2,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let maturity = calendar.advance(
            settlement,
            5,
            TimeUnit::Years,
            BusinessDayConvention::Following,
            false,
        );
        MakeSchedule::new()
            .from(settlement)
            .to(maturity)
            .with_frequency(Frequency::Semiannual)
            .with_calendar(calendar)
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .forwards()
            .end_of_month(false)
            .build()
    }

    fn attach_engine(
        swap: &mut FloatFloatSwap,
        discount: Handle<dyn YieldTermStructure>,
        s: &Shared<Settings<Date>>,
    ) {
        let engine = shared_mut(DiscountingSwapEngine::new(
            discount,
            None,
            None,
            None,
            Shared::clone(s),
        ));
        swap.base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
    }

    #[test]
    fn identical_legs_net_to_zero() {
        let settings = settings();
        let c = curve(0.03);
        let index = shared(Euribor::six_months(c.clone(), Shared::clone(&settings)));
        let mut swap = FloatFloatSwap::new(
            NOMINAL,
            schedule(),
            Shared::clone(&index),
            0.0,
            Actual360::new(),
            schedule(),
            Shared::clone(&index),
            0.0,
            Actual360::new(),
            Some(BusinessDayConvention::ModifiedFollowing),
            Shared::clone(&settings),
        )
        .unwrap();
        attach_engine(&mut swap, c, &settings);

        assert!(
            swap.npv().unwrap().abs() < 1e-6,
            "npv={}",
            swap.npv().unwrap()
        );
        // Pay and receive legs offset exactly.
        assert!((swap.pay_leg_npv().unwrap() + swap.receive_leg_npv().unwrap()).abs() < 1e-6);
    }

    #[test]
    fn a_positive_receive_spread_makes_the_swap_positive() {
        let settings = settings();
        let c = curve(0.03);
        let index = shared(Euribor::six_months(c.clone(), Shared::clone(&settings)));
        // Receive leg (leg 1) carries +50bp; identical pay leg otherwise.
        let mut swap = FloatFloatSwap::new(
            NOMINAL,
            schedule(),
            Shared::clone(&index),
            0.0,
            Actual360::new(),
            schedule(),
            Shared::clone(&index),
            0.005,
            Actual360::new(),
            Some(BusinessDayConvention::ModifiedFollowing),
            Shared::clone(&settings),
        )
        .unwrap();
        attach_engine(&mut swap, c, &settings);

        assert!(
            swap.npv().unwrap() > 0.0,
            "receiving extra spread should be positive: {}",
            swap.npv().unwrap()
        );
    }

    #[test]
    fn the_fair_pay_spread_zeroes_the_npv() {
        let settings = settings();
        let c = curve(0.03);
        let index = shared(Euribor::six_months(c.clone(), Shared::clone(&settings)));
        // Pay leg starts with 0 spread, receive leg with +30bp, so the swap is
        // off-market; the fair pay spread should rebalance it.
        let mut swap = FloatFloatSwap::new(
            NOMINAL,
            schedule(),
            Shared::clone(&index),
            0.0,
            Actual360::new(),
            schedule(),
            Shared::clone(&index),
            0.003,
            Actual360::new(),
            Some(BusinessDayConvention::ModifiedFollowing),
            Shared::clone(&settings),
        )
        .unwrap();
        attach_engine(&mut swap, c.clone(), &settings);
        let fair = swap.fair_pay_spread().unwrap();

        let mut repriced = FloatFloatSwap::new(
            NOMINAL,
            schedule(),
            Shared::clone(&index),
            fair,
            Actual360::new(),
            schedule(),
            Shared::clone(&index),
            0.003,
            Actual360::new(),
            Some(BusinessDayConvention::ModifiedFollowing),
            Shared::clone(&settings),
        )
        .unwrap();
        attach_engine(&mut repriced, c, &settings);
        assert!(
            repriced.npv().unwrap().abs() < 1e-6,
            "npv at the fair pay spread should vanish, got {}",
            repriced.npv().unwrap()
        );
    }
}
