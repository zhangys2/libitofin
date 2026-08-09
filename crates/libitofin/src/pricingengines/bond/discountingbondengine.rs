//! Discounting bond pricing engine.
//!
//! Port of `ql/pricingengines/bond/discountingbondengine.{hpp,cpp}`:
//! [`DiscountingBondEngine`] discounts a bond's cash flows over a
//! [`YieldTermStructure`] to fill the [`BondResults`] the base
//! [`Bond`](crate::instruments::Bond) reads its settlement value and prices
//! from. Changes to the discount-curve handle invalidate the attached bond
//! through the usual observable chain.
//!
//! Deviations, documented per D5/D10:
//! - The C++ global `Settings::instance()` becomes an explicit
//!   [`Settings`] handle the engine is built with, mirroring how the base
//!   bond carries its settings; it drives the `includeReferenceDateEvents`
//!   fall back for the reference-date flow decision.

use crate::cashflows::CashFlows;
use crate::errors::QlResult;
use crate::instruments::{BondArguments, BondEngine, BondResults};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::settings::Settings;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::calendars::nullcalendar::NullCalendar;
use crate::time::date::Date;
use crate::{fail, handle::Handle, require};

/// Discounting engine for bonds.
///
/// Discounts the bond's future cash flows to the discount curve's reference
/// date for the value, and to the settlement date for the settlement value.
pub struct DiscountingBondEngine {
    base: BondEngine,
    discount_curve: Handle<dyn YieldTermStructure>,
    include_settlement_date_flows: Option<bool>,
    settings: Shared<Settings<Date>>,
}

impl DiscountingBondEngine {
    /// Builds the engine over a discount-curve handle it registers with.
    ///
    /// `include_settlement_date_flows` overrides, when set, the settings'
    /// `include_reference_date_events` flag for the reference-date flow
    /// decision (the C++ `includeSettlementDateFlows` optional).
    pub fn new(
        discount_curve: Handle<dyn YieldTermStructure>,
        include_settlement_date_flows: Option<bool>,
        settings: Shared<Settings<Date>>,
    ) -> DiscountingBondEngine {
        let base = BondEngine::new(
            BondArguments {
                settlement_date: None,
                cashflows: Vec::new(),
                calendar: NullCalendar::new(),
            },
            BondResults::default(),
        );
        discount_curve.register_observer(&base.observer());
        DiscountingBondEngine {
            base,
            discount_curve,
            include_settlement_date_flows,
            settings,
        }
    }

    /// The discount-curve handle the engine prices over.
    pub fn discount_curve(&self) -> &Handle<dyn YieldTermStructure> {
        &self.discount_curve
    }
}

impl AsObservable for DiscountingBondEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for DiscountingBondEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn calculate(&mut self) -> QlResult<()> {
        require!(
            !self.discount_curve.is_empty(),
            "discounting term structure handle is empty"
        );
        let curve = self.discount_curve.current_link()?;
        let valuation_date = curve.reference_date()?;

        let include_ref_date_flows = self
            .include_settlement_date_flows
            .unwrap_or_else(|| self.settings.include_reference_date_events());

        let Some(settlement_date) = self.base.arguments().settlement_date else {
            fail!("no settlement date provided");
        };

        let value = CashFlows::npv(
            &self.base.arguments().cashflows,
            &*curve,
            &self.settings,
            Some(include_ref_date_flows),
            Some(valuation_date),
            Some(valuation_date),
        )?;

        let settlement_value = if !include_ref_date_flows && valuation_date == settlement_date {
            value
        } else {
            CashFlows::npv(
                &self.base.arguments().cashflows,
                &*curve,
                &self.settings,
                Some(false),
                Some(settlement_date),
                Some(settlement_date),
            )?
        };

        let results = self.base.results_mut();
        results.instrument.valuation_date = Some(valuation_date);
        results.instrument.value = Some(value);
        results.settlement_value = Some(settlement_value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexes::index::Index;
    use crate::indexes::{IborIndex, USDLibor};
    use crate::instrument::Instrument;
    use crate::instruments::{FixedRateBond, FloatingRateBond, ZeroCouponBond};
    use crate::interestrate::Compounding;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::calendars::unitedstates::{Market, UnitedStates};
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actualactual::{ActualActual, Convention};
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::schedule::{MakeSchedule, Schedule};
    use crate::time::timeunit::TimeUnit;
    use crate::types::Spread;

    fn today() -> Date {
        Date::new(22, Month::November, 2004)
    }

    fn settings_today() -> Shared<Settings<Date>> {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        settings
    }

    /// `bonds.cpp` testCachedZero (`:751`): three plain zero-coupon government
    /// bonds over a flat 3% Actual360 curve reproduce the cached clean prices
    /// to 1e-6.
    #[test]
    fn cached_zero_bonds_reproduce_the_c_clean_prices() {
        let settings = settings_today();
        let discount_curve: Handle<dyn YieldTermStructure> =
            Handle::new(shared(FlatForward::with_rate(
                today(),
                0.03,
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>);
        let engine = shared_mut(DiscountingBondEngine::new(
            discount_curve,
            None,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;

        let cases = [
            (Date::new(30, Month::November, 2008), 88.551726),
            (Date::new(30, Month::November, 2007), 91.278949),
            (Date::new(30, Month::November, 2006), 94.098006),
        ];
        let tol = 1.0e-6;
        for (maturity, cached) in cases {
            let mut bond = ZeroCouponBond::new(
                1,
                UnitedStates::new(Market::GovernmentBond),
                1_000_000.0,
                maturity,
                BusinessDayConvention::ModifiedFollowing,
                100.0,
                Some(Date::new(30, Month::November, 2004)),
                Shared::clone(&settings),
            )
            .unwrap();
            bond.bond_mut()
                .base_mut()
                .set_pricing_engine(SharedMut::clone(&engine));
            let price = bond.bond_mut().clean_price().unwrap();
            assert!(
                (price - cached).abs() <= tol,
                "maturity {maturity}: clean {price} vs cached {cached} (error {})",
                (price - cached).abs()
            );
        }
    }

    fn flat_actual360(rate: f64) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn floating_schedule(from: Date, to: Date) -> Schedule {
        MakeSchedule::new()
            .from(from)
            .to(to)
            .with_frequency(Frequency::Semiannual)
            .with_calendar(UnitedStates::new(Market::GovernmentBond))
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .backwards()
            .end_of_month(false)
            .build()
    }

    #[allow(clippy::too_many_arguments)]
    fn make_floating_bond(
        settings: &Shared<Settings<Date>>,
        schedule: Schedule,
        index: Shared<IborIndex>,
        spreads: Vec<Spread>,
        issue: Date,
        ex_coupon_period: Option<Period>,
        ex_coupon_convention: BusinessDayConvention,
    ) -> FloatingRateBond {
        FloatingRateBond::new(
            1,
            1_000_000.0,
            schedule,
            index,
            ActualActual::with_convention(Convention::ISMA),
            BusinessDayConvention::ModifiedFollowing,
            Some(1),
            Vec::new(),
            spreads,
            Vec::new(),
            Vec::new(),
            100.0,
            Some(issue),
            ex_coupon_period,
            NullCalendar::new(),
            ex_coupon_convention,
            false,
            Shared::clone(settings),
        )
        .unwrap()
    }

    fn assert_clean(
        bond: &mut FloatingRateBond,
        engine: SharedMut<dyn PricingEngine>,
        cached: f64,
        label: &str,
    ) {
        bond.bond_mut().base_mut().set_pricing_engine(engine);
        let price = bond.bond_mut().clean_price().unwrap();
        assert!(
            (price - cached).abs() <= 1.0e-6,
            "{label}: clean {price} vs cached {cached} (error {})",
            (price - cached).abs()
        );
    }

    /// `bonds.cpp` testCachedFloating (`:928`): USDLibor6M floaters reproduce
    /// the four cached clean prices to 1e-6 (at-par coupon branch).
    #[test]
    fn cached_floating_bonds_reproduce_the_c_clean_prices() {
        let settings = settings_today();
        assert!(
            settings.using_at_par_coupons(),
            "oracle expects the at-par coupon branch"
        );

        let risk_free = flat_actual360(0.025);
        let discount_curve = flat_actual360(0.03);
        let index: Shared<IborIndex> = shared(USDLibor::six_months(
            risk_free.clone(),
            Shared::clone(&settings),
        ));
        let sch = floating_schedule(
            Date::new(30, Month::November, 2004),
            Date::new(30, Month::November, 2008),
        );
        let engine_rf = shared_mut(DiscountingBondEngine::new(
            risk_free,
            None,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;
        let engine_disc = shared_mut(DiscountingBondEngine::new(
            discount_curve,
            None,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;

        // bond1: plain, discount = forecast curve
        let mut bond1 = make_floating_bond(
            &settings,
            sch.clone(),
            Shared::clone(&index),
            Vec::new(),
            Date::new(30, Month::November, 2004),
            None,
            BusinessDayConvention::Following,
        );
        assert_clean(
            &mut bond1,
            SharedMut::clone(&engine_rf),
            99.874646,
            "bond1 plain",
        );

        // bond2: dual curve (forecast 2.5%, discount 3%)
        let mut bond2 = make_floating_bond(
            &settings,
            sch.clone(),
            Shared::clone(&index),
            Vec::new(),
            Date::new(30, Month::November, 2004),
            None,
            BusinessDayConvention::Following,
        );
        assert_clean(
            &mut bond2,
            SharedMut::clone(&engine_disc),
            97.955904,
            "bond2 dual curve",
        );

        // bond3: varying spreads on the dual-curve engine
        let spreads = vec![0.001, 0.0012, 0.0014, 0.0016];
        let mut bond3 = make_floating_bond(
            &settings,
            sch,
            Shared::clone(&index),
            spreads.clone(),
            Date::new(30, Month::November, 2004),
            None,
            BusinessDayConvention::Following,
        );
        assert_clean(
            &mut bond3,
            SharedMut::clone(&engine_disc),
            98.495459,
            "bond3 spreads",
        );

        // bond4: earlier schedule, 6D ex-coupon, historical fixing
        let sch2 = floating_schedule(
            Date::new(26, Month::November, 2003),
            Date::new(26, Month::November, 2007),
        );
        let mut bond4 = make_floating_bond(
            &settings,
            sch2,
            Shared::clone(&index),
            spreads,
            Date::new(29, Month::October, 2004),
            Some(Period::new(6, TimeUnit::Days)),
            BusinessDayConvention::Unadjusted,
        );
        index
            .add_fixing(Date::new(25, Month::May, 2004), 0.0402)
            .unwrap();
        assert_clean(&mut bond4, engine_disc, 98.892055, "bond4 fixing+ex-coupon");
    }

    /// `bonds.cpp` testCachedFixed, bond1 (`:832`): a fixed-coupon government
    /// bond priced end to end over a flat 3% curve reproduces the cached clean
    /// price 99.298100 to 1e-6. This is the first bond priced against a real
    /// C++ number, closing the vertical slice.
    #[test]
    fn cached_fixed_bond1_reproduces_the_c_clean_price() {
        let settings = settings_today();

        let discount_curve: Handle<dyn YieldTermStructure> =
            Handle::new(shared(FlatForward::with_rate(
                today(),
                0.03,
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>);

        let schedule = MakeSchedule::new()
            .from(Date::new(30, Month::November, 2004))
            .to(Date::new(30, Month::November, 2008))
            .with_frequency(Frequency::Semiannual)
            .with_calendar(UnitedStates::new(Market::GovernmentBond))
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .backwards()
            .build();

        let mut bond = FixedRateBond::new(
            1,
            1_000_000.0,
            schedule,
            vec![0.02875],
            ActualActual::with_convention(Convention::ISMA),
            BusinessDayConvention::ModifiedFollowing,
            100.0,
            Some(Date::new(30, Month::November, 2004)),
            None,
            None,
            NullCalendar::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(&settings),
        )
        .unwrap();

        let engine = shared_mut(DiscountingBondEngine::new(
            discount_curve,
            None,
            Shared::clone(&settings),
        ));
        bond.bond_mut()
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

        let price = bond.bond_mut().clean_price().unwrap();
        assert!(
            (price - 99.298100).abs() <= 1.0e-6,
            "clean price {price} vs cached 99.298100 (error {})",
            (price - 99.298100).abs()
        );
    }

    /// An empty discount-curve handle is rejected before any discounting, as
    /// the C++ `QL_REQUIRE(!discountCurve_.empty(), ...)`.
    #[test]
    fn an_empty_discount_curve_is_rejected() {
        let mut engine = DiscountingBondEngine::new(Handle::empty(), None, settings_today());
        assert_eq!(
            engine.calculate().unwrap_err().message(),
            "discounting term structure handle is empty"
        );
    }
}
