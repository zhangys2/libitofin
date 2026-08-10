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
    use crate::instruments::{BondPrice, FixedRateBond, FloatingRateBond, ZeroCouponBond};
    use crate::interestrate::Compounding;
    use crate::pricingengines::BondFunctions;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::calendars::target::Target;
    use crate::time::calendars::unitedstates::{Market, UnitedStates};
    use crate::time::date::Month;
    use crate::time::dategenerationrule::DateGeneration;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actualactual::{ActualActual, Convention};
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::schedule::{MakeSchedule, Schedule};
    use crate::time::timeunit::TimeUnit;
    use crate::types::{Integer, Spread};

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

    fn fixed_schedule(from: Date, to: Date, next_to_last: Option<Date>) -> Schedule {
        let mut maker = MakeSchedule::new()
            .from(from)
            .to(to)
            .with_frequency(Frequency::Semiannual)
            .with_calendar(UnitedStates::new(Market::GovernmentBond))
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .backwards()
            .end_of_month(false);
        if let Some(d) = next_to_last {
            maker = maker.with_next_to_last_date(d);
        }
        maker.build()
    }

    fn make_fixed_bond(
        settings: &Shared<Settings<Date>>,
        schedule: Schedule,
        coupons: Vec<f64>,
    ) -> FixedRateBond {
        FixedRateBond::new(
            1,
            1_000_000.0,
            schedule,
            coupons,
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
            Shared::clone(settings),
        )
        .unwrap()
    }

    /// `bonds.cpp` testCachedFixed (`:831`): three fixed-coupon government
    /// bonds over a flat 3% Actual360 curve reproduce the cached clean prices
    /// to 1e-6 (plain, varying coupons, and next-to-last stub schedule).
    #[test]
    fn cached_fixed_bonds_reproduce_the_c_clean_prices() {
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

        let sch = fixed_schedule(
            Date::new(30, Month::November, 2004),
            Date::new(30, Month::November, 2008),
            None,
        );
        let coupon_rates = vec![0.02875, 0.03, 0.03125, 0.0325];

        // bond1: plain single coupon
        let mut bond1 = make_fixed_bond(&settings, sch.clone(), vec![0.02875]);
        bond1
            .bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine));
        let price1 = bond1.bond_mut().clean_price().unwrap();
        assert!(
            (price1 - 99.298100).abs() <= 1.0e-6,
            "bond1 plain: clean {price1} vs cached 99.298100 (error {})",
            (price1 - 99.298100).abs()
        );

        // bond2: varying coupons on the same schedule
        let mut bond2 = make_fixed_bond(&settings, sch, coupon_rates.clone());
        bond2
            .bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine));
        let price2 = bond2.bond_mut().clean_price().unwrap();
        assert!(
            (price2 - 100.334149).abs() <= 1.0e-6,
            "bond2 varying coupons: clean {price2} vs cached 100.334149 (error {})",
            (price2 - 100.334149).abs()
        );

        // bond3: same coupons, schedule to 30-Mar-2009 with next-to-last 30-Nov-2008
        let sch3 = fixed_schedule(
            Date::new(30, Month::November, 2004),
            Date::new(30, Month::March, 2009),
            Some(Date::new(30, Month::November, 2008)),
        );
        let mut bond3 = make_fixed_bond(&settings, sch3, coupon_rates);
        bond3.bond_mut().base_mut().set_pricing_engine(engine);
        let price3 = bond3.bond_mut().clean_price().unwrap();
        assert!(
            (price3 - 100.382794).abs() <= 1.0e-6,
            "bond3 stub schedule: clean {price3} vs cached 100.382794 (error {})",
            (price3 - 100.382794).abs()
        );
    }

    /// `bonds.cpp` testTheoretical (`:387`): engine clean/dirty prices on a
    /// quote-backed flat Continuous curve match Continuous yield prices to
    /// 1e-7, and `yield_rate` recovers the quoted rate. Pins a TARGET-adjusted
    /// evaluation date in place of C++ `Date::todaysDate()`.
    #[test]
    fn theoretical_bond_prices_match_continuous_yield_prices() {
        let calendar = Target::new();
        let today = calendar.adjust(
            Date::new(15, Month::June, 2026),
            BusinessDayConvention::Following,
        );
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);

        let tolerance = 1.0e-7;
        let max_evaluations = 100usize;
        let face_amount = 1_000_000.0;
        let settlement_days = 3;
        let bond_day_count = Actual360::new();
        let lengths: [Integer; 5] = [3, 5, 10, 15, 20];
        let coupons = [0.02, 0.05, 0.08];
        let frequencies = [Frequency::Semiannual, Frequency::Annual];
        let yields = [0.03, 0.04, 0.05, 0.06, 0.07];

        for &length in &lengths {
            for &coupon in &coupons {
                for &frequency in &frequencies {
                    let dated = today;
                    let issue = dated;
                    let maturity = calendar.advance(
                        issue,
                        length,
                        TimeUnit::Years,
                        BusinessDayConvention::Following,
                        false,
                    );
                    let rate = shared(SimpleQuote::new(0.0));
                    let discount_curve: Handle<dyn YieldTermStructure> =
                        Handle::new(shared(FlatForward::new(
                            today,
                            Handle::new(Shared::clone(&rate) as Shared<dyn Quote>),
                            bond_day_count.clone(),
                            Compounding::Continuous,
                            Frequency::Annual,
                        )) as Shared<dyn YieldTermStructure>);
                    let tenor = Period::try_from(frequency).unwrap();
                    let schedule = Schedule::new(
                        dated,
                        maturity,
                        tenor,
                        calendar.clone(),
                        BusinessDayConvention::Unadjusted,
                        BusinessDayConvention::Unadjusted,
                        DateGeneration::Backward,
                        false,
                        Date::null(),
                        Date::null(),
                    );
                    let mut bond = FixedRateBond::new(
                        settlement_days,
                        face_amount,
                        schedule,
                        vec![coupon],
                        bond_day_count.clone(),
                        BusinessDayConvention::ModifiedFollowing,
                        100.0,
                        Some(issue),
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
                    )) as SharedMut<dyn PricingEngine>;
                    bond.bond_mut()
                        .base_mut()
                        .set_pricing_engine(SharedMut::clone(&engine));

                    for &m in &yields {
                        rate.set_value(m);

                        let price = BondFunctions::clean_price_from_yield(
                            bond.bond(),
                            m,
                            bond_day_count.clone(),
                            Compounding::Continuous,
                            frequency,
                            None,
                        )
                        .unwrap();
                        let calculated_price = bond.bond_mut().clean_price().unwrap();
                        assert!(
                            (price - calculated_price).abs() <= tolerance,
                            "clean price mismatch: issue={issue} maturity={maturity} \
                             coupon={coupon} freq={frequency:?} yield={m} \
                             expected={price} calculated={calculated_price}"
                        );

                        let calculated_yield = BondFunctions::yield_rate(
                            bond.bond(),
                            BondPrice::Clean(calculated_price),
                            bond_day_count.clone(),
                            Compounding::Continuous,
                            frequency,
                            Some(bond.bond().settlement_date(None).unwrap()),
                            Some(tolerance),
                            Some(max_evaluations),
                            None,
                        )
                        .unwrap();
                        assert!(
                            (m - calculated_yield).abs() <= tolerance,
                            "clean yield recovery failed: issue={issue} maturity={maturity} \
                             coupon={coupon} freq={frequency:?} yield={m} \
                             clean={calculated_price} yield'={calculated_yield}"
                        );

                        let price = BondFunctions::dirty_price_from_yield(
                            bond.bond(),
                            m,
                            bond_day_count.clone(),
                            Compounding::Continuous,
                            frequency,
                            None,
                        )
                        .unwrap();
                        let calculated_price = bond.bond_mut().dirty_price().unwrap();
                        assert!(
                            (price - calculated_price).abs() <= tolerance,
                            "dirty price mismatch: issue={issue} maturity={maturity} \
                             coupon={coupon} freq={frequency:?} yield={m} \
                             expected={price} calculated={calculated_price}"
                        );

                        let calculated_yield = BondFunctions::yield_rate(
                            bond.bond(),
                            BondPrice::Dirty(calculated_price),
                            bond_day_count.clone(),
                            Compounding::Continuous,
                            frequency,
                            Some(bond.bond().settlement_date(None).unwrap()),
                            Some(tolerance),
                            Some(max_evaluations),
                            Some(0.05),
                        )
                        .unwrap();
                        assert!(
                            (m - calculated_yield).abs() <= tolerance,
                            "dirty yield recovery failed: issue={issue} maturity={maturity} \
                             coupon={coupon} freq={frequency:?} yield={m} \
                             dirty={calculated_price} yield'={calculated_yield}"
                        );
                    }
                }
            }
        }
    }

    /// `bonds.cpp` testCached (`:503`): market yield↔clean and engine clean↔yield
    /// cached values for three fixed bonds (schedule ISMA vs bare ISMA) to 1e-6.
    #[test]
    fn cached_bond_price_and_yield_values() {
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
        let tol = 1.0e-6;
        let freq = Frequency::Semiannual;
        let face = 1_000_000.0;
        let settlement_days = 1;

        let assert_close = |got: f64, cached: f64, label: &str| {
            assert!(
                (got - cached).abs() <= tol,
                "{label}: {got} vs cached {cached} (error {})",
                (got - cached).abs()
            );
        };

        // --- bond1: EOM NullCalendar short-first-coupon schedule ---
        let sch1 = MakeSchedule::new()
            .from(Date::new(31, Month::October, 2004))
            .to(Date::new(31, Month::October, 2006))
            .with_frequency(freq)
            .with_calendar(NullCalendar::new())
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .backwards()
            .end_of_month(true)
            .build();
        let dc1 = ActualActual::with_schedule(Convention::ISMA, sch1.clone());
        let dc1_bare = ActualActual::with_convention(Convention::ISMA);
        let make_bond1 = |day_counter: crate::time::daycounter::DayCounter| {
            FixedRateBond::new(
                settlement_days,
                face,
                sch1.clone(),
                vec![0.025],
                day_counter,
                BusinessDayConvention::ModifiedFollowing,
                100.0,
                Some(Date::new(1, Month::November, 2004)),
                None,
                None,
                NullCalendar::new(),
                BusinessDayConvention::Unadjusted,
                false,
                None,
                Shared::clone(&settings),
            )
            .unwrap()
        };
        let mut bond1 = make_bond1(dc1.clone());
        let mut bond1_bare = make_bond1(dc1_bare.clone());
        bond1
            .bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine));
        bond1_bare
            .bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine));

        let market_price1 = 99.203125;
        let market_yield1 = 0.02925;
        assert_close(
            BondFunctions::clean_price_from_yield(
                bond1.bond(),
                market_yield1,
                dc1.clone(),
                Compounding::Compounded,
                freq,
                None,
            )
            .unwrap(),
            99.204505,
            "bond1 schedule yield→clean",
        );
        assert_close(
            BondFunctions::clean_price_from_yield(
                bond1_bare.bond(),
                market_yield1,
                dc1_bare.clone(),
                Compounding::Compounded,
                freq,
                None,
            )
            .unwrap(),
            99.204505,
            "bond1 bare yield→clean",
        );
        let engine_clean1 = bond1.bond_mut().clean_price().unwrap();
        let engine_clean1_bare = bond1_bare.bond_mut().clean_price().unwrap();
        assert_close(engine_clean1, 98.943393, "bond1 schedule engine clean");
        assert_close(engine_clean1_bare, 98.943393, "bond1 bare engine clean");
        assert_close(
            BondFunctions::yield_rate(
                bond1.bond(),
                BondPrice::Clean(market_price1),
                dc1.clone(),
                Compounding::Compounded,
                freq,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
            0.029257,
            "bond1 schedule compounded yield",
        );
        assert_close(
            BondFunctions::yield_rate(
                bond1_bare.bond(),
                BondPrice::Clean(market_price1),
                dc1_bare.clone(),
                Compounding::Compounded,
                freq,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
            0.029257,
            "bond1 bare compounded yield",
        );
        assert_close(
            BondFunctions::yield_rate(
                bond1.bond(),
                BondPrice::Clean(market_price1),
                dc1.clone(),
                Compounding::Continuous,
                freq,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
            0.029045,
            "bond1 schedule continuous yield",
        );
        assert_close(
            BondFunctions::yield_rate(
                bond1_bare.bond(),
                BondPrice::Clean(market_price1),
                dc1_bare.clone(),
                Compounding::Continuous,
                freq,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
            0.029045,
            "bond1 bare continuous yield",
        );
        let settlement1 = bond1.bond().settlement_date(None).unwrap();
        assert_close(
            BondFunctions::yield_rate(
                bond1.bond(),
                BondPrice::Clean(engine_clean1),
                dc1,
                Compounding::Continuous,
                freq,
                Some(settlement1),
                None,
                None,
                None,
            )
            .unwrap(),
            0.030423,
            "bond1 schedule continuous yield from engine",
        );
        assert_close(
            BondFunctions::yield_rate(
                bond1_bare.bond(),
                BondPrice::Clean(engine_clean1_bare),
                dc1_bare,
                Compounding::Continuous,
                freq,
                Some(settlement1),
                None,
                None,
                None,
            )
            .unwrap(),
            0.030423,
            "bond1 bare continuous yield from engine",
        );

        // --- bond2: plain NullCalendar schedule ---
        let sch2 = MakeSchedule::new()
            .from(Date::new(15, Month::November, 2004))
            .to(Date::new(15, Month::November, 2009))
            .with_frequency(freq)
            .with_calendar(NullCalendar::new())
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .backwards()
            .end_of_month(false)
            .build();
        let dc2 = ActualActual::with_schedule(Convention::ISMA, sch2.clone());
        let dc2_bare = ActualActual::with_convention(Convention::ISMA);
        let make_bond2 = |day_counter: crate::time::daycounter::DayCounter| {
            FixedRateBond::new(
                settlement_days,
                face,
                sch2.clone(),
                vec![0.035],
                day_counter,
                BusinessDayConvention::ModifiedFollowing,
                100.0,
                Some(Date::new(15, Month::November, 2004)),
                None,
                None,
                NullCalendar::new(),
                BusinessDayConvention::Unadjusted,
                false,
                None,
                Shared::clone(&settings),
            )
            .unwrap()
        };
        let mut bond2 = make_bond2(dc2.clone());
        let mut bond2_bare = make_bond2(dc2_bare.clone());
        bond2
            .bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine));
        bond2_bare
            .bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine));

        let market_price2 = 99.6875;
        let market_yield2 = 0.03569;
        assert_close(
            BondFunctions::clean_price_from_yield(
                bond2.bond(),
                market_yield2,
                dc2.clone(),
                Compounding::Compounded,
                freq,
                None,
            )
            .unwrap(),
            99.687192,
            "bond2 schedule yield→clean",
        );
        assert_close(
            BondFunctions::clean_price_from_yield(
                bond2_bare.bond(),
                market_yield2,
                dc2_bare.clone(),
                Compounding::Compounded,
                freq,
                None,
            )
            .unwrap(),
            99.687192,
            "bond2 bare yield→clean",
        );
        let engine_clean2 = bond2.bond_mut().clean_price().unwrap();
        let engine_clean2_bare = bond2_bare.bond_mut().clean_price().unwrap();
        assert_close(engine_clean2, 101.986794, "bond2 schedule engine clean");
        assert_close(engine_clean2_bare, 101.986794, "bond2 bare engine clean");
        assert_close(
            BondFunctions::yield_rate(
                bond2.bond(),
                BondPrice::Clean(market_price2),
                dc2.clone(),
                Compounding::Compounded,
                freq,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
            0.035689,
            "bond2 schedule compounded yield",
        );
        assert_close(
            BondFunctions::yield_rate(
                bond2_bare.bond(),
                BondPrice::Clean(market_price2),
                dc2_bare.clone(),
                Compounding::Compounded,
                freq,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
            0.035689,
            "bond2 bare compounded yield",
        );
        assert_close(
            BondFunctions::yield_rate(
                bond2.bond(),
                BondPrice::Clean(market_price2),
                dc2.clone(),
                Compounding::Continuous,
                freq,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
            0.035375,
            "bond2 schedule continuous yield",
        );
        assert_close(
            BondFunctions::yield_rate(
                bond2_bare.bond(),
                BondPrice::Clean(market_price2),
                dc2_bare.clone(),
                Compounding::Continuous,
                freq,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
            0.035375,
            "bond2 bare continuous yield",
        );
        let settlement2 = bond2.bond().settlement_date(None).unwrap();
        assert_close(
            BondFunctions::yield_rate(
                bond2.bond(),
                BondPrice::Clean(engine_clean2),
                dc2,
                Compounding::Continuous,
                freq,
                Some(settlement2),
                None,
                None,
                None,
            )
            .unwrap(),
            0.030432,
            "bond2 schedule continuous yield from engine",
        );
        assert_close(
            BondFunctions::yield_rate(
                bond2_bare.bond(),
                BondPrice::Clean(engine_clean2_bare),
                dc2_bare,
                Compounding::Continuous,
                freq,
                Some(settlement2),
                None,
                None,
                None,
            )
            .unwrap(),
            0.030432,
            "bond2 bare continuous yield from engine",
        );

        // --- bond3: US GovBond, explicit settlement 30-Nov-2004 ---
        let sch3 = MakeSchedule::new()
            .from(Date::new(30, Month::November, 2004))
            .to(Date::new(30, Month::November, 2006))
            .with_frequency(freq)
            .with_calendar(UnitedStates::new(Market::GovernmentBond))
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .backwards()
            .end_of_month(false)
            .build();
        let dc3 = ActualActual::with_schedule(Convention::ISMA, sch3.clone());
        let dc3_bare = ActualActual::with_convention(Convention::ISMA);
        let make_bond3 = |day_counter: crate::time::daycounter::DayCounter| {
            FixedRateBond::new(
                settlement_days,
                face,
                sch3.clone(),
                vec![0.02875],
                day_counter,
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
            .unwrap()
        };
        let bond3 = make_bond3(dc3.clone());
        let bond3_bare = make_bond3(dc3_bare.clone());
        let market_yield3 = 0.02997;
        let settlement3 = Date::new(30, Month::November, 2004);
        assert_close(
            BondFunctions::clean_price_from_yield(
                bond3.bond(),
                market_yield3,
                dc3.clone(),
                Compounding::Compounded,
                freq,
                Some(settlement3),
            )
            .unwrap(),
            99.764759,
            "bond3 schedule yield→clean @ settle",
        );
        assert_close(
            BondFunctions::clean_price_from_yield(
                bond3_bare.bond(),
                market_yield3,
                dc3_bare.clone(),
                Compounding::Compounded,
                freq,
                Some(settlement3),
            )
            .unwrap(),
            99.764759,
            "bond3 bare yield→clean @ settle",
        );
        // earliest possible settlement equals issue; implicit settle matches
        assert_close(
            BondFunctions::clean_price_from_yield(
                bond3.bond(),
                market_yield3,
                dc3,
                Compounding::Compounded,
                freq,
                None,
            )
            .unwrap(),
            99.764759,
            "bond3 schedule yield→clean implicit settle",
        );
        assert_close(
            BondFunctions::clean_price_from_yield(
                bond3_bare.bond(),
                market_yield3,
                dc3_bare,
                Compounding::Compounded,
                freq,
                None,
            )
            .unwrap(),
            99.764759,
            "bond3 bare yield→clean implicit settle",
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
