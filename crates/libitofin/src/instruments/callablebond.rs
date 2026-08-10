//! Callable / puttable fixed-rate bond.
//!
//! Port of QuantLib's `ql/experimental/callablebonds/callablebond.{hpp,cpp}`
//! (fixed-rate path). A [`CallableFixedRateBond`] is a fixed-rate bond carrying
//! a [`CallabilitySchedule`] of European/Bermudan call or put dates; it prices
//! on a short-rate lattice through
//! [`TreeCallableFixedRateBondEngine`](crate::pricingengines::bond::TreeCallableFixedRateBondEngine).
//!
//! This slice mirrors QuantLib's argument setup (`setupArguments`): coupons and
//! a par redemption feed the discretized bond, and clean call prices are
//! converted to dirty with the accrued at the call date. Black/implied-vol and
//! OAS helpers are follow-ups.

use std::any::Any;

use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{Bond, BondPrice, BondResults, FixedRateBond};
use crate::pricingengine::{Arguments, Results};
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendars::nullcalendar::NullCalendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::schedule::Schedule;
use crate::types::{Natural, Rate, Real, Spread};

/// Whether a callability is a call (issuer's right to redeem) or a put
/// (holder's right to sell back).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallabilityType {
    /// The issuer may redeem at the callability price.
    Call,
    /// The holder may sell back at the callability price.
    Put,
}

/// One call/put right at a date and price (`ql/instruments/callabilityschedule.hpp`).
///
/// Soft calls (QuantLib's `SoftCallability`) are represented by a [`Call`](CallabilityType::Call)
/// with a [`trigger`](Self::trigger): the issuer may call only when the
/// underlying exceeds `trigger` times the conversion value.
#[derive(Clone, Debug)]
pub struct Callability {
    price: BondPrice,
    call_type: CallabilityType,
    date: Date,
    trigger: Option<Real>,
}

impl Callability {
    /// Builds a hard callability at `date` exercisable at `price`.
    pub fn new(price: BondPrice, call_type: CallabilityType, date: Date) -> Self {
        Self {
            price,
            call_type,
            date,
            trigger: None,
        }
    }

    /// Builds a soft call at `date` with soft-call `trigger`
    /// (`ql/instruments/bonds/convertiblebonds.hpp` `SoftCallability`).
    pub fn soft(price: BondPrice, date: Date, trigger: Real) -> Self {
        Self {
            price,
            call_type: CallabilityType::Call,
            date,
            trigger: Some(trigger),
        }
    }

    /// The callability price (clean or dirty, per 100).
    pub fn price(&self) -> BondPrice {
        self.price
    }

    /// Whether this is a call or a put.
    pub fn call_type(&self) -> CallabilityType {
        self.call_type
    }

    /// The callability date.
    pub fn date(&self) -> Date {
        self.date
    }

    /// Soft-call trigger multiple of the conversion value, when present.
    pub fn trigger(&self) -> Option<Real> {
        self.trigger
    }
}

/// A schedule of put/call rights.
pub type CallabilitySchedule = Vec<Callability>;

/// Engine arguments for a callable bond (`CallableBond::arguments`).
#[derive(Default)]
pub struct CallableBondArguments {
    /// The settlement date.
    pub settlement_date: Option<Date>,
    /// Coupon payment dates (redemption excluded).
    pub coupon_dates: Vec<Date>,
    /// Coupon amounts aligned with [`coupon_dates`](Self::coupon_dates).
    pub coupon_amounts: Vec<Real>,
    /// The bond's face amount.
    pub face_amount: Real,
    /// The redemption amount.
    pub redemption: Real,
    /// The redemption date.
    pub redemption_date: Option<Date>,
    /// Callability dirty prices (per 100), aligned with the dates/types.
    pub callability_prices: Vec<Real>,
    /// Callability types aligned with the dates/prices.
    pub callability_types: Vec<CallabilityType>,
    /// Callability dates.
    pub callability_dates: Vec<Date>,
    /// Continuous spread added to the model (0 unless set by an OAS solve).
    pub spread: Spread,
}

impl Arguments for CallableBondArguments {
    fn validate(&self) -> QlResult<()> {
        require_field(self.settlement_date.is_some(), "null settlement date")?;
        require_field(self.redemption_date.is_some(), "null redemption date")?;
        require_field(self.redemption >= 0.0, "negative redemption")?;
        require_field(
            self.callability_dates.len() == self.callability_prices.len()
                && self.callability_dates.len() == self.callability_types.len(),
            "callability dates/prices/types length mismatch",
        )?;
        require_field(
            self.coupon_dates.len() == self.coupon_amounts.len(),
            "coupon dates/amounts length mismatch",
        )
    }
}

#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn require_field(ok: bool, message: &str) -> QlResult<()> {
    if ok { Ok(()) } else { fail!("{message}") }
}

/// A callable / puttable fixed-rate bond.
pub struct CallableFixedRateBond {
    base: InstrumentBase,
    bond: FixedRateBond,
    put_call_schedule: CallabilitySchedule,
    face_amount: Real,
    settings: Shared<Settings<Date>>,
    settlement_value: Option<Real>,
}

impl CallableFixedRateBond {
    /// Builds a callable fixed-rate bond (`CallableFixedRateBond` ctor).
    ///
    /// # Errors
    ///
    /// Fails if a call/put date is after the bond maturity, or on any bond
    /// construction error.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        settlement_days: Natural,
        face_amount: Real,
        schedule: Schedule,
        coupons: Vec<Rate>,
        accrual_day_counter: DayCounter,
        payment_convention: BusinessDayConvention,
        redemption: Real,
        issue_date: Option<Date>,
        put_call_schedule: CallabilitySchedule,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<CallableFixedRateBond> {
        let maturity = schedule.end_date();
        for callability in &put_call_schedule {
            if callability.date() > maturity {
                fail!("bond cannot mature before the last call/put date");
            }
        }
        let bond = FixedRateBond::new(
            settlement_days,
            face_amount,
            schedule,
            coupons,
            accrual_day_counter,
            payment_convention,
            redemption,
            issue_date,
            None,
            None,
            NullCalendar::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(&settings),
        )?;
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(CallableFixedRateBond {
            base,
            bond,
            put_call_schedule,
            face_amount,
            settings,
            settlement_value: None,
        })
    }

    /// The put/call schedule.
    pub fn callability(&self) -> &CallabilitySchedule {
        &self.put_call_schedule
    }

    /// The underlying fixed-rate bond.
    pub fn bond(&self) -> &Bond {
        self.bond.bond()
    }

    /// The theoretical settlement value from the engine.
    pub fn settlement_value(&mut self) -> QlResult<Real> {
        self.calculate()?;
        let Some(value) = self.settlement_value else {
            fail!("settlement value not provided");
        };
        Ok(value)
    }

    /// The theoretical dirty price, per 100 of notional.
    pub fn dirty_price(&mut self) -> QlResult<Real> {
        let settlement = self.bond().settlement_date(None)?;
        let notional = self.bond().notional(Some(settlement))?;
        if notional == 0.0 {
            return Ok(0.0);
        }
        let value = self.settlement_value()?;
        Ok(value * 100.0 / notional)
    }

    /// The theoretical clean price, per 100 of notional.
    pub fn clean_price(&mut self) -> QlResult<Real> {
        let settlement = self.bond().settlement_date(None)?;
        let dirty = self.dirty_price()?;
        let accrued = self.bond().accrued_amount(Some(settlement))?;
        Ok(dirty - accrued)
    }
}

impl Instrument for CallableFixedRateBond {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        self.bond.bond().is_expired()
    }

    fn setup_expired(&mut self) {
        let expired = InstrumentResults {
            value: Some(0.0),
            ..InstrumentResults::default()
        };
        self.base.store_results(&expired);
        self.settlement_value = Some(0.0);
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(args) = (arguments as &mut dyn Any).downcast_mut::<CallableBondArguments>() else {
            fail!("wrong argument type");
        };
        let bond = self.bond.bond();
        let settlement = bond.settlement_date(None)?;
        args.settlement_date = Some(settlement);
        args.face_amount = self.face_amount;

        let cashflows = bond.cashflows();
        let count = cashflows.len();
        let redemption = &cashflows[count - 1];
        args.redemption = redemption.amount()?;
        args.redemption_date = Some(redemption.date());

        args.coupon_dates.clear();
        args.coupon_amounts.clear();
        for flow in &cashflows[..count - 1] {
            if !event_has_occurred(flow.date(), &self.settings, Some(settlement), Some(false))? {
                args.coupon_dates.push(flow.date());
                args.coupon_amounts.push(flow.amount()?);
            }
        }

        args.callability_dates.clear();
        args.callability_prices.clear();
        args.callability_types.clear();
        for callability in &self.put_call_schedule {
            if event_has_occurred(
                callability.date(),
                &self.settings,
                Some(settlement),
                Some(false),
            )? {
                continue;
            }
            let call_date = callability.date();
            args.callability_dates.push(call_date);
            args.callability_types.push(callability.call_type());
            let mut price = callability.price().amount();
            if let BondPrice::Clean(_) = callability.price() {
                // Convert the clean call price to dirty with the accrued at the
                // call date (`callablebond.cpp:453-477`).
                for flow in cashflows {
                    if !event_has_occurred(
                        flow.date(),
                        &self.settings,
                        Some(call_date),
                        Some(false),
                    )? {
                        if let Some(coupon) = flow.as_coupon() {
                            let accrued = coupon.accrued_amount(call_date)?;
                            let notional = bond.notional(Some(call_date))?;
                            if notional != 0.0 {
                                price += accrued / notional * 100.0;
                            }
                        }
                        break;
                    }
                }
            }
            args.callability_prices.push(price);
        }

        args.spread = 0.0;
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<BondResults>() else {
            fail!("wrong result type");
        };
        self.settlement_value = results.settlement_value;
        self.base.store_results(&results.instrument);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::instruments::FixedRateBond;
    use crate::interestrate::Compounding;
    use crate::models::shortrate::HullWhite;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::bond::{DiscountingBondEngine, TreeCallableFixedRateBondEngine};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;
    use crate::time::schedule::MakeSchedule;
    use crate::time::timeunit::TimeUnit;

    const FACE: Real = 100.0;
    const COUPON: Real = 0.05;
    const STEPS: usize = 400;

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
            0.03,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn schedule() -> Schedule {
        MakeSchedule::new()
            .from(today())
            .to(Target::new().advance(
                today(),
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

    fn straight_clean(s: &Shared<Settings<Date>>) -> Real {
        let mut bond = FixedRateBond::new(
            2,
            FACE,
            schedule(),
            vec![COUPON],
            Thirty360::with_convention(Convention::BondBasis),
            BusinessDayConvention::Unadjusted,
            100.0,
            Some(today()),
            None,
            None,
            Target::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(s),
        )
        .unwrap();
        let engine = shared_mut(DiscountingBondEngine::new(curve(), None, Shared::clone(s)));
        bond.bond_mut()
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        bond.bond_mut().clean_price().unwrap()
    }

    fn callable_clean(schedule_calls: CallabilitySchedule, s: &Shared<Settings<Date>>) -> Real {
        let mut bond = CallableFixedRateBond::new(
            2,
            FACE,
            schedule(),
            vec![COUPON],
            Thirty360::with_convention(Convention::BondBasis),
            BusinessDayConvention::Unadjusted,
            100.0,
            Some(today()),
            schedule_calls,
            Shared::clone(s),
        )
        .unwrap();
        let model = HullWhite::new(curve(), 0.03, 0.01).unwrap();
        let engine = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, STEPS, Shared::clone(s)).unwrap(),
        );
        bond.base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        bond.clean_price().unwrap()
    }

    fn option_date() -> Date {
        Target::new().advance(
            today(),
            3,
            TimeUnit::Years,
            BusinessDayConvention::Following,
            false,
        )
    }

    #[test]
    fn no_call_matches_the_straight_bond() {
        let s = settings();
        let straight = straight_clean(&s);
        let tree = callable_clean(Vec::new(), &s);
        assert!(
            (tree - straight).abs() < 0.1,
            "no-call tree price {tree} should match the straight bond {straight}"
        );
    }

    #[test]
    fn a_call_lowers_and_a_put_raises_the_value() {
        let s = settings();
        let straight = straight_clean(&s);

        let call = vec![Callability::new(
            BondPrice::Clean(100.0),
            CallabilityType::Call,
            option_date(),
        )];
        let put = vec![Callability::new(
            BondPrice::Clean(100.0),
            CallabilityType::Put,
            option_date(),
        )];
        let callable = callable_clean(call, &s);
        let puttable = callable_clean(put, &s);

        assert!(
            callable <= straight + 1e-6,
            "callable {callable} should not exceed the straight bond {straight}"
        );
        assert!(
            puttable >= straight - 1e-6,
            "puttable {puttable} should not fall below the straight bond {straight}"
        );
        // The 5% coupon trades above the 3% curve, so the issuer's call is
        // valuable: callable is strictly cheaper than puttable.
        assert!(
            callable < puttable - 1e-3,
            "callable {callable} should be strictly below puttable {puttable}"
        );
    }

    /// `callablebonds.cpp` testCached (`:472`): HW-tree clean prices for
    /// callable / puttable / both schedules reproduce the cached values to
    /// 1e-8.
    #[test]
    fn cached_callable_bond_prices_reproduce_the_c_values() {
        let calendar = Target::new();
        let today = Date::new(3, Month::June, 2004);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let settlement = calendar.advance(
            today,
            3,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let rolling = BusinessDayConvention::ModifiedFollowing;
        let issue = calendar.adjust(today - 100, BusinessDayConvention::Following);
        let maturity = calendar.advance(issue, 10, TimeUnit::Years, rolling, false);

        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.032,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let model = HullWhite::new(curve, 0.1, 0.01).unwrap();

        let schedule = MakeSchedule::new()
            .from(issue)
            .to(maturity)
            .with_calendar(calendar.clone())
            .with_frequency(Frequency::Semiannual)
            .with_convention(rolling)
            .with_termination_date_convention(rolling)
            .backwards()
            .build();

        let mut calls = Vec::new();
        let mut puts = Vec::new();
        let mut both = Vec::new();
        for i in (2..10).step_by(2) {
            let date = calendar.advance(issue, i, TimeUnit::Years, rolling, false);
            let exercise = Callability::new(BondPrice::Clean(110.0), CallabilityType::Call, date);
            calls.push(exercise.clone());
            both.push(exercise);
        }
        for i in (1..10).step_by(2) {
            let date = calendar.advance(issue, i, TimeUnit::Years, rolling, false);
            let exercise = Callability::new(BondPrice::Clean(100.0), CallabilityType::Put, date);
            puts.push(exercise.clone());
            both.push(exercise);
        }

        let engine = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, 240, Shared::clone(&settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;
        let day_counter = Thirty360::with_convention(Convention::BondBasis);
        let make_bond = |schedule_calls: CallabilitySchedule| {
            CallableFixedRateBond::new(
                3,
                10_000.0,
                schedule.clone(),
                vec![0.05],
                day_counter.clone(),
                rolling,
                100.0,
                Some(issue),
                schedule_calls,
                Shared::clone(&settings),
            )
            .unwrap()
        };

        let cases = [
            (calls, 110.60975477, "callable"),
            (puts, 115.16559362, "puttable"),
            (both, 110.97509625, "callable/puttable"),
        ];
        let tol = 1.0e-8;
        for (schedule_calls, cached, label) in cases {
            let mut bond = make_bond(schedule_calls);
            bond.base_mut()
                .set_pricing_engine(SharedMut::clone(&engine));
            let price = bond.clean_price().unwrap();
            assert!(
                (price - cached).abs() <= tol,
                "{label}: clean {price} vs cached {cached} (error {})",
                (price - cached).abs()
            );
        }
    }

    /// `callablebonds.cpp` `Globals` fixture with a pinned evaluation date
    /// (C++ uses `Date::todaysDate()`).
    struct Globals {
        settings: Shared<Settings<Date>>,
        calendar: crate::time::calendar::Calendar,
        rolling: BusinessDayConvention,
        issue: Date,
        schedule: Schedule,
        curve: Handle<dyn YieldTermStructure>,
    }

    impl Globals {
        fn new(rate: Real) -> Self {
            let calendar = Target::new();
            let today = Date::new(3, Month::June, 2004);
            let settings = shared(Settings::new());
            settings.set_evaluation_date(today);
            let rolling = BusinessDayConvention::ModifiedFollowing;
            let settlement = calendar.advance(
                today,
                2,
                TimeUnit::Days,
                BusinessDayConvention::Following,
                false,
            );
            let issue = calendar.adjust(today - 100, BusinessDayConvention::Following);
            let maturity = calendar.advance(issue, 10, TimeUnit::Years, rolling, false);
            let schedule = MakeSchedule::new()
                .from(issue)
                .to(maturity)
                .with_calendar(calendar.clone())
                .with_frequency(Frequency::Semiannual)
                .with_convention(rolling)
                .with_termination_date_convention(rolling)
                .backwards()
                .build();
            let curve: Handle<dyn YieldTermStructure> =
                Handle::new(shared(FlatForward::with_rate(
                    settlement,
                    rate,
                    Actual365Fixed::new(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>);
            Self {
                settings,
                calendar,
                rolling,
                issue,
                schedule,
                curve,
            }
        }

        fn even_years(&self) -> Vec<Date> {
            (2..10)
                .step_by(2)
                .map(|i| {
                    self.calendar
                        .advance(self.issue, i, TimeUnit::Years, self.rolling, false)
                })
                .collect()
        }

        fn odd_years(&self) -> Vec<Date> {
            (1..10)
                .step_by(2)
                .map(|i| {
                    self.calendar
                        .advance(self.issue, i, TimeUnit::Years, self.rolling, false)
                })
                .collect()
        }
    }

    /// `callablebonds.cpp` testConsistency (`:224`): callable clean < plain <
    /// puttable on the HW tree (calls @ 110 even years, puts @ 90 odd years).
    #[test]
    fn callable_bond_consistency_orders_call_plain_and_put() {
        let g = Globals::new(0.032);
        let model = HullWhite::new(g.curve.clone(), 0.1, 0.01).unwrap();
        let day_counter = Thirty360::with_convention(Convention::BondBasis);

        let mut plain = FixedRateBond::new(
            3,
            100.0,
            g.schedule.clone(),
            vec![0.05],
            day_counter.clone(),
            BusinessDayConvention::Following,
            100.0,
            Some(g.issue),
            None,
            None,
            NullCalendar::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(&g.settings),
        )
        .unwrap();
        let discounting = shared_mut(DiscountingBondEngine::new(
            g.curve.clone(),
            None,
            Shared::clone(&g.settings),
        )) as SharedMut<dyn PricingEngine>;
        plain
            .bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&discounting));
        let plain_clean = plain.bond_mut().clean_price().unwrap();

        let calls: CallabilitySchedule = g
            .even_years()
            .into_iter()
            .map(|d| Callability::new(BondPrice::Clean(110.0), CallabilityType::Call, d))
            .collect();
        let puts: CallabilitySchedule = g
            .odd_years()
            .into_iter()
            .map(|d| Callability::new(BondPrice::Clean(90.0), CallabilityType::Put, d))
            .collect();

        let tree = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, 240, Shared::clone(&g.settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;

        let mut callable = CallableFixedRateBond::new(
            3,
            100.0,
            g.schedule.clone(),
            vec![0.05],
            day_counter.clone(),
            g.rolling,
            100.0,
            Some(g.issue),
            calls,
            Shared::clone(&g.settings),
        )
        .unwrap();
        callable
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&tree));
        let callable_clean = callable.clean_price().unwrap();

        let mut puttable = CallableFixedRateBond::new(
            3,
            100.0,
            g.schedule,
            vec![0.05],
            day_counter,
            g.rolling,
            100.0,
            Some(g.issue),
            puts,
            Shared::clone(&g.settings),
        )
        .unwrap();
        puttable.base_mut().set_pricing_engine(tree);
        let puttable_clean = puttable.clean_price().unwrap();

        assert!(
            plain_clean > callable_clean,
            "plain {plain_clean} should exceed callable {callable_clean}"
        );
        assert!(
            plain_clean < puttable_clean,
            "plain {plain_clean} should be below puttable {puttable_clean}"
        );
    }

    /// `callablebonds.cpp` testDegenerate (`:359`): empty and deeply OTM
    /// callability schedules reprice the straight fixed bond to 1e-4 on the
    /// HW tree (zero-coupon path deferred — `CallableZeroCouponBond` not ported).
    #[test]
    fn degenerate_callable_fixed_bond_matches_the_straight_bond() {
        let g = Globals::new(0.034);
        let model = HullWhite::new(g.curve.clone(), 0.1, 0.01).unwrap();
        let day_counter = Thirty360::with_convention(Convention::BondBasis);
        let tol = 1.0e-4;

        let mut coupon_bond = FixedRateBond::new(
            3,
            100.0,
            g.schedule.clone(),
            vec![0.05],
            day_counter.clone(),
            BusinessDayConvention::Following,
            100.0,
            Some(g.issue),
            None,
            None,
            NullCalendar::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(&g.settings),
        )
        .unwrap();
        let discounting = shared_mut(DiscountingBondEngine::new(
            g.curve.clone(),
            None,
            Shared::clone(&g.settings),
        )) as SharedMut<dyn PricingEngine>;
        coupon_bond
            .bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&discounting));
        let expected = coupon_bond.bond_mut().clean_price().unwrap();

        let tree = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, 240, Shared::clone(&g.settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;

        let mut empty = CallableFixedRateBond::new(
            3,
            100.0,
            g.schedule.clone(),
            vec![0.05],
            day_counter.clone(),
            g.rolling,
            100.0,
            Some(g.issue),
            Vec::new(),
            Shared::clone(&g.settings),
        )
        .unwrap();
        empty
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&tree));
        let price = empty.clean_price().unwrap();
        assert!(
            (price - expected).abs() <= tol,
            "empty callability: tree {price} vs straight {expected} (error {})",
            (price - expected).abs()
        );

        let mut otm: CallabilitySchedule = g
            .even_years()
            .into_iter()
            .map(|d| Callability::new(BondPrice::Clean(10_000.0), CallabilityType::Call, d))
            .collect();
        otm.extend(
            g.odd_years()
                .into_iter()
                .map(|d| Callability::new(BondPrice::Clean(0.0), CallabilityType::Put, d)),
        );
        let mut worthless = CallableFixedRateBond::new(
            3,
            100.0,
            g.schedule,
            vec![0.05],
            day_counter,
            g.rolling,
            100.0,
            Some(g.issue),
            otm,
            Shared::clone(&g.settings),
        )
        .unwrap();
        worthless.base_mut().set_pricing_engine(tree);
        let price = worthless.clean_price().unwrap();
        assert!(
            (price - expected).abs() <= tol,
            "OTM callability: tree {price} vs straight {expected} (error {})",
            (price - expected).abs()
        );
    }
}
