//! Convertible bonds.
//!
//! Port of QuantLib's `ql/instruments/bonds/convertiblebonds.{hpp,cpp}`: a bond
//! that may be converted into equity, optionally with call/put rights (including
//! soft calls). Pricing is supplied by
//! [`BinomialConvertibleEngine`](crate::pricingengines::bond::BinomialConvertibleEngine).
//!
//! Zero-, fixed-, and floating-coupon convertibles are covered here. As in
//! QuantLib, most inherited bond yield / dirty-price helpers refer to the
//! underlying plain-vanilla bond and ignore convertibility.

use std::any::Any;

use crate::cashflow::Leg;
use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::exercise::Exercise;
use crate::fail;
use crate::indexes::IborIndex;
use crate::indexes::index::Index;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{
    Bond, BondPrice, BondResults, Callability, CallabilitySchedule, CallabilityType, FixedRateBond,
    FloatingRateBond, ZeroCouponBond,
};
use crate::pricingengine::{Arguments, Results};
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendar::Calendar;
use crate::time::calendars::nullcalendar::NullCalendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::period::Period;
use crate::time::schedule::Schedule;
use crate::types::{Natural, Rate, Real, Spread};

/// Soft callability (`SoftCallability` in QuantLib): a call with a trigger.
pub fn soft_callability(price: BondPrice, date: Date, trigger: Real) -> Callability {
    Callability::soft(price, date, trigger)
}

/// Engine arguments for a convertible bond (`ConvertibleBond::arguments`).
#[derive(Default)]
pub struct ConvertibleBondArguments {
    /// Conversion exercise schedule.
    pub exercise: Option<Shared<dyn Exercise>>,
    /// Shares received per 100 face on conversion.
    pub conversion_ratio: Real,
    /// Remaining callability dates.
    pub callability_dates: Vec<Date>,
    /// Callability types aligned with the dates.
    pub callability_types: Vec<CallabilityType>,
    /// Dirty callability prices (per 100), aligned with the dates.
    pub callability_prices: Vec<Real>,
    /// Soft-call triggers aligned with the dates (`None` = hard call/put).
    pub callability_triggers: Vec<Option<Real>>,
    /// Full cash-flow leg (coupons + redemption).
    pub cashflows: Leg,
    /// Issue date.
    pub issue_date: Option<Date>,
    /// Settlement date.
    pub settlement_date: Option<Date>,
    /// Settlement lag in business days.
    pub settlement_days: Natural,
    /// Redemption amount per 100 face.
    pub redemption: Real,
}

impl Arguments for ConvertibleBondArguments {
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn validate(&self) -> QlResult<()> {
        if self.exercise.is_none() {
            fail!("no exercise given");
        }
        if self.conversion_ratio <= 0.0 {
            fail!(
                "positive conversion ratio required: {} not allowed",
                self.conversion_ratio
            );
        }
        if self.redemption < 0.0 {
            fail!(
                "non-negative redemption required: {} not allowed",
                self.redemption
            );
        }
        if self.settlement_date.is_none() {
            fail!("null settlement date");
        }
        if self.callability_dates.len() != self.callability_types.len()
            || self.callability_dates.len() != self.callability_prices.len()
            || self.callability_dates.len() != self.callability_triggers.len()
        {
            fail!("different number of callability dates / types / prices / triggers");
        }
        if self.cashflows.is_empty() {
            fail!("no cashflows given");
        }
        Ok(())
    }
}

/// Shared convertible-bond state composed into the concrete coupon flavours.
struct ConvertibleBondBase {
    instrument: InstrumentBase,
    bond: Bond,
    exercise: Shared<dyn Exercise>,
    conversion_ratio: Real,
    callability: CallabilitySchedule,
    redemption: Real,
    settings: Shared<Settings<Date>>,
    settlement_value: Option<Real>,
}

impl ConvertibleBondBase {
    fn new(
        exercise: Shared<dyn Exercise>,
        conversion_ratio: Real,
        callability: CallabilitySchedule,
        bond: Bond,
        redemption: Real,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let maturity = bond.maturity_date()?;
        if callability
            .last()
            .is_some_and(|last| last.date() > maturity)
        {
            let last = callability.last().expect("checked");
            fail!(
                "last callability date ({}) later than maturity ({})",
                last.date(),
                maturity
            );
        }
        let instrument = InstrumentBase::new();
        settings.register_eval_date_observer(&instrument.observer());
        Ok(Self {
            instrument,
            bond,
            exercise,
            conversion_ratio,
            callability,
            redemption,
            settings,
            settlement_value: None,
        })
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(args) = (arguments as &mut dyn Any).downcast_mut::<ConvertibleBondArguments>()
        else {
            fail!("wrong argument type");
        };
        args.exercise = Some(Shared::clone(&self.exercise));
        args.conversion_ratio = self.conversion_ratio;

        let settlement = self.bond.settlement_date(None)?;
        args.callability_dates.clear();
        args.callability_types.clear();
        args.callability_prices.clear();
        args.callability_triggers.clear();
        for callability in &self.callability {
            if event_has_occurred(
                callability.date(),
                &self.settings,
                Some(settlement),
                Some(false),
            )? {
                continue;
            }
            args.callability_types.push(callability.call_type());
            args.callability_dates.push(callability.date());
            let mut price = callability.price().amount();
            if let BondPrice::Clean(_) = callability.price() {
                price += self.bond.accrued_amount(Some(callability.date()))?;
            }
            args.callability_prices.push(price);
            args.callability_triggers.push(callability.trigger());
        }

        args.cashflows = self.bond.cashflows().to_vec();
        args.issue_date = self.bond.issue_date();
        args.settlement_date = Some(settlement);
        args.settlement_days = self.bond.settlement_days();
        args.redemption = self.redemption;
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<BondResults>() else {
            fail!("wrong result type");
        };
        self.settlement_value = results.settlement_value;
        self.instrument.store_results(&results.instrument);
        Ok(())
    }
}

/// A convertible fixed-coupon bond (`ConvertibleFixedCouponBond`).
pub struct ConvertibleFixedCouponBond {
    inner: ConvertibleBondBase,
}

impl ConvertibleFixedCouponBond {
    /// Builds a convertible fixed-coupon bond. Notionals are forced to 100, as
    /// in QuantLib.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        exercise: Shared<dyn Exercise>,
        conversion_ratio: Real,
        callability: CallabilitySchedule,
        issue_date: Date,
        settlement_days: Natural,
        coupons: Vec<Rate>,
        day_counter: DayCounter,
        schedule: Schedule,
        redemption: Real,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        Self::with_ex_coupon(
            exercise,
            conversion_ratio,
            callability,
            issue_date,
            settlement_days,
            coupons,
            day_counter,
            schedule,
            redemption,
            None,
            NullCalendar::new(),
            BusinessDayConvention::Unadjusted,
            false,
            settings,
        )
    }

    /// Builds a convertible fixed-coupon bond with an optional ex-coupon period.
    #[allow(clippy::too_many_arguments)]
    pub fn with_ex_coupon(
        exercise: Shared<dyn Exercise>,
        conversion_ratio: Real,
        callability: CallabilitySchedule,
        issue_date: Date,
        settlement_days: Natural,
        coupons: Vec<Rate>,
        day_counter: DayCounter,
        schedule: Schedule,
        redemption: Real,
        ex_coupon_period: Option<Period>,
        ex_coupon_calendar: Calendar,
        ex_coupon_convention: BusinessDayConvention,
        ex_coupon_end_of_month: bool,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let payment_convention = schedule.business_day_convention();
        let fixed = FixedRateBond::new(
            settlement_days,
            100.0,
            schedule,
            coupons,
            day_counter,
            payment_convention,
            redemption,
            Some(issue_date),
            None,
            ex_coupon_period,
            ex_coupon_calendar,
            ex_coupon_convention,
            ex_coupon_end_of_month,
            None,
            Shared::clone(&settings),
        )?;
        Ok(Self {
            inner: ConvertibleBondBase::new(
                exercise,
                conversion_ratio,
                callability,
                fixed.into_bond(),
                redemption,
                settings,
            )?,
        })
    }

    /// Conversion ratio (shares per 100 face).
    pub fn conversion_ratio(&self) -> Real {
        self.inner.conversion_ratio
    }

    /// Call/put schedule.
    pub fn callability(&self) -> &CallabilitySchedule {
        &self.inner.callability
    }

    /// Underlying bond cash-flow view.
    pub fn bond(&self) -> &Bond {
        &self.inner.bond
    }

    /// Theoretical settlement value from the engine.
    pub fn settlement_value(&mut self) -> QlResult<Real> {
        self.calculate()?;
        let Some(value) = self.inner.settlement_value else {
            fail!("settlement value not provided");
        };
        Ok(value)
    }
}

impl Instrument for ConvertibleFixedCouponBond {
    fn base(&self) -> &InstrumentBase {
        &self.inner.instrument
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.inner.instrument
    }

    fn is_expired(&self) -> QlResult<bool> {
        self.inner.bond.is_expired()
    }

    fn setup_expired(&mut self) {
        let expired = InstrumentResults {
            value: Some(0.0),
            ..InstrumentResults::default()
        };
        self.inner.instrument.store_results(&expired);
        self.inner.settlement_value = Some(0.0);
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        self.inner.setup_arguments(arguments)
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        self.inner.fetch_results(results)
    }
}

/// A convertible zero-coupon bond (`ConvertibleZeroCouponBond`).
pub struct ConvertibleZeroCouponBond {
    inner: ConvertibleBondBase,
}

impl ConvertibleZeroCouponBond {
    /// Builds a convertible zero-coupon bond. Notionals are forced to 100, as
    /// in QuantLib.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        exercise: Shared<dyn Exercise>,
        conversion_ratio: Real,
        callability: CallabilitySchedule,
        issue_date: Date,
        settlement_days: Natural,
        _day_counter: DayCounter,
        schedule: Schedule,
        redemption: Real,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let maturity = schedule.end_date();
        let calendar = schedule.calendar().clone();
        let convention = schedule.business_day_convention();
        let zero = ZeroCouponBond::new(
            settlement_days,
            calendar,
            100.0,
            maturity,
            convention,
            redemption,
            Some(issue_date),
            Shared::clone(&settings),
        )?;
        // Keep maturity on the schedule end (QL sets maturityDate_ from the
        // schedule before the redemption adjust).
        let mut bond = zero.into_bond();
        bond.set_maturity_date(maturity);
        Ok(Self {
            inner: ConvertibleBondBase::new(
                exercise,
                conversion_ratio,
                callability,
                bond,
                redemption,
                settings,
            )?,
        })
    }

    /// Conversion ratio (shares per 100 face).
    pub fn conversion_ratio(&self) -> Real {
        self.inner.conversion_ratio
    }

    /// Call/put schedule.
    pub fn callability(&self) -> &CallabilitySchedule {
        &self.inner.callability
    }

    /// Underlying bond cash-flow view.
    pub fn bond(&self) -> &Bond {
        &self.inner.bond
    }

    /// Theoretical settlement value from the engine.
    pub fn settlement_value(&mut self) -> QlResult<Real> {
        self.calculate()?;
        let Some(value) = self.inner.settlement_value else {
            fail!("settlement value not provided");
        };
        Ok(value)
    }
}

impl Instrument for ConvertibleZeroCouponBond {
    fn base(&self) -> &InstrumentBase {
        &self.inner.instrument
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.inner.instrument
    }

    fn is_expired(&self) -> QlResult<bool> {
        self.inner.bond.is_expired()
    }

    fn setup_expired(&mut self) {
        let expired = InstrumentResults {
            value: Some(0.0),
            ..InstrumentResults::default()
        };
        self.inner.instrument.store_results(&expired);
        self.inner.settlement_value = Some(0.0);
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        self.inner.setup_arguments(arguments)
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        self.inner.fetch_results(results)
    }
}

/// A convertible floating-rate bond (`ConvertibleFloatingRateBond`).
pub struct ConvertibleFloatingRateBond {
    inner: ConvertibleBondBase,
}

impl ConvertibleFloatingRateBond {
    /// Builds a convertible floating-rate bond. Notionals are forced to 100, as
    /// in QuantLib.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        exercise: Shared<dyn Exercise>,
        conversion_ratio: Real,
        callability: CallabilitySchedule,
        issue_date: Date,
        settlement_days: Natural,
        index: Shared<IborIndex>,
        fixing_days: Natural,
        spreads: Vec<Spread>,
        day_counter: DayCounter,
        schedule: Schedule,
        redemption: Real,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        Self::with_ex_coupon(
            exercise,
            conversion_ratio,
            callability,
            issue_date,
            settlement_days,
            index,
            fixing_days,
            spreads,
            day_counter,
            schedule,
            redemption,
            None,
            NullCalendar::new(),
            BusinessDayConvention::Unadjusted,
            false,
            settings,
        )
    }

    /// Builds a convertible floating-rate bond with an optional ex-coupon period.
    #[allow(clippy::too_many_arguments)]
    pub fn with_ex_coupon(
        exercise: Shared<dyn Exercise>,
        conversion_ratio: Real,
        callability: CallabilitySchedule,
        issue_date: Date,
        settlement_days: Natural,
        index: Shared<IborIndex>,
        fixing_days: Natural,
        spreads: Vec<Spread>,
        day_counter: DayCounter,
        schedule: Schedule,
        redemption: Real,
        ex_coupon_period: Option<Period>,
        ex_coupon_calendar: Calendar,
        ex_coupon_convention: BusinessDayConvention,
        ex_coupon_end_of_month: bool,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let payment_convention = schedule.business_day_convention();
        let floating = FloatingRateBond::new(
            settlement_days,
            100.0,
            schedule,
            Shared::clone(&index),
            day_counter,
            payment_convention,
            Some(fixing_days),
            Vec::new(),
            spreads,
            Vec::new(),
            Vec::new(),
            redemption,
            Some(issue_date),
            ex_coupon_period,
            ex_coupon_calendar,
            ex_coupon_convention,
            ex_coupon_end_of_month,
            None,
            Shared::clone(&settings),
        )?;
        let inner = ConvertibleBondBase::new(
            exercise,
            conversion_ratio,
            callability,
            floating.into_bond(),
            redemption,
            settings,
        )?;
        inner.instrument.register_with(index.observable());
        Ok(Self { inner })
    }

    /// Conversion ratio (shares per 100 face).
    pub fn conversion_ratio(&self) -> Real {
        self.inner.conversion_ratio
    }

    /// Call/put schedule.
    pub fn callability(&self) -> &CallabilitySchedule {
        &self.inner.callability
    }

    /// Underlying bond cash-flow view.
    pub fn bond(&self) -> &Bond {
        &self.inner.bond
    }

    /// Theoretical settlement value from the engine.
    pub fn settlement_value(&mut self) -> QlResult<Real> {
        self.calculate()?;
        let Some(value) = self.inner.settlement_value else {
            fail!("settlement value not provided");
        };
        Ok(value)
    }
}

impl Instrument for ConvertibleFloatingRateBond {
    fn base(&self) -> &InstrumentBase {
        &self.inner.instrument
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.inner.instrument
    }

    fn is_expired(&self) -> QlResult<bool> {
        self.inner.bond.is_expired()
    }

    fn setup_expired(&mut self) {
        let expired = InstrumentResults {
            value: Some(0.0),
            ..InstrumentResults::default()
        };
        self.inner.instrument.store_results(&expired);
        self.inner.settlement_value = Some(0.0);
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        self.inner.setup_arguments(arguments)
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        self.inner.fetch_results(results)
    }
}
