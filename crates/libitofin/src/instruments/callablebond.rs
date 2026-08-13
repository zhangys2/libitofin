//! Callable / puttable bonds.
//!
//! Port of QuantLib's `ql/experimental/callablebonds/callablebond.{hpp,cpp}`.
//! [`CallableFixedRateBond`] and [`CallableZeroCouponBond`] carry a
//! [`CallabilitySchedule`] of European/Bermudan call or put dates and price on
//! a short-rate lattice through
//! [`TreeCallableFixedRateBondEngine`](crate::pricingengines::bond::TreeCallableFixedRateBondEngine)
//! (C++'s `TreeCallableZeroCouponBondEngine` is the same engine).
//!
//! This slice mirrors QuantLib's argument setup (`setupArguments`): coupons and
//! a par redemption feed the discretized bond, and clean call prices are
//! converted to dirty with the accrued at the call date. The Black European
//! engine is in
//! [`BlackCallableFixedRateBondEngine`](crate::pricingengines::bond::BlackCallableFixedRateBondEngine);
//! [`implied_volatility`](CallableFixedRateBond::implied_volatility) inverts it.
//! Tree OAS / clean-price-OAS / effective duration and convexity live on the
//! callable instruments (spread wired through
//! [`TreeCallableFixedRateBondEngine`](crate::pricingengines::bond::TreeCallableFixedRateBondEngine)).
//! Clean call prices use indenture accrued (GitHub issue #2236) so OAS stays
//! continuous through an ex-coupon window.

use std::any::Any;
use std::cell::RefCell;

use crate::cashflow::Leg;
use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::fail;
use crate::handle::Handle;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{Bond, BondPrice, BondResults, FixedRateBond, ZeroCouponBond};
use crate::interestrate::{Compounding, InterestRate};
use crate::math::solver1d::Solver1D;
use crate::math::solvers1d::brent::Brent;
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::bond::BlackCallableFixedRateBondEngine;
use crate::quotes::{Quote, SimpleQuote};
use crate::settings::Settings;
use crate::shared::{Shared, shared};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendar::Calendar;
use crate::time::calendars::nullcalendar::NullCalendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::frequency::Frequency;
use crate::time::period::Period;
use crate::time::schedule::Schedule;
use crate::types::{Natural, Rate, Real, Size, Spread, Volatility};

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
pub struct CallableBondArguments {
    /// The settlement date.
    pub settlement_date: Option<Date>,
    /// Full bond cash-flow leg (coupons + redemption), as `Bond::arguments`.
    pub cashflows: Leg,
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
    /// Accrual / payment day counter (yield and duration convention).
    pub payment_day_counter: Option<DayCounter>,
    /// Coupon frequency (`Once` for zeros; Black remaps to Annual).
    pub frequency: Frequency,
    /// Callability dirty prices (per 100), aligned with the dates/types.
    pub callability_prices: Vec<Real>,
    /// Callability types aligned with the dates/prices.
    pub callability_types: Vec<CallabilityType>,
    /// Callability dates.
    pub callability_dates: Vec<Date>,
    /// Continuous spread added to the model (0 unless set by an OAS solve).
    pub spread: Spread,
}

impl Default for CallableBondArguments {
    fn default() -> Self {
        Self {
            settlement_date: None,
            cashflows: Vec::new(),
            coupon_dates: Vec::new(),
            coupon_amounts: Vec::new(),
            face_amount: 0.0,
            redemption: 0.0,
            redemption_date: None,
            payment_day_counter: None,
            frequency: Frequency::NoFrequency,
            callability_prices: Vec::new(),
            callability_types: Vec::new(),
            callability_dates: Vec::new(),
            spread: 0.0,
        }
    }
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

/// Fills [`CallableBondArguments`] from a bond's cash flows and callability
/// schedule (`CallableBond::setupArguments`, `callablebond.cpp:410-480`).
fn fill_callable_bond_arguments(
    args: &mut CallableBondArguments,
    bond: &Bond,
    put_call_schedule: &[Callability],
    face_amount: Real,
    payment_day_counter: DayCounter,
    frequency: Frequency,
    settings: &Settings<Date>,
) -> QlResult<()> {
    let settlement = bond.settlement_date(None)?;
    args.settlement_date = Some(settlement);
    args.face_amount = face_amount;
    args.payment_day_counter = Some(payment_day_counter);
    args.frequency = frequency;
    args.cashflows = bond.cashflows().clone();

    let cashflows = bond.cashflows();
    let count = cashflows.len();
    let redemption = &cashflows[count - 1];
    args.redemption = redemption.amount()?;
    args.redemption_date = Some(redemption.date());

    args.coupon_dates.clear();
    args.coupon_amounts.clear();
    for flow in &cashflows[..count - 1] {
        if !flow.has_occurred(settings, Some(settlement), Some(false))?
            && !flow.trading_ex_coupon(settings, Some(settlement))?
        {
            args.coupon_dates.push(flow.date());
            args.coupon_amounts.push(flow.amount()?);
        }
    }

    args.callability_dates.clear();
    args.callability_prices.clear();
    args.callability_types.clear();
    for callability in put_call_schedule {
        if event_has_occurred(callability.date(), settings, Some(settlement), Some(false))? {
            continue;
        }
        let call_date = callability.date();
        args.callability_dates.push(call_date);
        args.callability_types.push(callability.call_type());
        let mut price = callability.price().amount();
        if let BondPrice::Clean(_) = callability.price() {
            // Convert the clean call price to dirty with indenture accrued at
            // the call date (`callablebond.cpp:453-477`, GitHub issue #2236):
            // if the coupon already trades ex-coupon, undo the market negative
            // accrued by adding the full coupon amount back.
            for flow in cashflows {
                if !event_has_occurred(flow.date(), settings, Some(call_date), Some(false))? {
                    if let Some(coupon) = flow.as_coupon() {
                        let mut accrued = coupon.accrued_amount(call_date)?;
                        if coupon.trades_ex_coupon_on(call_date) {
                            accrued += flow.amount()?;
                        }
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

/// Black implied forward yield volatility (`CallableBond::impliedVolatility`).
///
/// Inverts [`BlackCallableFixedRateBondEngine`] so the settlement value matches
/// `target_price` (clean quotes are converted to dirty with accrued).
#[allow(clippy::too_many_arguments)]
fn implied_volatility_from(
    instrument: &dyn Instrument,
    bond: &Bond,
    face_amount: Real,
    target_price: BondPrice,
    discount_curve: Handle<dyn YieldTermStructure>,
    settings: Shared<Settings<Date>>,
    accuracy: Real,
    max_evaluations: Size,
    min_vol: Volatility,
    max_vol: Volatility,
) -> QlResult<Volatility> {
    if instrument.is_expired()? {
        fail!("instrument expired");
    }

    let dirty_target = match target_price {
        BondPrice::Dirty(amount) => amount,
        BondPrice::Clean(amount) => amount + bond.accrued_amount(None)?,
    };
    let target_value = dirty_target * face_amount / 100.0;

    let vol = shared(SimpleQuote::new(0.0));
    let mut engine = BlackCallableFixedRateBondEngine::new(
        Handle::new(Shared::clone(&vol) as Shared<dyn Quote>),
        discount_curve,
        settings,
    );
    instrument.setup_arguments(engine.arguments_mut())?;
    engine.arguments_mut().validate()?;

    let failure = RefCell::new(None);
    let objective = |x: Volatility| {
        vol.set_value(x);
        match engine.calculate() {
            Ok(()) => {
                let Some(results) = (engine.results() as &dyn Any).downcast_ref::<BondResults>()
                else {
                    failure.borrow_mut().get_or_insert_with(|| {
                        crate::errors::QlError::new("wrong result type", file!(), line!())
                    });
                    return Real::NAN;
                };
                let Some(settlement) = results.settlement_value else {
                    failure.borrow_mut().get_or_insert_with(|| {
                        crate::errors::QlError::new(
                            "settlement value not provided",
                            file!(),
                            line!(),
                        )
                    });
                    return Real::NAN;
                };
                settlement - target_value
            }
            Err(error) => {
                failure.borrow_mut().get_or_insert(error);
                Real::NAN
            }
        }
    };

    let guess = 0.5 * (min_vol + max_vol);
    let mut solver = Brent::new().with_max_evaluations(max_evaluations);
    let root = solver.solve_bracketed(objective, accuracy, guess, min_vol, max_vol);
    match failure.into_inner() {
        Some(error) => Err(error),
        None => root,
    }
}

/// Converts a continuous model spread to a conventional OAS quote
/// (`callablebond.cpp` `continuousToConv`).
fn continuous_to_conv(
    oas: Spread,
    bond: &Bond,
    yts: &dyn YieldTermStructure,
    day_counter: DayCounter,
    compounding: Compounding,
    frequency: Frequency,
) -> QlResult<Spread> {
    let maturity = bond.maturity_date()?;
    let zz = yts
        .zero_rate_date(
            maturity,
            day_counter.clone(),
            Compounding::Continuous,
            Frequency::NoFrequency,
            true,
        )?
        .rate();
    let base_rate = InterestRate::new(
        zz,
        day_counter.clone(),
        Compounding::Continuous,
        Frequency::NoFrequency,
    )?;
    let spreaded_rate = InterestRate::new(
        oas + zz,
        day_counter.clone(),
        Compounding::Continuous,
        Frequency::NoFrequency,
    )?;
    let reference = yts.reference_date()?;
    let br = base_rate
        .equivalent_rate_between(
            day_counter.clone(),
            compounding,
            frequency,
            reference,
            maturity,
        )?
        .rate();
    let sr = spreaded_rate
        .equivalent_rate_between(day_counter, compounding, frequency, reference, maturity)?
        .rate();
    Ok(sr - br)
}

/// Converts a conventional OAS quote to a continuous model spread
/// (`callablebond.cpp` `convToContinuous`).
fn conv_to_continuous(
    oas: Spread,
    bond: &Bond,
    yts: &dyn YieldTermStructure,
    day_counter: DayCounter,
    compounding: Compounding,
    frequency: Frequency,
) -> QlResult<Spread> {
    let maturity = bond.maturity_date()?;
    let zz = yts
        .zero_rate_date(maturity, day_counter.clone(), compounding, frequency, true)?
        .rate();
    let base_rate = InterestRate::new(zz, day_counter.clone(), compounding, frequency)?;
    let spreaded_rate = InterestRate::new(oas + zz, day_counter.clone(), compounding, frequency)?;
    let reference = yts.reference_date()?;
    let br = base_rate
        .equivalent_rate_between(
            day_counter.clone(),
            Compounding::Continuous,
            Frequency::NoFrequency,
            reference,
            maturity,
        )?
        .rate();
    let sr = spreaded_rate
        .equivalent_rate_between(
            day_counter,
            Compounding::Continuous,
            Frequency::NoFrequency,
            reference,
            maturity,
        )?
        .rate();
    Ok(sr - br)
}

/// NPV under a continuous short-rate spread on the attached tree engine
/// (`CallableBond::NPVSpreadHelper`).
fn npv_at_continuous_spread(
    instrument: &dyn Instrument,
    continuous_spread: Spread,
) -> QlResult<Real> {
    let Some(engine) = instrument.base().pricing_engine().cloned() else {
        fail!("null pricing engine");
    };
    let mut eng = engine.borrow_mut();
    instrument.setup_arguments(eng.arguments_mut())?;
    {
        let Some(args) =
            (eng.arguments_mut() as &mut dyn Any).downcast_mut::<CallableBondArguments>()
        else {
            fail!("wrong argument type");
        };
        args.spread = continuous_spread;
    }
    eng.arguments_mut().validate()?;
    eng.calculate()?;
    let Some(results) = eng.results().as_instrument_results() else {
        fail!("no results returned from pricing engine");
    };
    let Some(value) = results.value else {
        fail!("null NPV from pricing engine");
    };
    Ok(value)
}

/// Option-adjusted spread that matches `clean_price` on the attached engine
/// (`CallableBond::OAS`).
#[allow(clippy::too_many_arguments)]
fn oas_from(
    instrument: &dyn Instrument,
    bond: &Bond,
    clean_price: Real,
    engine_ts: Handle<dyn YieldTermStructure>,
    day_counter: DayCounter,
    compounding: Compounding,
    frequency: Frequency,
    settlement: Option<Date>,
    accuracy: Real,
    max_iterations: Size,
    guess: Spread,
) -> QlResult<Spread> {
    let settlement = match settlement {
        Some(date) => date,
        None => bond.settlement_date(None)?,
    };
    let dirty_price = (clean_price + bond.accrued_amount(Some(settlement))?)
        * bond.notional(Some(settlement))?
        / 100.0;

    let failure = RefCell::new(None);
    let objective = |x: Spread| match npv_at_continuous_spread(instrument, x) {
        Ok(npv) => dirty_price - npv,
        Err(error) => {
            failure.borrow_mut().get_or_insert(error);
            Real::NAN
        }
    };
    let mut solver = Brent::new().with_max_evaluations(max_iterations);
    let continuous = match solver.solve(objective, accuracy, guess, 0.001) {
        Ok(root) => root,
        Err(error) => {
            if let Some(parked) = failure.into_inner() {
                return Err(parked);
            }
            return Err(error);
        }
    };
    if let Some(error) = failure.into_inner() {
        return Err(error);
    }

    continuous_to_conv(
        continuous,
        bond,
        &*engine_ts.current_link()?,
        day_counter,
        compounding,
        frequency,
    )
}

/// Clean price implied by a conventional OAS (`CallableBond::cleanPriceOAS`).
#[allow(clippy::too_many_arguments)]
fn clean_price_oas_from(
    instrument: &dyn Instrument,
    bond: &Bond,
    oas: Spread,
    engine_ts: Handle<dyn YieldTermStructure>,
    day_counter: DayCounter,
    compounding: Compounding,
    frequency: Frequency,
    settlement: Option<Date>,
) -> QlResult<Real> {
    let settlement = match settlement {
        Some(date) => date,
        None => bond.settlement_date(None)?,
    };
    let continuous = conv_to_continuous(
        oas,
        bond,
        &*engine_ts.current_link()?,
        day_counter,
        compounding,
        frequency,
    )?;
    let npv = npv_at_continuous_spread(instrument, continuous)?;
    Ok(npv * 100.0 / bond.notional(Some(settlement))? - bond.accrued_amount(Some(settlement))?)
}

/// Dirty-price finite-difference duration of a conventional OAS
/// (`CallableBond::effectiveDuration`).
#[allow(clippy::too_many_arguments)]
fn effective_duration_from(
    instrument: &dyn Instrument,
    bond: &Bond,
    oas: Spread,
    engine_ts: Handle<dyn YieldTermStructure>,
    day_counter: DayCounter,
    compounding: Compounding,
    frequency: Frequency,
    bump: Real,
) -> QlResult<Real> {
    let p = clean_price_oas_from(
        instrument,
        bond,
        oas,
        engine_ts.clone(),
        day_counter.clone(),
        compounding,
        frequency,
        None,
    )?;
    let p_up = clean_price_oas_from(
        instrument,
        bond,
        oas + bump,
        engine_ts.clone(),
        day_counter.clone(),
        compounding,
        frequency,
        None,
    )?;
    let p_down = clean_price_oas_from(
        instrument,
        bond,
        oas - bump,
        engine_ts,
        day_counter,
        compounding,
        frequency,
        None,
    )?;
    let dirty = p + bond.accrued_amount(None)?;
    if dirty == 0.0 {
        Ok(0.0)
    } else {
        Ok((p_down - p_up) / (2.0 * dirty * bump))
    }
}

/// Dirty-price finite-difference convexity of a conventional OAS
/// (`CallableBond::effectiveConvexity`).
#[allow(clippy::too_many_arguments)]
fn effective_convexity_from(
    instrument: &dyn Instrument,
    bond: &Bond,
    oas: Spread,
    engine_ts: Handle<dyn YieldTermStructure>,
    day_counter: DayCounter,
    compounding: Compounding,
    frequency: Frequency,
    bump: Real,
) -> QlResult<Real> {
    let p = clean_price_oas_from(
        instrument,
        bond,
        oas,
        engine_ts.clone(),
        day_counter.clone(),
        compounding,
        frequency,
        None,
    )?;
    let p_up = clean_price_oas_from(
        instrument,
        bond,
        oas + bump,
        engine_ts.clone(),
        day_counter.clone(),
        compounding,
        frequency,
        None,
    )?;
    let p_down = clean_price_oas_from(
        instrument,
        bond,
        oas - bump,
        engine_ts,
        day_counter,
        compounding,
        frequency,
        None,
    )?;
    let dirty = p + bond.accrued_amount(None)?;
    if dirty == 0.0 {
        Ok(0.0)
    } else {
        Ok((p_up + p_down - 2.0 * p) / (bump * bump * dirty))
    }
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
        Self::with_ex_coupon(
            settlement_days,
            face_amount,
            schedule,
            coupons,
            accrual_day_counter,
            payment_convention,
            redemption,
            issue_date,
            put_call_schedule,
            None,
            NullCalendar::new(),
            BusinessDayConvention::Unadjusted,
            false,
            settings,
        )
    }

    /// Builds a callable fixed-rate bond with an ex-coupon period
    /// (`CallableFixedRateBond` ctor trailing arguments).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    #[allow(clippy::too_many_arguments)]
    pub fn with_ex_coupon(
        settlement_days: Natural,
        face_amount: Real,
        schedule: Schedule,
        coupons: Vec<Rate>,
        accrual_day_counter: DayCounter,
        payment_convention: BusinessDayConvention,
        redemption: Real,
        issue_date: Option<Date>,
        put_call_schedule: CallabilitySchedule,
        ex_coupon_period: Option<Period>,
        ex_coupon_calendar: Calendar,
        ex_coupon_convention: BusinessDayConvention,
        ex_coupon_end_of_month: bool,
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
            ex_coupon_period,
            ex_coupon_calendar,
            ex_coupon_convention,
            ex_coupon_end_of_month,
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

    /// Black implied forward yield volatility for a European put/call schedule
    /// (`CallableBond::impliedVolatility`).
    ///
    /// # Errors
    ///
    /// Fails when the bond is expired, arguments are incomplete, or Brent cannot
    /// invert the Black engine onto `target_price`.
    #[allow(clippy::too_many_arguments)]
    pub fn implied_volatility(
        &self,
        target_price: BondPrice,
        discount_curve: Handle<dyn YieldTermStructure>,
        accuracy: Real,
        max_evaluations: Size,
        min_vol: Volatility,
        max_vol: Volatility,
    ) -> QlResult<Volatility> {
        implied_volatility_from(
            self,
            self.bond(),
            self.face_amount,
            target_price,
            discount_curve,
            Shared::clone(&self.settings),
            accuracy,
            max_evaluations,
            min_vol,
            max_vol,
        )
    }

    /// Option-adjusted spread that matches `clean_price` on the attached tree
    /// engine (`CallableBond::OAS`). Defaults: settlement = bond settlement,
    /// accuracy `1e-10`, max iterations `100`, guess `0.0`.
    ///
    /// # Errors
    ///
    /// Fails when no engine is attached, the tree cannot price, or Brent cannot
    /// invert the spread.
    #[allow(clippy::too_many_arguments)]
    pub fn oas(
        &self,
        clean_price: Real,
        engine_ts: Handle<dyn YieldTermStructure>,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement: Option<Date>,
        accuracy: Option<Real>,
        max_iterations: Option<Size>,
        guess: Option<Spread>,
    ) -> QlResult<Spread> {
        oas_from(
            self,
            self.bond(),
            clean_price,
            engine_ts,
            day_counter,
            compounding,
            frequency,
            settlement,
            accuracy.unwrap_or(1.0e-10),
            max_iterations.unwrap_or(100),
            guess.unwrap_or(0.0),
        )
    }

    /// Clean price implied by a conventional `oas` (`CallableBond::cleanPriceOAS`).
    ///
    /// # Errors
    ///
    /// As [`oas`](Self::oas).
    #[allow(clippy::too_many_arguments)]
    pub fn clean_price_oas(
        &self,
        oas: Spread,
        engine_ts: Handle<dyn YieldTermStructure>,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement: Option<Date>,
    ) -> QlResult<Real> {
        clean_price_oas_from(
            self,
            self.bond(),
            oas,
            engine_ts,
            day_counter,
            compounding,
            frequency,
            settlement,
        )
    }

    /// Effective duration of a conventional `oas` via a dirty-price bump
    /// (`CallableBond::effectiveDuration`). `bump` defaults to `2e-4`.
    ///
    /// # Errors
    ///
    /// As [`oas`](Self::oas).
    #[allow(clippy::too_many_arguments)]
    pub fn effective_duration(
        &self,
        oas: Spread,
        engine_ts: Handle<dyn YieldTermStructure>,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: Option<Real>,
    ) -> QlResult<Real> {
        effective_duration_from(
            self,
            self.bond(),
            oas,
            engine_ts,
            day_counter,
            compounding,
            frequency,
            bump.unwrap_or(2.0e-4),
        )
    }

    /// Effective convexity of a conventional `oas` via a dirty-price bump
    /// (`CallableBond::effectiveConvexity`). `bump` defaults to `2e-4`.
    ///
    /// # Errors
    ///
    /// As [`oas`](Self::oas).
    #[allow(clippy::too_many_arguments)]
    pub fn effective_convexity(
        &self,
        oas: Spread,
        engine_ts: Handle<dyn YieldTermStructure>,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: Option<Real>,
    ) -> QlResult<Real> {
        effective_convexity_from(
            self,
            self.bond(),
            oas,
            engine_ts,
            day_counter,
            compounding,
            frequency,
            bump.unwrap_or(2.0e-4),
        )
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
        fill_callable_bond_arguments(
            args,
            self.bond.bond(),
            &self.put_call_schedule,
            self.face_amount,
            self.bond.day_counter().clone(),
            self.bond.frequency(),
            &self.settings,
        )
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

/// A callable / puttable zero-coupon bond.
pub struct CallableZeroCouponBond {
    base: InstrumentBase,
    bond: ZeroCouponBond,
    put_call_schedule: CallabilitySchedule,
    face_amount: Real,
    payment_day_counter: DayCounter,
    settings: Shared<Settings<Date>>,
    settlement_value: Option<Real>,
}

impl CallableZeroCouponBond {
    /// Builds a callable zero-coupon bond (`CallableZeroCouponBond` ctor).
    ///
    /// # Errors
    ///
    /// Fails if a call/put date is after the bond maturity, or on any bond
    /// construction error.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        settlement_days: Natural,
        face_amount: Real,
        calendar: Calendar,
        maturity_date: Date,
        day_counter: DayCounter,
        payment_convention: BusinessDayConvention,
        redemption: Real,
        issue_date: Option<Date>,
        put_call_schedule: CallabilitySchedule,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<CallableZeroCouponBond> {
        for callability in &put_call_schedule {
            if callability.date() > maturity_date {
                fail!("bond cannot mature before the last call/put date");
            }
        }
        let bond = ZeroCouponBond::new(
            settlement_days,
            calendar,
            face_amount,
            maturity_date,
            payment_convention,
            redemption,
            issue_date,
            Shared::clone(&settings),
        )?;
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(CallableZeroCouponBond {
            base,
            bond,
            put_call_schedule,
            face_amount,
            payment_day_counter: day_counter,
            settings,
            settlement_value: None,
        })
    }

    /// The put/call schedule.
    pub fn callability(&self) -> &CallabilitySchedule {
        &self.put_call_schedule
    }

    /// The underlying [`Bond`] base.
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

    /// Black implied forward yield volatility for a European put/call schedule
    /// (`CallableBond::impliedVolatility`).
    ///
    /// # Errors
    ///
    /// Fails when the bond is expired, arguments are incomplete, or Brent cannot
    /// invert the Black engine onto `target_price`.
    #[allow(clippy::too_many_arguments)]
    pub fn implied_volatility(
        &self,
        target_price: BondPrice,
        discount_curve: Handle<dyn YieldTermStructure>,
        accuracy: Real,
        max_evaluations: Size,
        min_vol: Volatility,
        max_vol: Volatility,
    ) -> QlResult<Volatility> {
        implied_volatility_from(
            self,
            self.bond(),
            self.face_amount,
            target_price,
            discount_curve,
            Shared::clone(&self.settings),
            accuracy,
            max_evaluations,
            min_vol,
            max_vol,
        )
    }

    /// Option-adjusted spread that matches `clean_price` on the attached tree
    /// engine (`CallableBond::OAS`). Defaults: settlement = bond settlement,
    /// accuracy `1e-10`, max iterations `100`, guess `0.0`.
    ///
    /// # Errors
    ///
    /// Fails when no engine is attached, the tree cannot price, or Brent cannot
    /// invert the spread.
    #[allow(clippy::too_many_arguments)]
    pub fn oas(
        &self,
        clean_price: Real,
        engine_ts: Handle<dyn YieldTermStructure>,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement: Option<Date>,
        accuracy: Option<Real>,
        max_iterations: Option<Size>,
        guess: Option<Spread>,
    ) -> QlResult<Spread> {
        oas_from(
            self,
            self.bond(),
            clean_price,
            engine_ts,
            day_counter,
            compounding,
            frequency,
            settlement,
            accuracy.unwrap_or(1.0e-10),
            max_iterations.unwrap_or(100),
            guess.unwrap_or(0.0),
        )
    }

    /// Clean price implied by a conventional `oas` (`CallableBond::cleanPriceOAS`).
    ///
    /// # Errors
    ///
    /// As [`oas`](Self::oas).
    #[allow(clippy::too_many_arguments)]
    pub fn clean_price_oas(
        &self,
        oas: Spread,
        engine_ts: Handle<dyn YieldTermStructure>,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement: Option<Date>,
    ) -> QlResult<Real> {
        clean_price_oas_from(
            self,
            self.bond(),
            oas,
            engine_ts,
            day_counter,
            compounding,
            frequency,
            settlement,
        )
    }

    /// Effective duration of a conventional `oas` via a dirty-price bump
    /// (`CallableBond::effectiveDuration`). `bump` defaults to `2e-4`.
    ///
    /// # Errors
    ///
    /// As [`oas`](Self::oas).
    #[allow(clippy::too_many_arguments)]
    pub fn effective_duration(
        &self,
        oas: Spread,
        engine_ts: Handle<dyn YieldTermStructure>,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: Option<Real>,
    ) -> QlResult<Real> {
        effective_duration_from(
            self,
            self.bond(),
            oas,
            engine_ts,
            day_counter,
            compounding,
            frequency,
            bump.unwrap_or(2.0e-4),
        )
    }

    /// Effective convexity of a conventional `oas` via a dirty-price bump
    /// (`CallableBond::effectiveConvexity`). `bump` defaults to `2e-4`.
    ///
    /// # Errors
    ///
    /// As [`oas`](Self::oas).
    #[allow(clippy::too_many_arguments)]
    pub fn effective_convexity(
        &self,
        oas: Spread,
        engine_ts: Handle<dyn YieldTermStructure>,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: Option<Real>,
    ) -> QlResult<Real> {
        effective_convexity_from(
            self,
            self.bond(),
            oas,
            engine_ts,
            day_counter,
            compounding,
            frequency,
            bump.unwrap_or(2.0e-4),
        )
    }
}

impl Instrument for CallableZeroCouponBond {
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
        fill_callable_bond_arguments(
            args,
            self.bond.bond(),
            &self.put_call_schedule,
            self.face_amount,
            self.payment_day_counter.clone(),
            Frequency::Once,
            &self.settings,
        )
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
    use crate::instrument::Instrument;
    use crate::instruments::{FixedRateBond, ZeroCouponBond};
    use crate::interestrate::Compounding;
    use crate::models::shortrate::HullWhite;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::bond::{
        BlackCallableFixedRateBondEngine, BlackCallableZeroCouponBondEngine, DiscountingBondEngine,
        TreeCallableFixedRateBondEngine,
    };
    use crate::quotes::{Quote, SimpleQuote};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::target::Target;
    use crate::time::calendars::unitedstates::{Market, UnitedStates};
    use crate::time::date::Month;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::schedule::{MakeSchedule, Schedule};
    use crate::time::timeunit::TimeUnit;
    use crate::types::Natural;

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
        maturity: Date,
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
            let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
                settlement,
                rate,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            ))
                as Shared<dyn YieldTermStructure>);
            Self {
                settings,
                calendar,
                rolling,
                issue,
                maturity,
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
    /// HW tree.
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
        empty.base_mut().set_pricing_engine(SharedMut::clone(&tree));
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

    /// `callablebonds.cpp` testDegenerate (`:359`): empty and deeply OTM
    /// callability schedules reprice the straight zero-coupon bond to 1e-4.
    #[test]
    fn degenerate_callable_zero_bond_matches_the_straight_bond() {
        let g = Globals::new(0.034);
        let model = HullWhite::new(g.curve.clone(), 0.1, 0.01).unwrap();
        let day_counter = Thirty360::with_convention(Convention::BondBasis);
        let tol = 1.0e-4;

        let mut zero = ZeroCouponBond::new(
            3,
            g.calendar.clone(),
            100.0,
            g.maturity,
            g.rolling,
            100.0,
            Some(g.issue),
            Shared::clone(&g.settings),
        )
        .unwrap();
        let discounting = shared_mut(DiscountingBondEngine::new(
            g.curve.clone(),
            None,
            Shared::clone(&g.settings),
        )) as SharedMut<dyn PricingEngine>;
        zero.bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&discounting));
        let expected = zero.bond_mut().clean_price().unwrap();

        let tree = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, 240, Shared::clone(&g.settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;

        let mut empty = CallableZeroCouponBond::new(
            3,
            100.0,
            g.calendar.clone(),
            g.maturity,
            day_counter.clone(),
            g.rolling,
            100.0,
            Some(g.issue),
            Vec::new(),
            Shared::clone(&g.settings),
        )
        .unwrap();
        empty.base_mut().set_pricing_engine(SharedMut::clone(&tree));
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
        let mut worthless = CallableZeroCouponBond::new(
            3,
            100.0,
            g.calendar.clone(),
            g.maturity,
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

    /// `callablebonds.cpp` testObservability (`:300`): a quote-backed flat
    /// curve move invalidates the callable zero's cached NPV.
    #[test]
    fn callable_zero_bond_observes_the_discount_curve_quote() {
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

        let quote = shared(SimpleQuote::new(0.03));
        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::new(
            settlement,
            Handle::new(Shared::clone(&quote) as Shared<dyn Quote>),
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let model = HullWhite::new(curve, 0.1, 0.01).unwrap();

        let mut callabilities: CallabilitySchedule = (2..10)
            .step_by(2)
            .map(|i| {
                Callability::new(
                    BondPrice::Clean(110.0),
                    CallabilityType::Call,
                    calendar.advance(issue, i, TimeUnit::Years, rolling, false),
                )
            })
            .collect();
        callabilities.extend((1..10).step_by(2).map(|i| {
            Callability::new(
                BondPrice::Clean(90.0),
                CallabilityType::Put,
                calendar.advance(issue, i, TimeUnit::Years, rolling, false),
            )
        }));

        let mut bond = CallableZeroCouponBond::new(
            3,
            100.0,
            calendar,
            maturity,
            Thirty360::with_convention(Convention::BondBasis),
            rolling,
            100.0,
            Some(issue),
            callabilities,
            Shared::clone(&settings),
        )
        .unwrap();
        let engine = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, 240, Shared::clone(&settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;
        bond.base_mut().set_pricing_engine(engine);

        let original = bond.npv().unwrap();
        quote.set_value(0.04);
        let updated = bond.npv().unwrap();
        assert!(
            (original - updated).abs() > 1e-10,
            "callable zero NPV should change when the curve quote moves \
             (original={original}, updated={updated})"
        );
    }

    /// `callablebonds.cpp` testCallableFixedRateBondWithArbitrarySchedule
    /// (`:840`): a date-vector schedule with a mid-schedule call prices under
    /// the HW tree without error.
    #[test]
    fn callable_fixed_bond_with_arbitrary_schedule_prices() {
        let calendar = Target::new();
        let today = Date::new(10, Month::January, 2020);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let settlement_days: Natural = 2;
        let settlement = calendar.advance(
            today,
            2,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let rolling = BusinessDayConvention::ModifiedFollowing;
        let issue = calendar.adjust(today - 100, BusinessDayConvention::Following);
        let dates = vec![
            Date::new(20, Month::February, 2020),
            Date::new(15, Month::August, 2020),
            Date::new(25, Month::September, 2021),
            Date::new(27, Month::January, 2022),
        ];
        let schedule = Schedule::with_metadata(
            dates.clone(),
            calendar.clone(),
            BusinessDayConvention::Unadjusted,
            None,
            None,
            None,
            None,
            Vec::new(),
        );
        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.03,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let model = HullWhite::new(curve, 0.1, 0.01).unwrap();
        let engine = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, 240, Shared::clone(&settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;
        let callabilities = vec![Callability::new(
            BondPrice::Clean(100.0),
            CallabilityType::Call,
            dates[2],
        )];
        let mut bond = CallableFixedRateBond::new(
            settlement_days,
            100.0,
            schedule,
            vec![0.06],
            Actual365Fixed::new(),
            rolling,
            100.0,
            Some(issue),
            callabilities,
            Shared::clone(&settings),
        )
        .unwrap();
        bond.base_mut().set_pricing_engine(engine);
        let price = bond.clean_price().unwrap();
        assert!(
            price.is_finite() && price > 0.0,
            "arbitrary-schedule callable clean price should be positive finite, got {price}"
        );
    }

    /// `callablebonds.cpp` testInterplay (`:98`): an earlier exercise right
    /// that is in the money prevents a later opposite right — settlement value
    /// matches the discounted call/put price to 1e-2.
    #[test]
    fn callable_zero_interplay_of_call_and_put() {
        let g = Globals::new(0.03);
        let model = HullWhite::new(g.curve.clone(), 0.1, 0.01).unwrap();
        let engine = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, 240, Shared::clone(&g.settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;
        let day_counter = Thirty360::with_convention(Convention::BondBasis);
        let tol = 1.0e-2;

        let make_zero = |callabilities: CallabilitySchedule| {
            let mut bond = CallableZeroCouponBond::new(
                3,
                100.0,
                g.calendar.clone(),
                g.maturity,
                day_counter.clone(),
                g.rolling,
                100.0,
                Some(g.issue),
                callabilities,
                Shared::clone(&g.settings),
            )
            .unwrap();
            bond.base_mut()
                .set_pricing_engine(SharedMut::clone(&engine));
            bond
        };

        let expected_from = |exercise: &Callability, bond: &mut CallableZeroCouponBond| {
            let settlement = bond.bond().settlement_date(None).unwrap();
            let curve = g.curve.current_link().unwrap();
            let df_call = curve.discount_date(exercise.date(), true).unwrap();
            let df_settle = curve.discount_date(settlement, true).unwrap();
            exercise.price().amount() * df_call / df_settle
        };

        // Case 1: early OTM call blocks a later deep ITM put.
        let call_y4 = Callability::new(
            BondPrice::Clean(100.0),
            CallabilityType::Call,
            g.calendar
                .advance(g.issue, 4, TimeUnit::Years, g.rolling, false),
        );
        let put_y6 = Callability::new(
            BondPrice::Clean(1000.0),
            CallabilityType::Put,
            g.calendar
                .advance(g.issue, 6, TimeUnit::Years, g.rolling, false),
        );
        let mut bond = make_zero(vec![call_y4.clone(), put_y6]);
        let expected = expected_from(&call_y4, &mut bond);
        let value = bond.settlement_value().unwrap();
        assert!(
            (value - expected).abs() <= tol,
            "case1 call blocks put: settlement {value} vs expected {expected}"
        );

        // Case 2: same, with an added later call.
        let call_y8 = Callability::new(
            BondPrice::Clean(100.0),
            CallabilityType::Call,
            g.calendar
                .advance(g.issue, 8, TimeUnit::Years, g.rolling, false),
        );
        let mut bond = make_zero(vec![
            call_y4.clone(),
            Callability::new(
                BondPrice::Clean(1000.0),
                CallabilityType::Put,
                g.calendar
                    .advance(g.issue, 6, TimeUnit::Years, g.rolling, false),
            ),
            call_y8,
        ]);
        let expected = expected_from(&call_y4, &mut bond);
        let value = bond.settlement_value().unwrap();
        assert!(
            (value - expected).abs() <= tol,
            "case2 call blocks put (+later call): settlement {value} vs expected {expected}"
        );

        // Case 3: early ITM put blocks a later deep ITM call.
        let put_y4 = Callability::new(
            BondPrice::Clean(100.0),
            CallabilityType::Put,
            g.calendar
                .advance(g.issue, 4, TimeUnit::Years, g.rolling, false),
        );
        let call_y6 = Callability::new(
            BondPrice::Clean(10.0),
            CallabilityType::Call,
            g.calendar
                .advance(g.issue, 6, TimeUnit::Years, g.rolling, false),
        );
        let mut bond = make_zero(vec![put_y4.clone(), call_y6]);
        let expected = expected_from(&put_y4, &mut bond);
        let value = bond.settlement_value().unwrap();
        assert!(
            (value - expected).abs() <= tol,
            "case3 put blocks call: settlement {value} vs expected {expected}"
        );

        // Case 4: same, with an added later put.
        let put_y8 = Callability::new(
            BondPrice::Clean(100.0),
            CallabilityType::Put,
            g.calendar
                .advance(g.issue, 8, TimeUnit::Years, g.rolling, false),
        );
        let mut bond = make_zero(vec![
            put_y4.clone(),
            Callability::new(
                BondPrice::Clean(10.0),
                CallabilityType::Call,
                g.calendar
                    .advance(g.issue, 6, TimeUnit::Years, g.rolling, false),
            ),
            put_y8,
        ]);
        let expected = expected_from(&put_y4, &mut bond);
        let value = bond.settlement_value().unwrap();
        assert!(
            (value - expected).abs() <= tol,
            "case4 put blocks call (+later put): settlement {value} vs expected {expected}"
        );
    }

    /// `callablebonds.cpp` testBlackEngine (`:673`): European callable zero
    /// under the Black engine reproduces the cached clean price @ 1e-4.
    #[test]
    fn black_engine_callable_zero_matches_cached_price() {
        let calendar = Target::new();
        let today = Date::new(20, Month::September, 2022);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let rolling = BusinessDayConvention::ModifiedFollowing;
        let settlement = calendar.advance(
            today,
            3,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let issue = calendar.adjust(today - 100, BusinessDayConvention::Following);
        let maturity = calendar.advance(issue, 10, TimeUnit::Years, rolling, false);
        let call_date = calendar.advance(issue, 4, TimeUnit::Years, rolling, false);

        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.03,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);

        let callabilities = vec![Callability::new(
            BondPrice::Clean(100.0),
            CallabilityType::Call,
            call_date,
        )];
        let mut bond = CallableZeroCouponBond::new(
            3,
            10_000.0,
            calendar,
            maturity,
            Thirty360::with_convention(Convention::BondBasis),
            rolling,
            100.0,
            Some(issue),
            callabilities,
            Shared::clone(&settings),
        )
        .unwrap();

        let vol = Handle::new(shared(SimpleQuote::new(0.3)) as Shared<dyn Quote>);
        let engine = shared_mut(BlackCallableZeroCouponBondEngine::new(
            vol,
            curve,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;
        bond.base_mut().set_pricing_engine(engine);

        let cached = 74.54521578;
        let calculated = bond.clean_price().unwrap();
        assert!(
            (calculated - cached).abs() <= 1.0e-4,
            "Black zero: clean {calculated} vs cached {cached} (error {})",
            (calculated - cached).abs()
        );
    }

    /// `callablebonds.cpp` testBlackEngineDeepInTheMoney (`:783`): deep-ITM
    /// European call with near-zero vol collapses to the discounted strike.
    #[test]
    fn black_engine_deep_itm_collapses_to_discounted_strike() {
        let calendar = Target::new();
        let today = Date::new(20, Month::September, 2022);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let rolling = BusinessDayConvention::ModifiedFollowing;
        let settlement = calendar.advance(
            today,
            3,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let issue = calendar.adjust(today - 100, BusinessDayConvention::Following);
        let maturity = calendar.advance(issue, 10, TimeUnit::Years, rolling, false);

        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);

        let schedule = MakeSchedule::new()
            .from(issue)
            .to(maturity)
            .with_calendar(calendar)
            .with_frequency(Frequency::Semiannual)
            .with_convention(rolling)
            .with_termination_date_convention(rolling)
            .backwards()
            .build();
        let callability_date = schedule.date(6);
        let strike = 50.0;
        let callabilities = vec![Callability::new(
            BondPrice::Clean(strike),
            CallabilityType::Call,
            callability_date,
        )];

        let mut bond = CallableFixedRateBond::new(
            3,
            10_000.0,
            schedule,
            vec![0.0],
            Thirty360::with_convention(Convention::BondBasis),
            rolling,
            100.0,
            Some(issue),
            callabilities,
            Shared::clone(&settings),
        )
        .unwrap();

        let vol = Handle::new(shared(SimpleQuote::new(1.0e-10)) as Shared<dyn Quote>);
        let engine = shared_mut(BlackCallableFixedRateBondEngine::new(
            vol,
            curve.clone(),
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;
        bond.base_mut().set_pricing_engine(engine);

        let settle = bond.bond().settlement_date(None).unwrap();
        let expected = strike
            * curve
                .current_link()
                .unwrap()
                .discount_date(callability_date, true)
                .unwrap()
            / curve
                .current_link()
                .unwrap()
                .discount_date(settle, true)
                .unwrap();
        let calculated = bond.clean_price().unwrap();
        assert!(
            (calculated - expected).abs() <= 1.0e-8,
            "deep ITM: clean {calculated} vs expected {expected} (error {})",
            (calculated - expected).abs()
        );
    }

    /// `callablebonds.cpp` testImpliedVol (`:711`): Black implied yield vol
    /// round-trips dirty and clean target prices @ 1e-4.
    #[test]
    fn implied_vol_round_trips_dirty_and_clean_targets() {
        let g = Globals::new(0.03);
        let day_counter = Thirty360::with_convention(Convention::BondBasis);
        let callabilities = vec![Callability::new(
            BondPrice::Clean(100.0),
            CallabilityType::Call,
            g.schedule.date(8),
        )];
        let mut bond = CallableFixedRateBond::new(
            3,
            10_000.0,
            g.schedule.clone(),
            vec![0.01],
            day_counter,
            g.rolling,
            100.0,
            Some(g.issue),
            callabilities,
            Shared::clone(&g.settings),
        )
        .unwrap();

        let dirty_target = BondPrice::Dirty(78.50);
        let volatility = bond
            .implied_volatility(dirty_target, g.curve.clone(), 1e-8, 200, 1e-4, 1.0)
            .unwrap();
        let engine = shared_mut(BlackCallableFixedRateBondEngine::new(
            Handle::new(shared(SimpleQuote::new(volatility)) as Shared<dyn Quote>),
            g.curve.clone(),
            Shared::clone(&g.settings),
        )) as SharedMut<dyn PricingEngine>;
        bond.base_mut()
            .set_pricing_engine(SharedMut::clone(&engine));
        let dirty = bond.dirty_price().unwrap();
        assert!(
            (dirty - dirty_target.amount()).abs() <= 1.0e-4,
            "implied dirty: price {dirty} vs target {} (vol {volatility})",
            dirty_target.amount()
        );

        let clean_target = BondPrice::Clean(78.50);
        let volatility = bond
            .implied_volatility(clean_target, g.curve.clone(), 1e-8, 200, 1e-4, 1.0)
            .unwrap();
        let engine = shared_mut(BlackCallableFixedRateBondEngine::new(
            Handle::new(shared(SimpleQuote::new(volatility)) as Shared<dyn Quote>),
            g.curve.clone(),
            Shared::clone(&g.settings),
        )) as SharedMut<dyn PricingEngine>;
        bond.base_mut().set_pricing_engine(engine);
        let clean = bond.clean_price().unwrap();
        assert!(
            (clean - clean_target.amount()).abs() <= 1.0e-4,
            "implied clean: price {clean} vs target {} (vol {volatility})",
            clean_target.amount()
        );
    }

    /// `callablebonds.cpp` testCallableBondOasWithDifferentNotinals (`:881`):
    /// OAS and cleanPriceOAS are independent of face amount.
    #[test]
    fn oas_is_independent_of_notional() {
        let calendar = Target::new();
        let today = Date::new(10, Month::January, 2020);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let settlement_days = 2;
        let rolling = BusinessDayConvention::ModifiedFollowing;
        let settlement = calendar.advance(
            today,
            settlement_days as i32,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let issue = calendar.adjust(today - 100, BusinessDayConvention::Following);
        let maturity = calendar.advance(issue, 10, TimeUnit::Years, rolling, false);
        let day_counter = Actual365Fixed::new();
        let compounding = Compounding::Compounded;
        let frequency = Frequency::Semiannual;

        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.03,
            day_counter.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let model = HullWhite::new(curve.clone(), 0.1, 0.01).unwrap();
        let engine = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, 240, Shared::clone(&settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;

        let schedule = MakeSchedule::new()
            .from(issue)
            .to(maturity)
            .with_calendar(calendar)
            .with_frequency(frequency)
            .with_convention(rolling)
            .with_termination_date_convention(rolling)
            .backwards()
            .build();
        let first_call = schedule.date(schedule.len() - 5);
        let last_call = schedule.date(schedule.len() - 2);
        let call_dates = schedule.after(first_call).until(last_call);
        let call_schedule: CallabilitySchedule = call_dates
            .dates()
            .iter()
            .copied()
            .map(|d| Callability::new(BondPrice::Clean(100.0), CallabilityType::Call, d))
            .collect();

        let accrual = Actual365Fixed::new();
        let make_bond = |face: Real| {
            let mut bond = CallableFixedRateBond::new(
                settlement_days,
                face,
                schedule.clone(),
                vec![0.055],
                accrual.clone(),
                rolling,
                100.0,
                Some(issue),
                call_schedule.clone(),
                Shared::clone(&settings),
            )
            .unwrap();
            bond.base_mut()
                .set_pricing_engine(SharedMut::clone(&engine));
            bond
        };

        let bond100 = make_bond(100.0);
        let bond25 = make_bond(25.0);
        let clean_price = 96.0;
        let oas100 = bond100
            .oas(
                clean_price,
                curve.clone(),
                day_counter.clone(),
                compounding,
                frequency,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let oas25 = bond25
            .oas(
                clean_price,
                curve.clone(),
                day_counter.clone(),
                compounding,
                frequency,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(
            oas100, oas25,
            "OAS must match across notionals: 100 -> {oas100}, 25 -> {oas25}"
        );

        let oas = 0.0300;
        let clean100 = bond100
            .clean_price_oas(
                oas,
                curve.clone(),
                day_counter.clone(),
                compounding,
                frequency,
                None,
            )
            .unwrap();
        let clean25 = bond25
            .clean_price_oas(oas, curve, day_counter, compounding, frequency, None)
            .unwrap();
        assert_eq!(
            clean100, clean25,
            "cleanPriceOAS must match across notionals: 100 -> {clean100}, 25 -> {clean25}"
        );
    }

    /// `callablebonds.cpp` testEffectiveDurationAndConvexity (`:1049`):
    /// dirty-price finite differences match `effectiveDuration` /
    /// `effectiveConvexity`, and differ from a clean-price denominator.
    #[test]
    fn effective_duration_and_convexity_use_dirty_price() {
        let settlement_date = Date::new(30, Month::November, 2023);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(settlement_date);

        let effective = Date::new(20, Month::May, 2021);
        let maturity = Date::new(1, Month::June, 2029);
        let first_coupon = Date::new(1, Month::December, 2021);
        let calendar = UnitedStates::new(Market::GovernmentBond);
        let day_count = Thirty360::with_convention(Convention::ISDA);

        let schedule = MakeSchedule::new()
            .from(effective)
            .to(maturity)
            .with_tenor(Period::new(6, TimeUnit::Months))
            .with_calendar(calendar.clone())
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .backwards()
            .end_of_month(true)
            .with_first_date(first_coupon)
            .build();

        let call_schedule = vec![
            Callability::new(
                BondPrice::Clean(102.438),
                CallabilityType::Call,
                Date::new(1, Month::June, 2024),
            ),
            Callability::new(
                BondPrice::Clean(101.219),
                CallabilityType::Call,
                Date::new(1, Month::June, 2025),
            ),
            Callability::new(
                BondPrice::Clean(100.0),
                CallabilityType::Call,
                Date::new(1, Month::June, 2026),
            ),
            Callability::new(
                BondPrice::Clean(100.0),
                CallabilityType::Call,
                Date::new(1, Month::June, 2029),
            ),
        ];

        let mut bond = CallableFixedRateBond::new(
            2,
            100.0,
            schedule,
            vec![0.04875],
            day_count.clone(),
            BusinessDayConvention::Unadjusted,
            100.0,
            Some(effective),
            call_schedule,
            Shared::clone(&settings),
        )
        .unwrap();

        let flat: Handle<dyn YieldTermStructure> =
            Handle::new(shared(FlatForward::moving_with_rate(
                2,
                calendar,
                0.05,
                day_count.clone(),
                Compounding::Continuous,
                Frequency::Annual,
                Shared::clone(&settings),
            )) as Shared<dyn YieldTermStructure>);
        let hw = HullWhite::new(flat.clone(), 0.03, 0.012).unwrap();
        let grid_steps = ((maturity - settlement_date) / 30) as usize;
        let engine = shared_mut(
            TreeCallableFixedRateBondEngine::new(hw, grid_steps, Shared::clone(&settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;
        bond.base_mut().set_pricing_engine(engine);

        let compounding = Compounding::Compounded;
        let frequency = Frequency::Semiannual;
        let oas = bond
            .oas(
                70.926,
                flat.clone(),
                day_count.clone(),
                compounding,
                frequency,
                Some(settlement_date),
                None,
                None,
                None,
            )
            .unwrap();
        let shift = 0.001;
        let eff_dur = bond
            .effective_duration(
                oas,
                flat.clone(),
                day_count.clone(),
                compounding,
                frequency,
                Some(shift),
            )
            .unwrap();
        let eff_conv = bond
            .effective_convexity(
                oas,
                flat.clone(),
                day_count.clone(),
                compounding,
                frequency,
                Some(shift),
            )
            .unwrap();

        let accrued = bond.bond().accrued_amount(Some(settlement_date)).unwrap();
        let p0 = bond
            .clean_price_oas(
                oas,
                flat.clone(),
                day_count.clone(),
                compounding,
                frequency,
                Some(settlement_date),
            )
            .unwrap()
            + accrued;
        let p_up = bond
            .clean_price_oas(
                oas + shift,
                flat.clone(),
                day_count.clone(),
                compounding,
                frequency,
                Some(settlement_date),
            )
            .unwrap()
            + accrued;
        let p_down = bond
            .clean_price_oas(
                oas - shift,
                flat,
                day_count,
                compounding,
                frequency,
                Some(settlement_date),
            )
            .unwrap()
            + accrued;

        let expected_dur = (p_down - p_up) / (2.0 * p0 * shift);
        let expected_conv = (p_down + p_up - 2.0 * p0) / (p0 * shift * shift);
        let incorrect_dur = (p_down - p_up) / (2.0 * (p0 - accrued) * shift);

        // Boost `CHECK_CLOSE(..., 1e-4)` is a 1e-4 percent relative tolerance.
        let rel = 1.0e-6;
        assert!(
            (eff_dur - expected_dur).abs() <= expected_dur.abs() * rel,
            "effective duration {eff_dur} vs expected {expected_dur}"
        );
        assert!(
            (eff_conv - expected_conv).abs() <= expected_conv.abs() * rel,
            "effective convexity {eff_conv} vs expected {expected_conv}"
        );
        assert!(
            (eff_dur - incorrect_dur).abs() > 0.01,
            "duration {eff_dur} should differ from clean-denominator {incorrect_dur}"
        );
    }

    /// `callablebonds.cpp` testSnappingExerciseDate2ClosestCouponDate (`:571`):
    /// a call within a week of a coupon snaps so the callable NPV matches the
    /// truncated straight bond @ 1e-10, and OAS falls as the call date moves.
    #[test]
    fn snapping_exercise_date_to_closest_coupon_matches_truncated_bond() {
        let today = Date::new(18, Month::May, 2021);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);

        let calendar = UnitedStates::new(Market::FederalReserve);
        let accrual = Thirty360::with_convention(Convention::USA);
        let frequency = Frequency::Semiannual;
        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            today,
            0.02,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);

        let settlement_days = 2;
        let settlement = Date::new(20, Month::May, 2021);
        let coupon = 0.05;
        let face = 100.0;
        let maturity = Date::new(14, Month::February, 2026);
        let issue = settlement - 2 * 366;
        let schedule = MakeSchedule::new()
            .from(issue)
            .to(maturity)
            .with_frequency(frequency)
            .with_calendar(calendar.clone())
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .backwards()
            .end_of_month(false)
            .build();
        let coupons = vec![coupon; schedule.len() - 1];

        let initial_call = Date::new(14, Month::February, 2022);
        let tolerance = 1.0e-10;
        let mut prev_oas = 0.0266;
        let expected_oas_step = 0.00005;

        for i in -10..11 {
            let call_date = initial_call + i;
            if !calendar.is_business_day(call_date) {
                continue;
            }

            let callabilities = vec![Callability::new(
                BondPrice::Clean(face),
                CallabilityType::Call,
                call_date,
            )];
            let mut callable = CallableFixedRateBond::new(
                settlement_days,
                face,
                schedule.clone(),
                coupons.clone(),
                accrual.clone(),
                BusinessDayConvention::Following,
                face,
                Some(issue),
                callabilities,
                Shared::clone(&settings),
            )
            .unwrap();
            let model = HullWhite::new(curve.clone(), 1.0e-12, 0.003).unwrap();
            let tree = shared_mut(
                TreeCallableFixedRateBondEngine::new(model, 40, Shared::clone(&settings)).unwrap(),
            ) as SharedMut<dyn PricingEngine>;
            callable.base_mut().set_pricing_engine(tree);

            let truncated = schedule.until(call_date);
            let mut straight = FixedRateBond::new(
                settlement_days,
                face,
                truncated,
                coupons.clone(),
                accrual.clone(),
                BusinessDayConvention::Following,
                face,
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
            let discounting = shared_mut(DiscountingBondEngine::new(
                curve.clone(),
                None,
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>;
            straight
                .bond_mut()
                .base_mut()
                .set_pricing_engine(discounting);

            let npv_callable = callable.npv().unwrap();
            let npv_straight = straight.bond_mut().npv().unwrap();
            assert!(
                (npv_callable - npv_straight).abs() <= tolerance,
                "snap NPV at {call_date}: callable {npv_callable} vs truncated {npv_straight}"
            );

            let clean = callable.clean_price().unwrap() - 2.0;
            let oas = callable
                .oas(
                    clean,
                    curve.clone(),
                    accrual.clone(),
                    Compounding::Continuous,
                    frequency,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
            assert!(
                prev_oas - oas >= expected_oas_step,
                "OAS at {call_date}: {oas} should fall from {prev_oas} by at least {expected_oas_step}"
            );
            prev_oas = oas;
        }
    }

    /// `callablebonds.cpp` testOasContinuityThroughExCouponWindow (`:953`):
    /// OAS stays within 50 bps as the call date walks through the ex-coupon
    /// window (GitHub issue #2236).
    #[test]
    fn oas_is_continuous_through_the_ex_coupon_window() {
        let today = Date::new(31, Month::January, 2024);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);

        let calendar = UnitedStates::new(Market::Nyse);
        let dc = Thirty360::with_convention(Convention::BondBasis);
        let bdc = BusinessDayConvention::Unadjusted;
        let frequency = Frequency::Quarterly;
        let ex_coupon_period = Period::new(14, TimeUnit::Days);
        let issue = today;
        let maturity = Date::new(31, Month::January, 2029);

        let curve: Handle<dyn YieldTermStructure> = Handle::new(shared(FlatForward::with_rate(
            today,
            0.04,
            dc.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        ))
            as Shared<dyn YieldTermStructure>);
        let model = HullWhite::new(curve.clone(), 0.1, 0.01).unwrap();
        let engine = shared_mut(
            TreeCallableFixedRateBondEngine::new(model, 100, Shared::clone(&settings)).unwrap(),
        ) as SharedMut<dyn PricingEngine>;

        let schedule = MakeSchedule::new()
            .from(issue)
            .to(maturity)
            .with_frequency(frequency)
            .with_calendar(calendar)
            .with_convention(bdc)
            .with_termination_date_convention(bdc)
            .backwards()
            .end_of_month(true)
            .build();
        let first_payment = schedule.date(1);
        let ex_coupon_date = first_payment - 14;
        let sweep_start = ex_coupon_date - 7;
        let sweep_end = first_payment + 7;

        let mut max_oas = f64::NEG_INFINITY;
        let mut min_oas = f64::INFINITY;
        let mut call_date = sweep_start;
        while call_date <= sweep_end {
            let callabilities = vec![Callability::new(
                BondPrice::Clean(100.0),
                CallabilityType::Call,
                call_date,
            )];
            let mut bond = CallableFixedRateBond::with_ex_coupon(
                0,
                100.0,
                schedule.clone(),
                vec![0.06],
                dc.clone(),
                bdc,
                100.0,
                Some(issue),
                callabilities,
                Some(ex_coupon_period),
                NullCalendar::new(),
                BusinessDayConvention::Unadjusted,
                false,
                Shared::clone(&settings),
            )
            .unwrap();
            bond.base_mut()
                .set_pricing_engine(SharedMut::clone(&engine));
            let oas_bps = bond
                .oas(
                    100.0,
                    curve.clone(),
                    dc.clone(),
                    Compounding::Compounded,
                    frequency,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
                * 10_000.0;
            max_oas = max_oas.max(oas_bps);
            min_oas = min_oas.min(oas_bps);
            call_date += 1;
        }

        let range = max_oas - min_oas;
        let tolerance = 50.0;
        assert!(
            range <= tolerance,
            "OAS discontinuity across ex-coupon window: min {min_oas} bps, max {max_oas} bps, \
             range {range} bps (sweep {sweep_start} to {sweep_end})"
        );
    }
}
