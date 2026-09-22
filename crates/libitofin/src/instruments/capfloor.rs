//! Cap, floor and collar instruments.
//!
//! Port of `ql/instruments/capfloor.{hpp,cpp}`. A [`CapFloor`] is an
//! [`Instrument`] over a floating leg plus per-coupon cap and/or floor strike
//! vectors and a [`CapFloorType`]; [`CapFloor::cap`], [`CapFloor::floor`] and
//! [`CapFloor::collar`] are the thin constructors the C++ `Cap`/`Floor`/`Collar`
//! subclasses provide. [`setup_arguments`](Instrument::setup_arguments) fills the
//! [`CapFloorArguments`] a `CapFloor::engine` reads.
//!
//! ## Leg shape
//!
//! Ibor and overnight constructors retain their concrete coupons. Both feed the
//! same engine arguments using `(rate - spread) / gearing`; overnight coupons
//! supply their last fixing date. Legacy Ibor coupon inspectors remain unchanged.
//!
//! ## Divergences from QuantLib
//!
//! - Arbitrary user-defined floating coupon types are not accepted.
//! - [`MakeCapFloor`](super::MakeCapFloor) builds the market cap/floor and
//!   [`CapFloor::last_floating_rate_coupon`] exposes the trailing coupon the
//!   optionlet stripper reads; [`CapFloor::implied_volatility`] pins
//!   `testImpliedVolatility`; [`CapFloor::optionlet`] pins the
//!   `testConsistency` recomposition. [`CapFloor::deep_update`] pins the
//!   LazyObject invalidation half of C++ `CapFloor::deepUpdate` (coupon-level
//!   deepUpdate remains deferred with the cash-flow surface, as on
//!   [`Swap::deep_update`](crate::instruments::Swap::deep_update)).
//! - The `CapFloor::arguments` bundle carries `start_dates` (read by the analytic
//!   Hull-White engine to form each optionlet's exercise maturity, #438); the C++
//!   `spreads` and `indexes` are filled but unread by any ported engine, so they
//!   remain omitted.
//! - The D5 `Settings` handle replaces `Settings::instance()` for the evaluation
//!   date the expiry check and the forward guard read.

use std::any::Any;
use std::cell::RefCell;

use crate::cashflow::{CashFlow, Leg};
use crate::cashflows::{CashFlows, Coupon, IborCoupon, OvernightIndexedCoupon};
use crate::errors::QlResult;
use crate::event::Event;
use crate::handle::Handle;
use crate::instrument::{Instrument, InstrumentBase};
use crate::math::solver1d::{DerivativeSolver, Function1D};
use crate::math::solvers1d::newtonsafe::NewtonSafe;
use crate::patterns::observable::AsObservable;
use crate::pricingengine::{Arguments, PricingEngine};
use crate::pricingengines::capfloor::{BachelierCapFloorEngine, BlackCapFloorEngine};
use crate::quotes::{Quote, SimpleQuote};
use crate::settings::Settings;
use crate::shared::{Shared, shared};
use crate::termstructures::volatility::VolatilityType;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::time::daycounters::actual365fixed::Actual365Fixed;
use crate::types::{Rate, Real, Size, Time, Volatility};
use crate::{fail, require};

/// Whether the instrument caps, floors or collars its floating leg
/// (`CapFloor::Type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapFloorType {
    /// A cap: long a call on each coupon's rate at the cap strike.
    Cap,
    /// A floor: long a put on each coupon's rate at the floor strike.
    Floor,
    /// A collar: long the cap, short the floor.
    Collar,
}

/// Argument bundle a `CapFloor::engine` prices (`CapFloor::arguments`).
///
/// Per optionlet: the payment date (`end_dates`), fixing date, accrual time,
/// nominal, gearing, the coupon's adjusted forward (`None` for a past-fixing
/// coupon, the C++ `Null<Rate>`), and the de-spread cap and floor strikes
/// (`None` where the type has none).
#[derive(Clone, Default)]
pub struct CapFloorArguments {
    /// The instrument type, set by `setup_arguments`.
    pub cap_floor_type: Option<CapFloorType>,
    /// Each coupon's accrual start date (the C++ `startDates`), the exercise
    /// maturity the analytic Hull-White engine forms its bond option from.
    pub start_dates: Vec<Date>,
    /// Each coupon's fixing date.
    pub fixing_dates: Vec<Date>,
    /// Each coupon's payment date (the C++ `endDates`).
    pub end_dates: Vec<Date>,
    /// Each coupon's accrual period as a year fraction.
    pub accrual_times: Vec<Time>,
    /// Each coupon's de-spread cap strike, `None` for a pure floor.
    pub cap_rates: Vec<Option<Rate>>,
    /// Each coupon's de-spread floor strike, `None` for a pure cap.
    pub floor_rates: Vec<Option<Rate>>,
    /// Each coupon's adjusted forward, `None` when its fixing has passed.
    pub forwards: Vec<Option<Rate>>,
    /// Each coupon's gearing.
    pub gearings: Vec<Real>,
    /// Each coupon's nominal.
    pub nominals: Vec<Real>,
}

impl Arguments for CapFloorArguments {
    fn validate(&self) -> QlResult<()> {
        let n = self.end_dates.len();
        require!(self.cap_floor_type.is_some(), "cap/floor type not set");
        require!(self.start_dates.len() == n, "start-date count mismatch");
        require!(self.fixing_dates.len() == n, "fixing-date count mismatch");
        require!(self.accrual_times.len() == n, "accrual-time count mismatch");
        require!(self.cap_rates.len() == n, "cap-rate count mismatch");
        require!(self.floor_rates.len() == n, "floor-rate count mismatch");
        require!(self.forwards.len() == n, "forward count mismatch");
        require!(self.gearings.len() == n, "gearing count mismatch");
        require!(self.nominals.len() == n, "nominal count mismatch");
        Ok(())
    }
}

/// A cap, floor or collar over an Ibor or overnight leg.
pub struct CapFloor {
    base: InstrumentBase,
    cap_floor_type: CapFloorType,
    coupons: Vec<Shared<IborCoupon>>,
    overnight_coupons: Vec<Shared<OvernightIndexedCoupon>>,
    cap_rates: Vec<Rate>,
    floor_rates: Vec<Rate>,
    settings: Shared<Settings<Date>>,
}

impl CapFloor {
    /// Builds a cap/floor/collar over `coupons`, padding the strike vectors to
    /// the leg length by repeating the last strike (the C++ constructor's
    /// `while (rates.size() < leg.size()) push_back(rates.back())`).
    ///
    /// A `Cap` or `Collar` requires at least one cap rate, a `Floor` or `Collar`
    /// at least one floor rate.
    pub fn new(
        cap_floor_type: CapFloorType,
        coupons: Vec<Shared<IborCoupon>>,
        mut cap_rates: Vec<Rate>,
        mut floor_rates: Vec<Rate>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<CapFloor> {
        let n = coupons.len();
        if matches!(cap_floor_type, CapFloorType::Cap | CapFloorType::Collar) {
            require!(!cap_rates.is_empty(), "no cap rates given");
            while cap_rates.len() < n {
                cap_rates.push(*cap_rates.last().expect("non-empty"));
            }
        }
        if matches!(cap_floor_type, CapFloorType::Floor | CapFloorType::Collar) {
            require!(!floor_rates.is_empty(), "no floor rates given");
            while floor_rates.len() < n {
                floor_rates.push(*floor_rates.last().expect("non-empty"));
            }
        }

        let base = InstrumentBase::new();
        for coupon in &coupons {
            base.register_with(coupon.observable());
        }
        settings.register_eval_date_observer(&base.observer());

        Ok(CapFloor {
            base,
            cap_floor_type,
            coupons,
            overnight_coupons: Vec::new(),
            cap_rates,
            floor_rates,
            settings,
        })
    }

    /// Cap, floor or collar over retained overnight coupons.
    ///
    /// Engine arguments use each coupon's last fixing date and averaged rate,
    /// matching QuantLib's generic floating-coupon cap/floor path.
    pub fn from_overnight(
        kind: CapFloorType,
        coupons: Vec<Shared<OvernightIndexedCoupon>>,
        cap_rates: Vec<Rate>,
        floor_rates: Vec<Rate>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let mut result = Self::new(kind, Vec::new(), cap_rates, floor_rates, settings)?;
        if result.cap_rates.len() < coupons.len()
            && let Some(&last) = result.cap_rates.last()
        {
            result.cap_rates.resize(coupons.len(), last);
        }
        if result.floor_rates.len() < coupons.len()
            && let Some(&last) = result.floor_rates.last()
        {
            result.floor_rates.resize(coupons.len(), last);
        }
        for coupon in &coupons {
            result.base.register_with(coupon.observable());
        }
        result.overnight_coupons = coupons;
        Ok(result)
    }

    /// Overnight coupons; empty on a legacy Ibor-leg cap/floor.
    pub fn overnight_coupons(&self) -> &[Shared<OvernightIndexedCoupon>] {
        &self.overnight_coupons
    }

    /// A cap over `coupons` struck at `strikes` (the C++ `Cap`).
    pub fn cap(
        coupons: Vec<Shared<IborCoupon>>,
        strikes: Vec<Rate>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<CapFloor> {
        CapFloor::new(CapFloorType::Cap, coupons, strikes, Vec::new(), settings)
    }

    /// A floor over `coupons` struck at `strikes` (the C++ `Floor`).
    pub fn floor(
        coupons: Vec<Shared<IborCoupon>>,
        strikes: Vec<Rate>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<CapFloor> {
        CapFloor::new(CapFloorType::Floor, coupons, Vec::new(), strikes, settings)
    }

    /// A collar over `coupons`, long the cap at `cap_rates` and short the floor
    /// at `floor_rates` (the C++ `Collar`).
    pub fn collar(
        coupons: Vec<Shared<IborCoupon>>,
        cap_rates: Vec<Rate>,
        floor_rates: Vec<Rate>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<CapFloor> {
        CapFloor::new(
            CapFloorType::Collar,
            coupons,
            cap_rates,
            floor_rates,
            settings,
        )
    }

    /// The instrument type.
    pub fn cap_floor_type(&self) -> CapFloorType {
        self.cap_floor_type
    }

    /// The padded cap strikes.
    pub fn cap_rates(&self) -> &[Rate] {
        &self.cap_rates
    }

    /// The padded floor strikes.
    pub fn floor_rates(&self) -> &[Rate] {
        &self.floor_rates
    }

    /// Number of Ibor or overnight coupons in the instrument.
    pub fn coupon_count(&self) -> usize {
        self.coupons.len() + self.overnight_coupons.len()
    }

    /// The floating coupons.
    pub fn coupons(&self) -> &[Shared<IborCoupon>] {
        &self.coupons
    }

    /// The last Ibor coupon (`lastFloatingRateCoupon`), the coupon the
    /// optionlet stripper and [`MakeCapFloor`](super::MakeCapFloor) read.
    /// `None` on an overnight-only instrument; use
    /// [`last_overnight_coupon`](Self::last_overnight_coupon) there.
    pub fn last_floating_rate_coupon(&self) -> Option<&Shared<IborCoupon>> {
        self.coupons.last()
    }

    /// The last overnight coupon, or `None` on an Ibor-only instrument.
    pub fn last_overnight_coupon(&self) -> Option<&Shared<OvernightIndexedCoupon>> {
        self.overnight_coupons.last()
    }

    /// The `i`-th optionlet as a cap/floor over that one coupon
    /// (`CapFloor::optionlet`, `capfloor.cpp:195-208`).
    ///
    /// Index `i` runs over [`coupon_count`](Self::coupon_count): Ibor coupons
    /// first, then overnight coupons. Keeps the parent's type and carries only
    /// the strikes that type uses, so summing the optionlets' NPVs recomposes
    /// the parent's (`testConsistency` recomposition, un-nested here as in the
    /// YoY pin).
    ///
    /// # Errors
    ///
    /// When `i` is past the end of the leg.
    pub fn optionlet(&self, i: usize) -> QlResult<CapFloor> {
        require!(
            i < self.coupon_count(),
            "optionlet {i} does not exist, only {}",
            self.coupon_count()
        );
        let mut cap_rates = Vec::new();
        let mut floor_rates = Vec::new();
        if matches!(
            self.cap_floor_type,
            CapFloorType::Cap | CapFloorType::Collar
        ) {
            cap_rates.push(self.cap_rates[i]);
        }
        if matches!(
            self.cap_floor_type,
            CapFloorType::Floor | CapFloorType::Collar
        ) {
            floor_rates.push(self.floor_rates[i]);
        }
        if i < self.coupons.len() {
            CapFloor::new(
                self.cap_floor_type,
                vec![Shared::clone(&self.coupons[i])],
                cap_rates,
                floor_rates,
                Shared::clone(&self.settings),
            )
        } else {
            CapFloor::from_overnight(
                self.cap_floor_type,
                vec![Shared::clone(
                    &self.overnight_coupons[i - self.coupons.len()],
                )],
                cap_rates,
                floor_rates,
                Shared::clone(&self.settings),
            )
        }
    }

    /// The leg's earliest accrual start (`startDate`).
    pub fn start_date(&self) -> QlResult<Date> {
        CashFlows::start_date(&self.cash_flows())
    }

    /// The leg's latest accrual end (`maturityDate`).
    pub fn maturity_date(&self) -> QlResult<Date> {
        CashFlows::maturity_date(&self.cash_flows())
    }

    /// The at-the-money rate: the fixed rate that reprices the floating leg on
    /// `discount_curve` (`atmRate`).
    pub fn atm_rate(&self, discount_curve: &dyn YieldTermStructure) -> QlResult<Rate> {
        let reference = discount_curve.reference_date()?;
        CashFlows::atm_rate(
            &self.cash_flows(),
            discount_curve,
            &self.settings,
            Some(false),
            Some(reference),
            None,
            None,
        )
    }

    /// Implied term volatility that reprices this instrument to `target_value`
    /// (`CapFloor::impliedVolatility`).
    ///
    /// Uses [`NewtonSafe`] on a temporary Black or Bachelier engine whose flat
    /// vol is a [`SimpleQuote`], matching `ImpliedCapVolHelper` /
    /// `capfloor.cpp:39-109`. Defaults in C++: `accuracy = 1e-4`,
    /// `max_evaluations = 100`, `min_vol = 1e-7`, `max_vol = 4.0`,
    /// `ShiftedLognormal`, `displacement = 0.0`.
    #[allow(clippy::too_many_arguments)]
    pub fn implied_volatility(
        &self,
        target_value: Real,
        discount_curve: Handle<dyn YieldTermStructure>,
        guess: Volatility,
        accuracy: Real,
        max_evaluations: Size,
        min_vol: Volatility,
        max_vol: Volatility,
        vol_type: VolatilityType,
        displacement: Real,
    ) -> QlResult<Volatility> {
        require!(!self.is_expired()?, "instrument expired");

        // Implausible seed so the first evaluation always recalculates
        // (`ImpliedCapVolHelper` ctor, `capfloor.cpp:63-66`).
        let vol = shared(SimpleQuote::new(-1.0));
        let vol_handle = Handle::new(Shared::clone(&vol) as Shared<dyn Quote>);
        let mut engine: Box<dyn PricingEngine> = match vol_type {
            VolatilityType::ShiftedLognormal => Box::new(BlackCapFloorEngine::with_flat_vol(
                discount_curve,
                vol_handle,
                Actual365Fixed::new(),
                displacement,
                Shared::clone(&self.settings),
            )?),
            VolatilityType::Normal => Box::new(BachelierCapFloorEngine::with_flat_vol(
                discount_curve,
                vol_handle,
                Actual365Fixed::new(),
                Shared::clone(&self.settings),
            )?),
        };
        self.setup_arguments(engine.arguments_mut())?;
        engine.arguments_mut().validate()?;

        let failure = RefCell::new(None);
        let helper = ImpliedCapVolHelper {
            engine,
            vol,
            target_value,
            failure: &failure,
        };
        let solver = NewtonSafe::new().with_max_evaluations(max_evaluations);
        let root = solver.solve_bracketed(helper, accuracy, guess, min_vol, max_vol);
        match failure.into_inner() {
            Some(error) => Err(error),
            None => root,
        }
    }

    /// Invalidates the cached results and notifies observers (the C++
    /// `CapFloor::deepUpdate`, `capfloor.cpp:271-276`).
    ///
    /// C++ additionally walks `floatingLeg_` calling `deepUpdate` on each flow
    /// to refresh coupon pricer caches; the crate's cash-flow surface exposes no
    /// such hook yet, so the walk reduces to the instrument `update` step until
    /// one lands — the same reduction as [`Swap::deep_update`].
    pub fn deep_update(&mut self) {
        self.base().observer().borrow_mut().update();
    }

    /// The concrete coupons erased to a [`Leg`] for the [`CashFlows`] analytics.
    fn cash_flows(&self) -> Leg {
        self.coupons
            .iter()
            .map(|coupon| Shared::clone(coupon) as Shared<dyn CashFlow>)
            .chain(
                self.overnight_coupons
                    .iter()
                    .map(|coupon| Shared::clone(coupon) as Shared<dyn CashFlow>),
            )
            .collect()
    }
}

/// Newton objective for [`CapFloor::implied_volatility`] (`ImpliedCapVolHelper`).
struct ImpliedCapVolHelper<'a> {
    engine: Box<dyn PricingEngine>,
    vol: Shared<SimpleQuote>,
    target_value: Real,
    failure: &'a RefCell<Option<crate::errors::QlError>>,
}

impl ImpliedCapVolHelper<'_> {
    fn ensure_priced(&mut self, x: Volatility) {
        let current = self.vol.value().ok();
        if current != Some(x) {
            self.vol.set_value(x);
            if let Err(error) = self.engine.calculate() {
                self.failure.borrow_mut().get_or_insert(error);
            }
        }
    }

    fn npv(&self) -> Option<Real> {
        self.engine
            .results()
            .as_instrument_results()
            .and_then(|r| r.value)
    }

    fn vega(&self) -> Option<Real> {
        self.engine
            .results()
            .as_instrument_results()
            .and_then(|r| r.additional_results.get("vega"))
            .and_then(|v| v.as_ref().downcast_ref::<Real>().copied())
    }
}

impl Function1D for ImpliedCapVolHelper<'_> {
    fn value(&mut self, x: Real) -> Real {
        self.ensure_priced(x);
        match self.npv() {
            Some(value) => value - self.target_value,
            None => {
                self.failure.borrow_mut().get_or_insert_with(|| {
                    crate::errors::QlError::new(
                        "no results returned from pricing engine",
                        file!(),
                        line!(),
                    )
                });
                Real::NAN
            }
        }
    }

    fn derivative(&mut self, x: Real) -> Real {
        self.ensure_priced(x);
        match self.vega() {
            Some(vega) => vega,
            None => {
                self.failure.borrow_mut().get_or_insert_with(|| {
                    crate::errors::QlError::new("vega not provided", file!(), line!())
                });
                Real::NAN
            }
        }
    }
}

impl Instrument for CapFloor {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        for coupon in self.cash_flows().iter().rev() {
            if !coupon.has_occurred(&self.settings, None, None)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(args) = (arguments as &mut dyn Any).downcast_mut::<CapFloorArguments>() else {
            fail!("wrong argument type");
        };
        let today = match self.settings.evaluation_date() {
            Some(today) => today,
            None => fail!("no evaluation date set: a cap/floor needs a reference date"),
        };

        let n = self.coupons.len() + self.overnight_coupons.len();
        args.cap_floor_type = Some(self.cap_floor_type);
        args.start_dates = Vec::with_capacity(n);
        args.fixing_dates = Vec::with_capacity(n);
        args.end_dates = Vec::with_capacity(n);
        args.accrual_times = Vec::with_capacity(n);
        args.cap_rates = Vec::with_capacity(n);
        args.floor_rates = Vec::with_capacity(n);
        args.forwards = Vec::with_capacity(n);
        args.gearings = Vec::with_capacity(n);
        args.nominals = Vec::with_capacity(n);

        let has_cap = matches!(
            self.cap_floor_type,
            CapFloorType::Cap | CapFloorType::Collar
        );
        let has_floor = matches!(
            self.cap_floor_type,
            CapFloorType::Floor | CapFloorType::Collar
        );

        let coupons = self
            .coupons
            .iter()
            .map(|c| {
                (
                    c.as_ref() as &dyn Coupon,
                    c.fixing_date(),
                    c.spread(),
                    c.gearing(),
                    c.date(),
                )
            })
            .chain(self.overnight_coupons.iter().map(|c| {
                (
                    c.as_ref() as &dyn Coupon,
                    c.fixing_date(),
                    c.spread(),
                    c.gearing(),
                    c.date(),
                )
            }));
        for (i, (coupon, fixing_date, spread, gearing, end_date)) in coupons.enumerate() {
            args.start_dates.push(coupon.accrual_start_date());
            args.fixing_dates.push(fixing_date);
            args.end_dates.push(end_date);
            args.accrual_times.push(coupon.accrual_period());
            args.nominals.push(coupon.nominal());
            args.gearings.push(gearing);

            // Passed explicitly for precision, but only if the coupon can still
            // pay (`capfloor.cpp:245`): a past-fixing coupon has no forward.
            let forward = if end_date >= today {
                Some((coupon.rate()? - spread) / gearing)
            } else {
                None
            };
            args.forwards.push(forward);

            args.cap_rates.push(if has_cap {
                Some((self.cap_rates[i] - spread) / gearing)
            } else {
                None
            });
            args.floor_rates.push(if has_floor {
                Some((self.floor_rates[i] - spread) / gearing)
            } else {
                None
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflows::{IborLeg, OvernightLeg};
    use crate::handle::Handle;
    use crate::indexes::ibor::{Euribor, Sofr};
    use crate::shared::shared;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::target::Target;
    use crate::time::date::{Date, Month};
    use crate::time::frequency::Frequency;
    use crate::time::schedule::MakeSchedule;

    fn settings_on(today: Date) -> Shared<Settings<Date>> {
        let settings = shared(Settings::<Date>::new());
        settings.set_evaluation_date(today);
        settings
    }

    /// A three-coupon 18-month leg over an unlinked Euribor 6M.
    fn leg(settings: Shared<Settings<Date>>) -> Vec<Shared<IborCoupon>> {
        let index = shared(Euribor::six_months(
            Handle::<dyn YieldTermStructure>::empty(),
            settings,
        ));
        let schedule = MakeSchedule::new()
            .from(Date::new(15, Month::January, 2026))
            .to(Date::new(15, Month::July, 2027))
            .with_frequency(Frequency::Semiannual)
            .with_calendar(Target::new())
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .build();
        IborLeg::new(schedule, index)
            .with_notional(100.0)
            .coupons()
            .unwrap()
    }

    /// A three-coupon 9-month overnight leg over unlinked SOFR.
    fn overnight_leg(settings: Shared<Settings<Date>>) -> Vec<Shared<OvernightIndexedCoupon>> {
        let index = shared(Sofr::new(
            Handle::<dyn YieldTermStructure>::empty(),
            settings,
        ));
        let schedule = MakeSchedule::new()
            .from(Date::new(15, Month::January, 2026))
            .to(Date::new(15, Month::October, 2026))
            .with_frequency(Frequency::Quarterly)
            .with_calendar(Target::new())
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .build();
        OvernightLeg::new(schedule, index)
            .with_notional(100.0)
            .coupons()
            .unwrap()
    }

    #[test]
    fn a_cap_pads_the_strike_to_the_leg_length() {
        let settings = settings_on(Date::new(2, Month::January, 2026));
        let coupons = leg(settings.clone());
        let n = coupons.len();
        let cap = CapFloor::cap(coupons, vec![0.03], settings).unwrap();

        assert_eq!(cap.cap_floor_type(), CapFloorType::Cap);
        assert_eq!(cap.cap_rates(), vec![0.03; n].as_slice());
        assert!(cap.floor_rates().is_empty());
    }

    #[test]
    fn a_collar_keeps_both_padded_strike_vectors() {
        let settings = settings_on(Date::new(2, Month::January, 2026));
        let coupons = leg(settings.clone());
        let n = coupons.len();
        let collar = CapFloor::collar(coupons, vec![0.06], vec![0.02], settings).unwrap();

        assert_eq!(collar.cap_floor_type(), CapFloorType::Collar);
        assert_eq!(collar.cap_rates(), vec![0.06; n].as_slice());
        assert_eq!(collar.floor_rates(), vec![0.02; n].as_slice());
    }

    /// `optionlet(i)` keeps the parent's type and carries coupon `i` / strike `i`
    /// (`capfloor.cpp:195-208`). Distinct per-coupon strikes + `Shared::ptr_eq`
    /// pin the index map (a constant-strike / coupons[0] bug would still pass
    /// a padded-strike / len-only check).
    #[test]
    fn an_optionlet_carries_one_coupon_and_its_own_strike() {
        let settings = settings_on(Date::new(2, Month::January, 2026));
        let coupons = leg(settings.clone());
        let n = coupons.len();
        assert_eq!(n, 3, "fixture is the documented 3-coupon leg");
        let collar = CapFloor::collar(
            coupons.clone(),
            vec![0.04, 0.05, 0.06],
            vec![0.01, 0.02, 0.03],
            settings,
        )
        .unwrap();

        let optionlet = collar.optionlet(1).unwrap();
        assert_eq!(optionlet.cap_floor_type(), CapFloorType::Collar);
        assert_eq!(optionlet.coupons().len(), 1);
        assert!(Shared::ptr_eq(&optionlet.coupons()[0], &coupons[1]));
        assert_eq!(optionlet.cap_rates(), [0.05].as_slice());
        assert_eq!(optionlet.floor_rates(), [0.02].as_slice());

        let err = collar.optionlet(n).err().expect("past the leg");
        assert!(err.message().contains("does not exist"), "err was: {err}");
    }

    /// Overnight `from_overnight` instruments index `optionlet` by
    /// `coupon_count()`, not the empty Ibor store.
    #[test]
    fn an_overnight_optionlet_carries_one_coupon_and_its_own_strike() {
        let settings = settings_on(Date::new(2, Month::January, 2026));
        let coupons = overnight_leg(settings.clone());
        let n = coupons.len();
        assert_eq!(n, 3, "fixture is a 3-coupon overnight leg");
        let cap = CapFloor::from_overnight(
            CapFloorType::Cap,
            coupons.clone(),
            vec![0.04, 0.05, 0.06],
            Vec::new(),
            settings,
        )
        .unwrap();

        assert_eq!(cap.coupon_count(), n);
        assert!(cap.last_floating_rate_coupon().is_none());
        assert!(Shared::ptr_eq(
            cap.last_overnight_coupon().expect("overnight cap"),
            &coupons[n - 1]
        ));

        let optionlet = cap.optionlet(1).unwrap();
        assert_eq!(optionlet.cap_floor_type(), CapFloorType::Cap);
        assert!(optionlet.coupons().is_empty());
        assert_eq!(optionlet.overnight_coupons().len(), 1);
        assert!(Shared::ptr_eq(
            &optionlet.overnight_coupons()[0],
            &coupons[1]
        ));
        assert_eq!(optionlet.cap_rates(), [0.05].as_slice());

        let err = cap.optionlet(n).err().expect("past the leg");
        assert!(err.message().contains("does not exist"), "err was: {err}");
    }

    #[test]
    fn a_cap_needs_at_least_one_rate() {
        let settings = settings_on(Date::new(2, Month::January, 2026));
        let coupons = leg(settings.clone());
        let err = CapFloor::cap(coupons, Vec::new(), settings).err().unwrap();
        assert_eq!(err.message(), "no cap rates given");
    }

    #[test]
    fn a_floor_needs_at_least_one_rate() {
        let settings = settings_on(Date::new(2, Month::January, 2026));
        let coupons = leg(settings.clone());
        let err = CapFloor::floor(coupons, Vec::new(), settings)
            .err()
            .unwrap();
        assert_eq!(err.message(), "no floor rates given");
    }

    /// `setup_arguments` fills `start_dates` with each coupon's accrual start
    /// (`capfloor.cpp:237`), the maturity the analytic engine forms its bond
    /// option from.
    #[test]
    fn setup_arguments_fills_start_dates_with_the_accrual_starts() {
        // Evaluation date past the whole leg: every forward is `None`, so
        // `setup_arguments` never forecasts off the (unlinked) index curve, yet
        // `start_dates` is still populated unconditionally.
        let settings = settings_on(Date::new(2, Month::January, 2028));
        let coupons = leg(settings.clone());
        let expected: Vec<Date> = coupons.iter().map(|c| c.accrual_start_date()).collect();
        let cap = CapFloor::cap(coupons, vec![0.03], settings).unwrap();

        let mut args = CapFloorArguments::default();
        cap.setup_arguments(&mut args).unwrap();
        assert_eq!(args.start_dates, expected);
        assert_eq!(args.start_dates.len(), args.end_dates.len());
    }

    /// `CapFloor::arguments::validate` rejects a `start_dates` length that has
    /// desynced from `end_dates` (`capfloor.cpp:279`).
    #[test]
    fn validate_rejects_a_desynced_start_date_count() {
        let settings = settings_on(Date::new(2, Month::January, 2028));
        let coupons = leg(settings.clone());
        let cap = CapFloor::cap(coupons, vec![0.03], settings).unwrap();

        let mut args = CapFloorArguments::default();
        cap.setup_arguments(&mut args).unwrap();
        args.start_dates.pop();
        assert_eq!(
            args.validate().err().unwrap().message(),
            "start-date count mismatch"
        );
    }

    /// `CapFloor::isExpired`: expired once every coupon has paid.
    #[test]
    fn a_cap_is_expired_only_once_all_coupons_have_paid() {
        let settings = settings_on(Date::new(2, Month::January, 2026));
        let coupons = leg(settings.clone());
        let cap = CapFloor::cap(coupons, vec![0.03], settings.clone()).unwrap();
        assert!(!cap.is_expired().unwrap());

        settings.set_evaluation_date(Date::new(15, Month::August, 2027));
        assert!(cap.is_expired().unwrap());
    }
}
