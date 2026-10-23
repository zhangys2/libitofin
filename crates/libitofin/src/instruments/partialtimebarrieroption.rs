//! Partial-time barrier option instrument.
//!
//! Port of `ql/instruments/partialtimebarrieroption.{hpp,cpp}`.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{BarrierType, Greeks, PlainVanillaPayoff};
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Monitoring window for a partial-time barrier (QuantLib `PartialBarrier::Range`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PartialBarrierRange {
    /// From start until the cover event.
    Start,
    /// From cover event to expiry; knock-out on hit/cross from either side.
    EndB1,
    /// From cover event to expiry; knock-out if already on wrong side at cover event.
    EndB2,
}

/// Arguments for partial-time barrier engines.
#[derive(Default)]
pub struct PartialTimeBarrierArguments {
    pub barrier_type: Option<BarrierType>,
    pub barrier_range: Option<PartialBarrierRange>,
    pub barrier: Option<Real>,
    pub rebate: Option<Real>,
    pub cover_event_date: Option<Date>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for PartialTimeBarrierArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(self.barrier_type.is_some(), "no barrier type given");
        require!(self.barrier_range.is_some(), "no barrier range given");
        require!(self.barrier.is_some(), "no barrier given");
        require!(self.rebate.is_some(), "no rebate given");
        require!(self.cover_event_date.is_some(), "no cover event date given");
        Ok(())
    }
}

/// Engine results for partial-time barrier options.
#[derive(Default)]
pub struct PartialTimeBarrierResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for PartialTimeBarrierResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Partial-time barrier European option.
pub struct PartialTimeBarrierOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    barrier_type: BarrierType,
    barrier_range: PartialBarrierRange,
    barrier: Real,
    rebate: Real,
    cover_event_date: Date,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
    greeks: Greeks,
}

impl PartialTimeBarrierOption {
    /// `PartialTimeBarrierOption(barrierType, barrierRange, barrier, rebate, coverEventDate, payoff, exercise)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        barrier_type: BarrierType,
        barrier_range: PartialBarrierRange,
        barrier: Real,
        rebate: Real,
        cover_event_date: Date,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(barrier > 0.0, "barrier must be positive");
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            barrier_type,
            barrier_range,
            barrier,
            rebate,
            cover_event_date,
            payoff,
            exercise,
            greeks: Greeks::default(),
        })
    }

    pub fn barrier_type(&self) -> BarrierType {
        self.barrier_type
    }

    pub fn barrier_range(&self) -> PartialBarrierRange {
        self.barrier_range
    }

    pub fn barrier(&self) -> Real {
        self.barrier
    }

    pub fn rebate(&self) -> Real {
        self.rebate
    }

    pub fn cover_event_date(&self) -> Date {
        self.cover_event_date
    }
}

impl Instrument for PartialTimeBarrierOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        crate::event::event_has_occurred(self.exercise.last_date(), &self.settings, None, None)
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) =
            (arguments as &mut dyn Any).downcast_mut::<PartialTimeBarrierArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.barrier_type = Some(self.barrier_type);
        arguments.barrier_range = Some(self.barrier_range);
        arguments.barrier = Some(self.barrier);
        arguments.rebate = Some(self.rebate);
        arguments.cover_event_date = Some(self.cover_event_date);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<PartialTimeBarrierResults>()
        else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
