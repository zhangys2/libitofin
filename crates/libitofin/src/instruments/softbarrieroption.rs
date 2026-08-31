//! Soft barrier option instrument.
//!
//! Port of `ql/instruments/softbarrieroption.{hpp,cpp}`.

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

/// Arguments for soft-barrier engines.
#[derive(Default)]
pub struct SoftBarrierArguments {
    pub barrier_type: Option<BarrierType>,
    pub barrier_lo: Option<Real>,
    pub barrier_hi: Option<Real>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for SoftBarrierArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(self.barrier_type.is_some(), "no barrier type given");
        require!(self.barrier_lo.is_some(), "no low barrier given");
        require!(self.barrier_hi.is_some(), "no high barrier given");
        Ok(())
    }
}

/// Engine results for soft barrier options.
#[derive(Default)]
pub struct SoftBarrierResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for SoftBarrierResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Soft barrier European option (Hart–Ross / Haug).
pub struct SoftBarrierOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    barrier_type: BarrierType,
    barrier_lo: Real,
    barrier_hi: Real,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
    greeks: Greeks,
}

impl SoftBarrierOption {
    /// `SoftBarrierOption(barrierType, barrier_lo, barrier_hi, payoff, exercise)`.
    pub fn new(
        barrier_type: BarrierType,
        barrier_lo: Real,
        barrier_hi: Real,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            barrier_type,
            barrier_lo,
            barrier_hi,
            payoff,
            exercise,
            greeks: Greeks::default(),
        })
    }

    pub fn barrier_type(&self) -> BarrierType {
        self.barrier_type
    }

    pub fn barrier_lo(&self) -> Real {
        self.barrier_lo
    }

    pub fn barrier_hi(&self) -> Real {
        self.barrier_hi
    }
}

impl Instrument for SoftBarrierOption {
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
            (arguments as &mut dyn Any).downcast_mut::<SoftBarrierArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.barrier_type = Some(self.barrier_type);
        arguments.barrier_lo = Some(self.barrier_lo);
        arguments.barrier_hi = Some(self.barrier_hi);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<SoftBarrierResults>() else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
