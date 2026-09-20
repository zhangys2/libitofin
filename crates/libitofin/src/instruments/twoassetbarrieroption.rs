//! Two-asset barrier option (payoff from S1, barrier on S2).
//! Port of `ql/instruments/twoassetbarrieroption.{hpp,cpp}`.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{BarrierType, PlainVanillaPayoff};
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Arguments (`TwoAssetBarrierOption::arguments`).
#[derive(Default)]
pub struct TwoAssetBarrierArguments {
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
    pub barrier_type: Option<BarrierType>,
    pub barrier: Option<Real>,
}

impl Arguments for TwoAssetBarrierArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(self.barrier_type.is_some(), "unknown type");
        require!(self.barrier.is_some(), "no barrier given");
        Ok(())
    }
}

/// NPV-only results.
#[derive(Default)]
pub struct TwoAssetBarrierResults {
    pub instrument: InstrumentResults,
}

impl Results for TwoAssetBarrierResults {
    fn reset(&mut self) {
        self.instrument.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Two-asset barrier option (`ql/instruments/twoassetbarrieroption.hpp`).
pub struct TwoAssetBarrierOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    barrier_type: BarrierType,
    barrier: Real,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
}

impl TwoAssetBarrierOption {
    /// `TwoAssetBarrierOption(barrierType, barrier, payoff, exercise)`.
    pub fn new(
        barrier_type: BarrierType,
        barrier: Real,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            barrier_type,
            barrier,
            payoff,
            exercise,
        }
    }
}

impl Instrument for TwoAssetBarrierOption {
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
            (arguments as &mut dyn Any).downcast_mut::<TwoAssetBarrierArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        arguments.barrier_type = Some(self.barrier_type);
        arguments.barrier = Some(self.barrier);
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<TwoAssetBarrierResults>() else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
