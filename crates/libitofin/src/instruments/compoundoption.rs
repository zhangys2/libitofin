//! Compound option (option on option) on a single asset.
//!
//! Port of `ql/instruments/compoundoption.{hpp,cpp}`. NPV-only this slice
//! (greeks follow-up). Mother = compound; daughter = underlying option.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::PlainVanillaPayoff;
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;

/// Arguments for compound-option engines (`CompoundOption::arguments`).
#[derive(Default)]
pub struct CompoundArguments {
    pub mother_payoff: Option<PlainVanillaPayoff>,
    pub mother_exercise: Option<Shared<dyn Exercise>>,
    pub daughter_payoff: Option<PlainVanillaPayoff>,
    pub daughter_exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for CompoundArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.mother_payoff.is_some(), "no payoff given");
        require!(self.mother_exercise.is_some(), "no exercise given");
        require!(
            self.daughter_payoff.is_some(),
            "no payoff given for underlying option"
        );
        require!(
            self.daughter_exercise.is_some(),
            "no exercise given for underlying option"
        );
        require!(
            self.mother_exercise.as_ref().unwrap().last_date()
                <= self.daughter_exercise.as_ref().unwrap().last_date(),
            "maturity of compound option exceeds maturity of underlying option"
        );
        Ok(())
    }
}

/// NPV-only results (`CompoundOption::results`).
#[derive(Default)]
pub struct CompoundResults {
    pub instrument: InstrumentResults,
}

impl Results for CompoundResults {
    fn reset(&mut self) {
        self.instrument.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Compound option (`ql/instruments/compoundoption.hpp`).
pub struct CompoundOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    mother_payoff: PlainVanillaPayoff,
    mother_exercise: Shared<dyn Exercise>,
    daughter_payoff: PlainVanillaPayoff,
    daughter_exercise: Shared<dyn Exercise>,
}

impl CompoundOption {
    /// `CompoundOption(motherPayoff, motherExercise, daughterPayoff, daughterExercise)`.
    pub fn new(
        mother_payoff: PlainVanillaPayoff,
        mother_exercise: Shared<dyn Exercise>,
        daughter_payoff: PlainVanillaPayoff,
        daughter_exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            mother_payoff,
            mother_exercise,
            daughter_payoff,
            daughter_exercise,
        }
    }
}

impl Instrument for CompoundOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        crate::event::event_has_occurred(
            self.mother_exercise.last_date(),
            &self.settings,
            None,
            None,
        )
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<CompoundArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.mother_payoff = Some(self.mother_payoff);
        arguments.mother_exercise = Some(Shared::clone(&self.mother_exercise));
        arguments.daughter_payoff = Some(self.daughter_payoff);
        arguments.daughter_exercise = Some(Shared::clone(&self.daughter_exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<CompoundResults>() else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
