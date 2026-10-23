//! Writer-extensible option.
//!
//! Port of `ql/instruments/writerextensibleoption.{hpp,cpp}`. If OTM at the
//! first exercise date, the option is extended to a later date and strike.

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

/// Arguments (`WriterExtensibleOption::arguments`).
#[derive(Default)]
pub struct WriterExtensibleArguments {
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
    pub payoff2: Option<PlainVanillaPayoff>,
    pub exercise2: Option<Shared<dyn Exercise>>,
}

impl Arguments for WriterExtensibleArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(self.payoff2.is_some(), "no second payoff given");
        require!(self.exercise2.is_some(), "no second exercise given");
        let d1 = self.exercise.as_ref().expect("validated").last_date();
        let d2 = self.exercise2.as_ref().expect("validated").last_date();
        require!(d2 > d1, "second exercise date is not later than the first");
        Ok(())
    }
}

/// NPV-only results.
#[derive(Default)]
pub struct WriterExtensibleResults {
    pub instrument: InstrumentResults,
}

impl Results for WriterExtensibleResults {
    fn reset(&mut self) {
        self.instrument.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Writer-extensible option (`ql/instruments/writerextensibleoption.hpp`).
pub struct WriterExtensibleOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
    payoff2: PlainVanillaPayoff,
    exercise2: Shared<dyn Exercise>,
}

impl WriterExtensibleOption {
    /// `WriterExtensibleOption(payoff1, exercise1, payoff2, exercise2)`.
    pub fn new(
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        payoff2: PlainVanillaPayoff,
        exercise2: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            payoff,
            exercise,
            payoff2,
            exercise2,
        }
    }
}

impl Instrument for WriterExtensibleOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        crate::event::event_has_occurred(self.exercise2.last_date(), &self.settings, None, None)
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) =
            (arguments as &mut dyn Any).downcast_mut::<WriterExtensibleArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        arguments.payoff2 = Some(self.payoff2);
        arguments.exercise2 = Some(Shared::clone(&self.exercise2));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<WriterExtensibleResults>() else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
