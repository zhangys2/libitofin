//! Exchange option: exchange `Q2` of asset 2 for `Q1` of asset 1
//! (`ql/instruments/margrabeoption.{hpp,cpp}`). NPV-only; extra greeks follow-up.
//! Always carries a [`NullPayoff`] (QuantLib constructor).

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::NullPayoff;
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Integer;

/// Arguments for Margrabe engines (`MargrabeOption::arguments`).
#[derive(Default)]
pub struct MargrabeArguments {
    pub q1: Option<Integer>,
    pub q2: Option<Integer>,
    pub payoff: Option<NullPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for MargrabeArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.q1.is_some(), "unspecified quantity for asset 1");
        require!(self.q2.is_some(), "unspecified quantity for asset 2");
        require!(self.q1.unwrap() > 0, "quantity of asset 1 must be positive");
        require!(self.q2.unwrap() > 0, "quantity of asset 2 must be positive");
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        Ok(())
    }
}

/// NPV-only results (`MargrabeOption::results` without extra greeks).
#[derive(Default)]
pub struct MargrabeResults {
    pub instrument: InstrumentResults,
}

impl Results for MargrabeResults {
    fn reset(&mut self) {
        self.instrument.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Exchange option (`ql/instruments/margrabeoption.hpp`).
pub struct MargrabeOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    q1: Integer,
    q2: Integer,
    payoff: NullPayoff,
    exercise: Shared<dyn Exercise>,
}

impl MargrabeOption {
    /// `MargrabeOption(Q1, Q2, exercise)`.
    pub fn new(
        q1: Integer,
        q2: Integer,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            q1,
            q2,
            payoff: NullPayoff,
            exercise,
        }
    }
}

impl Instrument for MargrabeOption {
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
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<MargrabeArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.q1 = Some(self.q1);
        arguments.q2 = Some(self.q2);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<MargrabeResults>() else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
