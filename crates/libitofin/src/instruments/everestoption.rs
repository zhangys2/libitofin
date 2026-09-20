//! Everest option on a basket of assets.
//! Port of `ql/experimental/exoticoptions/everestoption.{hpp,cpp}`.

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
use crate::types::{Rate, Real};

/// Arguments (`EverestOption::arguments`).
#[derive(Default)]
pub struct EverestArguments {
    pub payoff: Option<NullPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
    pub notional: Option<Real>,
    pub guarantee: Option<Rate>,
}

impl Arguments for EverestArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(self.notional.is_some(), "no notional given");
        require!(self.notional.unwrap() != 0.0, "null notional given");
        require!(self.guarantee.is_some(), "no guarantee given");
        Ok(())
    }
}

/// NPV + Everest yield (`EverestOption::results`).
#[derive(Default)]
pub struct EverestResults {
    pub instrument: InstrumentResults,
    pub yield_rate: Option<Rate>,
}

impl Results for EverestResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.yield_rate = None;
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Everest option (`ql/experimental/exoticoptions/everestoption.hpp`).
pub struct EverestOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    notional: Real,
    guarantee: Rate,
    payoff: NullPayoff,
    exercise: Shared<dyn Exercise>,
    yield_rate: Option<Rate>,
}

impl EverestOption {
    /// `EverestOption(notional, guarantee, exercise)`.
    pub fn new(
        notional: Real,
        guarantee: Rate,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            notional,
            guarantee,
            payoff: NullPayoff,
            exercise,
            yield_rate: None,
        }
    }

    /// C++ `EverestOption::yield()`.
    pub fn yield_rate(&mut self) -> QlResult<Rate> {
        self.calculate()?;
        let Some(y) = self.yield_rate else {
            fail!("yield not provided");
        };
        Ok(y)
    }
}

impl Instrument for EverestOption {
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
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<EverestArguments>() else {
            fail!("wrong argument type");
        };
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        arguments.notional = Some(self.notional);
        arguments.guarantee = Some(self.guarantee);
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<EverestResults>() else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        self.yield_rate = results.yield_rate;
        Ok(())
    }
}
