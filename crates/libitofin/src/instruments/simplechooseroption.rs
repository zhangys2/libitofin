//! Simple chooser option instrument.
//!
//! Port of `ql/instruments/simplechooseroption.{hpp,cpp}`.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{Greeks, PlainVanillaPayoff};
use crate::option::OptionType;
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Arguments for simple-chooser engines.
#[derive(Default)]
pub struct SimpleChooserArguments {
    pub choosing_date: Option<Date>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for SimpleChooserArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(self.choosing_date.is_some(), "no choosing date given");
        let choosing_date = self.choosing_date.unwrap();
        require!(
            choosing_date < self.exercise.as_ref().expect("validated").last_date(),
            "choosing date later than or equal to maturity date"
        );
        Ok(())
    }
}

/// Engine results for simple chooser options.
#[derive(Default)]
pub struct SimpleChooserResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for SimpleChooserResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Simple chooser European option (same strike for call/put; choice at `choosing_date`).
pub struct SimpleChooserOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    choosing_date: Date,
    strike: Real,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
    greeks: Greeks,
}

impl SimpleChooserOption {
    /// `SimpleChooserOption(choosingDate, strike, exercise)`.
    pub fn new(
        choosing_date: Date,
        strike: Real,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(strike > 0.0, "strike must be positive");
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            choosing_date,
            strike,
            payoff: PlainVanillaPayoff::new(OptionType::Call, strike),
            exercise,
            greeks: Greeks::default(),
        })
    }

    pub fn choosing_date(&self) -> Date {
        self.choosing_date
    }

    pub fn strike(&self) -> Real {
        self.strike
    }
}

impl Instrument for SimpleChooserOption {
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
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<SimpleChooserArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.choosing_date = Some(self.choosing_date);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<SimpleChooserResults>() else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
