//! Complex chooser option instrument.
//!
//! Port of `ql/instruments/complexchooseroption.{hpp,cpp}`.

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

/// Arguments for complex-chooser engines.
#[derive(Default)]
pub struct ComplexChooserArguments {
    pub choosing_date: Option<Date>,
    pub strike_call: Option<Real>,
    pub strike_put: Option<Real>,
    pub exercise_call: Option<Shared<dyn Exercise>>,
    pub exercise_put: Option<Shared<dyn Exercise>>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for ComplexChooserArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(self.choosing_date.is_some(), "no choosing date given");
        require!(self.strike_call.is_some(), "no call strike given");
        require!(self.strike_put.is_some(), "no put strike given");
        require!(self.exercise_call.is_some(), "no call exercise given");
        require!(self.exercise_put.is_some(), "no put exercise given");
        let choosing_date = self.choosing_date.unwrap();
        require!(
            choosing_date < self.exercise_call.as_ref().expect("validated").last_date(),
            "choosing date later than or equal to Call maturity date"
        );
        require!(
            choosing_date < self.exercise_put.as_ref().expect("validated").last_date(),
            "choosing date later than or equal to Put maturity date"
        );
        Ok(())
    }
}

/// Engine results for complex chooser options.
#[derive(Default)]
pub struct ComplexChooserResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for ComplexChooserResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Complex chooser European option (distinct call/put strikes and maturities).
pub struct ComplexChooserOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    choosing_date: Date,
    strike_call: Real,
    strike_put: Real,
    exercise_call: Shared<dyn Exercise>,
    exercise_put: Shared<dyn Exercise>,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
    greeks: Greeks,
}

impl ComplexChooserOption {
    /// `ComplexChooserOption(choosingDate, strikeCall, strikePut, exerciseCall, exercisePut)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        choosing_date: Date,
        strike_call: Real,
        strike_put: Real,
        exercise_call: Shared<dyn Exercise>,
        exercise_put: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(strike_call > 0.0, "call strike must be positive");
        require!(strike_put > 0.0, "put strike must be positive");
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            choosing_date,
            strike_call,
            strike_put,
            exercise_call: Shared::clone(&exercise_call),
            exercise_put: Shared::clone(&exercise_put),
            payoff: PlainVanillaPayoff::new(OptionType::Call, strike_call),
            exercise: exercise_call,
            greeks: Greeks::default(),
        })
    }

    pub fn choosing_date(&self) -> Date {
        self.choosing_date
    }

    pub fn strike_call(&self) -> Real {
        self.strike_call
    }

    pub fn strike_put(&self) -> Real {
        self.strike_put
    }
}

impl Instrument for ComplexChooserOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        let call_expired =
            crate::event::event_has_occurred(self.exercise_call.last_date(), &self.settings, None, None)?;
        let put_expired =
            crate::event::event_has_occurred(self.exercise_put.last_date(), &self.settings, None, None)?;
        Ok(call_expired && put_expired)
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) =
            (arguments as &mut dyn Any).downcast_mut::<ComplexChooserArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.choosing_date = Some(self.choosing_date);
        arguments.strike_call = Some(self.strike_call);
        arguments.strike_put = Some(self.strike_put);
        arguments.exercise_call = Some(Shared::clone(&self.exercise_call));
        arguments.exercise_put = Some(Shared::clone(&self.exercise_put));
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<ComplexChooserResults>() else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
