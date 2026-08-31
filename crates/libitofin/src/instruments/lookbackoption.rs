//! Continuous lookback option instruments.
//!
//! Port of `ql/instruments/lookbackoption.{hpp,cpp}` (continuous floating/fixed slice).

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{FloatingTypePayoff, Greeks, PlainVanillaPayoff, TypePayoff};
use crate::option::OptionType;
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Arguments for continuous floating-strike lookback engines.
#[derive(Default)]
pub struct ContinuousFloatingLookbackArguments {
    pub minmax: Option<Real>,
    pub payoff: Option<FloatingTypePayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for ContinuousFloatingLookbackArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        let minmax = self.minmax.expect("minmax set by instrument");
        require!(
            minmax >= 0.0,
            "nonnegative prior extremum required: {minmax} not allowed"
        );
        Ok(())
    }
}

/// Engine results for continuous floating lookbacks.
#[derive(Default)]
pub struct ContinuousFloatingLookbackResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for ContinuousFloatingLookbackResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Continuous floating-strike lookback option.
pub struct ContinuousFloatingLookbackOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    minmax: Real,
    payoff: FloatingTypePayoff,
    exercise: Shared<dyn Exercise>,
    greeks: Greeks,
}

impl ContinuousFloatingLookbackOption {
    /// `ContinuousFloatingLookbackOption(currentMinmax, payoff, exercise)`.
    pub fn new(
        minmax: Real,
        payoff: FloatingTypePayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(
            minmax >= 0.0,
            "nonnegative prior extremum required: {minmax} not allowed"
        );
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            minmax,
            payoff,
            exercise,
            greeks: Greeks::default(),
        })
    }

    pub fn minmax(&self) -> Real {
        self.minmax
    }
}

impl Instrument for ContinuousFloatingLookbackOption {
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
        let Some(arguments) = (arguments as &mut dyn Any)
            .downcast_mut::<ContinuousFloatingLookbackArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.minmax = Some(self.minmax);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) =
            (results as &dyn Any).downcast_ref::<ContinuousFloatingLookbackResults>()
        else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}

/// Arguments for continuous fixed-strike lookback engines.
#[derive(Default)]
pub struct ContinuousFixedLookbackArguments {
    pub minmax: Option<Real>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for ContinuousFixedLookbackArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        let minmax = self.minmax.expect("minmax set by instrument");
        require!(
            minmax >= 0.0,
            "nonnegative prior extremum required: {minmax} not allowed"
        );
        Ok(())
    }
}

/// Engine results for continuous fixed lookbacks.
#[derive(Default)]
pub struct ContinuousFixedLookbackResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for ContinuousFixedLookbackResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Continuous fixed-strike lookback option.
pub struct ContinuousFixedLookbackOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    minmax: Real,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
    greeks: Greeks,
}

impl ContinuousFixedLookbackOption {
    /// `ContinuousFixedLookbackOption(currentMinmax, payoff, exercise)`.
    pub fn new(
        minmax: Real,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(
            minmax >= 0.0,
            "nonnegative prior extremum required: {minmax} not allowed"
        );
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            minmax,
            payoff,
            exercise,
            greeks: Greeks::default(),
        })
    }

    pub fn minmax(&self) -> Real {
        self.minmax
    }
}

impl Instrument for ContinuousFixedLookbackOption {
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
        let Some(arguments) = (arguments as &mut dyn Any)
            .downcast_mut::<ContinuousFixedLookbackArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.minmax = Some(self.minmax);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<ContinuousFixedLookbackResults>()
        else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}

/// Arguments for continuous partial floating-strike lookback engines.
#[derive(Default)]
pub struct ContinuousPartialFloatingLookbackArguments {
    pub minmax: Option<Real>,
    pub lambda: Option<Real>,
    pub lookback_period_end: Option<Date>,
    pub payoff: Option<FloatingTypePayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for ContinuousPartialFloatingLookbackArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(self.lambda.is_some(), "no lambda given");
        require!(self.lookback_period_end.is_some(), "no lookback period end given");
        let minmax = self.minmax.expect("minmax set by instrument");
        require!(
            minmax >= 0.0,
            "nonnegative prior extremum required: {minmax} not allowed"
        );
        let lookback_period_end = self.lookback_period_end.expect("validated");
        let exercise = self.exercise.as_ref().expect("validated");
        require!(
            lookback_period_end <= exercise.last_date(),
            "lookback start date must be earlier than exercise date"
        );
        let payoff = self.payoff.expect("validated");
        match payoff.option_type() {
            OptionType::Call => {
                require!(
                    self.lambda.expect("validated") >= 1.0,
                    "lambda should be greater than or equal to 1 for calls"
                );
            }
            OptionType::Put => {
                require!(
                    self.lambda.expect("validated") <= 1.0,
                    "lambda should be smaller than or equal to 1 for puts"
                );
            }
        }
        Ok(())
    }
}

/// Engine results for continuous partial floating lookbacks.
#[derive(Default)]
pub struct ContinuousPartialFloatingLookbackResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for ContinuousPartialFloatingLookbackResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Continuous partial-time floating-strike lookback option (Heynen–Kat).
pub struct ContinuousPartialFloatingLookbackOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    minmax: Real,
    lambda: Real,
    lookback_period_end: Date,
    payoff: FloatingTypePayoff,
    exercise: Shared<dyn Exercise>,
    greeks: Greeks,
}

impl ContinuousPartialFloatingLookbackOption {
    /// `ContinuousPartialFloatingLookbackOption(minmax, lambda, lookbackPeriodEnd, ...)`.
    pub fn new(
        minmax: Real,
        lambda: Real,
        lookback_period_end: Date,
        payoff: FloatingTypePayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(
            minmax >= 0.0,
            "nonnegative prior extremum required: {minmax} not allowed"
        );
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            minmax,
            lambda,
            lookback_period_end,
            payoff,
            exercise,
            greeks: Greeks::default(),
        })
    }

    pub fn minmax(&self) -> Real {
        self.minmax
    }

    pub fn lambda(&self) -> Real {
        self.lambda
    }

    pub fn lookback_period_end(&self) -> Date {
        self.lookback_period_end
    }
}

impl Instrument for ContinuousPartialFloatingLookbackOption {
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
        let Some(arguments) = (arguments as &mut dyn Any)
            .downcast_mut::<ContinuousPartialFloatingLookbackArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.minmax = Some(self.minmax);
        arguments.lambda = Some(self.lambda);
        arguments.lookback_period_end = Some(self.lookback_period_end);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any)
            .downcast_ref::<ContinuousPartialFloatingLookbackResults>()
        else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
