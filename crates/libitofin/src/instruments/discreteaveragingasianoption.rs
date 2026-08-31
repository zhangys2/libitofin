//! Discrete-averaging Asian option instrument.
//!
//! Port of `ql/instruments/asianoption.hpp` (discrete averaging subset).

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{AverageType, Greeks, PlainVanillaPayoff};
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::{Real, Size};

/// Arguments for discrete-averaging Asian engines.
#[derive(Default)]
pub struct DiscreteAveragingAsianArguments {
    pub average_type: Option<AverageType>,
    pub running_accumulator: Option<Real>,
    pub past_fixings: Option<Size>,
    pub fixing_dates: Vec<Date>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for DiscreteAveragingAsianArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.average_type.is_some(), "no average type given");
        require!(
            self.running_accumulator.is_some(),
            "no running accumulator given"
        );
        require!(self.past_fixings.is_some(), "no past fixings given");
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        Ok(())
    }
}

/// Engine results for discrete-averaging Asian options.
#[derive(Default)]
pub struct DiscreteAveragingAsianResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for DiscreteAveragingAsianResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// European option on the discrete average of the underlying.
pub struct DiscreteAveragingAsianOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    average_type: AverageType,
    running_accumulator: Real,
    past_fixings: Size,
    fixing_dates: Vec<Date>,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
    greeks: Greeks,
}

impl DiscreteAveragingAsianOption {
    /// Builds a discrete-averaging Asian option (future fixings only).
    pub fn new(
        average_type: AverageType,
        running_accumulator: Real,
        past_fixings: Size,
        fixing_dates: Vec<Date>,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            average_type,
            running_accumulator,
            past_fixings,
            fixing_dates,
            payoff,
            exercise,
            greeks: Greeks::default(),
        })
    }

    fn greek(value: Option<Real>, description: &str) -> QlResult<Real> {
        let Some(value) = value else {
            fail!("{description} not provided");
        };
        Ok(value)
    }

    pub fn theta(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.theta, "theta")
    }

    pub fn average_type(&self) -> AverageType {
        self.average_type
    }

    pub fn running_accumulator(&self) -> Real {
        self.running_accumulator
    }

    pub fn past_fixings(&self) -> Size {
        self.past_fixings
    }

    pub fn fixing_dates(&self) -> &[Date] {
        &self.fixing_dates
    }
}

impl Instrument for DiscreteAveragingAsianOption {
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
            .downcast_mut::<DiscreteAveragingAsianArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.average_type = Some(self.average_type);
        arguments.running_accumulator = Some(self.running_accumulator);
        arguments.past_fixings = Some(self.past_fixings);
        arguments.fixing_dates = self.fixing_dates.clone();
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<DiscreteAveragingAsianResults>()
        else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
