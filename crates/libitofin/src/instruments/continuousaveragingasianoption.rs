//! Continuous-averaging Asian option instrument.
//!
//! Port of `ql/instruments/asianoption.hpp` (continuous averaging subset).

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase};
use crate::instruments::PlainVanillaPayoff;
use crate::pricingengine::Arguments;
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;

/// Average type (`Average::Type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AverageType {
    Arithmetic,
    Geometric,
}

/// Arguments for continuous-averaging Asian engines.
#[derive(Default)]
pub struct ContinuousAveragingAsianArguments {
    pub average_type: Option<AverageType>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for ContinuousAveragingAsianArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.average_type.is_some(), "no average type given");
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        Ok(())
    }
}

/// European option on the continuous average of the underlying.
pub struct ContinuousAveragingAsianOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    average_type: AverageType,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
}

impl ContinuousAveragingAsianOption {
    /// Builds a continuous-averaging Asian option.
    pub fn new(
        average_type: AverageType,
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
            payoff,
            exercise,
        })
    }

    pub fn average_type(&self) -> AverageType {
        self.average_type
    }
}

impl Instrument for ContinuousAveragingAsianOption {
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
            .downcast_mut::<ContinuousAveragingAsianArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.average_type = Some(self.average_type);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }
}
