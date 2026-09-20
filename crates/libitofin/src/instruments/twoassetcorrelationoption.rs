//! Two-asset correlation option.
//!
//! Port of `ql/instruments/twoassetcorrelationoption.{hpp,cpp}`: pays the
//! second-asset vanilla payoff only if the first asset is also in the money.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::PlainVanillaPayoff;
use crate::option::OptionType;
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Arguments (`TwoAssetCorrelationOption::arguments`).
#[derive(Default)]
pub struct TwoAssetCorrelationArguments {
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
    pub x2: Option<Real>,
}

impl Arguments for TwoAssetCorrelationArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(self.x2.is_some(), "no X2 given");
        Ok(())
    }
}

/// NPV-only results.
#[derive(Default)]
pub struct TwoAssetCorrelationResults {
    pub instrument: InstrumentResults,
}

impl Results for TwoAssetCorrelationResults {
    fn reset(&mut self) {
        self.instrument.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Two-asset correlation option (`ql/instruments/twoassetcorrelationoption.hpp`).
pub struct TwoAssetCorrelationOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
    x2: Real,
}

impl TwoAssetCorrelationOption {
    /// `TwoAssetCorrelationOption(type, strike1, strike2, exercise)`.
    pub fn new(
        option_type: OptionType,
        strike1: Real,
        strike2: Real,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            payoff: PlainVanillaPayoff::new(option_type, strike1),
            exercise,
            x2: strike2,
        }
    }
}

impl Instrument for TwoAssetCorrelationOption {
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
            (arguments as &mut dyn Any).downcast_mut::<TwoAssetCorrelationArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        arguments.x2 = Some(self.x2);
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<TwoAssetCorrelationResults>()
        else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
