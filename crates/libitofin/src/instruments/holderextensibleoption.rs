//! Holder-extensible option.
//! Port of `ql/instruments/holderextensibleoption.{hpp,cpp}`.

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
use crate::types::Real;

/// Arguments (`HolderExtensibleOption::arguments`).
#[derive(Default)]
pub struct HolderExtensibleArguments {
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
    pub premium: Option<Real>,
    pub second_expiry: Option<Date>,
    pub second_strike: Option<Real>,
}

impl Arguments for HolderExtensibleArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        require!(
            self.premium.is_some() && self.premium.unwrap() > 0.0,
            "negative premium not allowed"
        );
        require!(self.second_expiry.is_some(), "no extending date given");
        let d2 = self.second_expiry.unwrap();
        require!(d2 != Date::null(), "no extending date given");
        let d1 = self.exercise.as_ref().expect("validated").last_date();
        require!(
            d2 >= d1,
            "extended date is earlier than or equal to first maturity date"
        );
        Ok(())
    }
}

/// NPV-only results.
#[derive(Default)]
pub struct HolderExtensibleResults {
    pub instrument: InstrumentResults,
}

impl Results for HolderExtensibleResults {
    fn reset(&mut self) {
        self.instrument.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Holder-extensible option (`ql/instruments/holderextensibleoption.hpp`).
pub struct HolderExtensibleOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    premium: Real,
    second_expiry: Date,
    second_strike: Real,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
}

impl HolderExtensibleOption {
    /// `HolderExtensibleOption(type, premium, secondExpiryDate, secondStrike, payoff, exercise)`.
    pub fn new(
        premium: Real,
        second_expiry: Date,
        second_strike: Real,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            premium,
            second_expiry,
            second_strike,
            payoff,
            exercise,
        }
    }
}

impl Instrument for HolderExtensibleOption {
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
            (arguments as &mut dyn Any).downcast_mut::<HolderExtensibleArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        arguments.premium = Some(self.premium);
        arguments.second_expiry = Some(self.second_expiry);
        arguments.second_strike = Some(self.second_strike);
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<HolderExtensibleResults>() else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
