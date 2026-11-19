//! Variance option on realized variance.
//!
//! Port of `ql/experimental/varianceoption/varianceoption.{hpp,cpp}`. Seasoned
//! options are not managed: the contract is the payoff, notional, and the
//! start and maturity dates.

use std::any::Any;

use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::instrument::{Instrument, InstrumentBase};
use crate::instruments::StrikedTypePayoff;
use crate::pricingengine::{Arguments, GenericEngine};
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;
use crate::{fail, require};

/// Inputs for a variance option.
#[derive(Default)]
pub struct VarianceOptionArguments {
    /// Strike payoff. A plain call uses the one-dimensional integral.
    pub payoff: Option<Shared<dyn StrikedTypePayoff>>,
    /// Cash amount per variance unit.
    pub notional: Option<Real>,
    /// Contract start. Stored, not used by the integral Heston engine.
    pub start_date: Option<Date>,
    /// Realized-variance horizon.
    pub maturity_date: Option<Date>,
}

impl Arguments for VarianceOptionArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no strike given");
        let Some(notional) = self.notional else {
            fail!("no notional given");
        };
        require!(notional > 0.0, "negative or null notional given");
        require!(self.start_date.is_some(), "null start date given");
        require!(self.maturity_date.is_some(), "null maturity date given");
        Ok(())
    }
}

/// Generic engine base for variance options.
pub type VarianceOptionEngine =
    GenericEngine<VarianceOptionArguments, crate::instrument::InstrumentResults>;

/// Option on realized variance.
pub struct VarianceOption {
    base: InstrumentBase,
    payoff: Shared<dyn StrikedTypePayoff>,
    notional: Real,
    start_date: Date,
    maturity_date: Date,
    settings: Shared<Settings<Date>>,
}

impl VarianceOption {
    /// Builds the contract. `notional` must be positive and both dates set.
    ///
    /// # Errors
    ///
    /// Returns an error when the notional is not positive or a date is missing.
    pub fn new(
        payoff: Shared<dyn StrikedTypePayoff>,
        notional: Real,
        start_date: Date,
        maturity_date: Date,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let arguments = VarianceOptionArguments {
            payoff: Some(Shared::clone(&payoff)),
            notional: Some(notional),
            start_date: Some(start_date),
            maturity_date: Some(maturity_date),
        };
        arguments.validate()?;
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            payoff,
            notional,
            start_date,
            maturity_date,
            settings,
        })
    }

    /// Cash amount per variance unit.
    pub fn notional(&self) -> Real {
        self.notional
    }

    /// Contract start.
    pub fn start_date(&self) -> Date {
        self.start_date
    }

    /// Realized-variance horizon.
    pub fn maturity_date(&self) -> Date {
        self.maturity_date
    }
}

impl Instrument for VarianceOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        event_has_occurred(self.maturity_date, &self.settings, None, None)
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<VarianceOptionArguments>()
        else {
            fail!("wrong argument type");
        };
        *arguments = VarianceOptionArguments {
            payoff: Some(Shared::clone(&self.payoff)),
            notional: Some(self.notional),
            start_date: Some(self.start_date),
            maturity_date: Some(self.maturity_date),
        };
        Ok(())
    }
}
