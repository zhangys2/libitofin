//! Cliquet (ratchet) option instrument.
//!
//! Port of `ql/instruments/cliquetoption.{hpp,cpp}`.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{Greeks, PercentageStrikePayoff, StrikedTypePayoff};
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Arguments for cliquet engines.
#[derive(Default)]
pub struct CliquetArguments {
    pub payoff: Option<PercentageStrikePayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
    pub reset_dates: Vec<Date>,
    pub accrued_coupon: Option<Real>,
    pub last_fixing: Option<Real>,
    pub local_cap: Option<Real>,
    pub local_floor: Option<Real>,
    pub global_cap: Option<Real>,
    pub global_floor: Option<Real>,
}

impl Arguments for CliquetArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        let payoff = self.payoff.expect("validated");
        require!(payoff.strike() > 0.0, "negative or zero moneyness given");
        if let Some(c) = self.accrued_coupon {
            require!(c >= 0.0, "negative accrued coupon");
        }
        if let Some(c) = self.local_cap {
            require!(c >= 0.0, "negative local cap");
        }
        if let Some(c) = self.local_floor {
            require!(c >= 0.0, "negative local floor");
        }
        if let Some(c) = self.global_cap {
            require!(c >= 0.0, "negative global cap");
        }
        if let Some(c) = self.global_floor {
            require!(c >= 0.0, "negative global floor");
        }
        require!(!self.reset_dates.is_empty(), "no reset dates given");
        let maturity = self.exercise.as_ref().expect("validated").last_date();
        for (i, reset) in self.reset_dates.iter().enumerate() {
            require!(maturity > *reset, "reset date greater or equal to maturity");
            if i > 0 {
                require!(*reset > self.reset_dates[i - 1], "unsorted reset dates");
            }
        }
        Ok(())
    }
}

/// Engine results for cliquet options.
#[derive(Default)]
pub struct CliquetResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for CliquetResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Cliquet (ratchet) option with percentage strikes at reset dates.
pub struct CliquetOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    payoff: PercentageStrikePayoff,
    exercise: Shared<dyn Exercise>,
    reset_dates: Vec<Date>,
    greeks: Greeks,
}

impl CliquetOption {
    /// `CliquetOption(payoff, maturity, resetDates)`.
    pub fn new(
        payoff: PercentageStrikePayoff,
        exercise: Shared<dyn Exercise>,
        reset_dates: Vec<Date>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            payoff,
            exercise,
            reset_dates,
            greeks: Greeks::default(),
        })
    }
}

impl Instrument for CliquetOption {
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
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<CliquetArguments>() else {
            fail!("wrong argument type");
        };
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        arguments.reset_dates = self.reset_dates.clone();
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<CliquetResults>() else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
