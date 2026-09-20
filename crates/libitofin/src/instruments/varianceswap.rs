//! Variance swap. Port of `ql/instruments/varianceswap.{hpp,cpp}`.
//!
//! Does not manage seasoned variance swaps.

use std::any::Any;

use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::position::Position;
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

#[derive(Default)]
pub struct VarianceSwapArguments {
    pub position: Option<Position>,
    pub strike: Option<Real>,
    pub notional: Option<Real>,
    pub start_date: Option<Date>,
    pub maturity_date: Option<Date>,
}

impl Arguments for VarianceSwapArguments {
    #[rustfmt::skip]
    fn validate(&self) -> QlResult<()> {
        let Some(strike) = self.strike else { fail!("no strike given"); };
        require!(strike > 0.0, "negative or null strike given");
        let Some(notional) = self.notional else { fail!("no notional given"); };
        require!(notional > 0.0, "negative or null notional given");
        require!(self.start_date.is_some(), "null start date given");
        require!(self.maturity_date.is_some(), "null maturity date given");
        require!(self.position.is_some(), "no position given");
        Ok(())
    }
}

#[derive(Default)]
pub struct VarianceSwapResults {
    pub instrument: InstrumentResults,
    pub variance: Option<Real>,
}

impl Results for VarianceSwapResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.variance = None;
    }
    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

pub struct VarianceSwap {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    position: Position,
    strike: Real,
    notional: Real,
    start_date: Date,
    maturity_date: Date,
    variance: Option<Real>,
}

impl VarianceSwap {
    #[rustfmt::skip]
    pub fn new(
        position: Position, strike: Real, notional: Real, start_date: Date, maturity_date: Date,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self { base, settings, position, strike, notional, start_date, maturity_date, variance: None }
    }

    pub fn strike(&self) -> Real {
        self.strike
    }
    pub fn position(&self) -> Position {
        self.position
    }
    pub fn start_date(&self) -> Date {
        self.start_date
    }
    pub fn maturity_date(&self) -> Date {
        self.maturity_date
    }
    pub fn notional(&self) -> Real {
        self.notional
    }

    pub fn variance(&mut self) -> QlResult<Real> {
        self.calculate()?;
        let Some(v) = self.variance else {
            fail!("result not available");
        };
        Ok(v)
    }
}

impl Instrument for VarianceSwap {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        event_has_occurred(self.maturity_date, &self.settings, None, None)
    }

    #[rustfmt::skip]
    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<VarianceSwapArguments>() else {
            fail!("wrong argument type");
        };
        arguments.position = Some(self.position);
        arguments.strike = Some(self.strike);
        arguments.notional = Some(self.notional);
        arguments.start_date = Some(self.start_date);
        arguments.maturity_date = Some(self.maturity_date);
        Ok(())
    }

    #[rustfmt::skip]
    fn setup_expired(&mut self) {
        self.base_mut().store_results(&InstrumentResults {
            value: Some(0.0), error_estimate: Some(0.0), ..InstrumentResults::default()
        });
        self.variance = None;
    }

    #[rustfmt::skip]
    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<VarianceSwapResults>() else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        self.variance = results.variance;
        Ok(())
    }
}
