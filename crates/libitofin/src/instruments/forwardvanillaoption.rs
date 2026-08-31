//! Forward (strike-resetting) vanilla option.
//!
//! Port of `ql/instruments/forwardvanillaoption.{hpp,cpp}`.

use std::any::Any;

use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{Greeks, MoreGreeks, OneAssetOptionResults, StrikedTypePayoff};
use crate::pricingengine::{Arguments, GenericEngine, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Arguments for forward vanilla engines.
#[derive(Default)]
pub struct ForwardVanillaArguments {
    pub payoff: Option<Shared<dyn StrikedTypePayoff>>,
    pub exercise: Option<Shared<dyn Exercise>>,
    pub moneyness: Option<Real>,
    pub reset_date: Option<Date>,
}

impl ForwardVanillaArguments {
    /// Validates payoff, exercise, moneyness and reset date.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn validate_with_evaluation_date(&self, evaluation_date: Date) -> QlResult<()> {
        if self.payoff.is_none() {
            fail!("no payoff given");
        }
        if self.exercise.is_none() {
            fail!("no exercise given");
        }
        if self.moneyness.is_none() {
            fail!("null moneyness given");
        }
        let moneyness = self.moneyness.unwrap();
        require!(moneyness > 0.0, "negative or zero moneyness given");
        if self.reset_date.is_none() {
            fail!("null reset date given");
        }
        let reset_date = self.reset_date.unwrap();
        require!(reset_date >= evaluation_date, "reset date in the past");
        let exercise = self.exercise.as_ref().unwrap();
        require!(
            exercise.last_date() > reset_date,
            "reset date later or equal to maturity"
        );
        Ok(())
    }
}

impl Arguments for ForwardVanillaArguments {
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn validate(&self) -> QlResult<()> {
        if self.payoff.is_none() {
            fail!("no payoff given");
        }
        if self.exercise.is_none() {
            fail!("no exercise given");
        }
        if self.moneyness.is_none() {
            fail!("null moneyness given");
        }
        let moneyness = self.moneyness.unwrap();
        require!(moneyness > 0.0, "negative or zero moneyness given");
        require!(self.reset_date.is_some(), "null reset date given");
        Ok(())
    }
}

/// Engine base for forward vanilla options.
pub type ForwardVanillaEngineBase = GenericEngine<ForwardVanillaArguments, OneAssetOptionResults>;

/// Forward version of a vanilla option on a single asset.
pub struct ForwardVanillaOption {
    base: InstrumentBase,
    payoff: Shared<dyn StrikedTypePayoff>,
    exercise: Shared<dyn Exercise>,
    settings: Shared<Settings<Date>>,
    moneyness: Real,
    reset_date: Date,
    greeks: Greeks,
    more_greeks: MoreGreeks,
}

impl ForwardVanillaOption {
    /// Builds a forward vanilla option.
    pub fn new(
        moneyness: Real,
        reset_date: Date,
        payoff: Shared<dyn StrikedTypePayoff>,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> ForwardVanillaOption {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        ForwardVanillaOption {
            base,
            payoff,
            exercise,
            settings,
            moneyness,
            reset_date,
            greeks: Greeks::default(),
            more_greeks: MoreGreeks::default(),
        }
    }
}

impl Instrument for ForwardVanillaOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        event_has_occurred(self.exercise.last_date(), &self.settings, None, None)
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<ForwardVanillaArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.payoff = Some(Shared::clone(&self.payoff));
        arguments.exercise = Some(Shared::clone(&self.exercise));
        arguments.moneyness = Some(self.moneyness);
        arguments.reset_date = Some(self.reset_date);
        if let Some(eval_date) = self.settings.evaluation_date() {
            arguments.validate_with_evaluation_date(eval_date)?;
        }
        Ok(())
    }

    fn setup_expired(&mut self) {
        let expired = InstrumentResults {
            value: Some(0.0),
            error_estimate: Some(0.0),
            ..InstrumentResults::default()
        };
        self.base_mut().store_results(&expired);
        self.greeks = Greeks {
            delta: Some(0.0),
            gamma: Some(0.0),
            theta: Some(0.0),
            vega: Some(0.0),
            rho: Some(0.0),
            dividend_rho: Some(0.0),
        };
        self.more_greeks = MoreGreeks {
            itm_cash_probability: Some(0.0),
            delta_forward: Some(0.0),
            elasticity: Some(0.0),
            theta_per_day: Some(0.0),
            strike_sensitivity: Some(0.0),
        };
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<OneAssetOptionResults>() else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.more_greeks = results.more_greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
