//! Forward (strike-resetting) vanilla option.
//!
//! Port of `ql/instruments/forwardvanillaoption.{hpp,cpp}`: a one-asset
//! option whose strike is set at a future reset date as a moneyness of the
//! then spot.

use std::any::Any;

use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{Greeks, MoreGreeks, OneAssetOptionResults, StrikedTypePayoff};
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Arguments for a forward (strike-resetting) option.
///
/// Port of `ForwardOptionArguments<OneAssetOption::arguments>`.
#[derive(Default)]
pub struct ForwardOptionArguments {
    /// The payoff the option is written on (strike is ignored; moneyness wins).
    pub payoff: Option<Shared<dyn StrikedTypePayoff>>,
    /// The exercise schedule.
    pub exercise: Option<Shared<dyn Exercise>>,
    /// Strike as a fraction of the spot at the reset date.
    pub moneyness: Option<Real>,
    /// Date on which the strike is fixed.
    pub reset_date: Date,
    /// Settings used to validate the reset date against the evaluation date.
    pub settings: Option<Shared<Settings<Date>>>,
}

impl Arguments for ForwardOptionArguments {
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn validate(&self) -> QlResult<()> {
        if self.payoff.is_none() {
            fail!("no payoff given");
        }
        let Some(exercise) = self.exercise.as_ref() else {
            fail!("no exercise given");
        };
        let Some(moneyness) = self.moneyness else {
            fail!("null moneyness given");
        };
        require!(moneyness > 0.0, "negative or zero moneyness given");
        require!(self.reset_date != Date::null(), "null reset date given");
        let Some(settings) = self.settings.as_ref() else {
            fail!("no settings given for forward option arguments");
        };
        let Some(today) = settings.evaluation_date() else {
            fail!("no evaluation date set");
        };
        require!(self.reset_date >= today, "reset date in the past");
        require!(
            exercise.last_date() > self.reset_date,
            "reset date later or equal to maturity"
        );
        Ok(())
    }
}

/// Forward version of a vanilla option.
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
    /// `ForwardVanillaOption(moneyness, resetDate, payoff, exercise)`.
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

    /// Strike moneyness at the reset date.
    pub fn moneyness(&self) -> Real {
        self.moneyness
    }

    /// Strike-fix date.
    pub fn reset_date(&self) -> Date {
        self.reset_date
    }

    fn greek(value: Option<Real>, description: &str) -> QlResult<Real> {
        let Some(value) = value else {
            fail!("{description} not provided");
        };
        Ok(value)
    }

    /// The option delta.
    pub fn delta(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.delta, "delta")
    }

    /// The option gamma.
    pub fn gamma(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.gamma, "gamma")
    }

    /// The option theta.
    pub fn theta(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.theta, "theta")
    }

    /// The option vega.
    pub fn vega(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.vega, "vega")
    }

    /// The option rho.
    pub fn rho(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.rho, "rho")
    }

    /// The option dividend rho.
    pub fn dividend_rho(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.dividend_rho, "dividend rho")
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
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<ForwardOptionArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.payoff = Some(Shared::clone(&self.payoff));
        arguments.exercise = Some(Shared::clone(&self.exercise));
        arguments.moneyness = Some(self.moneyness);
        arguments.reset_date = self.reset_date;
        arguments.settings = Some(Shared::clone(&self.settings));
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
