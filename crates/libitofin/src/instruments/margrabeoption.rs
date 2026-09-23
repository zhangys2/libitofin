//! Exchange option: exchange `Q2` of asset 2 for `Q1` of asset 1
//! (`ql/instruments/margrabeoption.{hpp,cpp}`). Always carries a [`NullPayoff`].

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::NullPayoff;
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::{Integer, Real};

/// Arguments for Margrabe engines (`MargrabeOption::arguments`).
#[derive(Default)]
pub struct MargrabeArguments {
    pub q1: Option<Integer>,
    pub q2: Option<Integer>,
    pub payoff: Option<NullPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for MargrabeArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.q1.is_some(), "unspecified quantity for asset 1");
        require!(self.q2.is_some(), "unspecified quantity for asset 2");
        require!(self.q1.unwrap() > 0, "quantity of asset 1 must be positive");
        require!(self.q2.unwrap() > 0, "quantity of asset 2 must be positive");
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        Ok(())
    }
}

/// Results for Margrabe options (`MargrabeOption::results`).
#[derive(Default)]
pub struct MargrabeResults {
    pub instrument: InstrumentResults,
    pub delta1: Option<Real>,
    pub delta2: Option<Real>,
    pub gamma1: Option<Real>,
    pub gamma2: Option<Real>,
    pub theta: Option<Real>,
    pub rho: Option<Real>,
}

impl Results for MargrabeResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.delta1 = None;
        self.delta2 = None;
        self.gamma1 = None;
        self.gamma2 = None;
        self.theta = None;
        self.rho = None;
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Exchange option (`ql/instruments/margrabeoption.hpp`).
pub struct MargrabeOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    q1: Integer,
    q2: Integer,
    payoff: NullPayoff,
    exercise: Shared<dyn Exercise>,
    delta1: Option<Real>,
    delta2: Option<Real>,
    gamma1: Option<Real>,
    gamma2: Option<Real>,
    theta: Option<Real>,
    rho: Option<Real>,
}

impl MargrabeOption {
    /// `MargrabeOption(Q1, Q2, exercise)`.
    pub fn new(
        q1: Integer,
        q2: Integer,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            q1,
            q2,
            payoff: NullPayoff,
            exercise,
            delta1: None,
            delta2: None,
            gamma1: None,
            gamma2: None,
            theta: None,
            rho: None,
        }
    }

    pub fn q1(&self) -> Integer {
        self.q1
    }

    pub fn q2(&self) -> Integer {
        self.q2
    }

    pub fn payoff(&self) -> &NullPayoff {
        &self.payoff
    }

    pub fn exercise(&self) -> &Shared<dyn Exercise> {
        &self.exercise
    }

    fn greek(val: Option<Real>, name: &str) -> QlResult<Real> {
        match val {
            Some(v) => Ok(v),
            None => fail!("{name} not provided"),
        }
    }

    pub fn delta1(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.delta1, "delta1")
    }

    pub fn delta2(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.delta2, "delta2")
    }

    pub fn gamma1(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.gamma1, "gamma1")
    }

    pub fn gamma2(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.gamma2, "gamma2")
    }

    pub fn theta(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.theta, "theta")
    }

    pub fn rho(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.rho, "rho")
    }
}

impl Instrument for MargrabeOption {
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
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<MargrabeArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.q1 = Some(self.q1);
        arguments.q2 = Some(self.q2);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn setup_expired(&mut self) {
        self.base_mut().store_results(&InstrumentResults {
            value: Some(0.0),
            error_estimate: Some(0.0),
            ..Default::default()
        });
        self.delta1 = Some(0.0);
        self.delta2 = Some(0.0);
        self.gamma1 = Some(0.0);
        self.gamma2 = Some(0.0);
        self.theta = Some(0.0);
        self.rho = Some(0.0);
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<MargrabeResults>() else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        self.delta1 = results.delta1;
        self.delta2 = results.delta2;
        self.gamma1 = results.gamma1;
        self.gamma2 = results.gamma2;
        self.theta = results.theta;
        self.rho = results.rho;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::shared::shared;

    #[test]
    fn test_expired_margrabe_option() {
        let settings = shared(Settings::new());
        let today = Date::new(15, crate::time::date::Month::May, 2020);
        settings.set_evaluation_date(today);
        let exercise = shared(EuropeanExercise::new(Date::new(
            10,
            crate::time::date::Month::May,
            2020,
        )));
        let mut opt = MargrabeOption::new(1, 1, exercise, settings);
        assert!(opt.is_expired().unwrap());
        assert_eq!(opt.npv().unwrap(), 0.0);
        assert_eq!(opt.delta1().unwrap(), 0.0);
        assert_eq!(opt.delta2().unwrap(), 0.0);
        assert_eq!(opt.gamma1().unwrap(), 0.0);
        assert_eq!(opt.gamma2().unwrap(), 0.0);
        assert_eq!(opt.theta().unwrap(), 0.0);
        assert_eq!(opt.rho().unwrap(), 0.0);
    }

    #[test]
    fn test_margrabe_missing_greeks() {
        let settings = shared(Settings::new());
        let today = Date::new(15, crate::time::date::Month::May, 2020);
        settings.set_evaluation_date(today);
        let exercise = shared(EuropeanExercise::new(Date::new(
            20,
            crate::time::date::Month::May,
            2020,
        )));
        let mut opt = MargrabeOption::new(1, 1, exercise, settings);
        assert!(opt.delta1().is_err());
        assert!(opt.delta2().is_err());
        assert!(opt.gamma1().is_err());
        assert!(opt.gamma2().is_err());
        assert!(opt.theta().is_err());
        assert!(opt.rho().is_err());
    }
}
