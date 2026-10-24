//! Compound option (option on option) on a single asset.
//!
//! Port of `ql/instruments/compoundoption.{hpp,cpp}`: [`CompoundArguments`],
//! [`CompoundResults`], and [`CompoundOption`] with its greek accessors.
//! Mother = compound; daughter = underlying option.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{Greeks, MoreGreeks, OneAssetOptionResults, PlainVanillaPayoff};
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Arguments for compound-option engines (`CompoundOption::arguments`).
#[derive(Default)]
pub struct CompoundArguments {
    pub mother_payoff: Option<PlainVanillaPayoff>,
    pub mother_exercise: Option<Shared<dyn Exercise>>,
    pub daughter_payoff: Option<PlainVanillaPayoff>,
    pub daughter_exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for CompoundArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.mother_payoff.is_some(), "no payoff given");
        require!(self.mother_exercise.is_some(), "no exercise given");
        require!(
            self.daughter_payoff.is_some(),
            "no payoff given for underlying option"
        );
        require!(
            self.daughter_exercise.is_some(),
            "no exercise given for underlying option"
        );
        require!(
            self.mother_exercise.as_ref().unwrap().last_date()
                <= self.daughter_exercise.as_ref().unwrap().last_date(),
            "maturity of compound option exceeds maturity of underlying option"
        );
        Ok(())
    }
}

/// Compound option results (`ql/instruments/compoundoption.hpp`).
///
/// In QuantLib, `CompoundOption` inherits from `OneAssetOption`, sharing its
/// results class which bundles base instrument results, primary greeks, and
/// more greeks.
pub type CompoundResults = OneAssetOptionResults;

/// Compound option (`ql/instruments/compoundoption.hpp`).
pub struct CompoundOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    mother_payoff: PlainVanillaPayoff,
    mother_exercise: Shared<dyn Exercise>,
    daughter_payoff: PlainVanillaPayoff,
    daughter_exercise: Shared<dyn Exercise>,
    greeks: Greeks,
    more_greeks: MoreGreeks,
}

macro_rules! impl_greek {
    ($fn:ident, $group:ident, $desc:expr) => {
        #[doc = concat!("Sensitivity / greek `", $desc, "`.")]
        pub fn $fn(&mut self) -> QlResult<Real> {
            self.calculate()?;
            Self::greek(self.$group.$fn, $desc)
        }
    };
}

impl CompoundOption {
    /// `CompoundOption(motherPayoff, motherExercise, daughterPayoff, daughterExercise)`.
    pub fn new(
        mother_payoff: PlainVanillaPayoff,
        mother_exercise: Shared<dyn Exercise>,
        daughter_payoff: PlainVanillaPayoff,
        daughter_exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            mother_payoff,
            mother_exercise,
            daughter_payoff,
            daughter_exercise,
            greeks: Greeks::default(),
            more_greeks: MoreGreeks::default(),
        }
    }

    /// Payoff of the compound (mother) option.
    pub fn mother_payoff(&self) -> &PlainVanillaPayoff {
        &self.mother_payoff
    }

    /// Exercise schedule of the compound (mother) option.
    pub fn mother_exercise(&self) -> &Shared<dyn Exercise> {
        &self.mother_exercise
    }

    /// Payoff of the underlying (daughter) option.
    pub fn daughter_payoff(&self) -> &PlainVanillaPayoff {
        &self.daughter_payoff
    }

    /// Exercise schedule of the underlying (daughter) option.
    pub fn daughter_exercise(&self) -> &Shared<dyn Exercise> {
        &self.daughter_exercise
    }

    /// The payoff the option is written on (convenience alias for mother payoff).
    pub fn payoff(&self) -> &PlainVanillaPayoff {
        &self.mother_payoff
    }

    /// The exercise schedule (convenience alias for mother exercise).
    pub fn exercise(&self) -> &Shared<dyn Exercise> {
        &self.mother_exercise
    }

    fn greek(value: Option<Real>, description: &str) -> QlResult<Real> {
        let Some(value) = value else {
            fail!("{description} not provided");
        };
        Ok(value)
    }

    impl_greek!(delta, greeks, "delta");
    impl_greek!(gamma, greeks, "gamma");
    impl_greek!(theta, greeks, "theta");
    impl_greek!(vega, greeks, "vega");
    impl_greek!(rho, greeks, "rho");
    impl_greek!(dividend_rho, greeks, "dividend rho");
    impl_greek!(delta_forward, more_greeks, "forward delta");
    impl_greek!(elasticity, more_greeks, "elasticity");
    impl_greek!(theta_per_day, more_greeks, "theta per-day");
    impl_greek!(strike_sensitivity, more_greeks, "strike sensitivity");
    impl_greek!(
        itm_cash_probability,
        more_greeks,
        "in-the-money cash probability"
    );
}

impl Instrument for CompoundOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        crate::event::event_has_occurred(
            self.mother_exercise.last_date(),
            &self.settings,
            None,
            None,
        )
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<CompoundArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.mother_payoff = Some(self.mother_payoff);
        arguments.mother_exercise = Some(Shared::clone(&self.mother_exercise));
        arguments.daughter_payoff = Some(self.daughter_payoff);
        arguments.daughter_exercise = Some(Shared::clone(&self.daughter_exercise));
        Ok(())
    }

    fn setup_expired(&mut self) {
        self.base_mut().store_results(&InstrumentResults {
            value: Some(0.0),
            error_estimate: Some(0.0),
            ..Default::default()
        });
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
        let Some(results) = (results as &dyn Any).downcast_ref::<CompoundResults>() else {
            fail!("wrong result type");
        };
        self.base_mut().store_results(&results.instrument);
        self.greeks = results.greeks;
        self.more_greeks = results.more_greeks;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::option::OptionType;
    use crate::patterns::observable::{AsObservable, Observable};
    use crate::pricingengine::{GenericEngine, PricingEngine};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::time::date::Month;

    struct MockEngine {
        base: GenericEngine<CompoundArguments, CompoundResults>,
        populate: fn(&mut CompoundResults),
    }

    impl AsObservable for MockEngine {
        fn observable(&self) -> &Observable {
            self.base.observable()
        }
    }

    impl PricingEngine for MockEngine {
        fn arguments_mut(&mut self) -> &mut dyn Arguments {
            self.base.arguments_mut()
        }
        fn results(&self) -> &dyn Results {
            self.base.results()
        }
        fn reset(&mut self) {
            self.base.reset();
        }
        fn calculate(&mut self) -> QlResult<()> {
            (self.populate)(self.base.results_mut());
            Ok(())
        }
    }

    fn dummy_option(settings: Shared<Settings<Date>>) -> CompoundOption {
        let today = Date::new(15, Month::May, 1998);
        CompoundOption::new(
            PlainVanillaPayoff::new(OptionType::Call, 50.0),
            shared(EuropeanExercise::new(today + 90)),
            PlainVanillaPayoff::new(OptionType::Call, 520.0),
            shared(EuropeanExercise::new(today + 180)),
            settings,
        )
    }

    #[test]
    fn test_expired_compound_option() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today + 100);
        let mut opt = dummy_option(settings);

        assert!(opt.is_expired().unwrap());
        assert_eq!(opt.npv().unwrap(), 0.0);
        for g in [
            opt.delta(),
            opt.gamma(),
            opt.theta(),
            opt.vega(),
            opt.rho(),
            opt.dividend_rho(),
            opt.delta_forward(),
            opt.elasticity(),
            opt.theta_per_day(),
            opt.strike_sensitivity(),
            opt.itm_cash_probability(),
        ] {
            assert_eq!(g.unwrap(), 0.0);
        }
    }

    #[test]
    fn test_mock_compound_engine_and_missing_greeks() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(Date::new(15, Month::May, 1998));
        let mut opt = dummy_option(settings);

        let engine = shared_mut(MockEngine {
            base: GenericEngine::new(CompoundArguments::default(), CompoundResults::default()),
            populate: |res| {
                res.instrument.value = Some(10.0);
                res.greeks.delta = Some(0.42);
            },
        });
        opt.base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

        assert_eq!(opt.npv().unwrap(), 10.0);
        assert_eq!(opt.delta().unwrap(), 0.42);
        for (g, msg) in [
            (opt.gamma(), "gamma not provided"),
            (opt.theta(), "theta not provided"),
            (opt.vega(), "vega not provided"),
            (opt.rho(), "rho not provided"),
        ] {
            assert_eq!(g.unwrap_err().message(), msg);
        }
    }
}
