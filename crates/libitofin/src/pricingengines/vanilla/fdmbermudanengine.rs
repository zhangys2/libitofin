//! One-dimensional finite-difference pricing for Bermudan vanilla options.
//!
//! Port of the Bermudan path through QuantLib's `FdBlackScholesVanillaEngine`
//! (via `FdmBermudanStepCondition`). It mirrors [`FdmAmericanEngine`] but the
//! early-exercise floor is applied only on the discrete exercise dates rather
//! than at every timestep.
//!
//! [`FdmAmericanEngine`]: super::FdmAmericanEngine

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
    PlainVanillaPayoff, StrikedTypePayoff,
};
use crate::math::array::Array;
use crate::methods::finitedifferences::StepCondition;
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher,
};
use crate::methods::finitedifferences::operators::{FdmBlackScholesOp, FdmLinearOpComposite};
use crate::methods::finitedifferences::solvers::{FdmBackwardSolver, FdmSchemeDesc};
use crate::methods::finitedifferences::stepconditions::{
    FdmBermudanStepCondition, FdmStepConditionComposite,
};
use crate::methods::finitedifferences::utilities::{FdmInnerValueCalculator, fdm_log_inner_value};
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size, Time};

/// A one-dimensional Douglas Black-Scholes finite-difference engine for
/// Bermudan vanillas (QuantLib's `FdBlackScholesVanillaEngine` Bermudan path).
pub struct FdmBermudanEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    grid_points: Size,
    time_steps: Size,
    damping_steps: Size,
    mesher_eps: Real,
    mesher_scale_factor: Real,
}

impl FdmBermudanEngine {
    /// Builds an engine with the same default grid as [`FdmAmericanEngine`]
    /// (400 spatial points, 200 time steps).
    ///
    /// [`FdmAmericanEngine`]: super::FdmAmericanEngine
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_grid(process, 400, 200)
    }

    /// Builds an engine with explicit spatial (`grid_points`) and temporal
    /// (`time_steps`) resolution.
    pub fn with_grid(
        process: Shared<GeneralizedBlackScholesProcess>,
        grid_points: Size,
        time_steps: Size,
    ) -> Self {
        assert!(grid_points >= 3, "the FDM grid needs at least three points");
        assert!(time_steps > 0, "the FDM rollback needs at least one step");
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            grid_points,
            time_steps,
            damping_steps: 0,
            mesher_eps: 0.0001,
            mesher_scale_factor: 1.5,
        }
    }

    /// Sets the number of implicit-Euler damping steps before Douglas.
    pub fn with_damping_steps(mut self, damping_steps: Size) -> Self {
        self.damping_steps = damping_steps;
        self
    }
}

impl AsObservable for FdmBermudanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdmBermudanEngine {
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
        let arguments = self.base.arguments();
        let Some(exercise) = &arguments.exercise else {
            fail!("no exercise given");
        };
        if exercise.exercise_type() != ExerciseType::Bermudan {
            fail!("not a Bermudan option");
        }
        let Some(payoff) = &arguments.payoff else {
            fail!("no payoff given");
        };
        let Some(payoff) = (&**payoff as &dyn Any).downcast_ref::<PlainVanillaPayoff>() else {
            fail!("the FDM Bermudan engine needs a plain vanilla payoff");
        };
        let payoff = *payoff;

        let maturity_date = exercise.last_date();
        let maturity = self.process.time(&maturity_date)?;
        if maturity <= 0.0 {
            fail!("the FDM Bermudan engine needs a positive maturity");
        }

        // Compute the exercise times with the risk-free curve's day counter and
        // reference date - the exact convention `process.time` uses - so the
        // stopping times fed to the rollback coincide (bit-for-bit) with the
        // times `FdmBermudanStepCondition` recomputes, and the condition fires
        // on the exact grid points the model lands on. Past/today exercise
        // opportunities (t <= 0) do not enter the present-value rollback.
        let risk_free = self.process.risk_free_rate().current_link()?;
        let reference_date = risk_free.reference_date()?;
        let day_counter = risk_free.require_day_counter()?;
        let mut exercise_dates = Vec::new();
        let mut exercise_times: Vec<Time> = Vec::new();
        for date in exercise.dates() {
            let t = day_counter.year_fraction(reference_date, *date);
            if t > 0.0 {
                exercise_dates.push(*date);
                exercise_times.push(t);
            }
        }
        if exercise_times.is_empty() {
            fail!("the FDM Bermudan engine needs at least one future exercise date");
        }

        let spot = self.process.x0()?;
        let strike = payoff.strike();

        let equity = fdm_black_scholes_mesher(
            self.grid_points,
            &self.process,
            maturity,
            strike,
            None,
            None,
            self.mesher_eps,
            self.mesher_scale_factor,
            Some((strike, 0.01)),
            &[],
            0.0,
        )?;
        let mesher = shared(FdmMesherComposite::new(vec![equity]));
        let locations = mesher.locations(0);
        let mut values = terminal_payoff(&payoff, &locations);

        let payoff_dyn: Shared<dyn Payoff> = shared(payoff);
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            Shared::clone(&payoff_dyn),
            mesher.clone() as Shared<dyn FdmMesher>,
            0,
        ));
        let bermudan: Shared<dyn StepCondition> = shared(FdmBermudanStepCondition::new(
            &exercise_dates,
            reference_date,
            &day_counter,
            mesher.clone() as Shared<dyn FdmMesher>,
            calculator,
        ));
        // The exercise times enter as the composite's stopping times so the
        // rollback is cut to land exactly on each exercise date.
        let conditions = shared(FdmStepConditionComposite::new(
            &[exercise_times],
            vec![bermudan],
        ));

        let map = shared_mut(FdmBlackScholesOp::new(
            mesher.clone() as Shared<dyn FdmMesher>,
            &self.process,
            strike,
            0,
        )?);
        let mut solver = FdmBackwardSolver::new(
            map as SharedMut<dyn FdmLinearOpComposite>,
            Vec::new(),
            Some(conditions),
            FdmSchemeDesc::douglas(),
        );
        solver.rollback(
            &mut values,
            maturity,
            0.0,
            self.time_steps,
            self.damping_steps,
        )?;

        let value = interpolate(&locations, &values, spot.ln());
        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.greeks = Greeks::default();
        results.more_greeks = MoreGreeks::default();
        Ok(())
    }
}

fn terminal_payoff(payoff: &PlainVanillaPayoff, locations: &Array) -> Array {
    locations.iter().map(|x| payoff.value(x.exp())).collect()
}

fn interpolate(x: &Array, values: &Array, point: Real) -> Real {
    if point <= x[0] {
        return values[0];
    }
    let last = x.size() - 1;
    if point >= x[last] {
        return values[last];
    }
    for i in 0..last {
        if point <= x[i + 1] {
            let weight = (point - x[i]) / (x[i + 1] - x[i]);
            return values[i] + weight * (values[i + 1] - values[i]);
        }
    }
    values[last]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, BermudanExercise, EuropeanExercise, Exercise};
    use crate::instrument::Instrument;
    use crate::instruments::OneAssetOption;
    use crate::option::OptionType;
    use crate::pricingengines::vanilla::test_market::{Market, market, time_to_days, today};
    use crate::pricingengines::vanilla::{AnalyticEuropeanEngine, FdmAmericanEngine};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::time::date::Date;

    fn bermudan_npv(
        market: &Market,
        option_type: OptionType,
        strike: Real,
        dates: Vec<Date>,
    ) -> Real {
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(option_type, strike));
        let exercise: Shared<dyn Exercise> = shared(BermudanExercise::new(dates, false).unwrap());
        let mut option = OneAssetOption::new(payoff, exercise, Shared::clone(&market.settings));
        let engine = shared_mut(FdmBermudanEngine::new(Shared::clone(&market.process)))
            as SharedMut<dyn PricingEngine>;
        option.base_mut().set_pricing_engine(engine);
        option.npv().unwrap()
    }

    #[test]
    fn bermudan_with_only_the_expiry_matches_the_european_price() {
        let market = market();
        market.set(40.0, 0.0, 0.0488, 0.3);
        let expiry = today() + time_to_days(0.5833);

        let berm = bermudan_npv(&market, OptionType::Put, 40.0, vec![expiry]);

        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, 40.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));
        let mut european = OneAssetOption::new(payoff, exercise, Shared::clone(&market.settings));
        let eng = shared_mut(AnalyticEuropeanEngine::new(Shared::clone(&market.process)))
            as SharedMut<dyn PricingEngine>;
        european.base_mut().set_pricing_engine(eng);
        let eur = european.npv().unwrap();

        assert!(
            (berm - eur).abs() < 0.15,
            "bermudan(expiry-only) {berm} vs european {eur}"
        );
    }

    #[test]
    fn more_exercise_dates_raise_value_towards_the_american_bound() {
        let market = market();
        market.set(40.0, 0.0, 0.0488, 0.3);
        let expiry = today() + time_to_days(0.5833);

        // A single exercise at expiry is the no-early-exercise (European) case.
        let single = bermudan_npv(&market, OptionType::Put, 45.0, vec![expiry]);

        // Roughly monthly exercise opportunities up to and including expiry.
        let mut dates = Vec::new();
        let mut days = 30;
        while today() + days < expiry {
            dates.push(today() + days);
            days += 30;
        }
        dates.push(expiry);
        let berm = bermudan_npv(&market, OptionType::Put, 45.0, dates);

        // The continuous-exercise American value is the upper bound.
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, 45.0));
        let exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today(), expiry, false).unwrap());
        let mut american = OneAssetOption::new(payoff, exercise, Shared::clone(&market.settings));
        let eng = shared_mut(FdmAmericanEngine::new(Shared::clone(&market.process)))
            as SharedMut<dyn PricingEngine>;
        american.base_mut().set_pricing_engine(eng);
        let amer = american.npv().unwrap();

        assert!(
            berm >= single - 1e-9,
            "adding exercise dates lowered the value: {berm} < {single}"
        );
        assert!(
            berm > single + 1e-3,
            "monthly early exercise added no value for an ITM put: {berm} vs {single}"
        );
        assert!(
            berm <= amer + 5e-2,
            "bermudan {berm} exceeded the american upper bound {amer}"
        );
    }
}
