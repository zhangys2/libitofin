//! One-dimensional finite-difference pricing for American vanilla options.
//!
//! Port of the American path through QuantLib's `FdBlackScholesVanillaEngine`
//! (via `FdmAmericanStepCondition`), exercised against
//! `test-suite/americanoption.cpp` `testFdValues` / Ju (1999) table values.

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
    FdmAmericanStepCondition, FdmStepConditionComposite,
};
use crate::methods::finitedifferences::utilities::{FdmInnerValueCalculator, fdm_log_inner_value};
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size};

/// A one-dimensional Douglas Black-Scholes finite-difference engine for
/// American vanillas (QuantLib's `FdBlackScholesVanillaEngine` American path).
pub struct FdmAmericanEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    grid_points: Size,
    time_steps: Size,
    damping_steps: Size,
    mesher_eps: Real,
    mesher_scale_factor: Real,
}

impl FdmAmericanEngine {
    /// Builds an engine with QuantLib's American FD spatial grid (400) and a
    /// slightly denser time grid (200) so the near-zero-rate Ju cases stay
    /// inside the `americanoption.cpp` `testFdValues` 8e-2 tolerance.
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

impl AsObservable for FdmAmericanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdmAmericanEngine {
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
        if exercise.exercise_type() != ExerciseType::American {
            fail!("not an American option");
        }
        let Some(payoff) = &arguments.payoff else {
            fail!("no payoff given");
        };
        let Some(payoff) = (&**payoff as &dyn Any).downcast_ref::<PlainVanillaPayoff>() else {
            fail!("the FDM American engine needs a plain vanilla payoff");
        };
        let payoff = *payoff;

        let maturity_date = exercise.last_date();
        let maturity = self.process.time(&maturity_date)?;
        if maturity <= 0.0 {
            fail!("the FDM American engine needs a positive maturity");
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
        // Match QuantLib's FdBlackScholesVanillaEngine: empty Dirichlet set;
        // the American step condition supplies the intrinsic floor.

        let payoff_dyn: Shared<dyn Payoff> = shared(payoff);
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            Shared::clone(&payoff_dyn),
            mesher.clone() as Shared<dyn FdmMesher>,
            0,
        ));
        let american: Shared<dyn StepCondition> = shared(FdmAmericanStepCondition::new(
            mesher.clone() as Shared<dyn FdmMesher>,
            calculator,
            0.0,
        ));
        let conditions = shared(FdmStepConditionComposite::new(&[], vec![american]));

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
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::instrument::Instrument;
    use crate::instruments::{OneAssetOption, PlainVanillaPayoff};
    use crate::option::OptionType;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::vanilla::test_market::{market, time_to_days, today};
    use crate::shared::{SharedMut, shared, shared_mut};

    /// Ju (1999) table values from `americanoption.cpp` `juValues[]`.
    /// Columns: type, strike, spot, q, r, t, vol, value.
    #[allow(clippy::type_complexity)]
    const JU_VALUES: &[(OptionType, Real, Real, Real, Real, Real, Real, Real)] = &[
        (OptionType::Put, 35.0, 40.0, 0.0, 0.0488, 0.0833, 0.2, 0.006),
        (OptionType::Put, 35.0, 40.0, 0.0, 0.0488, 0.3333, 0.2, 0.201),
        (OptionType::Put, 35.0, 40.0, 0.0, 0.0488, 0.5833, 0.2, 0.433),
        (OptionType::Put, 40.0, 40.0, 0.0, 0.0488, 0.0833, 0.2, 0.851),
        (OptionType::Put, 40.0, 40.0, 0.0, 0.0488, 0.3333, 0.2, 1.576),
        (OptionType::Put, 40.0, 40.0, 0.0, 0.0488, 0.5833, 0.2, 1.984),
        (OptionType::Put, 45.0, 40.0, 0.0, 0.0488, 0.0833, 0.2, 5.000),
        (OptionType::Put, 45.0, 40.0, 0.0, 0.0488, 0.3333, 0.2, 5.084),
        (OptionType::Put, 45.0, 40.0, 0.0, 0.0488, 0.5833, 0.2, 5.260),
        (OptionType::Put, 35.0, 40.0, 0.0, 0.0488, 0.0833, 0.3, 0.078),
        (OptionType::Put, 35.0, 40.0, 0.0, 0.0488, 0.3333, 0.3, 0.697),
        (OptionType::Put, 35.0, 40.0, 0.0, 0.0488, 0.5833, 0.3, 1.218),
        (OptionType::Put, 40.0, 40.0, 0.0, 0.0488, 0.0833, 0.3, 1.309),
        (OptionType::Put, 40.0, 40.0, 0.0, 0.0488, 0.3333, 0.3, 2.477),
        (OptionType::Put, 40.0, 40.0, 0.0, 0.0488, 0.5833, 0.3, 3.161),
        (OptionType::Put, 45.0, 40.0, 0.0, 0.0488, 0.0833, 0.3, 5.059),
        (OptionType::Put, 45.0, 40.0, 0.0, 0.0488, 0.3333, 0.3, 5.699),
        (OptionType::Put, 45.0, 40.0, 0.0, 0.0488, 0.5833, 0.3, 6.231),
        (OptionType::Put, 35.0, 40.0, 0.0, 0.0488, 0.0833, 0.4, 0.247),
        (OptionType::Put, 35.0, 40.0, 0.0, 0.0488, 0.3333, 0.4, 1.344),
        (OptionType::Put, 35.0, 40.0, 0.0, 0.0488, 0.5833, 0.4, 2.150),
        (OptionType::Put, 40.0, 40.0, 0.0, 0.0488, 0.0833, 0.4, 1.767),
        (OptionType::Put, 40.0, 40.0, 0.0, 0.0488, 0.3333, 0.4, 3.381),
        (OptionType::Put, 40.0, 40.0, 0.0, 0.0488, 0.5833, 0.4, 4.342),
        (OptionType::Put, 45.0, 40.0, 0.0, 0.0488, 0.0833, 0.4, 5.288),
        (OptionType::Put, 45.0, 40.0, 0.0, 0.0488, 0.3333, 0.4, 6.501),
        (OptionType::Put, 45.0, 40.0, 0.0, 0.0488, 0.5833, 0.4, 7.367),
        (OptionType::Call, 100.0, 80.0, 0.07, 0.03, 3.0, 0.2, 2.605),
        (OptionType::Call, 100.0, 90.0, 0.07, 0.03, 3.0, 0.2, 5.182),
        (OptionType::Call, 100.0, 100.0, 0.07, 0.03, 3.0, 0.2, 9.065),
        (OptionType::Call, 100.0, 110.0, 0.07, 0.03, 3.0, 0.2, 14.430),
        (OptionType::Call, 100.0, 120.0, 0.07, 0.03, 3.0, 0.2, 21.398),
        (OptionType::Call, 100.0, 80.0, 0.07, 0.03, 3.0, 0.4, 11.336),
        (OptionType::Call, 100.0, 90.0, 0.07, 0.03, 3.0, 0.4, 15.711),
        (OptionType::Call, 100.0, 100.0, 0.07, 0.03, 3.0, 0.4, 20.760),
        (OptionType::Call, 100.0, 110.0, 0.07, 0.03, 3.0, 0.4, 26.440),
        (OptionType::Call, 100.0, 120.0, 0.07, 0.03, 3.0, 0.4, 32.709),
        (
            OptionType::Call,
            100.0,
            80.0,
            0.07,
            0.00001,
            3.0,
            0.3,
            5.552,
        ),
        (
            OptionType::Call,
            100.0,
            90.0,
            0.07,
            0.00001,
            3.0,
            0.3,
            8.868,
        ),
        (
            OptionType::Call,
            100.0,
            100.0,
            0.07,
            0.00001,
            3.0,
            0.3,
            13.158,
        ),
        (
            OptionType::Call,
            100.0,
            110.0,
            0.07,
            0.00001,
            3.0,
            0.3,
            18.458,
        ),
        (
            OptionType::Call,
            100.0,
            120.0,
            0.07,
            0.00001,
            3.0,
            0.3,
            24.786,
        ),
        (OptionType::Call, 100.0, 80.0, 0.03, 0.07, 3.0, 0.3, 12.177),
        (OptionType::Call, 100.0, 90.0, 0.03, 0.07, 3.0, 0.3, 17.411),
        (OptionType::Call, 100.0, 100.0, 0.03, 0.07, 3.0, 0.3, 23.402),
        (OptionType::Call, 100.0, 110.0, 0.03, 0.07, 3.0, 0.3, 30.028),
        (OptionType::Call, 100.0, 120.0, 0.03, 0.07, 3.0, 0.3, 37.177),
    ];

    fn american_option(
        market: &crate::pricingengines::vanilla::test_market::Market,
        option_type: OptionType,
        strike: Real,
        expiry: crate::time::date::Date,
    ) -> OneAssetOption {
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(option_type, strike));
        let exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today(), expiry, false).unwrap());
        OneAssetOption::new(payoff, exercise, Shared::clone(&market.settings))
    }

    #[test]
    fn fdm_american_matches_ju_values_within_8e_2() {
        let market = market();
        let mut worst: Real = 0.0;
        for &(option_type, strike, spot, q, r, t, vol, expected) in JU_VALUES {
            market.set(spot, q, r, vol);
            let expiry = today() + time_to_days(t);
            let mut option = american_option(&market, option_type, strike, expiry);
            let engine = shared_mut(FdmAmericanEngine::new(Shared::clone(&market.process)))
                as SharedMut<dyn PricingEngine>;
            option.base_mut().set_pricing_engine(engine);
            let actual = option.npv().unwrap();
            let err = (actual - expected).abs();
            worst = worst.max(err);
            assert!(
                err <= 8.0e-2,
                "{option_type:?} K={strike} S={spot} q={q} r={r} t={t} vol={vol}: {actual} vs {expected} (err={err})"
            );
        }
        assert!(worst <= 8.0e-2);
    }

    #[test]
    fn american_put_dominates_european_put() {
        let market = market();
        market.set(40.0, 0.0, 0.0488, 0.3);
        let expiry = today() + time_to_days(0.5833);

        let mut american = american_option(&market, OptionType::Put, 40.0, expiry);
        let engine = shared_mut(FdmAmericanEngine::new(Shared::clone(&market.process)))
            as SharedMut<dyn PricingEngine>;
        american.base_mut().set_pricing_engine(engine);
        let american_npv = american.npv().unwrap();

        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, 40.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));
        let mut european = OneAssetOption::new(payoff, exercise, Shared::clone(&market.settings));
        let eur_engine = shared_mut(crate::pricingengines::vanilla::AnalyticEuropeanEngine::new(
            Shared::clone(&market.process),
        )) as SharedMut<dyn PricingEngine>;
        european.base_mut().set_pricing_engine(eur_engine);
        let european_npv = european.npv().unwrap();

        assert!(
            american_npv + 1e-8 >= european_npv,
            "american {american_npv} < european {european_npv}"
        );
    }
}
