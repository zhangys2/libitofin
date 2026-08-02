//! One-dimensional finite-difference pricing for European vanilla options.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
    PlainVanillaPayoff, StrikedTypePayoff, TypePayoff,
};
use crate::math::array::Array;
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher,
};
use crate::methods::finitedifferences::operators::{FdmBlackScholesOp, FdmLinearOpComposite};
use crate::methods::finitedifferences::solvers::{FdmBackwardSolver, FdmSchemeDesc};
use crate::methods::finitedifferences::{BoundarySide, TimeDependentDirichletBoundary};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size};

/// A one-dimensional Crank-Nicolson Black-Scholes finite-difference engine.
pub struct FdmEuropeanEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    grid_points: Size,
    time_steps: Size,
    damping_steps: Size,
    mesher_eps: Real,
    mesher_scale_factor: Real,
}

impl FdmEuropeanEngine {
    /// Builds an engine with a concentrated 201-point grid and 400 rollback steps.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_grid(process, 201, 400)
    }

    /// Builds an engine with explicit spatial and temporal resolution.
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
            damping_steps: 4,
            mesher_eps: 0.0001,
            mesher_scale_factor: 1.5,
        }
    }

    /// Sets the number of implicit-Euler damping steps before Crank-Nicolson.
    pub fn with_damping_steps(mut self, damping_steps: Size) -> Self {
        self.damping_steps = damping_steps;
        self
    }
}

impl AsObservable for FdmEuropeanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdmEuropeanEngine {
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
        if exercise.exercise_type() != ExerciseType::European {
            fail!("not an European option");
        }
        let Some(payoff) = &arguments.payoff else {
            fail!("no payoff given");
        };
        let Some(payoff) = (&**payoff as &dyn Any).downcast_ref::<PlainVanillaPayoff>() else {
            fail!("the FDM European engine needs a plain vanilla payoff");
        };
        let payoff = *payoff;

        let maturity_date = exercise.last_date();
        let maturity = self.process.time(&maturity_date)?;
        if maturity <= 0.0 {
            fail!("the FDM European engine needs a positive maturity");
        }
        let spot = self.process.x0()?;
        let risk_free = self.process.risk_free_rate().current_link()?;
        let dividend = self.process.dividend_yield().current_link()?;
        let discount = risk_free.discount(maturity, false)?;
        let dividend_discount = dividend.discount(maturity, false)?;
        let rate = -discount.ln() / maturity;
        let dividend_rate = -dividend_discount.ln() / maturity;
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
        let lower_spot = locations[0].exp();
        let upper_spot = locations[locations.size() - 1].exp();
        let lower_boundary = TimeDependentDirichletBoundary::new(BoundarySide::Lower, move |t| {
            boundary_value(
                payoff.option_type(),
                lower_spot,
                strike,
                rate,
                dividend_rate,
                maturity,
                t,
            )
        });
        let upper_boundary = TimeDependentDirichletBoundary::new(BoundarySide::Upper, move |t| {
            boundary_value(
                payoff.option_type(),
                upper_spot,
                strike,
                rate,
                dividend_rate,
                maturity,
                t,
            )
        });

        let map = shared_mut(FdmBlackScholesOp::new(
            mesher.clone() as Shared<dyn FdmMesher>,
            &self.process,
            strike,
            0,
        )?);
        let conditions: Vec<Shared<dyn crate::methods::finitedifferences::BoundaryCondition>> =
            vec![lower_boundary, upper_boundary];
        let mut solver = FdmBackwardSolver::new(
            map as SharedMut<dyn FdmLinearOpComposite>,
            conditions,
            None,
            FdmSchemeDesc::crank_nicolson(),
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

fn boundary_value(
    option_type: OptionType,
    spot: Real,
    strike: Real,
    rate: Real,
    dividend_rate: Real,
    maturity: Real,
    time: Real,
) -> Real {
    let tau = maturity - time;
    match option_type {
        OptionType::Call => {
            (spot * (-dividend_rate * tau).exp() - strike * (-rate * tau).exp()).max(0.0)
        }
        OptionType::Put => {
            (strike * (-rate * tau).exp() - spot * (-dividend_rate * tau).exp()).max(0.0)
        }
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
    use crate::instrument::Instrument;
    use crate::instruments::PlainVanillaPayoff;
    use crate::option::OptionType;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::vanilla::test_market::{market, time_to_days, today};
    use crate::shared::{SharedMut, shared_mut};

    #[test]
    fn terminal_payoff_is_evaluated_at_exp_log_grid_points() {
        let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let x = Array::from([4.0, (100.0_f64).ln(), 5.0]);
        let values = terminal_payoff(&payoff, &x);
        assert!(values[1].abs() < 1e-12);
        assert!(values[2] > values[0]);
    }

    #[test]
    fn interpolation_uses_adjacent_log_grid_points() {
        let x = Array::from([0.0, 1.0, 2.0]);
        let values = Array::from([0.0, 10.0, 20.0]);
        assert!((interpolate(&x, &values, 0.25) - 2.5).abs() < 1e-14);
    }

    #[test]
    fn fdm_call_is_close_to_the_analytic_engine() {
        let market = market();
        market.set(100.0, 0.02, 0.05, 0.20);
        let expiry = today() + time_to_days(1.0);
        let mut analytic = market.option(OptionType::Call, 100.0, expiry);
        let expected = analytic.npv().unwrap();

        let mut fdm = market.option(OptionType::Call, 100.0, expiry);
        let engine = shared_mut(FdmEuropeanEngine::new(Shared::clone(&market.process)))
            as SharedMut<dyn PricingEngine>;
        fdm.base_mut().set_pricing_engine(engine);
        let actual = fdm.npv().unwrap();

        assert!(actual.is_finite());
        assert!((actual - expected).abs() < 0.05, "{actual} vs {expected}");
    }

    #[test]
    fn fdm_put_is_close_to_the_analytic_engine() {
        let market = market();
        market.set(100.0, 0.02, 0.05, 0.20);
        let expiry = today() + time_to_days(1.0);
        let mut analytic = market.option(OptionType::Put, 100.0, expiry);
        let expected = analytic.npv().unwrap();

        let mut fdm = market.option(OptionType::Put, 100.0, expiry);
        let engine = shared_mut(FdmEuropeanEngine::new(Shared::clone(&market.process)))
            as SharedMut<dyn PricingEngine>;
        fdm.base_mut().set_pricing_engine(engine);
        let actual = fdm.npv().unwrap();

        assert!(actual.is_finite());
        assert!((actual - expected).abs() < 0.05, "{actual} vs {expected}");
    }
}
