//! Finite-difference Black–Scholes engine for European vanillas with cash
//! dividends.
//!
//! Port of `ql/pricingengines/vanilla/fdblackscholesvanillaengine.{hpp,cpp}`
//! on the Spot cash-dividend model (the default). Escrowed dividends, local
//! vol, and quanto are follow-up.

use crate::errors::QlResult;
use crate::exercise::{Exercise, ExerciseType};
use crate::fail;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
    StrikedTypePayoff,
};
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher,
};
use crate::methods::finitedifferences::solvers::{
    FdmBlackScholesSolver, FdmSchemeDesc, FdmSolverDesc,
};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{FdmInnerValueCalculator, fdm_log_inner_value};
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::DividendSchedule;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size};

/// Finite-difference Black–Scholes vanilla engine (European + Spot dividends).
pub struct FdBlackScholesVanillaEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    dividends: DividendSchedule,
    t_grid: Size,
    x_grid: Size,
    damping_steps: Size,
    scheme_desc: FdmSchemeDesc,
}

impl FdBlackScholesVanillaEngine {
    /// `FdBlackScholesVanillaEngine(process)` with QuantLib defaults:
    /// `tGrid = xGrid = 100`, no damping, Douglas, no dividends.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_params(process, Vec::new(), 100, 100, 0, FdmSchemeDesc::douglas())
    }

    /// `FdBlackScholesVanillaEngine(process, dividends)` with the same grid
    /// defaults.
    pub fn with_dividends(
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
    ) -> Self {
        Self::with_params(process, dividends, 100, 100, 0, FdmSchemeDesc::douglas())
    }

    /// Full constructor matching the C++ six-argument form (local-vol /
    /// escrowed / quanto omitted).
    pub fn with_params(
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
        t_grid: Size,
        x_grid: Size,
        damping_steps: Size,
        scheme_desc: FdmSchemeDesc,
    ) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            dividends,
            t_grid,
            x_grid,
            damping_steps,
            scheme_desc,
        }
    }

    /// Fills the arguments and returns the NPV.
    pub fn price(
        &mut self,
        payoff: Shared<dyn StrikedTypePayoff>,
        exercise: Shared<dyn Exercise>,
    ) -> QlResult<Real> {
        {
            let args = self.base.arguments_mut();
            args.payoff = Some(payoff);
            args.exercise = Some(exercise);
        }
        self.calculate()?;
        match self.base.results().instrument.value {
            Some(value) => Ok(value),
            None => fail!("no results returned"),
        }
    }
}

impl AsObservable for FdBlackScholesVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdBlackScholesVanillaEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn calculate(&mut self) -> QlResult<()> {
        let arguments = self.base.arguments();
        let Some(exercise) = arguments.exercise.as_ref() else {
            fail!("no exercise given");
        };
        if exercise.exercise_type() != ExerciseType::European {
            fail!("only european style option are supported");
        }
        let Some(payoff) = arguments.payoff.as_ref() else {
            fail!("no payoff given");
        };
        let strike = payoff.strike();
        require!(strike > 0.0, "strike must be positive");

        let maturity = self.process.time(&exercise.last_date())?;
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");

        let equity = fdm_black_scholes_mesher(
            self.x_grid,
            &self.process,
            maturity,
            strike,
            None,
            None,
            0.0001,
            1.5,
            Some((strike, 0.1)),
            &self.dividends,
            0.0,
        )?;
        let mesher = shared(FdmMesherComposite::new(vec![equity]));
        let mesher_dyn: Shared<dyn FdmMesher> = mesher.clone() as Shared<dyn FdmMesher>;

        let payoff_dyn: Shared<dyn Payoff> = Shared::clone(payoff) as Shared<dyn Payoff>;
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            payoff_dyn,
            Shared::clone(&mesher_dyn),
            0,
        ));

        let r_ts = self.process.risk_free_rate().current_link()?;
        let conditions = FdmStepConditionComposite::vanilla_composite(
            &self.dividends,
            &**exercise,
            Shared::clone(&mesher_dyn),
            Shared::clone(&calculator),
            r_ts.reference_date()?,
            r_ts.require_day_counter()?,
        )?;

        let solver_desc = FdmSolverDesc {
            mesher: mesher_dyn,
            bc_set: Vec::new(),
            condition: conditions,
            calculator,
            maturity,
            time_steps: self.t_grid,
            damping_steps: self.damping_steps,
        };
        let solver =
            FdmBlackScholesSolver::new(&self.process, strike, solver_desc, self.scheme_desc)?;
        let results = self.base.results_mut();
        results.instrument.value = Some(solver.value_at(spot)?);
        results.greeks = Greeks::default();
        results.more_greeks = MoreGreeks::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::PlainVanillaPayoff;
    use crate::instruments::VanillaOption;
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared_mut;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    fn process(
        today: Date,
        spot: Real,
        q: Real,
        r: Real,
        vol: Real,
    ) -> Shared<GeneralizedBlackScholesProcess> {
        let dc = Actual365Fixed::new();
        shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            Handle::new(shared(FlatForward::with_rate(
                today,
                q,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(FlatForward::with_rate(
                today,
                r,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(BlackConstantVol::new(today, None, vol, dc))
                as Shared<dyn BlackVolTermStructure>),
        ))
    }

    #[test]
    fn no_dividend_european_is_close_to_analytic() {
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let process = process(today, 100.0, 0.0, 0.05, 0.20);
        let expiry = today + Period::new(1, TimeUnit::Years);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, 105.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));

        let mut analytic =
            VanillaOption::new(Shared::clone(&payoff), Shared::clone(&exercise), settings);
        analytic
            .base_mut()
            .set_pricing_engine(
                shared_mut(AnalyticEuropeanEngine::new(Shared::clone(&process)))
                    as crate::shared::SharedMut<dyn PricingEngine>,
            );
        let expected = analytic.npv().unwrap();

        let mut fd = FdBlackScholesVanillaEngine::with_params(
            process,
            Vec::new(),
            100,
            100,
            0,
            FdmSchemeDesc::douglas(),
        );
        let calculated = fd.price(payoff, exercise).unwrap();
        assert!(
            (calculated - expected).abs() < 0.05,
            "{calculated} vs {expected}"
        );
    }
}
