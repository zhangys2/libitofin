//! Finite-difference Black–Scholes engine for European knock-out barriers.
//!
//! Port of `ql/pricingengines/barrier/fdblackscholesbarrierengine.{hpp,cpp}`
//! through the knock-out path (`calculate` steps 1–5). Knock-in barriers
//! (`cpp` step 6: vanilla + rebate option − out-value) need
//! `FdBlackScholesVanillaEngine` and `FdBlackScholesRebateEngine` and are
//! follow-up.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{BarrierArguments, BarrierType, StrikedTypePayoff};
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher,
};
use crate::methods::finitedifferences::solvers::{
    FdmBlackScholesSolver, FdmSchemeDesc, FdmSolverDesc,
};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{
    FdmDividendHandler, FdmInnerValueCalculator, fdm_log_inner_value,
};
use crate::methods::finitedifferences::{BoundaryCondition, BoundarySide, DirichletBoundary};
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::DividendSchedule;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size};

type BarrierEngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// Finite-difference Black–Scholes barrier engine (knock-out + discrete
/// dividends).
pub struct FdBlackScholesBarrierEngine {
    base: BarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    dividends: DividendSchedule,
    t_grid: Size,
    x_grid: Size,
    damping_steps: Size,
    scheme_desc: FdmSchemeDesc,
}

impl FdBlackScholesBarrierEngine {
    /// `FdBlackScholesBarrierEngine(process)` with QuantLib defaults:
    /// `tGrid = xGrid = 100`, no damping, Douglas, no dividends.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_params(process, Vec::new(), 100, 100, 0, FdmSchemeDesc::douglas())
    }

    /// `FdBlackScholesBarrierEngine(process, dividends)` with the same
    /// grid defaults.
    pub fn with_dividends(
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
    ) -> Self {
        Self::with_params(process, dividends, 100, 100, 0, FdmSchemeDesc::douglas())
    }

    /// Full constructor matching the C++ six-argument form (local-vol
    /// arguments omitted).
    pub fn with_params(
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
        t_grid: Size,
        x_grid: Size,
        damping_steps: Size,
        scheme_desc: FdmSchemeDesc,
    ) -> Self {
        let base =
            BarrierEngineBase::new(BarrierArguments::default(), InstrumentResults::default());
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
}

impl AsObservable for FdBlackScholesBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdBlackScholesBarrierEngine {
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
        let args = self.base.arguments();
        let Some(payoff) = args.payoff else {
            fail!("non-striked type payoff given");
        };
        require!(payoff.strike() > 0.0, "strike must be positive");
        let Some(exercise) = args.exercise.as_ref() else {
            fail!("no exercise given");
        };
        if exercise.exercise_type() != ExerciseType::European {
            fail!("only european style option are supported");
        }
        let barrier_type = args.barrier_type.expect("validated");
        let barrier = args.barrier.expect("validated");
        let rebate = args.rebate.expect("validated");

        match barrier_type {
            BarrierType::DownIn | BarrierType::UpIn => {
                fail!(
                    "FdBlackScholesBarrierEngine knock-in path \
                     (vanilla − out + rebate) is follow-up"
                );
            }
            BarrierType::DownOut | BarrierType::UpOut => {}
        }

        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");
        require!(!triggered(barrier_type, spot, barrier), "barrier touched");

        let maturity = self.process.time(&exercise.last_date())?;

        let (x_min, x_max) = match barrier_type {
            BarrierType::DownIn | BarrierType::DownOut => (Some(barrier.ln()), None),
            BarrierType::UpIn | BarrierType::UpOut => (None, Some(barrier.ln())),
        };

        let equity = fdm_black_scholes_mesher(
            self.x_grid,
            &self.process,
            maturity,
            payoff.strike(),
            x_min,
            x_max,
            0.0001,
            1.5,
            None,
            &self.dividends,
            0.0,
        )?;
        let mesher = shared(FdmMesherComposite::new(vec![equity]));
        let mesher_dyn: Shared<dyn FdmMesher> = mesher.clone() as Shared<dyn FdmMesher>;

        let payoff_dyn: Shared<dyn Payoff> = shared(payoff);
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            Shared::clone(&payoff_dyn),
            Shared::clone(&mesher_dyn),
            0,
        ));

        let r_ts = self.process.risk_free_rate().current_link()?;
        let ref_date = r_ts.reference_date()?;
        let day_counter = r_ts.require_day_counter()?;

        let mut stopping_times = Vec::new();
        let mut step_conditions = Vec::new();
        if !self.dividends.is_empty() {
            let dividend_condition = shared(FdmDividendHandler::new(
                &self.dividends,
                Shared::clone(&mesher_dyn),
                ref_date,
                day_counter,
                0,
            )?);
            let mut dividend_times = dividend_condition.dividend_times().to_vec();
            for t in &mut dividend_times {
                *t = (*t).min(maturity);
            }
            stopping_times.push(dividend_times);
            step_conditions.push(
                dividend_condition as Shared<dyn crate::methods::finitedifferences::StepCondition>,
            );
        }
        let conditions = shared(FdmStepConditionComposite::new(
            &stopping_times,
            step_conditions,
        ));

        let mut bc_set = Vec::new();
        match barrier_type {
            BarrierType::DownIn | BarrierType::DownOut => {
                bc_set.push(shared(DirichletBoundary::new(BoundarySide::Lower, rebate))
                    as Shared<dyn BoundaryCondition>);
            }
            BarrierType::UpIn | BarrierType::UpOut => {
                bc_set.push(shared(DirichletBoundary::new(BoundarySide::Upper, rebate))
                    as Shared<dyn BoundaryCondition>);
            }
        }

        let solver_desc = FdmSolverDesc {
            mesher: mesher_dyn,
            bc_set,
            condition: conditions,
            calculator,
            maturity,
            time_steps: self.t_grid,
            damping_steps: self.damping_steps,
        };
        let solver = FdmBlackScholesSolver::new(
            &self.process,
            payoff.strike(),
            solver_desc,
            self.scheme_desc,
        )?;
        self.base.results_mut().value = Some(solver.value_at(spot)?);
        Ok(())
    }
}

fn triggered(barrier_type: BarrierType, spot: Real, barrier: Real) -> bool {
    match barrier_type {
        BarrierType::DownIn | BarrierType::DownOut => spot < barrier,
        BarrierType::UpIn | BarrierType::UpOut => spot > barrier,
    }
}

/// Attach the FD knock-out barrier engine to an option.
pub fn set_fd_black_scholes_barrier_engine(
    option: &mut crate::instruments::BarrierOption,
    engine: SharedMut<FdBlackScholesBarrierEngine>,
) {
    option
        .base_mut()
        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflows::dividend_vector;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{BarrierOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::payoff::Payoff;
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

    fn flat_bs_process(
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

    /// `barrieroption.cpp` `testDividendBarrierOption` knock-out arms
    /// (Douglas / Crank–Nicolson / Hundsdorfer) @ `relTol = 2e-4`.
    #[test]
    fn dividend_knock_out_matches_quantlib_oracle() {
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let maturity = today + Period::new(1, TimeUnit::Years);
        let spot = 100.0;
        let strike = 105.0;
        let rebate = 5.0;
        let r = 0.05;
        let process = flat_bs_process(today, spot, 0.0, r, 0.02);
        let div_date = today + Period::new(6, TimeUnit::Months);
        let div_amount = 30.0;
        let dividends = dividend_vector(&[div_date], &[div_amount]).unwrap();
        let r_ts = process.risk_free_rate().current_link().unwrap();
        let payoff = PlainVanillaPayoff::new(OptionType::Put, strike);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));

        let expected_down_out = r_ts.discount_date(div_date, false).unwrap() * rebate;
        let expected_up_out = {
            let df_div = r_ts.discount_date(div_date, false).unwrap();
            let df_mat = r_ts.discount_date(maturity, false).unwrap();
            payoff.value((spot - div_amount * df_div) / df_mat) * df_mat
        };

        let cases = [
            (BarrierType::DownOut, 80.0, expected_down_out),
            (BarrierType::UpOut, 120.0, expected_up_out),
        ];
        let schemes = [
            FdmSchemeDesc::douglas(),
            FdmSchemeDesc::crank_nicolson(),
            FdmSchemeDesc::hundsdorfer(),
        ];
        let rel_tol = 2e-4;

        for (barrier_type, barrier, expected) in cases {
            for scheme in schemes {
                let mut option = BarrierOption::with_rebate(
                    barrier_type,
                    barrier,
                    rebate,
                    payoff,
                    Shared::clone(&exercise),
                    Shared::clone(&settings),
                )
                .unwrap();
                let engine = shared_mut(FdBlackScholesBarrierEngine::with_params(
                    Shared::clone(&process),
                    dividends.clone(),
                    100,
                    100,
                    0,
                    scheme,
                ));
                set_fd_black_scholes_barrier_engine(&mut option, engine);
                let calculated = option.npv().unwrap();
                let diff = (calculated - expected).abs();
                let tol = rel_tol * expected;
                eprintln!(
                    "{barrier_type:?} {:?} H={barrier}: calculated={calculated:.8} \
                     expected={expected:.8} diff={diff:.2e} tol={tol:.2e}",
                    scheme.scheme_type
                );
                assert!(
                    calculated.is_finite() && diff <= tol,
                    "{barrier_type:?} {:?}: {calculated} vs {expected} \
                     (diff {diff}, tol {tol})",
                    scheme.scheme_type
                );
            }
        }
    }

    #[test]
    fn knock_in_is_rejected_until_the_vanilla_decomposition_lands() {
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let process = flat_bs_process(today, 100.0, 0.0, 0.05, 0.20);
        let mut option = BarrierOption::new(
            BarrierType::DownIn,
            90.0,
            PlainVanillaPayoff::new(OptionType::Put, 105.0),
            shared(EuropeanExercise::new(
                today + Period::new(1, TimeUnit::Years),
            )),
            settings,
        )
        .unwrap();
        set_fd_black_scholes_barrier_engine(
            &mut option,
            shared_mut(FdBlackScholesBarrierEngine::new(process)),
        );
        let err = option.npv().unwrap_err();
        assert!(err.message().contains("knock-in"));
    }
}
