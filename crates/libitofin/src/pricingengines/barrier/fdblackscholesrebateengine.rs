//! Finite-difference helper that prices the cash rebate of a knock-in barrier.
//!
//! Port of `ql/pricingengines/barrier/fdblackscholesrebateengine.{hpp,cpp}`.
//! The terminal payoff is a cash-or-nothing call struck at 0 that pays the
//! rebate; Dirichlet conditions pin the barrier edge to the same cash amount.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::InstrumentResults;
use crate::instruments::{BarrierArguments, CashOrNothingPayoff, StrikedTypePayoff};
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher,
};
use crate::methods::finitedifferences::solvers::{
    FdmBlackScholesSolver, FdmSchemeDesc, FdmSolverDesc,
};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{FdmInnerValueCalculator, fdm_log_inner_value};
use crate::methods::finitedifferences::{BoundaryCondition, BoundarySide, DirichletBoundary};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::DividendSchedule;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size};
use crate::utilities::null::Null;

use crate::instruments::BarrierType;

type BarrierEngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// Finite-difference Black–Scholes rebate engine.
pub struct FdBlackScholesRebateEngine {
    base: BarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    dividends: DividendSchedule,
    t_grid: Size,
    x_grid: Size,
    damping_steps: Size,
    scheme_desc: FdmSchemeDesc,
    local_vol: bool,
    illegal_local_vol_overwrite: Real,
}

impl FdBlackScholesRebateEngine {
    /// `FdBlackScholesRebateEngine(process)` with QuantLib defaults.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_params(process, Vec::new(), 100, 100, 0, FdmSchemeDesc::douglas())
    }

    /// Full constructor matching the C++ six-argument form (local-vol off).
    pub fn with_params(
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
        t_grid: Size,
        x_grid: Size,
        damping_steps: Size,
        scheme_desc: FdmSchemeDesc,
    ) -> Self {
        Self::with_local_vol(
            process,
            dividends,
            t_grid,
            x_grid,
            damping_steps,
            scheme_desc,
            false,
            -Real::null(),
        )
    }

    /// As [`with_params`](Self::with_params), with the C++ `localVol` /
    /// `illegalLocalVolOverwrite` arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn with_local_vol(
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
        t_grid: Size,
        x_grid: Size,
        damping_steps: Size,
        scheme_desc: FdmSchemeDesc,
        local_vol: bool,
        illegal_local_vol_overwrite: Real,
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
            local_vol,
            illegal_local_vol_overwrite,
        }
    }

    /// Fills the arguments and returns the NPV.
    pub fn price(
        &mut self,
        barrier_type: BarrierType,
        barrier: Real,
        rebate: Real,
        payoff: crate::instruments::PlainVanillaPayoff,
        exercise: Shared<dyn crate::exercise::Exercise>,
    ) -> QlResult<Real> {
        {
            let args = self.base.arguments_mut();
            args.barrier_type = Some(barrier_type);
            args.barrier = Some(barrier);
            args.rebate = Some(rebate);
            args.payoff = Some(payoff);
            args.exercise = Some(exercise);
        }
        self.calculate()?;
        match self.base.results().value {
            Some(value) => Ok(value),
            None => fail!("no results returned"),
        }
    }
}

impl AsObservable for FdBlackScholesRebateEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdBlackScholesRebateEngine {
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

        let maturity = self.process.time(&exercise.last_date())?;
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");

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

        let rebate_payoff: Shared<dyn Payoff> =
            shared(CashOrNothingPayoff::new(OptionType::Call, 0.0, rebate));
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            rebate_payoff,
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
        let solver = FdmBlackScholesSolver::with_local_vol(
            &self.process,
            payoff.strike(),
            solver_desc,
            self.scheme_desc,
            self.local_vol,
            self.illegal_local_vol_overwrite,
        )?;
        self.base.results_mut().value = Some(solver.value_at(spot)?);
        Ok(())
    }
}
