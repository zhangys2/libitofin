//! Finite-difference helper that prices the cash rebate of a Heston knock-in.
//!
//! Port of `ql/pricingengines/barrier/fdhestonrebateengine.{hpp,cpp}` without
//! leverage / mixing-factor arguments. The terminal payoff is a cash-or-nothing
//! call struck at 0 that pays the rebate; 2-D Dirichlet conditions pin the
//! barrier face to the same cash amount.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::InstrumentResults;
use crate::instruments::{BarrierArguments, CashOrNothingPayoff};
use crate::methods::finitedifferences::meshers::{
    FdmHestonVarianceMesher, FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher,
    process_helper,
};
use crate::methods::finitedifferences::solvers::{FdmHestonSolver, FdmSchemeDesc, FdmSolverDesc};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{
    FdmDirichletBoundary, FdmInnerValueCalculator, fdm_log_inner_value,
};
use crate::methods::finitedifferences::{BoundaryCondition, BoundarySide};
use crate::models::equity::HestonModel;
use crate::models::model::CalibratedModelHolder;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::DividendSchedule;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared};
use crate::stochasticprocess::StochasticProcess;
use crate::types::{Real, Size};

use crate::instruments::BarrierType;

type BarrierEngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// Finite-difference Heston rebate engine.
pub struct FdHestonRebateEngine {
    base: BarrierEngineBase,
    model: SharedMut<HestonModel>,
    dividends: DividendSchedule,
    t_grid: Size,
    x_grid: Size,
    v_grid: Size,
    damping_steps: Size,
    scheme_desc: FdmSchemeDesc,
}

impl FdHestonRebateEngine {
    /// `FdHestonRebateEngine(model)` with QuantLib defaults:
    /// `tGrid = xGrid = 100`, `vGrid = 50`, no damping, Hundsdorfer.
    pub fn new(model: SharedMut<HestonModel>) -> Self {
        Self::with_params(
            model,
            Vec::new(),
            100,
            100,
            50,
            0,
            FdmSchemeDesc::hundsdorfer(),
        )
    }

    /// Full constructor matching the C++ dividends form, without leverage /
    /// mixing-factor arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        model: SharedMut<HestonModel>,
        dividends: DividendSchedule,
        t_grid: Size,
        x_grid: Size,
        v_grid: Size,
        damping_steps: Size,
        scheme_desc: FdmSchemeDesc,
    ) -> Self {
        let base =
            BarrierEngineBase::new(BarrierArguments::default(), InstrumentResults::default());
        base.register_with(model.borrow().calibrated_model().observable());
        Self {
            base,
            model,
            dividends,
            t_grid,
            x_grid,
            v_grid,
            damping_steps,
            scheme_desc,
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

impl AsObservable for FdHestonRebateEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdHestonRebateEngine {
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

        let process = self.model.borrow().process();
        let maturity = process.time(&exercise.last_date())?;
        let spot = process.s0().current_link()?.value()?;
        require!(spot > 0.0, "negative or null underlying given");

        let t_avg_steps = 5.max(self.t_grid / 50);
        let v_mesher = FdmHestonVarianceMesher::new(
            self.v_grid,
            &process,
            maturity,
            t_avg_steps,
            0.0001,
            1.0,
        )?;

        let (x_min, x_max) = match barrier_type {
            BarrierType::DownIn | BarrierType::DownOut => (Some(barrier.ln()), None),
            BarrierType::UpIn | BarrierType::UpOut => (None, Some(barrier.ln())),
        };

        let bs_process = process_helper(
            process.s0(),
            process.risk_free_rate(),
            process.dividend_yield(),
            v_mesher.vola_estimate(),
        )?;
        let equity = fdm_black_scholes_mesher(
            self.x_grid,
            &bs_process,
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
        let mesher = shared(FdmMesherComposite::new(vec![
            equity,
            v_mesher.into_mesher(),
        ]));
        let mesher_dyn: Shared<dyn FdmMesher> = mesher.clone() as Shared<dyn FdmMesher>;

        let rebate_payoff: Shared<dyn Payoff> =
            shared(CashOrNothingPayoff::new(OptionType::Call, 0.0, rebate));
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            rebate_payoff,
            Shared::clone(&mesher_dyn),
            0,
        ));

        let r_ts = process.risk_free_rate().current_link()?;
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
                bc_set.push(shared(FdmDirichletBoundary::new(
                    Shared::clone(&mesher_dyn),
                    rebate,
                    0,
                    BoundarySide::Lower,
                )) as Shared<dyn BoundaryCondition>);
            }
            BarrierType::UpIn | BarrierType::UpOut => {
                bc_set.push(shared(FdmDirichletBoundary::new(
                    Shared::clone(&mesher_dyn),
                    rebate,
                    0,
                    BoundarySide::Upper,
                )) as Shared<dyn BoundaryCondition>);
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
        let solver = FdmHestonSolver::new(process, solver_desc, self.scheme_desc, 1.0);
        let v0 = self.model.borrow().v0();
        self.base.results_mut().value = Some(solver.value_at(spot, v0)?);
        Ok(())
    }
}
