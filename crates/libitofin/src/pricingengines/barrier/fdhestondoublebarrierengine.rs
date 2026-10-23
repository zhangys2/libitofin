//! Finite-difference Heston engine for European knock-out double barriers.
//!
//! Port of `ql/pricingengines/barrier/fdhestondoublebarrierengine.{hpp,cpp}`
//! without leverage / mixing-factor arguments. Knock-in, KIKO/KOKI, and
//! greeks are omitted (C++ `calculate` requires KnockOut + European).

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{
    DoubleBarrierArguments, DoubleBarrierType, StrikedTypePayoff, double_barrier_triggered,
};
use crate::methods::finitedifferences::meshers::{
    FdmHestonLocalVolatilityVarianceMesher, FdmMesher, FdmMesherComposite,
    fdm_black_scholes_mesher, process_helper,
};
use crate::methods::finitedifferences::solvers::{FdmHestonSolver, FdmSchemeDesc, FdmSolverDesc};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{
    FdmDirichletBoundary, FdmInnerValueCalculator, fdm_log_inner_value,
};
use crate::methods::finitedifferences::{BoundaryCondition, BoundarySide};
use crate::models::equity::HestonModel;
use crate::models::model::CalibratedModelHolder;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::require;
use crate::shared::{Shared, SharedMut, shared};
use crate::stochasticprocess::StochasticProcess;
use crate::types::Size;

type EngineBase = GenericEngine<DoubleBarrierArguments, InstrumentResults>;

/// Finite-difference Heston knock-out double-barrier engine.
pub struct FdHestonDoubleBarrierEngine {
    base: EngineBase,
    model: SharedMut<HestonModel>,
    t_grid: Size,
    x_grid: Size,
    v_grid: Size,
    damping_steps: Size,
    scheme_desc: FdmSchemeDesc,
}

impl FdHestonDoubleBarrierEngine {
    /// QuantLib defaults: `tGrid = xGrid = 100`, `vGrid = 50`, Hundsdorfer.
    pub fn new(model: SharedMut<HestonModel>) -> Self {
        Self::with_params(model, 100, 100, 50, 0, FdmSchemeDesc::hundsdorfer())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        model: SharedMut<HestonModel>,
        t_grid: Size,
        x_grid: Size,
        v_grid: Size,
        damping_steps: Size,
        scheme_desc: FdmSchemeDesc,
    ) -> Self {
        let base = EngineBase::new(
            DoubleBarrierArguments::default(),
            InstrumentResults::default(),
        );
        base.register_with(model.borrow().calibrated_model().observable());
        Self {
            base,
            model,
            t_grid,
            x_grid,
            v_grid,
            damping_steps,
            scheme_desc,
        }
    }
}

impl AsObservable for FdHestonDoubleBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdHestonDoubleBarrierEngine {
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
    #[rustfmt::skip]
    fn calculate(&mut self) -> QlResult<()> {
        let args = self.base.arguments();
        require!(args.barrier_type == Some(DoubleBarrierType::KnockOut), "only Knock-Out double barrier options are supported");
        let Some(payoff) = args.payoff else { fail!("non-striked type payoff given"); };
        require!(payoff.strike() > 0.0, "strike must be positive");
        let Some(exercise) = args.exercise.as_ref() else { fail!("no exercise given"); };
        if exercise.exercise_type() != ExerciseType::European {
            fail!("only european style option are supported");
        }
        let barrier_lo = args.barrier_lo.expect("validated");
        let barrier_hi = args.barrier_hi.expect("validated");
        let rebate = args.rebate.expect("validated");

        let process = self.model.borrow().process();
        let spot = process.s0().current_link()?.value()?;
        require!(spot > 0.0, "negative or null underlying given");
        require!(!double_barrier_triggered(spot, barrier_lo, barrier_hi), "barrier touched");

        let maturity = process.time(&exercise.last_date())?;
        let t_avg_steps = 5.max(self.t_grid / 50);
        let v_mesher = FdmHestonLocalVolatilityVarianceMesher::new(
            self.v_grid, &process, None, maturity, t_avg_steps, 0.0001, 1.0,
        )?;
        let bs_process = process_helper(
            process.s0(), process.dividend_yield(), process.risk_free_rate(), v_mesher.vola_estimate(),
        )?;
        let equity = fdm_black_scholes_mesher(
            self.x_grid, &bs_process, maturity, payoff.strike(),
            Some(barrier_lo.ln()), Some(barrier_hi.ln()), 0.0001, 1.5, None, &[], 0.0,
        )?;
        let mesher = shared(FdmMesherComposite::new(vec![equity, v_mesher.into_mesher()]));
        let mesher_dyn: Shared<dyn FdmMesher> = mesher.clone() as Shared<dyn FdmMesher>;
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            shared(payoff) as Shared<dyn Payoff>, Shared::clone(&mesher_dyn), 0,
        ));
        let conditions = shared(FdmStepConditionComposite::new(&[], Vec::new()));
        let bc_set = vec![
            shared(FdmDirichletBoundary::new(Shared::clone(&mesher_dyn), rebate, 0, BoundarySide::Lower))
                as Shared<dyn BoundaryCondition>,
            shared(FdmDirichletBoundary::new(Shared::clone(&mesher_dyn), rebate, 0, BoundarySide::Upper))
                as Shared<dyn BoundaryCondition>,
        ];
        let solver = FdmHestonSolver::new(
            process,
            FdmSolverDesc {
                mesher: mesher_dyn, bc_set, condition: conditions, calculator, maturity,
                time_steps: self.t_grid, damping_steps: self.damping_steps,
            },
            self.scheme_desc,
            1.0,
        );
        self.base.results_mut().value = Some(solver.value_at(spot, self.model.borrow().v0())?);
        Ok(())
    }
}

/// Attach the FD Heston knock-out double-barrier engine.
#[rustfmt::skip]
pub fn set_fd_heston_double_barrier_engine(
    option: &mut crate::instruments::DoubleBarrierOption,
    engine: SharedMut<FdHestonDoubleBarrierEngine>,
) {
    option.base_mut().set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{DoubleBarrierOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::models::equity::HestonModel;
    use crate::option::OptionType;
    use crate::processes::HestonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared_mut;
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::types::Real;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn heston(spot: Real, r: Real, q: Real, vol: Real) -> SharedMut<HestonModel> {
        let dc = Actual360::new();
        let flat = |rate| {
            Handle::new(shared(FlatForward::with_rate(
                today(),
                rate,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let v0 = vol * vol;
        HestonModel::new(shared(HestonProcess::new(
            flat(r),
            flat(q),
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            v0,
            1.0,
            v0,
            0.001,
            0.0,
        )))
        .unwrap()
    }

    /// `doublebarrieroption.cpp` `testEuropeanHaugValues` Heston KnockOut arm
    /// (`FdHestonDoubleBarrierEngine` 251×76×3, vol-of-vol 0.001) @ 0.025.
    #[test]
    fn haug_knock_out_subset() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let exercise = |t: Real| -> Shared<dyn Exercise> {
            shared(EuropeanExercise::new(today() + (t * 360.0).round() as i32))
        };
        #[rustfmt::skip]
        let rows: &[(Real, Real, OptionType, Real, Real, Real)] = &[
            (50.0, 150.0, OptionType::Call, 0.15, 0.25, 4.3515),
            (90.0, 110.0, OptionType::Call, 0.15, 0.25, 1.2055),
            (50.0, 150.0, OptionType::Put,  0.15, 0.25, 1.8825),
            (80.0, 120.0, OptionType::Call, 0.25, 0.50, 1.5098),
        ];
        for &(lo, hi, opt, vol, t, expected) in rows {
            let mut option = DoubleBarrierOption::new(
                DoubleBarrierType::KnockOut,
                lo,
                hi,
                0.0,
                PlainVanillaPayoff::new(opt, 100.0),
                exercise(t),
                Shared::clone(&settings),
            )
            .unwrap();
            set_fd_heston_double_barrier_engine(
                &mut option,
                shared_mut(FdHestonDoubleBarrierEngine::with_params(
                    heston(100.0, 0.1, 0.0, vol),
                    251,
                    76,
                    3,
                    0,
                    FdmSchemeDesc::hundsdorfer(),
                )),
            );
            let calculated = option.npv().unwrap();
            let diff = (calculated - expected).abs();
            assert!(
                calculated.is_finite() && diff <= 0.025,
                "{opt:?} lo={lo} hi={hi} v={vol} t={t}: {calculated} vs {expected} (diff {diff})"
            );
        }
    }

    #[test]
    fn rejects_knock_in_american_zero_spot_and_triggered() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let european: Shared<dyn Exercise> = shared(EuropeanExercise::new(today() + 90));
        let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let mut knock_in = DoubleBarrierOption::new(
            DoubleBarrierType::KnockIn,
            50.0,
            150.0,
            0.0,
            payoff,
            Shared::clone(&european),
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_heston_double_barrier_engine(
            &mut knock_in,
            shared_mut(FdHestonDoubleBarrierEngine::new(heston(
                100.0, 0.1, 0.0, 0.15,
            ))),
        );
        let err = knock_in.npv().unwrap_err().to_string();
        assert!(err.contains("only Knock-Out"), "knock-in: {err}");

        let american: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today(), today() + 90, false).unwrap());
        let mut am = DoubleBarrierOption::new(
            DoubleBarrierType::KnockOut,
            50.0,
            150.0,
            0.0,
            payoff,
            american,
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_heston_double_barrier_engine(
            &mut am,
            shared_mut(FdHestonDoubleBarrierEngine::new(heston(
                100.0, 0.1, 0.0, 0.15,
            ))),
        );
        let err = am.npv().unwrap_err().to_string();
        assert!(err.contains("only european style"), "american: {err}");

        let mut zero = DoubleBarrierOption::new(
            DoubleBarrierType::KnockOut,
            50.0,
            150.0,
            0.0,
            payoff,
            Shared::clone(&european),
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_heston_double_barrier_engine(
            &mut zero,
            shared_mut(FdHestonDoubleBarrierEngine::new(heston(
                0.0, 0.1, 0.0, 0.15,
            ))),
        );
        let err = zero.npv().unwrap_err().to_string();
        assert!(err.contains("negative or null underlying"), "zero: {err}");

        let mut touched = DoubleBarrierOption::new(
            DoubleBarrierType::KnockOut,
            50.0,
            150.0,
            0.0,
            payoff,
            european,
            settings,
        )
        .unwrap();
        set_fd_heston_double_barrier_engine(
            &mut touched,
            shared_mut(FdHestonDoubleBarrierEngine::new(heston(
                150.0, 0.1, 0.0, 0.15,
            ))),
        );
        let err = touched.npv().unwrap_err().to_string();
        assert!(err.contains("barrier touched"), "triggered: {err}");
    }
}
