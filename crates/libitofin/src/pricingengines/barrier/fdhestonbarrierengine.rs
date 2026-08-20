//! Finite-difference Heston engine for European barrier options.
//!
//! Port of `ql/pricingengines/barrier/fdhestonbarrierengine.{hpp,cpp}`.
//! Knock-out uses steps 1–5 (mesher pinned at `ln(H)`, 2-D Dirichlet rebate,
//! dividend handler). Knock-in is step 6: vanilla + rebate − out-value.
//! Leverage / mixing factor ≠ 1 are omitted.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{BarrierArguments, BarrierType, StrikedTypePayoff};
use crate::pricingengines::vanilla::FdHestonVanillaEngine;

use super::fdhestonrebateengine::FdHestonRebateEngine;
use crate::methods::finitedifferences::meshers::{
    FdmHestonVarianceMesher, FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher,
    process_helper,
};
use crate::methods::finitedifferences::solvers::{FdmHestonSolver, FdmSchemeDesc, FdmSolverDesc};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{
    FdmDirichletBoundary, FdmDividendHandler, FdmInnerValueCalculator, fdm_log_inner_value,
};
use crate::methods::finitedifferences::{BoundaryCondition, BoundarySide};
use crate::models::equity::HestonModel;
use crate::models::model::CalibratedModelHolder;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::DividendSchedule;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared};
use crate::stochasticprocess::StochasticProcess;
use crate::types::Size;

use super::fdblackscholesbarrierengine::triggered;

type BarrierEngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// Finite-difference Heston barrier engine (knock-out and knock-in).
pub struct FdHestonBarrierEngine {
    base: BarrierEngineBase,
    model: SharedMut<HestonModel>,
    dividends: DividendSchedule,
    t_grid: Size,
    x_grid: Size,
    v_grid: Size,
    damping_steps: Size,
    scheme_desc: FdmSchemeDesc,
}

impl FdHestonBarrierEngine {
    /// `FdHestonBarrierEngine(model)` with QuantLib defaults:
    /// `tGrid = 100`, `xGrid = 100`, `vGrid = 50`, no damping, Hundsdorfer.
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
}

impl AsObservable for FdHestonBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdHestonBarrierEngine {
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
        let is_in = matches!(barrier_type, BarrierType::DownIn | BarrierType::UpIn);
        let exercise = Shared::clone(exercise);

        let process = self.model.borrow().process();
        let spot = process.s0().current_link()?.value()?;
        require!(spot > 0.0, "negative or null underlying given");
        require!(!triggered(barrier_type, spot, barrier), "barrier touched");

        let maturity = process.time(&exercise.last_date())?;

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

        let payoff_dyn: Shared<dyn Payoff> = shared(payoff);
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            Shared::clone(&payoff_dyn),
            Shared::clone(&mesher_dyn),
            0,
        ));

        let r_ts = process.risk_free_rate().current_link()?;
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
        let mut value = solver.value_at(spot, v0)?;
        if is_in {
            let vanilla = {
                let mut engine = FdHestonVanillaEngine::with_params(
                    SharedMut::clone(&self.model),
                    self.dividends.clone(),
                    self.t_grid,
                    self.x_grid,
                    self.v_grid,
                    self.damping_steps,
                    self.scheme_desc,
                );
                engine.price(
                    shared(payoff) as Shared<dyn StrikedTypePayoff>,
                    Shared::clone(&exercise),
                )?
            };
            let rebate_x_grid = 20.max(self.x_grid / 4);
            let rebate_v_grid = 10.max(self.v_grid / 4);
            let rebate_damping = if self.damping_steps > 0 {
                1.min(self.damping_steps / 2)
            } else {
                0
            };
            let rebate_npv = {
                let mut engine = FdHestonRebateEngine::with_params(
                    SharedMut::clone(&self.model),
                    self.dividends.clone(),
                    self.t_grid,
                    rebate_x_grid,
                    rebate_v_grid,
                    rebate_damping,
                    self.scheme_desc,
                );
                engine.price(barrier_type, barrier, rebate, payoff, exercise)?
            };
            value = vanilla + rebate_npv - value;
        }
        self.base.results_mut().value = Some(value);
        Ok(())
    }
}

/// Attach the FD Heston knock-out barrier engine to an option.
pub fn set_fd_heston_barrier_engine(
    option: &mut crate::instruments::BarrierOption,
    engine: SharedMut<FdHestonBarrierEngine>,
) {
    option
        .base_mut()
        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflows::dividend_vector;
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{BarrierOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::math::interpolations::linear::Linear;
    use crate::option::OptionType;
    use crate::payoff::Payoff;
    use crate::processes::HestonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared_mut;
    use crate::termstructures::yields::{FlatForward, ZeroCurve};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;
    use crate::types::Real;

    #[allow(clippy::too_many_arguments)]
    fn heston_model(
        spot: Real,
        v0: Real,
        kappa: Real,
        theta: Real,
        sigma: Real,
        rho: Real,
        r_ts: Handle<dyn YieldTermStructure>,
        q_ts: Handle<dyn YieldTermStructure>,
    ) -> SharedMut<HestonModel> {
        let process = shared(HestonProcess::new(
            r_ts,
            q_ts,
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            v0,
            kappa,
            theta,
            sigma,
            rho,
        ));
        HestonModel::new(process).unwrap()
    }

    /// `barrieroption.cpp` `testLocalVolAndHestonComparison` Heston arm:
    /// DownOut put, expected NPV 111.5 at 1% relative.
    #[test]
    fn local_vol_and_heston_comparison_heston_arm() {
        let settlement = Date::new(5, Month::July, 2002);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(settlement);
        let dc = Actual365Fixed::new();

        let t = [13, 41, 75, 165, 256, 345, 524, 703];
        let r = [
            0.0357, 0.0349, 0.0341, 0.0355, 0.0359, 0.0368, 0.0386, 0.0401,
        ];
        let mut dates = vec![settlement];
        let mut rates = vec![0.0357];
        for i in 0..8 {
            dates.push(settlement + Period::new(t[i], TimeUnit::Days));
            rates.push(r[i]);
        }
        let r_ts = Handle::new(
            shared(ZeroCurve::new(dates, rates, dc.clone(), Linear).unwrap())
                as Shared<dyn YieldTermStructure>,
        );
        let q_ts = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.0,
            dc,
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);

        let spot = 4500.0;
        let model = heston_model(
            spot, 0.195662, 5.6628, 0.0745911, 1.1619, -0.511493, r_ts, q_ts,
        );
        let expiry = settlement + Period::new(20, TimeUnit::Months);
        let payoff = PlainVanillaPayoff::new(OptionType::Put, spot);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));
        let mut option = BarrierOption::with_rebate(
            BarrierType::DownOut,
            3000.0,
            100.0,
            payoff,
            exercise,
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_heston_barrier_engine(
            &mut option,
            shared_mut(FdHestonBarrierEngine::with_params(
                model,
                Vec::new(),
                100,
                400,
                50,
                0,
                FdmSchemeDesc::hundsdorfer(),
            )),
        );
        let calculated = option.npv().unwrap();
        let expected = 111.5;
        let diff = (calculated - expected).abs();
        let tol = 0.01 * expected;
        eprintln!(
            "Heston DownOut: calculated={calculated:.8} expected={expected} diff={diff:.4} tol={tol:.4}"
        );
        assert!(
            calculated.is_finite() && diff <= tol,
            "Heston barrier: {calculated} vs {expected} (diff {diff}, tol {tol})"
        );
    }

    #[test]
    fn heston_barrier_rejects_zero_spot_triggered_and_american() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let dc = Actual365Fixed::new();
        let flat = |rate| {
            Handle::new(shared(FlatForward::with_rate(
                today,
                rate,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let expiry = today + Period::new(6, TimeUnit::Months);
        let payoff = PlainVanillaPayoff::new(OptionType::Put, 100.0);
        let european: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));

        let mut knock_in = BarrierOption::with_rebate(
            BarrierType::DownIn,
            90.0,
            5.0,
            payoff,
            Shared::clone(&european),
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_heston_barrier_engine(
            &mut knock_in,
            shared_mut(FdHestonBarrierEngine::with_params(
                heston_model(100.0, 0.04, 1.0, 0.04, 0.2, -0.5, flat(0.05), flat(0.0)),
                Vec::new(),
                20,
                40,
                10,
                0,
                FdmSchemeDesc::hundsdorfer(),
            )),
        );
        let knock_in_npv = knock_in.npv().unwrap();
        assert!(
            knock_in_npv.is_finite() && knock_in_npv > 0.0,
            "knock-in should price, got {knock_in_npv}"
        );

        let mut zero_spot = BarrierOption::with_rebate(
            BarrierType::DownOut,
            90.0,
            5.0,
            payoff,
            Shared::clone(&european),
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_heston_barrier_engine(
            &mut zero_spot,
            shared_mut(FdHestonBarrierEngine::new(heston_model(
                0.0,
                0.04,
                1.0,
                0.04,
                0.2,
                -0.5,
                flat(0.05),
                flat(0.0),
            ))),
        );
        let err = zero_spot.npv().unwrap_err().to_string();
        assert!(
            err.contains("negative or null underlying"),
            "zero spot: {err}"
        );

        let mut already = BarrierOption::with_rebate(
            BarrierType::DownOut,
            101.0,
            5.0,
            payoff,
            Shared::clone(&european),
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_heston_barrier_engine(
            &mut already,
            shared_mut(FdHestonBarrierEngine::new(heston_model(
                100.0,
                0.04,
                1.0,
                0.04,
                0.2,
                -0.5,
                flat(0.05),
                flat(0.0),
            ))),
        );
        let err = already.npv().unwrap_err().to_string();
        assert!(err.contains("barrier touched"), "triggered: {err}");

        let american: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, expiry, false).unwrap());
        let mut american_opt = BarrierOption::with_rebate(
            BarrierType::DownOut,
            90.0,
            5.0,
            payoff,
            american,
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_heston_barrier_engine(
            &mut american_opt,
            shared_mut(FdHestonBarrierEngine::new(heston_model(
                100.0,
                0.04,
                1.0,
                0.04,
                0.2,
                -0.5,
                flat(0.05),
                flat(0.0),
            ))),
        );
        let err = american_opt.npv().unwrap_err().to_string();
        assert!(
            err.contains("only european style option are supported"),
            "american: {err}"
        );
    }

    fn nearly_bs_heston(today: Date, spot: Real, r: Real, vol: Real) -> SharedMut<HestonModel> {
        let dc = Actual365Fixed::new();
        let flat = |rate| {
            Handle::new(shared(FlatForward::with_rate(
                today,
                rate,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        heston_model(
            spot,
            vol * vol,
            1.0,
            vol * vol,
            0.005,
            0.0,
            flat(r),
            flat(0.0),
        )
    }

    /// `barrieroption.cpp` `testDividendBarrierOption` Heston arm:
    /// `FdHestonBarrierEngine` 50×101×3, vol-of-vol 0.005, `relTol = 2e-4`.
    #[test]
    fn dividend_barrier_matches_quantlib_heston_oracle() {
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let maturity = today + Period::new(1, TimeUnit::Years);
        let spot = 100.0;
        let strike = 105.0;
        let rebate = 5.0;
        let r = 0.05;
        let vol = 0.02;
        let model = nearly_bs_heston(today, spot, r, vol);
        let div_date = today + Period::new(6, TimeUnit::Months);
        let div_amount = 30.0;
        let dividends = dividend_vector(&[div_date], &[div_amount]).unwrap();
        let r_ts = model
            .borrow()
            .process()
            .risk_free_rate()
            .current_link()
            .unwrap();
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
            (BarrierType::DownIn, 80.0, 29.154),
            (BarrierType::UpIn, 120.0, 4.765),
        ];
        let rel_tol = 2e-4;

        for (barrier_type, barrier, expected) in cases {
            let mut option = BarrierOption::with_rebate(
                barrier_type,
                barrier,
                rebate,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_fd_heston_barrier_engine(
                &mut option,
                shared_mut(FdHestonBarrierEngine::with_params(
                    SharedMut::clone(&model),
                    dividends.clone(),
                    50,
                    101,
                    3,
                    0,
                    FdmSchemeDesc::hundsdorfer(),
                )),
            );
            let calculated = option.npv().unwrap();
            let diff = (calculated - expected).abs();
            let tol = rel_tol * expected;
            eprintln!(
                "Heston {barrier_type:?} H={barrier}: calculated={calculated:.8} \
                 expected={expected:.8} diff={diff:.2e} tol={tol:.2e}"
            );
            assert!(
                calculated.is_finite() && diff <= tol,
                "Heston {barrier_type:?} H={barrier}: {calculated} vs {expected} \
                 (diff {diff}, tol {tol})"
            );
        }
    }

    /// `barrieroption.cpp` `testDividendBarrierOptionWithDividendsPastMaturity`
    /// Heston arm @ 1e-12.
    #[test]
    fn past_maturity_dividends_do_not_change_heston_npv() {
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let maturity = today + Period::new(1, TimeUnit::Years);
        let model = nearly_bs_heston(today, 100.0, 0.05, 0.02);
        let dividends =
            dividend_vector(&[today + Period::new(18, TimeUnit::Months)], &[30.0]).unwrap();
        let payoff = PlainVanillaPayoff::new(OptionType::Put, 105.0);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));
        let cases = [(BarrierType::DownOut, 90.0), (BarrierType::UpOut, 110.0)];
        for (barrier_type, barrier) in cases {
            let mut without = BarrierOption::with_rebate(
                barrier_type,
                barrier,
                5.0,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_fd_heston_barrier_engine(
                &mut without,
                shared_mut(FdHestonBarrierEngine::with_params(
                    SharedMut::clone(&model),
                    Vec::new(),
                    50,
                    101,
                    3,
                    0,
                    FdmSchemeDesc::hundsdorfer(),
                )),
            );
            let without_npv = without.npv().unwrap();

            let mut with_div = BarrierOption::with_rebate(
                barrier_type,
                barrier,
                5.0,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_fd_heston_barrier_engine(
                &mut with_div,
                shared_mut(FdHestonBarrierEngine::with_params(
                    SharedMut::clone(&model),
                    dividends.clone(),
                    50,
                    101,
                    3,
                    0,
                    FdmSchemeDesc::hundsdorfer(),
                )),
            );
            let with_npv = with_div.npv().unwrap();
            let diff = (with_npv - without_npv).abs();
            eprintln!(
                "Heston {barrier_type:?} H={barrier}: without={without_npv:.12} \
                 with={with_npv:.12} diff={diff:.2e}"
            );
            assert!(
                diff <= 1e-12,
                "Heston {barrier_type:?} H={barrier}: {with_npv} vs {without_npv} (diff {diff})"
            );
        }
    }
}
