//! Finite-difference Black–Scholes engine for European barrier options.
//!
//! Port of `ql/pricingengines/barrier/fdblackscholesbarrierengine.{hpp,cpp}`.
//! Knock-out uses steps 1–5 (mesher pinned at `ln(H)`, Dirichlet rebate,
//! dividend handler). Knock-in is step 6: vanilla + rebate − out-value.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{BarrierArguments, BarrierType, StrikedTypePayoff};
use crate::pricingengines::vanilla::FdBlackScholesVanillaEngine;

use super::fdblackscholesrebateengine::FdBlackScholesRebateEngine;
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
use crate::utilities::null::Null;

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
    local_vol: bool,
    illegal_local_vol_overwrite: Real,
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
        let exercise = Shared::clone(exercise);
        let is_in = matches!(barrier_type, BarrierType::DownIn | BarrierType::UpIn);
        if is_in && self.local_vol {
            fail!("knock-in local-vol barrier engine not yet ported");
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
        let solver = FdmBlackScholesSolver::with_local_vol(
            &self.process,
            payoff.strike(),
            solver_desc,
            self.scheme_desc,
            self.local_vol,
            self.illegal_local_vol_overwrite,
        )?;
        let mut value = solver.value_at(spot)?;
        if is_in {
            let vanilla = {
                let mut engine = FdBlackScholesVanillaEngine::with_params(
                    Shared::clone(&self.process),
                    self.dividends.clone(),
                    self.t_grid,
                    self.x_grid,
                    0,
                    self.scheme_desc,
                );
                engine.price(
                    shared(payoff) as Shared<dyn StrikedTypePayoff>,
                    Shared::clone(&exercise),
                )?
            };
            let rebate_x_grid = self.x_grid / 5;
            let rebate_x_grid = rebate_x_grid.max(50);
            let rebate_damping = if self.damping_steps > 0 {
                1.min(self.damping_steps / 2)
            } else {
                0
            };
            let rebate_npv = {
                let mut engine = FdBlackScholesRebateEngine::with_params(
                    Shared::clone(&self.process),
                    self.dividends.clone(),
                    self.t_grid,
                    rebate_x_grid,
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

pub(crate) fn triggered(barrier_type: BarrierType, spot: Real, barrier: Real) -> bool {
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
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{BarrierOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::math::interpolations::bicubic::Bicubic;
    use crate::math::interpolations::linear::Linear;
    use crate::math::matrix::Matrix;
    use crate::option::OptionType;
    use crate::payoff::Payoff;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared_mut;
    use crate::termstructures::volatility::{
        BlackConstantVol, BlackVarianceSurface, BlackVolTermStructure,
    };
    use crate::termstructures::yields::{FlatForward, ZeroCurve};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::target::Target;
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

    /// `barrieroption.cpp` `testDividendBarrierOption`
    /// (Douglas / Crank–Nicolson / Hundsdorfer / Craig–Sneyd /
    /// Modified Craig–Sneyd / Method of Lines / TrBDF2) @ `relTol = 2e-4`.
    #[test]
    fn dividend_barrier_matches_quantlib_oracle() {
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
            (BarrierType::DownIn, 80.0, 29.154),
            (BarrierType::UpIn, 120.0, 4.765),
        ];
        let schemes = [
            FdmSchemeDesc::douglas(),
            FdmSchemeDesc::crank_nicolson(),
            FdmSchemeDesc::hundsdorfer(),
            FdmSchemeDesc::craig_sneyd(),
            FdmSchemeDesc::modified_craig_sneyd(),
            FdmSchemeDesc::method_of_lines(),
            FdmSchemeDesc::tr_bdf2(),
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

    /// `barrieroption.cpp` `testDividendBarrierOptionWithDividendsPastMaturity`
    /// (FD Black–Scholes arm; Heston follow-up).
    #[test]
    fn past_maturity_dividends_do_not_change_npv() {
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let maturity = today + Period::new(1, TimeUnit::Years);
        let process = flat_bs_process(today, 100.0, 0.0, 0.05, 0.02);
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
            set_fd_black_scholes_barrier_engine(
                &mut without,
                shared_mut(FdBlackScholesBarrierEngine::new(Shared::clone(&process))),
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
            set_fd_black_scholes_barrier_engine(
                &mut with_div,
                shared_mut(FdBlackScholesBarrierEngine::with_dividends(
                    Shared::clone(&process),
                    dividends.clone(),
                )),
            );
            let with_npv = with_div.npv().unwrap();
            let diff = (with_npv - without_npv).abs();
            eprintln!(
                "{barrier_type:?} H={barrier}: without={without_npv:.12} \
                 with={with_npv:.12} diff={diff:.2e}"
            );
            assert!(
                diff <= 1e-12,
                "{barrier_type:?} H={barrier}: {with_npv} vs {without_npv} (diff {diff})"
            );
        }
    }

    /// `barrieroption.cpp` `testHaugValues` reject paths for the FD engine
    /// (zero spot, already-triggered barrier, American exercise).
    #[test]
    fn fd_barrier_rejects_zero_spot_triggered_and_american() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let expiry = today + Period::new(6, TimeUnit::Months);
        let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let european: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));

        let mut zero_spot = BarrierOption::with_rebate(
            BarrierType::DownOut,
            95.0,
            3.0,
            payoff,
            Shared::clone(&european),
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_black_scholes_barrier_engine(
            &mut zero_spot,
            shared_mut(FdBlackScholesBarrierEngine::new(flat_bs_process(
                today, 0.0, 0.04, 0.08, 0.25,
            ))),
        );
        let err = zero_spot.npv().unwrap_err().to_string();
        assert!(
            err.contains("negative or null underlying"),
            "zero spot: {err}"
        );

        let mut triggered = BarrierOption::with_rebate(
            BarrierType::DownOut,
            101.0,
            3.0,
            payoff,
            Shared::clone(&european),
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_black_scholes_barrier_engine(
            &mut triggered,
            shared_mut(FdBlackScholesBarrierEngine::new(flat_bs_process(
                today, 100.0, 0.04, 0.08, 0.25,
            ))),
        );
        let err = triggered.npv().unwrap_err().to_string();
        assert!(err.contains("barrier touched"), "triggered: {err}");

        let american: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, expiry, false).unwrap());
        let mut american_opt = BarrierOption::with_rebate(
            BarrierType::DownOut,
            95.0,
            3.0,
            payoff,
            american,
            Shared::clone(&settings),
        )
        .unwrap();
        set_fd_black_scholes_barrier_engine(
            &mut american_opt,
            shared_mut(FdBlackScholesBarrierEngine::new(flat_bs_process(
                today, 100.0, 0.04, 0.08, 0.25,
            ))),
        );
        let err = american_opt.npv().unwrap_err().to_string();
        assert!(
            err.contains("only european style option are supported"),
            "american: {err}"
        );
    }

    /// `barrieroption.cpp` `testLocalVolAndHestonComparison` local-vol arm:
    /// DownOut put, expected NPV 132.8 at 1% relative.
    #[test]
    fn local_vol_and_heston_comparison_local_vol_arm() {
        let settlement = Date::new(5, Month::July, 2002);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(settlement);
        let dc = Actual365Fixed::new();

        let t = [13, 41, 75, 165, 256, 345, 524, 703];
        let r = [
            0.0357, 0.0349, 0.0341, 0.0355, 0.0359, 0.0368, 0.0386, 0.0401,
        ];
        let mut curve_dates = vec![settlement];
        let mut rates = vec![0.0357];
        for i in 0..8 {
            curve_dates.push(settlement + Period::new(t[i], TimeUnit::Days));
            rates.push(r[i]);
        }
        let vol_dates = curve_dates[1..].to_vec();
        let r_ts = Handle::new(shared(
            ZeroCurve::new(curve_dates, rates, dc.clone(), Linear).unwrap(),
        ) as Shared<dyn YieldTermStructure>);
        let q_ts = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.0,
            dc.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);

        let strikes = [
            100.0, 500.0, 2000.0, 3400.0, 3600.0, 3800.0, 4000.0, 4200.0, 4400.0, 4500.0, 4600.0,
            4800.0, 5000.0, 5200.0, 5400.0, 5600.0, 7500.0, 10000.0, 20000.0, 30000.0,
        ];
        let v = [
            1.015873, 1.015873, 1.015873, 0.89729, 0.796493, 0.730914, 0.631335, 0.568895,
            0.711309, 0.711309, 0.711309, 0.641309, 0.635593, 0.583653, 0.508045, 0.463182,
            0.516034, 0.500534, 0.500534, 0.500534, 0.448706, 0.416661, 0.375470, 0.353442,
            0.516034, 0.482263, 0.447713, 0.387703, 0.355064, 0.337438, 0.316966, 0.306859,
            0.497587, 0.464373, 0.430764, 0.374052, 0.344336, 0.328607, 0.310619, 0.301865,
            0.479511, 0.446815, 0.414194, 0.361010, 0.334204, 0.320301, 0.304664, 0.297180,
            0.461866, 0.429645, 0.398092, 0.348638, 0.324680, 0.312512, 0.299082, 0.292785,
            0.444801, 0.413014, 0.382634, 0.337026, 0.315788, 0.305239, 0.293855, 0.288660,
            0.428604, 0.397219, 0.368109, 0.326282, 0.307555, 0.298483, 0.288972, 0.284791,
            0.420971, 0.389782, 0.361317, 0.321274, 0.303697, 0.295302, 0.286655, 0.282948,
            0.413749, 0.382754, 0.354917, 0.316532, 0.300016, 0.292251, 0.284420, 0.281164,
            0.400889, 0.370272, 0.343525, 0.307904, 0.293204, 0.286549, 0.280189, 0.277767,
            0.390685, 0.360399, 0.334344, 0.300507, 0.287149, 0.281380, 0.276271, 0.274588,
            0.383477, 0.353434, 0.327580, 0.294408, 0.281867, 0.276746, 0.272655, 0.271617,
            0.379106, 0.349214, 0.323160, 0.289618, 0.277362, 0.272641, 0.269332, 0.268846,
            0.377073, 0.347258, 0.320776, 0.286077, 0.273617, 0.269057, 0.266293, 0.266265,
            0.399925, 0.369232, 0.338895, 0.289042, 0.265509, 0.255589, 0.249308, 0.249665,
            0.423432, 0.406891, 0.373720, 0.314667, 0.281009, 0.263281, 0.246451, 0.242166,
            0.453704, 0.453704, 0.453704, 0.381255, 0.334578, 0.305527, 0.268909, 0.251367,
            0.517748, 0.517748, 0.517748, 0.416577, 0.364770, 0.331595, 0.287423, 0.264285,
        ];
        let n_vol_dates = vol_dates.len();
        let mut black_vol = Matrix::with_size(strikes.len(), n_vol_dates);
        for i in 0..strikes.len() {
            for j in 0..n_vol_dates {
                black_vol[(i, j)] = v[i * n_vol_dates + j];
            }
        }
        let mut vol_ts = BlackVarianceSurface::new(
            settlement,
            Some(Target::new()),
            &vol_dates,
            strikes.to_vec(),
            &black_vol,
            dc,
        )
        .unwrap();
        vol_ts.set_interpolation(&Bicubic).unwrap();
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(4500.0)) as Shared<dyn Quote>),
            q_ts,
            r_ts,
            Handle::new(shared(vol_ts) as Shared<dyn BlackVolTermStructure>),
        ));

        let expiry = settlement + Period::new(20, TimeUnit::Months);
        let payoff = PlainVanillaPayoff::new(OptionType::Put, 4500.0);
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
        set_fd_black_scholes_barrier_engine(
            &mut option,
            shared_mut(FdBlackScholesBarrierEngine::with_local_vol(
                process,
                Vec::new(),
                100,
                400,
                0,
                FdmSchemeDesc::douglas(),
                true,
                0.35,
            )),
        );
        let calculated = option.npv().unwrap();
        let expected = 132.8;
        let diff = (calculated - expected).abs();
        let tol = 0.01 * expected;
        eprintln!(
            "local-vol DownOut: calculated={calculated:.8} expected={expected} \
             diff={diff:.4} tol={tol:.4}"
        );
        assert!(
            calculated.is_finite() && diff <= tol,
            "local-vol barrier: {calculated} vs {expected} (diff {diff}, tol {tol})"
        );
    }
}
