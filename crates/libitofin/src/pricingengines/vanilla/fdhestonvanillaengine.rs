//! Finite-difference Heston engine for European vanillas with cash dividends.
//!
//! Port of `ql/pricingengines/vanilla/fdhestonvanillaengine.{hpp,cpp}` on the
//! single-strike path (no multiple-strikes cache, leverage function, quanto,
//! or mixing factor ≠ 1). American/Bermudan exercise is follow-up.

use crate::errors::QlResult;
use crate::exercise::{Exercise, ExerciseType};
use crate::fail;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
    StrikedTypePayoff,
};
use crate::methods::finitedifferences::meshers::{
    FdmHestonVarianceMesher, FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher,
    process_helper,
};
use crate::methods::finitedifferences::solvers::{FdmHestonSolver, FdmSchemeDesc, FdmSolverDesc};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{FdmInnerValueCalculator, fdm_log_inner_value};
use crate::models::equity::HestonModel;
use crate::models::model::CalibratedModelHolder;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::DividendSchedule;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared};
use crate::stochasticprocess::StochasticProcess;
use crate::types::{Real, Size};

/// Finite-difference Heston vanilla engine (European + Spot dividends).
pub struct FdHestonVanillaEngine {
    base: OneAssetOptionEngine,
    model: SharedMut<HestonModel>,
    dividends: DividendSchedule,
    t_grid: Size,
    x_grid: Size,
    v_grid: Size,
    damping_steps: Size,
    scheme_desc: FdmSchemeDesc,
}

impl FdHestonVanillaEngine {
    /// `FdHestonVanillaEngine(model)` with QuantLib defaults:
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

    /// `FdHestonVanillaEngine(model, dividends)` with the same grid defaults.
    pub fn with_dividends(model: SharedMut<HestonModel>, dividends: DividendSchedule) -> Self {
        Self::with_params(
            model,
            dividends,
            100,
            100,
            50,
            0,
            FdmSchemeDesc::hundsdorfer(),
        )
    }

    /// Full constructor matching the C++ dividends form, without leverage /
    /// mixing-factor / quanto arguments.
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
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
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

impl AsObservable for FdHestonVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdHestonVanillaEngine {
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
            strike,
            None,
            None,
            0.0001,
            2.0,
            Some((strike, 0.1)),
            &self.dividends,
            0.0,
        )?;
        let mesher = shared(FdmMesherComposite::new(vec![
            equity,
            v_mesher.into_mesher(),
        ]));
        let mesher_dyn: Shared<dyn FdmMesher> = mesher.clone() as Shared<dyn FdmMesher>;

        let payoff_dyn: Shared<dyn Payoff> = Shared::clone(payoff) as Shared<dyn Payoff>;
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            payoff_dyn,
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

        let solver_desc = FdmSolverDesc {
            mesher: mesher_dyn,
            bc_set: Vec::new(),
            condition: conditions,
            calculator,
            maturity,
            time_steps: self.t_grid,
            damping_steps: self.damping_steps,
        };
        let solver = FdmHestonSolver::new(process, solver_desc, self.scheme_desc, 1.0);
        let v0 = self.model.borrow().v0();
        let value = solver.value_at(spot, v0)?;
        let results = self.base.results_mut();
        results.instrument.value = Some(value);
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
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::vanilla::analytichestonengine::AnalyticHestonEngine;
    use crate::processes::HestonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{Shared, SharedMut, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    fn heston_model(
        spot: Real,
        v0: Real,
        kappa: Real,
        theta: Real,
        sigma: Real,
        rho: Real,
        r: Real,
        q: Real,
        today: Date,
    ) -> SharedMut<HestonModel> {
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
        let process = shared(HestonProcess::new(
            flat(r),
            flat(q),
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            v0,
            kappa,
            theta,
            sigma,
            rho,
        ));
        HestonModel::new(process).unwrap()
    }

    /// No-dividend European FD vs [`AnalyticHestonEngine`] at 1% relative.
    #[test]
    fn no_dividend_european_is_close_to_analytic() {
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let model = heston_model(100.0, 0.04, 2.0, 0.04, 0.2, -0.5, 0.05, 0.03, today);
        let expiry = today + Period::new(1, TimeUnit::Years);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, 100.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));

        let mut analytic = VanillaOption::new(
            Shared::clone(&payoff),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        analytic.base_mut().set_pricing_engine(shared_mut(
            AnalyticHestonEngine::with_default_order(SharedMut::clone(&model)).unwrap(),
        ) as SharedMut<dyn PricingEngine>);
        let expected = analytic.npv().unwrap();

        let mut fd = FdHestonVanillaEngine::with_params(
            model,
            Vec::new(),
            40,
            80,
            25,
            0,
            FdmSchemeDesc::hundsdorfer(),
        );
        let calculated = fd.price(payoff, exercise).unwrap();
        let diff = (calculated - expected).abs();
        let tol = 0.01 * expected;
        eprintln!(
            "Heston vanilla FD: calculated={calculated:.8} analytic={expected:.8} \
             diff={diff:.4} tol={tol:.4}"
        );
        assert!(
            calculated.is_finite() && expected > 0.0 && diff <= tol,
            "Heston vanilla FD {calculated} vs analytic {expected} (diff {diff}, tol {tol})"
        );
    }

    #[test]
    fn non_european_exercise_is_rejected() {
        use crate::exercise::AmericanExercise;

        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let model = heston_model(100.0, 0.04, 1.0, 0.04, 0.2, -0.5, 0.05, 0.0, today);
        let expiry = today + Period::new(6, TimeUnit::Months);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, 100.0));
        let exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, expiry, false).unwrap());
        let mut engine = FdHestonVanillaEngine::new(model);
        let err = engine.price(payoff, exercise).unwrap_err().to_string();
        assert!(
            err.contains("only european style option are supported"),
            "american: {err}"
        );
        let _ = settings;
    }
}
