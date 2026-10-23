//! FD Black–Scholes engine for simple swing options.
//!
//! Port of `ql/pricingengines/vanilla/fdsimplebsswingengine.{hpp,cpp}` plus
//! `FdmSimple2dBSSolver` (`fdmsimple2dbssolver.{hpp,cpp}`).

use crate::errors::QlResult;
use crate::exercise::{Exercise, ExerciseType};
use crate::instrument::Instrument;
use crate::instruments::{VanillaSwingArguments, VanillaSwingOption, VanillaSwingResults};
use crate::methods::finitedifferences::StepCondition;
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher, uniform_1d_mesher,
};
use crate::methods::finitedifferences::operators::FdmLinearOpIterator;
use crate::methods::finitedifferences::operators::{FdmBlackScholesOp, FdmLinearOpComposite};
use crate::methods::finitedifferences::solvers::{Fdm2DimSolver, FdmSchemeDesc, FdmSolverDesc};
use crate::methods::finitedifferences::stepconditions::{
    FdmSimpleSwingCondition, FdmStepConditionComposite,
};
use crate::methods::finitedifferences::utilities::{FdmInnerValueCalculator, fdm_log_inner_value};
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size, Time};

type EngineBase = GenericEngine<VanillaSwingArguments, VanillaSwingResults>;

struct FdmZeroInnerValue;
#[rustfmt::skip]
impl FdmInnerValueCalculator for FdmZeroInnerValue {
    fn inner_value(&self, _: &FdmLinearOpIterator, _: Time) -> Real { 0.0 }
    fn avg_inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real { self.inner_value(iter, t) }
}

pub struct FdSimpleBSSwingEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    t_grid: Size,
    x_grid: Size,
    scheme_desc: FdmSchemeDesc,
}

impl FdSimpleBSSwingEngine {
    #[rustfmt::skip]
    pub fn with_grid(process: Shared<GeneralizedBlackScholesProcess>, t_grid: Size, x_grid: Size) -> Self {
        let base = EngineBase::new(VanillaSwingArguments::default(), VanillaSwingResults::default());
        base.register_with(process.observable());
        Self { base, process, t_grid, x_grid, scheme_desc: FdmSchemeDesc::douglas() }
    }
}

#[rustfmt::skip]
impl AsObservable for FdSimpleBSSwingEngine {
    fn observable(&self) -> &Observable { self.base.observable() }
}

#[rustfmt::skip]
impl PricingEngine for FdSimpleBSSwingEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments { self.base.arguments_mut() }
    fn results(&self) -> &dyn Results { self.base.results() }
    fn reset(&mut self) { self.base.reset(); }

    fn calculate(&mut self) -> QlResult<()> {
        let args = self.base.arguments();
        let exercise = Shared::clone(args.exercise.as_ref().expect("validated"));
        require!(exercise.exercise_type() == ExerciseType::Bermudan, "Bermudan exercise supported only");
        let payoff = Shared::clone(args.payoff.as_ref().expect("validated"));
        let max_rights = args.max_exercise_rights.expect("validated");
        let min_rights = args.min_exercise_rights.expect("validated");
        let maturity = self.process.time(&exercise.last_date())?;
        let equity = fdm_black_scholes_mesher(self.x_grid, &self.process, maturity, payoff.strike(), None, None, 0.0001, 1.5, None, &[], 0.0)?;
        let rights = uniform_1d_mesher(0.0, max_rights as Real, max_rights + 1)?;
        let mesher = shared(FdmMesherComposite::new(vec![equity, rights])) as Shared<dyn FdmMesher>;
        let mut exercise_times = Vec::new();
        for d in exercise.dates() {
            let t = self.process.time(d)?;
            require!(t >= 0.0, "exercise dates must not contain past date");
            exercise_times.push(t);
        }
        let calc: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(Shared::clone(&payoff) as Shared<dyn Payoff>, Shared::clone(&mesher), 0));
        let swing: Shared<dyn StepCondition> = shared(FdmSimpleSwingCondition::new(exercise_times.clone(), Shared::clone(&mesher), calc, 1, min_rights));
        let conditions = shared(FdmStepConditionComposite::new(&[exercise_times], vec![swing]));
        let desc = FdmSolverDesc { mesher: Shared::clone(&mesher), bc_set: Vec::new(), condition: conditions, calculator: shared(FdmZeroInnerValue), maturity, time_steps: self.t_grid, damping_steps: 0 };
        let op = shared_mut(FdmBlackScholesOp::new(Shared::clone(&mesher), &self.process, payoff.strike(), 0)?) as SharedMut<dyn FdmLinearOpComposite>;
        let solver = Fdm2DimSolver::new(desc, self.scheme_desc, op);
        let spot = self.process.x0()?;
        self.base.results_mut().instrument.value = Some(solver.interpolate_at(spot.ln(), 1.0_f64.ln())?);
        Ok(())
    }
}

#[rustfmt::skip]
pub fn set_fd_simple_bs_swing_engine(option: &mut VanillaSwingOption, process: Shared<GeneralizedBlackScholesProcess>, t_grid: Size, x_grid: Size) {
    option.base_mut().set_pricing_engine(shared_mut(FdSimpleBSSwingEngine::with_grid(process, t_grid, x_grid)) as SharedMut<dyn PricingEngine>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instruments::{
        EuropeanOption, PlainVanillaPayoff, StrikedTypePayoff, SwingExercise, VanillaForwardPayoff,
        VanillaOption,
    };
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengines::vanilla::{AnalyticEuropeanEngine, FdBlackScholesVanillaEngine};
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::Date;
    use crate::time::date::Month;
    use crate::time::daycounters::actualactual::{ActualActual, Convention};
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    #[test]
    #[rustfmt::skip]
    fn fd_bs_swing_option_bounds() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);
        let dc = ActualActual::with_convention(Convention::ISDA);
        let qh = |q: &Shared<SimpleQuote>| Handle::new(Shared::clone(q) as Shared<dyn Quote>);
        let spot = shared(SimpleQuote::new(30.0));
        let r_ts = Handle::new(shared(FlatForward::new(today, qh(&shared(SimpleQuote::new(0.14))), dc.clone(), Compounding::Continuous, Frequency::Annual)) as Shared<dyn YieldTermStructure>);
        let q_ts = Handle::new(shared(FlatForward::new(today, qh(&shared(SimpleQuote::new(0.02))), dc.clone(), Compounding::Continuous, Frequency::Annual)) as Shared<dyn YieldTermStructure>);
        let vol = Handle::new(shared(BlackConstantVol::new(today, None, 0.4, dc.clone())) as Shared<dyn BlackVolTermStructure>);
        let process = shared(GeneralizedBlackScholesProcess::new(qh(&spot), q_ts, r_ts, vol));
        let maturity = today + Period::new(12, TimeUnit::Months);
        let mut dates = vec![today + Period::new(1, TimeUnit::Months)];
        while dates[dates.len() - 1] < maturity { dates.push(dates[dates.len() - 1] + Period::new(1, TimeUnit::Months)); }
        let swing = shared(SwingExercise::new(dates.clone()).unwrap());
        let put = shared(PlainVanillaPayoff::new(OptionType::Put, 30.0));
        let mut bermudan = VanillaOption::new(Shared::clone(&put) as Shared<dyn StrikedTypePayoff>, Shared::clone(&swing) as Shared<dyn Exercise>, Shared::clone(&settings));
        bermudan.base_mut().set_pricing_engine(shared_mut(FdBlackScholesVanillaEngine::with_params(Shared::clone(&process), Vec::new(), 50, 200, 0, FdmSchemeDesc::douglas())) as SharedMut<dyn PricingEngine>);
        let bermudan_npv = bermudan.npv().unwrap();
        let forward = shared(VanillaForwardPayoff::new(OptionType::Put, 30.0)) as Shared<dyn StrikedTypePayoff>;
        for i in 0..dates.len() {
            let rights = i + 1;
            let mut swing_opt = VanillaSwingOption::new(Shared::clone(&forward), Shared::clone(&swing), 0, rights, Shared::clone(&settings));
            set_fd_simple_bs_swing_engine(&mut swing_opt, Shared::clone(&process), 50, 200);
            let price = swing_opt.npv().unwrap();
            let upper = rights as Real * bermudan_npv;
            assert!(price - upper <= 0.01, "rights={rights} price {price} > upper {upper}");
            let mut lower = 0.0;
            for d in dates.iter().skip(dates.len() - rights) {
                let mut euro = EuropeanOption::new(Shared::clone(&put) as Shared<dyn StrikedTypePayoff>, shared(EuropeanExercise::new(*d)) as Shared<dyn Exercise>, Shared::clone(&settings));
                euro.base_mut().set_pricing_engine(shared_mut(AnalyticEuropeanEngine::new(Shared::clone(&process))) as SharedMut<dyn PricingEngine>);
                lower += euro.npv().unwrap();
            }
            assert!(lower - price <= 4e-2, "rights={rights} lower {lower} > price {price}");
        }
        let n = dates.len();
        let mut forced = VanillaSwingOption::new(Shared::clone(&forward), Shared::clone(&swing), n, n, Shared::clone(&settings));
        set_fd_simple_bs_swing_engine(&mut forced, Shared::clone(&process), 50, 200);
        let mut analytic = 0.0;
        for d in &dates {
            let t = process.time(d).unwrap();
            analytic += 30.0 * process.risk_free_rate().current_link().unwrap().discount(t, false).unwrap() - 30.0 * process.dividend_yield().current_link().unwrap().discount(t, false).unwrap();
        }
        assert!((forced.npv().unwrap() - analytic).abs() < 2e-3);
        assert_eq!(VanillaForwardPayoff::new(OptionType::Put, 30.0).value(40.0), -10.0);
    }
}
