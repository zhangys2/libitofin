//! Monte Carlo Everest-option engine.
//! Port of `ql/experimental/exoticoptions/mceverestengine.{hpp,cpp}` (`PseudoRandom`).

use crate::errors::QlResult;
use crate::instrument::Instrument;
use crate::instruments::{EverestArguments, EverestResults};
use crate::math::randomnumbers::rngtraits::{McRngTraits, PseudoRandom};
use crate::math::statistics::MeanStdDev;
use crate::math::timegrid::TimeGrid;
use crate::methods::montecarlo::{McSimulation, MultiPath, MultiPathGenerator, PathPricer};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::{GeneralizedBlackScholesProcess, StochasticProcessArray};
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::{StochasticProcess, StochasticProcess1D};
use crate::types::{DiscountFactor, Rate, Real, Size};

type EngineBase = GenericEngine<EverestArguments, EverestResults>;

/// `(1 + min_j(S_j(T)/S_j(0) - 1) + guarantee) * notional * discount`.
pub struct EverestMultiPathPricer {
    notional: Real,
    guarantee: Rate,
    discount: DiscountFactor,
}

impl PathPricer<MultiPath> for EverestMultiPathPricer {
    fn price(&self, multi_path: &MultiPath) -> Real {
        let mut min_yield = multi_path[0].back() / multi_path[0].front() - 1.0;
        for j in 1..multi_path.asset_number() {
            min_yield = min_yield.min(multi_path[j].back() / multi_path[j].front() - 1.0);
        }
        (1.0 + min_yield + self.guarantee) * self.notional * self.discount
    }
}

/// Monte Carlo engine for European Everest options (`MCEverestEngine<PseudoRandom>`).
pub struct MCEverestEngine {
    base: EngineBase,
    process: Shared<StochasticProcessArray>,
    first: Shared<GeneralizedBlackScholesProcess>,
    time_steps: Option<Size>,
    time_steps_per_year: Option<Size>,
    samples: Option<Size>,
    tolerance: Option<Real>,
    max_samples: Option<Size>,
    antithetic: bool,
    seed: u32,
}

impl MCEverestEngine {
    #[allow(clippy::too_many_arguments)]
    #[rustfmt::skip]
    fn new(
        process: Shared<StochasticProcessArray>,
        first: Shared<GeneralizedBlackScholesProcess>,
        time_steps: Option<Size>,
        time_steps_per_year: Option<Size>,
        samples: Option<Size>,
        tolerance: Option<Real>,
        max_samples: Option<Size>,
        antithetic: bool,
        seed: u32,
    ) -> QlResult<Self> {
        require!(time_steps.is_some() || time_steps_per_year.is_some(), "no time steps provided");
        require!(time_steps.is_none() || time_steps_per_year.is_none(), "both time steps and time steps per year were provided");
        require!(time_steps != Some(0), "timeSteps must be positive, 0 not allowed");
        require!(time_steps_per_year != Some(0), "timeStepsPerYear must be positive, 0 not allowed");
        let p0 = process.process(0);
        let first_1d: Shared<dyn StochasticProcess1D> = Shared::clone(&first) as _;
        require!(Shared::ptr_eq(&first_1d, &p0), "Black-Scholes process required");
        let base = EngineBase::new(EverestArguments::default(), EverestResults::default());
        base.register_with(process.observable());
        Ok(Self { base, process, first, time_steps, time_steps_per_year, samples, tolerance, max_samples, antithetic, seed })
    }
}

impl AsObservable for MCEverestEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for MCEverestEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    #[rustfmt::skip]
    fn calculate(&mut self) -> QlResult<()> {
        let a = self.base.arguments();
        let last = a.exercise.as_ref().expect("validated").last_date();
        let notional = a.notional.expect("validated");
        let guarantee = a.guarantee.expect("validated");
        let t = self.process.process(0).time(&last)?;
        let steps = if let Some(n) = self.time_steps { n } else { ((self.time_steps_per_year.unwrap() as Real) * t) as Size }.max(1);
        let grid = TimeGrid::new(t, steps)?;
        let dim = self.process.size() * (grid.size() - 1);
        let rsg = PseudoRandom::make_sequence_generator(dim, self.seed)?;
        let path_gen = MultiPathGenerator::new(Shared::clone(&self.process) as Shared<dyn StochasticProcess>, grid, rsg, false)?;
        let disc = self.first.risk_free_rate().current_link()?.discount_date(last, false)?;
        let pricer = EverestMultiPathPricer { notional, guarantee, discount: disc };
        let mut sim: McSimulation<_, EverestMultiPathPricer> = McSimulation::new(self.antithetic, false);
        sim.calculate(path_gen, pricer, self.tolerance, self.samples, self.max_samples)?;
        let value = sim.sample_accumulator()?.mean()?;
        let r = self.base.results_mut();
        r.instrument.value = Some(value);
        r.instrument.error_estimate = Some(sim.error_estimate()?);
        r.yield_rate = Some(value / (notional * disc) - 1.0);
        Ok(())
    }
}

/// Factory (`MakeMCEverestEngine<PseudoRandom>`). `first` must be `process(0)`.
pub struct MakeMcEverestEngine {
    process: Shared<StochasticProcessArray>,
    first: Shared<GeneralizedBlackScholesProcess>,
    steps: Option<Size>,
    steps_per_year: Option<Size>,
    samples: Option<Size>,
    tolerance: Option<Real>,
    max_samples: Option<Size>,
    antithetic: bool,
    seed: u32,
}

impl MakeMcEverestEngine {
    #[rustfmt::skip]
    pub fn new(process: Shared<StochasticProcessArray>, first: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self { process, first, steps: None, steps_per_year: None, samples: None, tolerance: None, max_samples: None, antithetic: false, seed: 0 }
    }

    #[must_use]
    #[rustfmt::skip]
    pub fn with_steps(mut self, steps: Size) -> Self { self.steps = Some(steps); self }

    #[must_use]
    #[rustfmt::skip]
    pub fn with_steps_per_year(mut self, steps: Size) -> Self { self.steps_per_year = Some(steps); self }

    #[must_use]
    #[rustfmt::skip]
    pub fn with_samples(mut self, samples: Size) -> Self { self.samples = Some(samples); self }

    #[must_use]
    #[rustfmt::skip]
    pub fn with_absolute_tolerance(mut self, tolerance: Real) -> Self { self.tolerance = Some(tolerance); self }

    #[must_use]
    #[rustfmt::skip]
    pub fn with_max_samples(mut self, samples: Size) -> Self { self.max_samples = Some(samples); self }

    #[must_use]
    #[rustfmt::skip]
    pub fn with_antithetic_variate(mut self, antithetic: bool) -> Self { self.antithetic = antithetic; self }

    #[must_use]
    #[rustfmt::skip]
    pub fn with_seed(mut self, seed: u32) -> Self { self.seed = seed; self }

    #[rustfmt::skip]
    pub fn build(self) -> QlResult<MCEverestEngine> {
        require!(self.steps.is_some() || self.steps_per_year.is_some(), "number of steps not given");
        require!(self.steps.is_none() || self.steps_per_year.is_none(), "number of steps overspecified");
        require!(!(self.samples.is_some() && self.tolerance.is_some()), "number of samples already set");
        if self.tolerance.is_some() {
            require!(PseudoRandom::ALLOWS_ERROR_ESTIMATE, "chosen random generator policy does not allow an error estimate");
        }
        MCEverestEngine::new(self.process, self.first, self.steps, self.steps_per_year, self.samples, self.tolerance, self.max_samples, self.antithetic, self.seed)
    }
}

#[rustfmt::skip]
pub fn set_mc_everest_engine(
    option: &mut crate::instruments::EverestOption,
    process: Shared<StochasticProcessArray>,
    first: Shared<GeneralizedBlackScholesProcess>,
    steps_per_year: Size,
    samples: Size,
    seed: u32,
) -> QlResult<()> {
    let engine = shared_mut(MakeMcEverestEngine::new(process, first).with_steps_per_year(steps_per_year).with_samples(samples).with_seed(seed).build()?) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instruments::EverestOption;
    use crate::interestrate::Compounding;
    use crate::math::matrix::Matrix;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn quote_h(v: Real) -> Handle<dyn Quote> {
        Handle::new(shared(SimpleQuote::new(v)) as Shared<dyn Quote>)
    }

    #[rustfmt::skip]
    fn flat_rate(today: Date, r: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(today, quote_h(r), Actual360::new(), Compounding::Continuous, Frequency::Annual)) as Shared<dyn YieldTermStructure>)
    }

    #[rustfmt::skip]
    fn flat_vol(today: Date, v: Real) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(today, None, quote_h(v), Actual360::new())) as Shared<dyn BlackVolTermStructure>)
    }

    /// `everestoption.cpp` `testCached` NPV @ 1e-8.
    #[test]
    #[rustfmt::skip]
    fn everest_cached_npv() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);
        let rf = flat_rate(today, 0.05);
        let dummy = quote_h(1.0);
        let bs = |q: Real, v: Real| shared(BlackScholesMertonProcess::new(dummy.clone(), flat_rate(today, q), rf.clone(), flat_vol(today, v)));
        let first = bs(0.01, 0.30);
        let processes: Vec<Shared<dyn StochasticProcess1D>> = vec![
            Shared::clone(&first) as _, bs(0.05, 0.35) as _, bs(0.04, 0.25) as _, bs(0.03, 0.20) as _,
        ];
        let corr = Matrix::from([
            [1.00, 0.50, 0.30, 0.10],
            [0.50, 1.00, 0.20, 0.40],
            [0.30, 0.20, 1.00, 0.60],
            [0.10, 0.40, 0.60, 1.00],
        ]);
        let array = shared(StochasticProcessArray::new(processes, &corr).unwrap());
        let ex: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 360));
        let mut option = EverestOption::new(1.0, 0.0, ex, settings);
        set_mc_everest_engine(&mut option, Shared::clone(&array), Shared::clone(&first), 1, 1023, 86421).unwrap();
        let value = option.npv().unwrap();
        assert!((value - 0.75784944).abs() <= 1e-8, "cached 0.75784944 vs {value}");
        let tol = (option.error_estimate().unwrap() / 2.0).min(1e-2 * value);
        let engine = shared_mut(MakeMcEverestEngine::new(array, first).with_steps_per_year(1).with_absolute_tolerance(tol).with_seed(86421).build().unwrap()) as SharedMut<dyn PricingEngine>;
        option.base_mut().set_pricing_engine(engine);
        option.npv().unwrap();
        let acc = option.error_estimate().unwrap();
        assert!(acc <= tol, "errorEstimate {acc} vs {tol}");
    }
}
