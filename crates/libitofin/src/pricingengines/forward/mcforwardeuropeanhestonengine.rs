//! Monte Carlo engine for forward-starting European options under Heston.
//!
//! Port of `ql/pricingengines/forward/mcforwardeuropeanhestonengine.{hpp,cpp}`
//! plus `MCForwardVanillaEngine::timeGrid` (`mcforwardvanillaengine.hpp:120-140`).
//! Brownian-bridge / control-variate knobs are deferred (always off).

use std::any::Any;
use std::marker::PhantomData;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instruments::{
    ForwardOptionArguments, OneAssetOptionResults, PlainVanillaPayoff, StrikedTypePayoff,
    TypePayoff,
};
use crate::math::randomnumbers::rngtraits::McRngTraits;
use crate::math::statistics::MeanStdDev;
use crate::math::timegrid::TimeGrid;
use crate::methods::montecarlo::{McSimulation, McTraits, MultiPath, MultiVariate, PathPricer};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::HestonProcess;
use crate::shared::Shared;
use crate::stochasticprocess::StochasticProcess;
use crate::types::{DiscountFactor, Real, Size};
use crate::{fail, require};

type ForwardEngineBase = GenericEngine<ForwardOptionArguments, OneAssetOptionResults>;

/// Discounted payoff of a strike-reset European on asset 0
/// (`mcforwardeuropeanhestonengine.cpp:33-43`).
pub struct ForwardEuropeanHestonPathPricer {
    option_type: OptionType,
    moneyness: Real,
    reset_index: Size,
    discount: DiscountFactor,
}

impl ForwardEuropeanHestonPathPricer {
    /// `ForwardEuropeanHestonPathPricer(type, moneyness, resetIndex, discount)`.
    pub fn new(
        option_type: OptionType,
        moneyness: Real,
        reset_index: Size,
        discount: DiscountFactor,
    ) -> QlResult<Self> {
        require!(moneyness >= 0.0, "moneyness less than zero not allowed");
        Ok(Self {
            option_type,
            moneyness,
            reset_index,
            discount,
        })
    }
}

impl PathPricer<MultiPath> for ForwardEuropeanHestonPathPricer {
    fn price(&self, multi_path: &MultiPath) -> Real {
        let path = &multi_path[0];
        let strike = path[self.reset_index] * self.moneyness;
        PlainVanillaPayoff::new(self.option_type, strike).value(path.back()) * self.discount
    }
}

/// `MCForwardEuropeanHestonEngine<RNG>`.
pub struct McForwardEuropeanHestonEngine<RNG> {
    base: ForwardEngineBase,
    process: Shared<HestonProcess>,
    time_steps: Option<Size>,
    time_steps_per_year: Option<Size>,
    required_samples: Option<Size>,
    max_samples: Option<Size>,
    required_tolerance: Option<Real>,
    antithetic_variate: bool,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> McForwardEuropeanHestonEngine<RNG> {
    /// Prefer [`MakeMcForwardEuropeanHestonEngine`] for the named-parameter path.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        process: Shared<HestonProcess>,
        time_steps: Option<Size>,
        time_steps_per_year: Option<Size>,
        antithetic_variate: bool,
        required_samples: Option<Size>,
        required_tolerance: Option<Real>,
        max_samples: Option<Size>,
        seed: u32,
    ) -> QlResult<Self> {
        require!(
            time_steps.is_some() || time_steps_per_year.is_some(),
            "no time steps provided"
        );
        require!(
            time_steps.is_none() || time_steps_per_year.is_none(),
            "both time steps and time steps per year were provided"
        );
        require!(
            time_steps != Some(0),
            "timeSteps must be positive, 0 not allowed"
        );
        require!(
            time_steps_per_year != Some(0),
            "timeStepsPerYear must be positive, 0 not allowed"
        );
        let base = ForwardEngineBase::new(
            ForwardOptionArguments::default(),
            OneAssetOptionResults::default(),
        );
        base.register_with(process.observable());
        Ok(Self {
            base,
            process,
            time_steps,
            time_steps_per_year,
            required_samples,
            max_samples,
            required_tolerance,
            antithetic_variate,
            seed,
            _rng: PhantomData,
        })
    }
}

impl<RNG: McRngTraits> AsObservable for McForwardEuropeanHestonEngine<RNG> {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl<RNG: McRngTraits> PricingEngine for McForwardEuropeanHestonEngine<RNG> {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn calculate(&mut self) -> QlResult<()> {
        let args = self.base.arguments();
        let Some(payoff) = args.payoff.as_ref() else {
            fail!("no payoff given");
        };
        let payoff: &dyn StrikedTypePayoff = &**payoff;
        let Some(payoff) = (payoff as &dyn Any).downcast_ref::<PlainVanillaPayoff>() else {
            fail!("non-plain payoff given");
        };
        let Some(exercise) = args.exercise.as_ref() else {
            fail!("no exercise given");
        };
        if exercise.exercise_type() != ExerciseType::European {
            fail!("wrong exercise given");
        }
        let Some(moneyness) = args.moneyness else {
            fail!("null moneyness given");
        };
        let option_type = payoff.option_type();
        let reset_date = args.reset_date;
        let exercise = Shared::clone(exercise);

        let t_reset = self.process.time(&reset_date)?;
        let t_expiry = self.process.time(&exercise.last_date())?;
        let steps = match (self.time_steps, self.time_steps_per_year) {
            (Some(steps), None) => steps,
            (None, Some(per_year)) => (per_year as Real * t_expiry) as Size,
            _ => fail!("time steps not specified"),
        };
        let grid = TimeGrid::with_mandatory_times(&[t_reset, t_expiry], steps)?;
        let reset_index = grid.closest_index(t_reset);
        let Some(last_time) = grid.back() else {
            fail!("empty time grid");
        };
        let discount = self
            .process
            .risk_free_rate()
            .current_link()?
            .discount(last_time, false)?;
        let pricer =
            ForwardEuropeanHestonPathPricer::new(option_type, moneyness, reset_index, discount)?;
        let process: Shared<dyn StochasticProcess> =
            Shared::clone(&self.process) as Shared<dyn StochasticProcess>;
        let generator = RNG::make_sequence_generator(
            MultiVariate::factors(&process) * (grid.size() - 1),
            self.seed,
        )?;
        let path_gen = MultiVariate::path_generator(process, grid, generator, false)?;
        let mut simulation = McSimulation::<_, _>::new(self.antithetic_variate, false);
        simulation.calculate(
            path_gen,
            pricer,
            self.required_tolerance,
            self.required_samples,
            self.max_samples,
        )?;
        let results = self.base.results_mut();
        results.instrument.value = Some(simulation.sample_accumulator()?.mean()?);
        if RNG::ALLOWS_ERROR_ESTIMATE {
            results.instrument.error_estimate = Some(simulation.error_estimate()?);
        }
        Ok(())
    }
}

/// Factory for [`McForwardEuropeanHestonEngine`] (`MakeMCForwardEuropeanHestonEngine`).
pub struct MakeMcForwardEuropeanHestonEngine<RNG> {
    process: Shared<HestonProcess>,
    steps: Option<Size>,
    steps_per_year: Option<Size>,
    samples: Option<Size>,
    max_samples: Option<Size>,
    tolerance: Option<Real>,
    antithetic: bool,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MakeMcForwardEuropeanHestonEngine<RNG> {
    /// `MakeMCForwardEuropeanHestonEngine(process)`.
    pub fn new(process: Shared<HestonProcess>) -> Self {
        Self {
            process,
            steps: None,
            steps_per_year: None,
            samples: None,
            max_samples: None,
            tolerance: None,
            antithetic: false,
            seed: 0,
            _rng: PhantomData,
        }
    }

    #[must_use]
    pub fn with_steps(mut self, steps: Size) -> Self {
        self.steps = Some(steps);
        self
    }

    #[must_use]
    pub fn with_steps_per_year(mut self, steps: Size) -> Self {
        self.steps_per_year = Some(steps);
        self
    }

    #[must_use]
    pub fn with_samples(mut self, samples: Size) -> Self {
        self.samples = Some(samples);
        self
    }

    #[must_use]
    pub fn with_absolute_tolerance(mut self, tolerance: Real) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    #[must_use]
    pub fn with_max_samples(mut self, samples: Size) -> Self {
        self.max_samples = Some(samples);
        self
    }

    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    #[must_use]
    pub fn with_antithetic_variate(mut self, antithetic: bool) -> Self {
        self.antithetic = antithetic;
        self
    }

    /// Builds the engine (`mcforwardeuropeanhestonengine.hpp:283-299`).
    pub fn build(self) -> QlResult<McForwardEuropeanHestonEngine<RNG>> {
        require!(
            !(self.samples.is_some() && self.tolerance.is_some()),
            "number of samples already set"
        );
        if self.tolerance.is_some() {
            require!(
                RNG::ALLOWS_ERROR_ESTIMATE,
                "chosen random generator policy does not allow an error estimate"
            );
        }
        McForwardEuropeanHestonEngine::new(
            self.process,
            self.steps,
            self.steps_per_year,
            self.antithetic,
            self.samples,
            self.tolerance,
            self.max_samples,
            self.seed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::instrument::Instrument;
    use crate::instruments::ForwardVanillaOption;
    use crate::math::randomnumbers::rngtraits::{LowDiscrepancy, PseudoRandom};
    use crate::pricingengines::forward::set_analytic_forward_vanilla_engine;
    use crate::pricingengines::vanilla::test_market::{market, today};
    use crate::shared::{Shared, SharedMut, shared, shared_mut};
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    fn heston() -> Shared<HestonProcess> {
        let mkt = market();
        let sigma_bs = 0.245;
        mkt.set(100.0, 0.04, 0.01, sigma_bs);
        shared(HestonProcess::new(
            mkt.process.risk_free_rate(),
            mkt.process.dividend_yield(),
            mkt.process.state_variable(),
            sigma_bs * sigma_bs,
            1e-8,
            sigma_bs * sigma_bs,
            1e-8,
            -0.93,
        ))
    }

    fn maker() -> MakeMcForwardEuropeanHestonEngine<PseudoRandom> {
        MakeMcForwardEuropeanHestonEngine::new(heston())
    }

    /// `forwardoption.cpp` `testHestonMCPrices` Test 1: flat Heston vs analytic BS.
    #[test]
    fn forward_heston_mc_vs_analytic_bs() {
        let mkt = market();
        let sigma_bs = 0.245;
        mkt.set(100.0, 0.04, 0.01, sigma_bs);
        let heston = shared(HestonProcess::new(
            mkt.process.risk_free_rate(),
            mkt.process.dividend_yield(),
            mkt.process.state_variable(),
            sigma_bs * sigma_bs,
            1e-8,
            sigma_bs * sigma_bs,
            1e-8,
            -0.93,
        ));
        let exercise = shared(EuropeanExercise::new(
            today() + Period::new(1, TimeUnit::Years),
        ));
        let reset = today() + Period::new(6, TimeUnit::Months);
        let moneyness = [0.8, 0.9, 1.0, 1.1, 1.2];
        let call_tols = [7e-4, 8e-4, 6e-4, 5e-4, 5e-4];
        let put_tols = [6e-4, 5e-4, 6e-4, 1e-3, 1e-3];
        for (option_type, tols) in [(OptionType::Call, call_tols), (OptionType::Put, put_tols)] {
            let payoff =
                shared(PlainVanillaPayoff::new(option_type, 0.0)) as Shared<dyn StrikedTypePayoff>;
            for (m, tol) in moneyness.into_iter().zip(tols) {
                let mut option = ForwardVanillaOption::new(
                    m,
                    reset,
                    Shared::clone(&payoff),
                    Shared::clone(&exercise) as _,
                    Shared::clone(&mkt.settings),
                );
                set_analytic_forward_vanilla_engine(&mut option, Shared::clone(&mkt.process));
                let analytic = option.npv().unwrap();
                let engine = MakeMcForwardEuropeanHestonEngine::<LowDiscrepancy>::new(
                    Shared::clone(&heston),
                )
                .with_steps(50)
                .with_samples(4095)
                .with_seed(42)
                .build()
                .unwrap();
                option
                    .base_mut()
                    .set_pricing_engine(shared_mut(engine) as SharedMut<dyn PricingEngine>);
                let mc = option.npv().unwrap();
                let error = (analytic - mc).abs() / 100.0;
                assert!(
                    error <= tol,
                    "{option_type:?} moneyness={m}: analytic={analytic} mc={mc} rel={error} tol={tol}"
                );
            }
        }
    }

    #[test]
    fn factory_rejects_samples_and_tolerance() {
        assert!(
            maker()
                .with_steps(2)
                .with_samples(10)
                .with_absolute_tolerance(1e-4)
                .build()
                .is_err()
        );
    }

    #[test]
    fn factory_threads_antithetic_and_max_samples() {
        assert!(
            maker()
                .with_steps(2)
                .with_samples(10)
                .with_max_samples(20)
                .with_antithetic_variate(true)
                .build()
                .is_ok()
        );
    }

    #[test]
    fn factory_rejects_tolerance_on_low_discrepancy() {
        assert!(
            MakeMcForwardEuropeanHestonEngine::<LowDiscrepancy>::new(heston())
                .with_steps(2)
                .with_absolute_tolerance(1e-4)
                .build()
                .is_err()
        );
    }
}
