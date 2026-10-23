//! Monte Carlo engine for barrier options.
//!
//! Port of `ql/pricingengines/barrier/mcbarrierengine.{hpp,cpp}`: a
//! single-factor [`McSimulation`] over a Black–Scholes process, with the
//! Beaglehole–Dybvig–Zhou / Babsiri–Noel Brownian-bridge correction for the
//! intra-step extremum (`BarrierPathPricer`) and a node-only biased variant
//! (`BiasedBarrierPathPricer`).
//!
//! Divergences from `mcbarrierengine.hpp`, all deliberate:
//! - **composition, not MI**: C++ derives from `BarrierOption::engine` and
//!   `McSimulation` (`mcbarrierengine.hpp:58`). Rust holds a
//!   [`GenericEngine`] plus a fresh [`McSimulation`] per `calculate`, matching
//!   [`McVanillaEngineBase`].
//! - **`Null` sentinels become [`Option`]**: `timeSteps`, `timeStepsPerYear`,
//!   `requiredSamples`, `requiredTolerance`, and `maxSamples` are `Option`.
//! - **payoff is already plain**: C++ `dynamic_pointer_cast`s the argument
//!   payoff (`mcbarrierengine.hpp:237`); [`BarrierArguments`] stores a
//!   [`PlainVanillaPayoff`] so that downcast cannot fail.
//! - **uniform extrema RNG is interior-mutable**: C++ mutates
//!   `sequenceGen_` from a `const operator()` (`mcbarrierengine.cpp:58`).
//!   [`PathPricer::price`] takes `&self`, so the uniforms live in a
//!   [`RefCell`].
//!
//! Deferred, rejected visibly rather than silently ignored:
//! - **antithetic variate**: [`MakeMcBarrierEngine::with_antithetic_variate`]
//!   is kept; requesting it makes [`build`](MakeMcBarrierEngine::build)
//!   return `Err` (the single-factor generator's antithetic draw is deferred).

use std::cell::RefCell;
use std::marker::PhantomData;

use crate::errors::QlResult;
use crate::fail;
use crate::instrument::{Instrument, InstrumentResults};
use crate::instruments::{
    BarrierArguments, BarrierOption, BarrierType, PlainVanillaPayoff, StrikedTypePayoff, TypePayoff,
};
use crate::math::randomnumbers::mt19937uniformrng::MersenneTwisterUniformRng;
use crate::math::randomnumbers::randomsequencegenerator::RandomSequenceGenerator;
use crate::math::randomnumbers::rngtraits::{McRngTraits, SequenceGenerator};
use crate::math::statistics::MeanStdDev;
use crate::math::timegrid::TimeGrid;
use crate::methods::montecarlo::{McSimulation, Path, PathGenerator, PathPricer};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{DiscountFactor, Real, Size};

use super::fdblackscholesbarrierengine::triggered;

type BarrierEngineBase = GenericEngine<BarrierArguments, InstrumentResults>;

/// Unbiased barrier path pricer: samples the intra-step geometric-Brownian
/// extremum with a uniform deviate (`mcbarrierengine.cpp:45-157`).
pub struct BarrierPathPricer {
    barrier_type: BarrierType,
    barrier: Real,
    rebate: Real,
    diff_process: Shared<dyn StochasticProcess1D>,
    sequence_gen: RefCell<RandomSequenceGenerator<MersenneTwisterUniformRng>>,
    payoff: PlainVanillaPayoff,
    discounts: Vec<DiscountFactor>,
}

impl BarrierPathPricer {
    /// Builds the unbiased pricer (`mcbarrierengine.cpp:28-43`).
    ///
    /// # Errors
    ///
    /// Errors if `strike` is negative or `barrier` is not strictly positive.
    #[allow(clippy::neg_cmp_op_on_partial_ord, clippy::too_many_arguments)]
    pub fn new(
        barrier_type: BarrierType,
        barrier: Real,
        rebate: Real,
        option_type: OptionType,
        strike: Real,
        discounts: Vec<DiscountFactor>,
        diff_process: Shared<dyn StochasticProcess1D>,
        sequence_gen: RandomSequenceGenerator<MersenneTwisterUniformRng>,
    ) -> QlResult<Self> {
        require!(strike >= 0.0, "strike less than zero not allowed");
        require!(barrier > 0.0, "barrier less/equal zero not allowed");
        Ok(BarrierPathPricer {
            barrier_type,
            barrier,
            rebate,
            diff_process,
            sequence_gen: RefCell::new(sequence_gen),
            payoff: PlainVanillaPayoff::new(option_type, strike),
            discounts,
        })
    }
}

fn sampled_extremum(
    down: bool,
    asset: Real,
    new_asset: Real,
    vol: Real,
    dt: Real,
    u: Real,
) -> Real {
    let x = (new_asset / asset).ln();
    let log_u = if down { u.ln() } else { (1.0 - u).ln() };
    let inner = (x * x - 2.0 * vol * vol * dt * log_u).sqrt();
    let y = if down {
        0.5 * (x - inner)
    } else {
        0.5 * (x + inner)
    };
    asset * y.exp()
}

fn barrier_payoff(
    is_option_active: bool,
    barrier_type: BarrierType,
    asset_price: Real,
    rebate: Real,
    payoff: &PlainVanillaPayoff,
    discounts: &[DiscountFactor],
    knock_node: Option<Size>,
) -> Real {
    if is_option_active {
        payoff.value(asset_price) * discounts[discounts.len() - 1]
    } else {
        match barrier_type {
            BarrierType::UpIn | BarrierType::DownIn => rebate * discounts[discounts.len() - 1],
            BarrierType::UpOut | BarrierType::DownOut => {
                rebate * discounts[knock_node.expect("knock-out rebate needs a knock node")]
            }
        }
    }
}

impl PathPricer<Path> for BarrierPathPricer {
    fn price(&self, path: &Path) -> Real {
        let n = path.length();
        debug_assert!(n > 1, "the path cannot be empty");

        let u = self.sequence_gen.borrow_mut().next_sequence().value.clone();
        let time_grid = path.time_grid();
        let mut is_option_active =
            matches!(self.barrier_type, BarrierType::DownOut | BarrierType::UpOut);
        let mut knock_node = None;
        let mut asset_price = path.front();

        for i in 0..n - 1 {
            let new_asset_price = path[i + 1];
            let vol = self
                .diff_process
                .diffusion(time_grid[i], asset_price)
                .expect("diffusion lookup failed during barrier path pricing");
            let dt = time_grid.dt(i);
            let down = matches!(
                self.barrier_type,
                BarrierType::DownIn | BarrierType::DownOut
            );
            let y = sampled_extremum(down, asset_price, new_asset_price, vol, dt, u[i]);
            let hit = if down {
                y <= self.barrier
            } else {
                y >= self.barrier
            };
            if hit {
                is_option_active =
                    matches!(self.barrier_type, BarrierType::DownIn | BarrierType::UpIn);
                if knock_node.is_none() {
                    knock_node = Some(i + 1);
                }
            }
            asset_price = new_asset_price;
        }

        barrier_payoff(
            is_option_active,
            self.barrier_type,
            asset_price,
            self.rebate,
            &self.payoff,
            &self.discounts,
            knock_node,
        )
    }
}

/// Node-only barrier path pricer (`mcbarrierengine.cpp:173-246`): a hit is
/// recorded only when a sampled node itself crosses the barrier.
pub struct BiasedBarrierPathPricer {
    barrier_type: BarrierType,
    barrier: Real,
    rebate: Real,
    payoff: PlainVanillaPayoff,
    discounts: Vec<DiscountFactor>,
}

impl BiasedBarrierPathPricer {
    /// Builds the biased pricer (`mcbarrierengine.cpp:159-171`).
    ///
    /// # Errors
    ///
    /// Errors if `strike` is negative or `barrier` is not strictly positive.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn new(
        barrier_type: BarrierType,
        barrier: Real,
        rebate: Real,
        option_type: OptionType,
        strike: Real,
        discounts: Vec<DiscountFactor>,
    ) -> QlResult<Self> {
        require!(strike >= 0.0, "strike less than zero not allowed");
        require!(barrier > 0.0, "barrier less/equal zero not allowed");
        Ok(BiasedBarrierPathPricer {
            barrier_type,
            barrier,
            rebate,
            payoff: PlainVanillaPayoff::new(option_type, strike),
            discounts,
        })
    }
}

impl PathPricer<Path> for BiasedBarrierPathPricer {
    fn price(&self, path: &Path) -> Real {
        let n = path.length();
        debug_assert!(n > 1, "the path cannot be empty");

        let mut is_option_active =
            matches!(self.barrier_type, BarrierType::DownOut | BarrierType::UpOut);
        let mut knock_node = None;
        let mut asset_price = path.front();

        for i in 1..n {
            asset_price = path[i];
            let hit = match self.barrier_type {
                BarrierType::DownIn | BarrierType::DownOut => asset_price <= self.barrier,
                BarrierType::UpIn | BarrierType::UpOut => asset_price >= self.barrier,
            };
            if hit {
                is_option_active =
                    matches!(self.barrier_type, BarrierType::DownIn | BarrierType::UpIn);
                if knock_node.is_none() {
                    knock_node = Some(i);
                }
            }
        }

        barrier_payoff(
            is_option_active,
            self.barrier_type,
            asset_price,
            self.rebate,
            &self.payoff,
            &self.discounts,
            knock_node,
        )
    }
}

/// Monte Carlo pricing engine for barrier options (`mcbarrierengine.hpp:57`),
/// generic over the RNG policy `RNG`.
pub struct MCBarrierEngine<RNG> {
    base: BarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    time_steps: Option<Size>,
    time_steps_per_year: Option<Size>,
    required_samples: Option<Size>,
    max_samples: Option<Size>,
    required_tolerance: Option<Real>,
    is_biased: bool,
    brownian_bridge: bool,
    antithetic_variate: bool,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MCBarrierEngine<RNG> {
    /// Builds the engine (`mcbarrierengine.hpp:184-213`). Prefer
    /// [`MakeMcBarrierEngine`] for the validated construction path.
    ///
    /// # Errors
    ///
    /// Errors if neither `time_steps` nor `time_steps_per_year` is set, if both
    /// are set, or if either is `Some(0)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        time_steps: Option<Size>,
        time_steps_per_year: Option<Size>,
        brownian_bridge: bool,
        antithetic_variate: bool,
        required_samples: Option<Size>,
        required_tolerance: Option<Real>,
        max_samples: Option<Size>,
        is_biased: bool,
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

        let base =
            BarrierEngineBase::new(BarrierArguments::default(), InstrumentResults::default());
        base.register_with(process.observable());

        Ok(MCBarrierEngine {
            base,
            process,
            time_steps,
            time_steps_per_year,
            required_samples,
            max_samples,
            required_tolerance,
            is_biased,
            brownian_bridge,
            antithetic_variate,
            seed,
            _rng: PhantomData,
        })
    }

    fn time_grid(&self) -> QlResult<TimeGrid> {
        let Some(exercise) = &self.base.arguments().exercise else {
            fail!("no exercise given");
        };
        let residual = self.process.time(&exercise.last_date())?;
        if let Some(steps) = self.time_steps {
            TimeGrid::new(residual, steps)
        } else if let Some(per_year) = self.time_steps_per_year {
            let steps = (per_year as Real * residual) as Size;
            TimeGrid::new(residual, steps.max(1))
        } else {
            fail!("time steps not specified")
        }
    }

    fn path_generator(&self) -> QlResult<PathGenerator<RNG::RsgType>> {
        let grid = self.time_grid()?;
        let dimension = grid.size() - 1;
        let generator = RNG::make_sequence_generator(dimension, self.seed)?;
        PathGenerator::from_time_grid(
            Shared::clone(&self.process) as Shared<dyn StochasticProcess1D>,
            grid,
            generator,
            self.brownian_bridge,
        )
    }

    fn discounts(&self, grid: &TimeGrid) -> QlResult<Vec<DiscountFactor>> {
        let curve = self.process.risk_free_rate().current_link()?;
        let mut discounts = Vec::with_capacity(grid.size());
        for i in 0..grid.size() {
            discounts.push(curve.discount(grid[i], false)?);
        }
        Ok(discounts)
    }
}

impl<RNG: McRngTraits> AsObservable for MCBarrierEngine<RNG> {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl<RNG: McRngTraits> PricingEngine for MCBarrierEngine<RNG> {
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
        let payoff = args.payoff.expect("validated");
        let barrier_type = args.barrier_type.expect("validated");
        let barrier = args.barrier.expect("validated");
        let rebate = args.rebate.expect("validated");

        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");
        require!(!triggered(barrier_type, spot, barrier), "barrier touched");

        let grid = self.time_grid()?;
        let discounts = self.discounts(&grid)?;
        let generator = self.path_generator()?;

        let mean = if self.is_biased {
            let pricer = BiasedBarrierPathPricer::new(
                barrier_type,
                barrier,
                rebate,
                payoff.option_type(),
                payoff.strike(),
                discounts,
            )?;
            run_simulation::<RNG, _>(
                generator,
                pricer,
                self.antithetic_variate,
                self.required_tolerance,
                self.required_samples,
                self.max_samples,
            )?
        } else {
            let sequence_gen = RandomSequenceGenerator::with_seed(grid.size() - 1, 5)?;
            let pricer = BarrierPathPricer::new(
                barrier_type,
                barrier,
                rebate,
                payoff.option_type(),
                payoff.strike(),
                discounts,
                Shared::clone(&self.process) as Shared<dyn StochasticProcess1D>,
                sequence_gen,
            )?;
            run_simulation::<RNG, _>(
                generator,
                pricer,
                self.antithetic_variate,
                self.required_tolerance,
                self.required_samples,
                self.max_samples,
            )?
        };

        self.base.results_mut().value = Some(mean.0);
        self.base.results_mut().error_estimate = mean.1;
        Ok(())
    }
}

fn run_simulation<RNG, P>(
    generator: PathGenerator<RNG::RsgType>,
    path_pricer: P,
    antithetic_variate: bool,
    required_tolerance: Option<Real>,
    required_samples: Option<Size>,
    max_samples: Option<Size>,
) -> QlResult<(Real, Option<Real>)>
where
    RNG: McRngTraits,
    P: PathPricer<Path>,
{
    let mut simulation: McSimulation<PathGenerator<RNG::RsgType>, P> =
        McSimulation::new(antithetic_variate, false);
    simulation.calculate(
        generator,
        path_pricer,
        required_tolerance,
        required_samples,
        max_samples,
    )?;
    let mean = simulation.sample_accumulator()?.mean()?;
    let error = if RNG::ALLOWS_ERROR_ESTIMATE {
        Some(simulation.error_estimate()?)
    } else {
        None
    };
    Ok((mean, error))
}

/// Factory for [`MCBarrierEngine`] (`mcbarrierengine.hpp:116`).
pub struct MakeMcBarrierEngine<RNG> {
    process: Shared<GeneralizedBlackScholesProcess>,
    brownian_bridge: bool,
    antithetic: bool,
    biased: bool,
    steps: Option<Size>,
    steps_per_year: Option<Size>,
    samples: Option<Size>,
    max_samples: Option<Size>,
    tolerance: Option<Real>,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MakeMcBarrierEngine<RNG> {
    /// Starts a builder on the given Black–Scholes process
    /// (`mcbarrierengine.hpp:266-268`).
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        MakeMcBarrierEngine {
            process,
            brownian_bridge: false,
            antithetic: false,
            biased: false,
            steps: None,
            steps_per_year: None,
            samples: None,
            max_samples: None,
            tolerance: None,
            seed: 0,
            _rng: PhantomData,
        }
    }

    /// Sets the fixed number of time steps (`mcbarrierengine.hpp:272`).
    #[must_use]
    pub fn with_steps(mut self, steps: Size) -> Self {
        self.steps = Some(steps);
        self
    }

    /// Sets the number of time steps per year (`mcbarrierengine.hpp:279`).
    #[must_use]
    pub fn with_steps_per_year(mut self, steps: Size) -> Self {
        self.steps_per_year = Some(steps);
        self
    }

    /// Enables the Brownian-bridge increment construction
    /// (`mcbarrierengine.hpp:286`; default `true` in C++).
    #[must_use]
    pub fn with_brownian_bridge(mut self, brownian_bridge: bool) -> Self {
        self.brownian_bridge = brownian_bridge;
        self
    }

    /// Requests the antithetic-variate variance reduction
    /// (`mcbarrierengine.hpp:293`). Deferred: setting `true` makes
    /// [`build`](Self::build) return `Err`.
    #[must_use]
    pub fn with_antithetic_variate(mut self, antithetic: bool) -> Self {
        self.antithetic = antithetic;
        self
    }

    /// Sets the required number of samples (`mcbarrierengine.hpp:300`).
    #[must_use]
    pub fn with_samples(mut self, samples: Size) -> Self {
        self.samples = Some(samples);
        self
    }

    /// Sets the required absolute tolerance (`mcbarrierengine.hpp:308`).
    #[must_use]
    pub fn with_absolute_tolerance(mut self, tolerance: Real) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    /// Sets the maximum number of samples (`mcbarrierengine.hpp:319`).
    #[must_use]
    pub fn with_max_samples(mut self, samples: Size) -> Self {
        self.max_samples = Some(samples);
        self
    }

    /// Selects the node-only (biased) path pricer (`mcbarrierengine.hpp:326`).
    #[must_use]
    pub fn with_bias(mut self, biased: bool) -> Self {
        self.biased = biased;
        self
    }

    /// Sets the RNG seed (`mcbarrierengine.hpp:333`).
    #[must_use]
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Builds the configured [`MCBarrierEngine`]
    /// (`mcbarrierengine.hpp:341-355`).
    ///
    /// # Errors
    ///
    /// Errors if neither or both of `steps`/`steps_per_year` are set, if both
    /// `samples` and `tolerance` are set, if a tolerance is set on an RNG
    /// policy without an error estimate, or if the deferred antithetic variate
    /// is requested.
    pub fn build(self) -> QlResult<MCBarrierEngine<RNG>> {
        require!(
            self.steps.is_some() || self.steps_per_year.is_some(),
            "number of steps not given"
        );
        require!(
            self.steps.is_none() || self.steps_per_year.is_none(),
            "number of steps overspecified"
        );
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
        require!(!self.antithetic, "antithetic variate not yet supported");

        MCBarrierEngine::new(
            self.process,
            self.steps,
            self.steps_per_year,
            self.brownian_bridge,
            false,
            self.samples,
            self.tolerance,
            self.max_samples,
            self.biased,
            self.seed,
        )
    }
}

/// Attach a Monte Carlo barrier engine to an option.
pub fn set_mc_barrier_engine<RNG: McRngTraits + 'static>(
    option: &mut BarrierOption,
    engine: SharedMut<MCBarrierEngine<RNG>>,
) {
    option
        .base_mut()
        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::interestrate::Compounding;
    use crate::math::array::Array;
    use crate::math::randomnumbers::rngtraits::{LowDiscrepancy, PseudoRandom};
    use crate::methods::montecarlo::Path;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::nullcalendar::NullCalendar;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn flat_bs_process(
        spot: Real,
        q: Real,
        r: Real,
        vol: Real,
    ) -> Shared<GeneralizedBlackScholesProcess> {
        let dc = Actual360::new();
        let ref_date = today();
        shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            Handle::new(shared(FlatForward::with_rate(
                ref_date,
                q,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(FlatForward::with_rate(
                ref_date,
                r,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(BlackConstantVol::new(
                ref_date,
                Some(NullCalendar::new()),
                vol,
                dc,
            )) as Shared<dyn BlackVolTermStructure>),
        ))
    }

    fn two_step_path(values: [Real; 2]) -> Path {
        let grid = TimeGrid::new(1.0, 1).unwrap();
        Path::new(grid, Array::from(values)).unwrap()
    }

    #[test]
    fn biased_down_out_pays_terminal_when_nodes_stay_above_the_barrier() {
        let pricer = BiasedBarrierPathPricer::new(
            BarrierType::DownOut,
            90.0,
            0.0,
            OptionType::Call,
            100.0,
            vec![1.0, 0.95],
        )
        .unwrap();
        let path = two_step_path([100.0, 105.0]);
        let price = pricer.price(&path);
        assert!((price - 5.0 * 0.95).abs() < 1e-14);
    }

    #[test]
    fn biased_down_out_pays_rebate_when_a_node_crosses() {
        let pricer = BiasedBarrierPathPricer::new(
            BarrierType::DownOut,
            90.0,
            3.0,
            OptionType::Call,
            100.0,
            vec![1.0, 0.95],
        )
        .unwrap();
        let path = two_step_path([100.0, 85.0]);
        let price = pricer.price(&path);
        assert!((price - 3.0 * 0.95).abs() < 1e-14);
    }

    #[test]
    fn builder_rejects_missing_and_duplicate_steps() {
        let process = flat_bs_process(100.0, 0.02, 0.05, 0.20);
        assert!(
            MakeMcBarrierEngine::<PseudoRandom>::new(Shared::clone(&process))
                .with_samples(128)
                .build()
                .is_err()
        );
        assert!(
            MakeMcBarrierEngine::<PseudoRandom>::new(process)
                .with_steps(4)
                .with_steps_per_year(1)
                .with_samples(128)
                .build()
                .is_err()
        );
    }

    #[test]
    fn builder_rejects_antithetic_and_low_discrepancy_tolerance() {
        let process = flat_bs_process(100.0, 0.02, 0.05, 0.20);
        assert!(
            MakeMcBarrierEngine::<PseudoRandom>::new(Shared::clone(&process))
                .with_steps(1)
                .with_samples(128)
                .with_antithetic_variate(true)
                .build()
                .is_err()
        );
        assert!(
            MakeMcBarrierEngine::<LowDiscrepancy>::new(process)
                .with_steps(1)
                .with_absolute_tolerance(1e-3)
                .build()
                .is_err()
        );
    }

    #[test]
    fn zero_spot_and_triggered_barrier_are_rejected() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today() + 360));

        let mut option = BarrierOption::new(
            BarrierType::DownOut,
            90.0,
            PlainVanillaPayoff::new(OptionType::Call, 100.0),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        let engine =
            MakeMcBarrierEngine::<PseudoRandom>::new(flat_bs_process(0.0, 0.02, 0.05, 0.20))
                .with_steps(1)
                .with_samples(8)
                .build()
                .unwrap();
        set_mc_barrier_engine(&mut option, shared_mut(engine));
        let err = option.npv().unwrap_err();
        assert!(err.message().contains("negative or null underlying"));

        let mut option = BarrierOption::new(
            BarrierType::DownOut,
            90.0,
            PlainVanillaPayoff::new(OptionType::Call, 100.0),
            exercise,
            settings,
        )
        .unwrap();
        let engine =
            MakeMcBarrierEngine::<PseudoRandom>::new(flat_bs_process(85.0, 0.02, 0.05, 0.20))
                .with_steps(1)
                .with_samples(8)
                .build()
                .unwrap();
        set_mc_barrier_engine(&mut option, shared_mut(engine));
        let err = option.npv().unwrap_err();
        assert!(err.message().contains("barrier touched"));
    }

    #[test]
    fn babsiri_mc_matches_published_calls() {
        // QuantLib barrieroption.cpp `testBabsiriValues` MC arm:
        // MakeMCBarrierEngine<LowDiscrepancy>.withStepsPerYear(1)
        // .withBrownianBridge().withSamples(131071).withMaxSamples(1048575)
        // .withSeed(5); relative 2e-2.
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today() + 360));
        #[rustfmt::skip]
        let cases = [
            (BarrierType::DownIn, 0.10, 100.0,  90.0,  0.07187),
            (BarrierType::DownIn, 0.15, 100.0,  90.0,  0.60638),
            (BarrierType::DownIn, 0.20, 100.0,  90.0,  1.64005),
            (BarrierType::DownIn, 0.25, 100.0,  90.0,  2.98495),
            (BarrierType::DownIn, 0.30, 100.0,  90.0,  4.50952),
            (BarrierType::UpIn,   0.10, 100.0, 110.0,  4.79148),
            (BarrierType::UpIn,   0.15, 100.0, 110.0,  7.08268),
            (BarrierType::UpIn,   0.20, 100.0, 110.0,  9.11008),
            (BarrierType::UpIn,   0.25, 100.0, 110.0, 11.06148),
            (BarrierType::UpIn,   0.30, 100.0, 110.0, 12.98351),
        ];
        let mut max_rel = 0.0;
        let mut worst = String::new();
        for (barrier_type, vol, strike, barrier, expected) in cases {
            let mut option = BarrierOption::new(
                barrier_type,
                barrier,
                PlainVanillaPayoff::new(OptionType::Call, strike),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            let engine =
                MakeMcBarrierEngine::<LowDiscrepancy>::new(flat_bs_process(100.0, 0.02, 0.05, vol))
                    .with_steps_per_year(1)
                    .with_brownian_bridge(true)
                    .with_samples(131_071)
                    .with_max_samples(1_048_575)
                    .with_seed(5)
                    .build()
                    .unwrap();
            set_mc_barrier_engine(&mut option, shared_mut(engine));
            let calculated = option.npv().unwrap();
            let rel = (calculated - expected).abs() / expected;
            if rel > max_rel {
                max_rel = rel;
                worst = format!("{barrier_type:?} H={barrier} v={vol}: {calculated} vs {expected}");
            }
            assert!(
                rel <= 2.0e-2,
                "{barrier_type:?} H={barrier} v={vol}: {calculated} vs Babsiri {expected} \
                 (rel {rel})"
            );
        }
        eprintln!("babsiri mc max relative error {max_rel:.2e} at {worst}");
    }

    #[test]
    fn beaglehole_mc_matches_published_down_out_call() {
        // QuantLib barrieroption.cpp `testBeagleholeValues` MC arm:
        // same builder as Babsiri, seed 10, relative 1e-2.
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let mut option = BarrierOption::new(
            BarrierType::DownOut,
            45.0,
            PlainVanillaPayoff::new(OptionType::Call, 50.0),
            shared(EuropeanExercise::new(today() + 360)),
            settings,
        )
        .unwrap();
        let engine = MakeMcBarrierEngine::<LowDiscrepancy>::new(flat_bs_process(
            50.0,
            0.0,
            1.1_f64.ln(),
            0.50,
        ))
        .with_steps_per_year(1)
        .with_brownian_bridge(true)
        .with_samples(131_071)
        .with_max_samples(1_048_575)
        .with_seed(10)
        .build()
        .unwrap();
        set_mc_barrier_engine(&mut option, shared_mut(engine));
        let calculated = option.npv().unwrap();
        let expected = 5.477;
        let rel = (calculated - expected).abs() / expected;
        eprintln!("beaglehole mc relative error {rel:.2e} ({calculated} vs {expected})");
        assert!(
            rel <= 0.01,
            "DownOut K=50 H=45: {calculated} vs Beaglehole {expected} (rel {rel})"
        );
    }
}
