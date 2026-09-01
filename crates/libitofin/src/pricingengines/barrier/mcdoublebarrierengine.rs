//! Monte Carlo engine for double-barrier options.
//!
//! Port of `ql/experimental/barrieroption/mcdoublebarrierengine.{hpp,cpp}`:
//! a single-factor [`McSimulation`] over a Black–Scholes process with
//! node-sampled double-barrier knock-in/out logic
//! ([`DoubleBarrierPathPricer`]).

use std::marker::PhantomData;

use crate::errors::QlResult;
use crate::fail;
use crate::instrument::Instrument;
use crate::instrument::InstrumentResults;
use crate::instruments::{
    DoubleBarrierArguments, DoubleBarrierOption, DoubleBarrierType, PlainVanillaPayoff,
    StrikedTypePayoff, TypePayoff, double_barrier_triggered,
};
use crate::math::randomnumbers::rngtraits::McRngTraits;
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

type DoubleBarrierEngineBase = GenericEngine<DoubleBarrierArguments, InstrumentResults>;

/// Node-sampled double-barrier path pricer (`mcdoublebarrierengine.cpp:42-96`).
pub struct DoubleBarrierPathPricer {
    barrier_type: DoubleBarrierType,
    barrier_lo: Real,
    barrier_hi: Real,
    rebate: Real,
    payoff: PlainVanillaPayoff,
    discounts: Vec<DiscountFactor>,
}

impl DoubleBarrierPathPricer {
    /// Builds the path pricer (`mcdoublebarrierengine.cpp:25-40`).
    ///
    /// # Errors
    ///
    /// Errors if `strike` is negative or either barrier is not strictly positive.
    pub fn new(
        barrier_type: DoubleBarrierType,
        barrier_lo: Real,
        barrier_hi: Real,
        rebate: Real,
        option_type: OptionType,
        strike: Real,
        discounts: Vec<DiscountFactor>,
    ) -> QlResult<Self> {
        require!(strike >= 0.0, "strike less than zero not allowed");
        require!(barrier_lo > 0.0, "low barrier less/equal zero not allowed");
        require!(barrier_hi > 0.0, "high barrier less/equal zero not allowed");
        Ok(Self {
            barrier_type,
            barrier_lo,
            barrier_hi,
            rebate,
            payoff: PlainVanillaPayoff::new(option_type, strike),
            discounts,
        })
    }
}

impl PathPricer<Path> for DoubleBarrierPathPricer {
    fn price(&self, path: &Path) -> Real {
        let n = path.length();
        if n <= 1 {
            return 0.0;
        }

        let mut is_option_active;
        let mut knock_node: Option<Size> = None;
        let terminal_price = path.back();

        match self.barrier_type {
            DoubleBarrierType::KnockOut => {
                is_option_active = true;
                for i in 0..n - 1 {
                    let new_asset_price = path[i + 1];
                    if new_asset_price >= self.barrier_hi || new_asset_price <= self.barrier_lo {
                        is_option_active = false;
                        if knock_node.is_none() {
                            knock_node = Some(i + 1);
                        }
                        break;
                    }
                }
            }
            DoubleBarrierType::KnockIn => {
                is_option_active = false;
                for i in 0..n - 1 {
                    let new_asset_price = path[i + 1];
                    if new_asset_price >= self.barrier_hi || new_asset_price <= self.barrier_lo {
                        is_option_active = true;
                        if knock_node.is_none() {
                            knock_node = Some(i + 1);
                        }
                        break;
                    }
                }
            }
        }

        if is_option_active {
            self.payoff.value(terminal_price) * self.discounts[n - 1]
        } else {
            match self.barrier_type {
                DoubleBarrierType::KnockOut => {
                    let node = knock_node.expect("knock-out path must record knock node");
                    self.rebate * self.discounts[node]
                }
                DoubleBarrierType::KnockIn => self.rebate * self.discounts[n - 1],
            }
        }
    }
}

/// Monte Carlo pricing engine for double-barrier options.
pub struct MCDoubleBarrierEngine<RNG> {
    base: DoubleBarrierEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    time_steps: Option<Size>,
    time_steps_per_year: Option<Size>,
    required_samples: Option<Size>,
    max_samples: Option<Size>,
    required_tolerance: Option<Real>,
    brownian_bridge: bool,
    antithetic_variate: bool,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MCDoubleBarrierEngine<RNG> {
    /// Builds the engine. Prefer [`MakeMcDoubleBarrierEngine`] for validation.
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

        let base = DoubleBarrierEngineBase::new(
            DoubleBarrierArguments::default(),
            InstrumentResults::default(),
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

impl<RNG: McRngTraits> AsObservable for MCDoubleBarrierEngine<RNG> {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl<RNG: McRngTraits> PricingEngine for MCDoubleBarrierEngine<RNG> {
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
        let payoff = args.payoff.expect("validated");
        let barrier_type = args.barrier_type.expect("validated");
        let barrier_lo = args.barrier_lo.expect("validated");
        let barrier_hi = args.barrier_hi.expect("validated");
        let rebate = args.rebate.expect("validated");

        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");
        require!(
            !double_barrier_triggered(spot, barrier_lo, barrier_hi),
            "barrier(s) already touched"
        );

        let grid = self.time_grid()?;
        let discounts = self.discounts(&grid)?;
        let generator = self.path_generator()?;
        let pricer = DoubleBarrierPathPricer::new(
            barrier_type,
            barrier_lo,
            barrier_hi,
            rebate,
            payoff.option_type(),
            payoff.strike(),
            discounts,
        )?;

        let mean = run_simulation::<RNG, _>(
            generator,
            pricer,
            self.antithetic_variate,
            self.required_tolerance,
            self.required_samples,
            self.max_samples,
        )?;

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

/// Factory for [`MCDoubleBarrierEngine`].
pub struct MakeMcDoubleBarrierEngine<RNG> {
    process: Shared<GeneralizedBlackScholesProcess>,
    brownian_bridge: bool,
    antithetic: bool,
    steps: Option<Size>,
    steps_per_year: Option<Size>,
    samples: Option<Size>,
    max_samples: Option<Size>,
    tolerance: Option<Real>,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MakeMcDoubleBarrierEngine<RNG> {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self {
            process,
            brownian_bridge: false,
            antithetic: false,
            steps: None,
            steps_per_year: None,
            samples: None,
            max_samples: None,
            tolerance: None,
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
    pub fn with_brownian_bridge(mut self, brownian_bridge: bool) -> Self {
        self.brownian_bridge = brownian_bridge;
        self
    }

    #[must_use]
    pub fn with_antithetic_variate(mut self, antithetic: bool) -> Self {
        self.antithetic = antithetic;
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

    pub fn build(self) -> QlResult<MCDoubleBarrierEngine<RNG>> {
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

        MCDoubleBarrierEngine::new(
            self.process,
            self.steps,
            self.steps_per_year,
            self.brownian_bridge,
            self.antithetic,
            self.samples,
            self.tolerance,
            self.max_samples,
            self.seed,
        )
    }
}

/// Attach an [`MCDoubleBarrierEngine`] to a double-barrier option.
pub fn set_mc_double_barrier_engine<RNG: McRngTraits + 'static>(
    option: &mut DoubleBarrierOption,
    engine: SharedMut<MCDoubleBarrierEngine<RNG>>,
) {
    option
        .base_mut()
        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::interestrate::Compounding;
    use crate::math::randomnumbers::rngtraits::PseudoRandom;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{Shared, shared, shared_mut};
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::yields::FlatForward;
    use crate::time::calendars::target::Target;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::types::Real;

    fn oracle_market() -> (
        Shared<Settings<Date>>,
        Shared<BlackScholesMertonProcess>,
        Date,
        Real,
        Real,
        Real,
        Real,
    ) {
        let today = Date::new(15, Month::May, 1998);
        let settlement = Date::new(17, Month::May, 1998);
        let maturity = Date::new(17, Month::May, 1999);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);

        let underlying = 36.0;
        let strike = 40.0;
        let q = 0.0;
        let r = 0.06;
        let vol = 0.20;
        let dc = Actual365Fixed::new();

        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(underlying)) as Shared<dyn crate::quotes::Quote>),
            Handle::new(shared(FlatForward::with_rate(
                settlement,
                q,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            ))
                as Shared<
                    dyn crate::termstructures::yieldtermstructure::YieldTermStructure,
                >),
            Handle::new(shared(FlatForward::with_rate(
                settlement,
                r,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            ))
                as Shared<
                    dyn crate::termstructures::yieldtermstructure::YieldTermStructure,
                >),
            Handle::new(shared(BlackConstantVol::new(
                settlement,
                Some(Target::new()),
                vol,
                dc,
            ))
                as Shared<
                    dyn crate::termstructures::volatility::BlackVolTermStructure,
                >),
        ));

        (
            settings,
            process,
            maturity,
            underlying,
            strike,
            underlying * 0.9,
            underlying * 1.1,
        )
    }

    /// `doublebarrieroption.cpp` `testMonteCarloDoubleBarrierWithAnalytical`.
    #[test]
    fn monte_carlo_double_barrier_matches_analytical() {
        let (settings, process, maturity, _, strike, barrier_lo, barrier_hi) = oracle_market();
        let payoff = PlainVanillaPayoff::new(OptionType::Put, strike);
        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(maturity));

        let analytic = |barrier_type: DoubleBarrierType| -> Real {
            let mut option = DoubleBarrierOption::new(
                barrier_type,
                barrier_lo,
                barrier_hi,
                0.0,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            crate::pricingengines::set_analytic_double_barrier_engine(
                &mut option,
                Shared::clone(&process),
            );
            option.npv().unwrap()
        };

        // Knock-in: relative diff <= 1%
        let ki_analytic = analytic(DoubleBarrierType::KnockIn);
        let mut ki = DoubleBarrierOption::new(
            DoubleBarrierType::KnockIn,
            barrier_lo,
            barrier_hi,
            0.0,
            PlainVanillaPayoff::new(OptionType::Put, strike),
            exercise.clone(),
            Shared::clone(&settings),
        )
        .unwrap();
        set_mc_double_barrier_engine(
            &mut ki,
            shared_mut(
                MakeMcDoubleBarrierEngine::<PseudoRandom>::new(Shared::clone(&process))
                    .with_steps(5000)
                    .with_antithetic_variate(true)
                    .with_absolute_tolerance(0.5)
                    .with_seed(1)
                    .build()
                    .unwrap(),
            ),
        );
        let ki_mc = ki.npv().unwrap();
        let ki_rel = (ki_analytic - ki_mc).abs() / ki_analytic;
        assert!(
            ki_rel <= 0.01,
            "KnockIn analytic {ki_analytic} vs MC {ki_mc} (relative {ki_rel})"
        );

        // Knock-out: absolute diff <= 0.01
        let ko_analytic = analytic(DoubleBarrierType::KnockOut);
        let mut ko = DoubleBarrierOption::new(
            DoubleBarrierType::KnockOut,
            barrier_lo,
            barrier_hi,
            0.0,
            PlainVanillaPayoff::new(OptionType::Put, strike),
            exercise,
            settings,
        )
        .unwrap();
        set_mc_double_barrier_engine(
            &mut ko,
            shared_mut(
                MakeMcDoubleBarrierEngine::<PseudoRandom>::new(process)
                    .with_steps(5000)
                    .with_antithetic_variate(true)
                    .with_absolute_tolerance(0.01)
                    .with_seed(10)
                    .build()
                    .unwrap(),
            ),
        );
        let ko_mc = ko.npv().unwrap();
        let ko_abs = (ko_analytic - ko_mc).abs();
        assert!(
            ko_abs <= 0.01,
            "KnockOut analytic {ko_analytic} vs MC {ko_mc} (absolute {ko_abs})"
        );
    }
}
