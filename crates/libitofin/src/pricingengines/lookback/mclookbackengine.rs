//! Monte Carlo engine for continuous lookback options.
//!
//! Port of `ql/pricingengines/lookback/mclookbackengine.{hpp,cpp}`.

use std::marker::PhantomData;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{
    ContinuousFixedLookbackArguments, ContinuousFixedLookbackResults,
    ContinuousFloatingLookbackArguments, ContinuousFloatingLookbackResults,
    ContinuousPartialFixedLookbackArguments, ContinuousPartialFixedLookbackResults,
    ContinuousPartialFloatingLookbackArguments, ContinuousPartialFloatingLookbackResults,
    FloatingTypePayoff, PlainVanillaPayoff, StrikedTypePayoff, TypePayoff,
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
use crate::types::{DiscountFactor, Real, Size, Time};

/// Fixed-strike lookback path pricer (`mclookbackengine.cpp:25-165`).
pub struct LookbackFixedPathPricer {
    payoff: PlainVanillaPayoff,
    discount: DiscountFactor,
}

impl LookbackFixedPathPricer {
    pub fn new(option_type: OptionType, strike: Real, discount: DiscountFactor) -> QlResult<Self> {
        require!(strike >= 0.0, "strike less than zero not allowed");
        Ok(Self {
            payoff: PlainVanillaPayoff::new(option_type, strike),
            discount,
        })
    }
}

impl PathPricer<Path> for LookbackFixedPathPricer {
    fn price(&self, path: &Path) -> Real {
        assert!(!path.empty(), "the path cannot be empty");
        let values = path.values();
        let underlying = match self.payoff.option_type() {
            OptionType::Put => values[1..].iter().copied().fold(f64::INFINITY, f64::min),
            OptionType::Call => values[1..]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        };
        self.payoff.value(underlying) * self.discount
    }
}

/// Partial fixed-strike lookback path pricer (`mclookbackengine.cpp:37-196`).
pub struct LookbackPartialFixedPathPricer {
    lookback_start: Time,
    payoff: PlainVanillaPayoff,
    discount: DiscountFactor,
}

impl LookbackPartialFixedPathPricer {
    pub fn new(
        lookback_start: Time,
        option_type: OptionType,
        strike: Real,
        discount: DiscountFactor,
    ) -> QlResult<Self> {
        require!(strike >= 0.0, "strike less than zero not allowed");
        Ok(Self {
            lookback_start,
            payoff: PlainVanillaPayoff::new(option_type, strike),
            discount,
        })
    }
}

impl PathPricer<Path> for LookbackPartialFixedPathPricer {
    fn price(&self, path: &Path) -> Real {
        assert!(!path.empty(), "the path cannot be empty");
        let start_index = path.time_grid().closest_index(self.lookback_start);
        let values = path.values();
        let slice = &values[(start_index + 1)..];
        let underlying = match self.payoff.option_type() {
            OptionType::Put => slice.iter().copied().fold(f64::INFINITY, f64::min),
            OptionType::Call => slice.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        };
        self.payoff.value(underlying) * self.discount
    }
}

/// Floating-strike lookback path pricer (`mclookbackengine.cpp:51-221`).
pub struct LookbackFloatingPathPricer {
    payoff: FloatingTypePayoff,
    discount: DiscountFactor,
}

impl LookbackFloatingPathPricer {
    pub fn new(option_type: OptionType, discount: DiscountFactor) -> Self {
        Self {
            payoff: FloatingTypePayoff::new(option_type),
            discount,
        }
    }
}

impl PathPricer<Path> for LookbackFloatingPathPricer {
    fn price(&self, path: &Path) -> Real {
        assert!(!path.empty(), "the path cannot be empty");
        let values = path.values();
        let terminal_price = path.back();
        let strike = match self.payoff.option_type() {
            OptionType::Call => values[1..].iter().copied().fold(f64::INFINITY, f64::min),
            OptionType::Put => values[1..]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        };
        self.payoff.value_with_strike(terminal_price, strike) * self.discount
    }
}

/// Partial floating-strike lookback path pricer (`mclookbackengine.cpp:62-250`).
pub struct LookbackPartialFloatingPathPricer {
    lookback_end: Time,
    payoff: FloatingTypePayoff,
    discount: DiscountFactor,
}

impl LookbackPartialFloatingPathPricer {
    pub fn new(lookback_end: Time, option_type: OptionType, discount: DiscountFactor) -> Self {
        Self {
            lookback_end,
            payoff: FloatingTypePayoff::new(option_type),
            discount,
        }
    }
}

impl PathPricer<Path> for LookbackPartialFloatingPathPricer {
    fn price(&self, path: &Path) -> Real {
        assert!(!path.empty(), "the path cannot be empty");
        let end_index = path.time_grid().closest_index(self.lookback_end);
        let values = path.values();
        let slice = &values[1..=end_index];
        let terminal_price = path.back();
        let strike = match self.payoff.option_type() {
            OptionType::Call => slice.iter().copied().fold(f64::INFINITY, f64::min),
            OptionType::Put => slice.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        };
        self.payoff.value_with_strike(terminal_price, strike) * self.discount
    }
}

enum LookbackMcPathPricer {
    Fixed(LookbackFixedPathPricer),
    PartialFixed(LookbackPartialFixedPathPricer),
    Floating(LookbackFloatingPathPricer),
    PartialFloating(LookbackPartialFloatingPathPricer),
}

impl PathPricer<Path> for LookbackMcPathPricer {
    fn price(&self, path: &Path) -> Real {
        match self {
            Self::Fixed(p) => p.price(path),
            Self::PartialFixed(p) => p.price(path),
            Self::Floating(p) => p.price(path),
            Self::PartialFloating(p) => p.price(path),
        }
    }
}

trait McLookbackArguments: Arguments {
    fn exercise(&self) -> &Shared<dyn Exercise>;

    fn build_path_pricer(
        &self,
        process: &GeneralizedBlackScholesProcess,
        discount: DiscountFactor,
    ) -> QlResult<LookbackMcPathPricer>;
}

trait McLookbackResults: Results {
    fn set_mc_value(&mut self, value: Real, error_estimate: Option<Real>);
}

impl McLookbackArguments for ContinuousFixedLookbackArguments {
    fn exercise(&self) -> &Shared<dyn Exercise> {
        self.exercise.as_ref().expect("validated")
    }

    fn build_path_pricer(
        &self,
        _process: &GeneralizedBlackScholesProcess,
        discount: DiscountFactor,
    ) -> QlResult<LookbackMcPathPricer> {
        let payoff = self.payoff.expect("validated");
        Ok(LookbackMcPathPricer::Fixed(LookbackFixedPathPricer::new(
            payoff.option_type(),
            payoff.strike(),
            discount,
        )?))
    }
}

impl McLookbackArguments for ContinuousPartialFixedLookbackArguments {
    fn exercise(&self) -> &Shared<dyn Exercise> {
        self.exercise.as_ref().expect("validated")
    }

    fn build_path_pricer(
        &self,
        process: &GeneralizedBlackScholesProcess,
        discount: DiscountFactor,
    ) -> QlResult<LookbackMcPathPricer> {
        let payoff = self.payoff.expect("validated");
        let lookback_start = self.lookback_period_start.expect("validated");
        let lookback_start_time = process.time(&lookback_start)?;
        Ok(LookbackMcPathPricer::PartialFixed(
            LookbackPartialFixedPathPricer::new(
                lookback_start_time,
                payoff.option_type(),
                payoff.strike(),
                discount,
            )?,
        ))
    }
}

impl McLookbackArguments for ContinuousFloatingLookbackArguments {
    fn exercise(&self) -> &Shared<dyn Exercise> {
        self.exercise.as_ref().expect("validated")
    }

    fn build_path_pricer(
        &self,
        _process: &GeneralizedBlackScholesProcess,
        discount: DiscountFactor,
    ) -> QlResult<LookbackMcPathPricer> {
        let payoff = self.payoff.expect("validated");
        Ok(LookbackMcPathPricer::Floating(
            LookbackFloatingPathPricer::new(payoff.option_type(), discount),
        ))
    }
}

impl McLookbackArguments for ContinuousPartialFloatingLookbackArguments {
    fn exercise(&self) -> &Shared<dyn Exercise> {
        self.exercise.as_ref().expect("validated")
    }

    fn build_path_pricer(
        &self,
        process: &GeneralizedBlackScholesProcess,
        discount: DiscountFactor,
    ) -> QlResult<LookbackMcPathPricer> {
        let payoff = self.payoff.expect("validated");
        let lookback_end = self.lookback_period_end.expect("validated");
        let lookback_end_time = process.time(&lookback_end)?;
        Ok(LookbackMcPathPricer::PartialFloating(
            LookbackPartialFloatingPathPricer::new(
                lookback_end_time,
                payoff.option_type(),
                discount,
            ),
        ))
    }
}

impl McLookbackResults for ContinuousFixedLookbackResults {
    fn set_mc_value(&mut self, value: Real, error_estimate: Option<Real>) {
        self.instrument.value = Some(value);
        self.instrument.error_estimate = error_estimate;
    }
}

impl McLookbackResults for ContinuousPartialFixedLookbackResults {
    fn set_mc_value(&mut self, value: Real, error_estimate: Option<Real>) {
        self.instrument.value = Some(value);
        self.instrument.error_estimate = error_estimate;
    }
}

impl McLookbackResults for ContinuousFloatingLookbackResults {
    fn set_mc_value(&mut self, value: Real, error_estimate: Option<Real>) {
        self.instrument.value = Some(value);
        self.instrument.error_estimate = error_estimate;
    }
}

impl McLookbackResults for ContinuousPartialFloatingLookbackResults {
    fn set_mc_value(&mut self, value: Real, error_estimate: Option<Real>) {
        self.instrument.value = Some(value);
        self.instrument.error_estimate = error_estimate;
    }
}

/// Monte Carlo pricing engine for continuous lookback options.
#[allow(private_bounds)]
pub struct MCLookbackEngine<Args, Res, RNG> {
    base: GenericEngine<Args, Res>,
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

#[allow(private_bounds)]
impl<Args, Res, RNG> MCLookbackEngine<Args, Res, RNG>
where
    Args: McLookbackArguments + Default,
    Res: McLookbackResults + Default,
    RNG: McRngTraits,
{
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

        let base = GenericEngine::new(Args::default(), Res::default());
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
        let exercise = self.base.arguments().exercise();
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
}

#[allow(private_bounds)]
impl<Args, Res, RNG> AsObservable for MCLookbackEngine<Args, Res, RNG>
where
    Args: McLookbackArguments + Default,
    Res: McLookbackResults + Default,
    RNG: McRngTraits,
{
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

#[allow(private_bounds)]
impl<Args, Res, RNG> PricingEngine for MCLookbackEngine<Args, Res, RNG>
where
    Args: McLookbackArguments + Default,
    Res: McLookbackResults + Default,
    RNG: McRngTraits,
{
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
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");

        let grid = self.time_grid()?;
        let maturity = grid.back().expect("non-empty time grid");
        let discount = self
            .process
            .risk_free_rate()
            .current_link()?
            .discount(maturity, false)?;
        let args = self.base.arguments();
        let path_pricer = args.build_path_pricer(&self.process, discount)?;
        let generator = self.path_generator()?;

        let (mean, error) = run_simulation::<RNG, _>(
            generator,
            path_pricer,
            self.antithetic_variate,
            self.required_tolerance,
            self.required_samples,
            self.max_samples,
        )?;

        self.base.results_mut().set_mc_value(mean, error);
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

pub type McContinuousFixedLookbackEngine<RNG> =
    MCLookbackEngine<ContinuousFixedLookbackArguments, ContinuousFixedLookbackResults, RNG>;
pub type McContinuousPartialFixedLookbackEngine<RNG> = MCLookbackEngine<
    ContinuousPartialFixedLookbackArguments,
    ContinuousPartialFixedLookbackResults,
    RNG,
>;
pub type McContinuousFloatingLookbackEngine<RNG> =
    MCLookbackEngine<ContinuousFloatingLookbackArguments, ContinuousFloatingLookbackResults, RNG>;
pub type McContinuousPartialFloatingLookbackEngine<RNG> = MCLookbackEngine<
    ContinuousPartialFloatingLookbackArguments,
    ContinuousPartialFloatingLookbackResults,
    RNG,
>;

/// Factory for [`MCLookbackEngine`].
#[allow(private_bounds)]
pub struct MakeMcLookbackEngine<Args, Res, RNG> {
    process: Shared<GeneralizedBlackScholesProcess>,
    brownian_bridge: bool,
    antithetic: bool,
    steps: Option<Size>,
    steps_per_year: Option<Size>,
    samples: Option<Size>,
    max_samples: Option<Size>,
    tolerance: Option<Real>,
    seed: u32,
    _args: PhantomData<Args>,
    _results: PhantomData<Res>,
    _rng: PhantomData<RNG>,
}

#[allow(private_bounds)]
impl<Args, Res, RNG> MakeMcLookbackEngine<Args, Res, RNG>
where
    Args: McLookbackArguments + Default,
    Res: McLookbackResults + Default,
    RNG: McRngTraits,
{
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
            _args: PhantomData,
            _results: PhantomData,
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

    pub fn build(self) -> QlResult<MCLookbackEngine<Args, Res, RNG>> {
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

        MCLookbackEngine::new(
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

pub fn set_mc_continuous_fixed_lookback_engine<RNG: McRngTraits + 'static>(
    option: &mut crate::instruments::ContinuousFixedLookbackOption,
    engine: SharedMut<McContinuousFixedLookbackEngine<RNG>>,
) {
    option
        .base_mut()
        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

pub fn set_mc_continuous_partial_fixed_lookback_engine<RNG: McRngTraits + 'static>(
    option: &mut crate::instruments::ContinuousPartialFixedLookbackOption,
    engine: SharedMut<McContinuousPartialFixedLookbackEngine<RNG>>,
) {
    option
        .base_mut()
        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

pub fn set_mc_continuous_floating_lookback_engine<RNG: McRngTraits + 'static>(
    option: &mut crate::instruments::ContinuousFloatingLookbackOption,
    engine: SharedMut<McContinuousFloatingLookbackEngine<RNG>>,
) {
    option
        .base_mut()
        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

pub fn set_mc_continuous_partial_floating_lookback_engine<RNG: McRngTraits + 'static>(
    option: &mut crate::instruments::ContinuousPartialFloatingLookbackOption,
    engine: SharedMut<McContinuousPartialFloatingLookbackEngine<RNG>>,
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
    use crate::instruments::{
        ContinuousFixedLookbackOption, ContinuousFloatingLookbackOption,
        ContinuousPartialFixedLookbackOption, ContinuousPartialFloatingLookbackOption,
    };
    use crate::interestrate::Compounding;
    use crate::math::randomnumbers::rngtraits::PseudoRandom;
    use crate::pricingengines::lookback::{
        set_analytic_continuous_fixed_lookback_engine,
        set_analytic_continuous_floating_lookback_engine,
        set_analytic_continuous_partial_fixed_lookback_engine,
        set_analytic_continuous_partial_floating_lookback_engine,
    };
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::types::Time;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn flat_rate(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            reference,
            quote_handle(quote),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(
            reference,
            None,
            quote_handle(quote),
            Actual360::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    fn time_to_days(t: Time) -> i32 {
        (t * 360.0).round() as i32
    }

    fn mc_engine<Args, Res>() -> SharedMut<MCLookbackEngine<Args, Res, PseudoRandom>>
    where
        Args: McLookbackArguments + Default,
        Res: McLookbackResults + Default,
    {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.06));
        let vol = shared(SimpleQuote::new(0.1));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        shared_mut(
            MakeMcLookbackEngine::<Args, Res, PseudoRandom>::new(process)
                .with_steps(2000)
                .with_antithetic_variate(true)
                .with_seed(1)
                .with_absolute_tolerance(0.1)
                .build()
                .unwrap(),
        )
    }

    /// `lookbackoptions.cpp` `testMonteCarloLookback`.
    #[test]
    fn monte_carlo_lookback_matches_analytic() {
        let tolerance = 0.1;
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let strike = 90.0;
        let t = 1.0;
        let t1 = 0.25;
        let minmax = 100.0;
        let lambda = 1.0;

        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.06));
        let vol = shared(SimpleQuote::new(0.1));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(today + time_to_days(t)));
        let lookback_start = today + time_to_days(t1);
        let lookback_end = today + time_to_days(t1);

        for option_type in [OptionType::Call, OptionType::Put] {
            let payoff = PlainVanillaPayoff::new(option_type, strike);
            let floating_payoff = FloatingTypePayoff::new(option_type);

            // Partial fixed
            let mut partial_fixed = ContinuousPartialFixedLookbackOption::new(
                lookback_start,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_continuous_partial_fixed_lookback_engine(
                &mut partial_fixed,
                Shared::clone(&process),
            );
            let analytical = partial_fixed.npv().unwrap();
            set_mc_continuous_partial_fixed_lookback_engine(
                &mut partial_fixed,
                mc_engine::<
                    ContinuousPartialFixedLookbackArguments,
                    ContinuousPartialFixedLookbackResults,
                >(),
            );
            let monte_carlo = partial_fixed.npv().unwrap();
            assert!(
                (analytical - monte_carlo).abs() <= tolerance,
                "Partial Fixed {option_type:?}: analytic {analytical} vs MC {monte_carlo}"
            );

            // Fixed
            let mut fixed = ContinuousFixedLookbackOption::new(
                minmax,
                payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_continuous_fixed_lookback_engine(&mut fixed, Shared::clone(&process));
            let analytical = fixed.npv().unwrap();
            set_mc_continuous_fixed_lookback_engine(
                &mut fixed,
                mc_engine::<ContinuousFixedLookbackArguments, ContinuousFixedLookbackResults>(),
            );
            let monte_carlo = fixed.npv().unwrap();
            assert!(
                (analytical - monte_carlo).abs() <= tolerance,
                "Fixed {option_type:?}: analytic {analytical} vs MC {monte_carlo}"
            );

            // Partial floating
            let mut partial_floating = ContinuousPartialFloatingLookbackOption::new(
                minmax,
                lambda,
                lookback_end,
                floating_payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_continuous_partial_floating_lookback_engine(
                &mut partial_floating,
                Shared::clone(&process),
            );
            let analytical = partial_floating.npv().unwrap();
            set_mc_continuous_partial_floating_lookback_engine(
                &mut partial_floating,
                mc_engine::<
                    ContinuousPartialFloatingLookbackArguments,
                    ContinuousPartialFloatingLookbackResults,
                >(),
            );
            let monte_carlo = partial_floating.npv().unwrap();
            assert!(
                (analytical - monte_carlo).abs() <= tolerance,
                "Partial Floating {option_type:?}: analytic {analytical} vs MC {monte_carlo}"
            );

            // Floating
            let mut floating = ContinuousFloatingLookbackOption::new(
                minmax,
                floating_payoff,
                Shared::clone(&exercise),
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_continuous_floating_lookback_engine(
                &mut floating,
                Shared::clone(&process),
            );
            let analytical = floating.npv().unwrap();
            set_mc_continuous_floating_lookback_engine(
                &mut floating,
                mc_engine::<ContinuousFloatingLookbackArguments, ContinuousFloatingLookbackResults>(
                ),
            );
            let monte_carlo = floating.npv().unwrap();
            assert!(
                (analytical - monte_carlo).abs() <= tolerance,
                "Floating {option_type:?}: analytic {analytical} vs MC {monte_carlo}"
            );
        }
    }
}
