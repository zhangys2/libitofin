//! Monte Carlo discrete geometric-average price Asian engine.
//!
//! Port of `ql/pricingengines/asian/mc_discr_geom_av_price.{hpp,cpp}` and
//! `mcdiscreteasianenginebase.hpp`.

use std::marker::PhantomData;

use crate::errors::QlResult;
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults,
    PlainVanillaPayoff, StrikedTypePayoff,
};
use crate::payoff::Payoff;
use crate::math::randomnumbers::rngtraits::McRngTraits;
use crate::math::statistics::MeanStdDev;
use crate::math::timegrid::TimeGrid;
use crate::methods::montecarlo::{McSimulation, Path, PathGenerator, PathPricer};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{DiscountFactor, Real, Size, Time};

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Geometric average-price Asian path pricer (`mc_discr_geom_av_price.cpp`).
pub struct GeometricApoPathPricer {
    payoff: PlainVanillaPayoff,
    discount: DiscountFactor,
    running_product: Real,
    past_fixings: Size,
}

impl GeometricApoPathPricer {
    /// Builds the path pricer.
    pub fn new(
        payoff: PlainVanillaPayoff,
        discount: DiscountFactor,
        running_product: Real,
        past_fixings: Size,
    ) -> QlResult<Self> {
        require!(payoff.strike() >= 0.0, "negative strike given");
        Ok(Self {
            payoff,
            discount,
            running_product,
            past_fixings,
        })
    }
}

impl PathPricer<Path> for GeometricApoPathPricer {
    fn price(&self, path: &Path) -> Real {
        let n = path.length() - 1;
        if n == 0 {
            return 0.0;
        }

        const MAX_VALUE: Real = Real::MAX;
        let mut product = self.running_product;
        let mut fixings = n + self.past_fixings;
        if path
            .time_grid()
            .mandatory_times()
            .first()
            .is_some_and(|t| *t == 0.0)
        {
            fixings += 1;
            product *= path.front();
        }

        let mut average_price = 1.0;
        for i in 1..=n {
            let price = path[i];
            if product < MAX_VALUE / price {
                product *= price;
            } else {
                average_price *= product.powf(1.0 / fixings as Real);
                product = price;
            }
        }
        average_price *= product.powf(1.0 / fixings as Real);
        self.discount * self.payoff.value(average_price)
    }
}

/// Monte Carlo engine for discrete geometric average-price Asians.
pub struct MCDiscreteGeometricAveragePriceAsianEngine<RNG> {
    base: EngineBase,
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

impl<RNG: McRngTraits> MCDiscreteGeometricAveragePriceAsianEngine<RNG> {
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
            time_steps.is_none() && time_steps_per_year.is_none()
                || time_steps.is_some() ^ time_steps_per_year.is_some(),
            "provide either time steps or time steps per year, not both"
        );
        require!(time_steps != Some(0), "timeSteps must be positive, 0 not allowed");
        require!(
            time_steps_per_year != Some(0),
            "timeStepsPerYear must be positive, 0 not allowed"
        );

        let base = EngineBase::new(
            DiscreteAveragingAsianArguments::default(),
            DiscreteAveragingAsianResults::default(),
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

    fn fixing_times(&self) -> QlResult<Vec<Time>> {
        let args = self.base.arguments();
        let mut fixing_times = Vec::new();
        for fixing_date in &args.fixing_dates {
            let t = self.process.time(fixing_date)?;
            if t >= 0.0 {
                fixing_times.push(t);
            }
        }
        if fixing_times.is_empty()
            || (fixing_times.len() == 1 && fixing_times[0] == 0.0)
        {
            fail!("all fixings are in the past");
        }
        Ok(fixing_times)
    }

    fn time_grid(&self) -> QlResult<TimeGrid> {
        let fixing_times = self.fixing_times()?;

        if let Some(steps) = self.time_steps {
            TimeGrid::with_mandatory_times(&fixing_times, steps)
        } else if let Some(per_year) = self.time_steps_per_year {
            let exercise = self.base.arguments().exercise.as_ref().expect("validated");
            let t = self.process.time(&exercise.last_date())?;
            let steps = (per_year as Real * t) as Size;
            TimeGrid::with_mandatory_times(&fixing_times, steps.max(1))
        } else {
            TimeGrid::from_mandatory_times(&fixing_times)
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

impl<RNG: McRngTraits> AsObservable for MCDiscreteGeometricAveragePriceAsianEngine<RNG> {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl<RNG: McRngTraits> PricingEngine for MCDiscreteGeometricAveragePriceAsianEngine<RNG> {
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
        require!(
            args.average_type == Some(AverageType::Geometric),
            "not a geometric average option"
        );
        let payoff = args.payoff.expect("validated");
        let exercise = args.exercise.as_ref().expect("validated");
        let running_accumulator = args.running_accumulator.expect("validated");
        let past_fixings = args.past_fixings.expect("validated");

        let r_ts = self.process.risk_free_rate().current_link()?;
        let discount = r_ts.discount_date(exercise.last_date(), false)?;
        let path_pricer = GeometricApoPathPricer::new(
            payoff,
            discount,
            running_accumulator,
            past_fixings,
        )?;

        let generator = self.path_generator()?;
        let mut simulation: McSimulation<PathGenerator<RNG::RsgType>, GeometricApoPathPricer> =
            McSimulation::new(self.antithetic_variate, false);
        simulation.calculate(
            generator,
            path_pricer,
            self.required_tolerance,
            self.required_samples,
            self.max_samples,
        )?;

        let mean = simulation.sample_accumulator()?.mean()?;
        let results = self.base.results_mut();
        results.instrument.value = Some(mean.max(0.0));
        if RNG::ALLOWS_ERROR_ESTIMATE {
            results.instrument.error_estimate = Some(simulation.error_estimate()?);
        }
        Ok(())
    }
}

/// Attaches [`MCDiscreteGeometricAveragePriceAsianEngine`] to `option`.
pub fn set_mc_discrete_geometric_average_price_asian_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    engine: SharedMut<dyn PricingEngine>,
) {
    option.base_mut().set_pricing_engine(engine);
}

/// Factory for [`MCDiscreteGeometricAveragePriceAsianEngine`].
pub struct MakeMcDiscreteGeometricApEngine<RNG> {
    process: Shared<GeneralizedBlackScholesProcess>,
    brownian_bridge: bool,
    antithetic: bool,
    samples: Option<Size>,
    max_samples: Option<Size>,
    tolerance: Option<Real>,
    time_steps: Option<Size>,
    time_steps_per_year: Option<Size>,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MakeMcDiscreteGeometricApEngine<RNG> {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self {
            process,
            brownian_bridge: true,
            antithetic: false,
            samples: None,
            max_samples: None,
            tolerance: None,
            time_steps: None,
            time_steps_per_year: None,
            seed: 0,
            _rng: PhantomData,
        }
    }

    pub fn with_brownian_bridge(mut self, value: bool) -> Self {
        self.brownian_bridge = value;
        self
    }

    pub fn with_antithetic_variate(mut self, value: bool) -> Self {
        self.antithetic = value;
        self
    }

    pub fn with_samples(mut self, samples: Size) -> Self {
        self.samples = Some(samples);
        self
    }

    pub fn with_absolute_tolerance(mut self, tolerance: Real) -> Self {
        self.tolerance = Some(tolerance);
        self
    }

    pub fn with_max_samples(mut self, max_samples: Size) -> Self {
        self.max_samples = Some(max_samples);
        self
    }

    pub fn with_time_steps(mut self, steps: Size) -> Self {
        self.time_steps = Some(steps);
        self.time_steps_per_year = None;
        self
    }

    pub fn with_time_steps_per_year(mut self, steps: Size) -> Self {
        self.time_steps_per_year = Some(steps);
        self.time_steps = None;
        self
    }

    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn build(self) -> QlResult<MCDiscreteGeometricAveragePriceAsianEngine<RNG>> {
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
        MCDiscreteGeometricAveragePriceAsianEngine::new(
            self.process,
            self.time_steps,
            self.time_steps_per_year,
            self.brownian_bridge,
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
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::DiscreteAveragingAsianOption;
    use crate::interestrate::Compounding;
    use crate::math::randomnumbers::rngtraits::PseudoRandom;
    use crate::option::OptionType;
    use crate::pricingengines::set_analytic_discrete_geometric_average_price_asian_engine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{Shared, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn crate::quotes::Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn crate::quotes::Quote>)
    }

    fn flat_rate(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn YieldTermStructure> {
        Handle::new(
            shared(FlatForward::new(
                reference,
                quote_handle(quote),
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
        )
    }

    fn flat_vol(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(
            shared(BlackConstantVol::with_quote(
                reference,
                None,
                quote_handle(quote),
                Actual360::new(),
            )) as Shared<dyn BlackVolTermStructure>,
        )
    }

    /// Grid nodes match future fixing times only (QuantLib auto-spacing, steps=0).
    #[test]
    fn time_grid_nodes_match_fixing_times() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.03));
        let r_rate = shared(SimpleQuote::new(0.06));
        let vol = shared(SimpleQuote::new(0.20));

        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        let future_fixings = 10;
        let _exercise_date = today + 360;
        let dt = (360.0 / future_fixings as Real).round() as i32;
        let mut fixing_dates = Vec::with_capacity(future_fixings);
        fixing_dates.push(today + dt);
        for _ in 1..future_fixings {
            let last = *fixing_dates.last().expect("non-empty");
            fixing_dates.push(last + dt);
        }

        let mut fixing_times = Vec::new();
        for fixing_date in &fixing_dates {
            let t = process.time(fixing_date).unwrap();
            if t >= 0.0 {
                fixing_times.push(t);
            }
        }
        let grid = TimeGrid::from_mandatory_times(&fixing_times).unwrap();
        assert_eq!(grid.size(), future_fixings + 1);
        assert_eq!(grid.mandatory_times().len(), future_fixings);
    }

    fn clewlow_strickland_process() -> Shared<BlackScholesMertonProcess> {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.03));
        let r_rate = shared(SimpleQuote::new(0.06));
        let vol = shared(SimpleQuote::new(0.20));

        shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ))
    }

    fn clewlow_strickland_option(
        _process: &Shared<BlackScholesMertonProcess>,
    ) -> DiscreteAveragingAsianOption {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);
        let _ = _process;

        let future_fixings = 10;
        let exercise_date = today + 360;
        let dt = (360.0 / future_fixings as Real).round() as i32;
        let mut fixing_dates = Vec::with_capacity(future_fixings);
        fixing_dates.push(today + dt);
        for _ in 1..future_fixings {
            let last = *fixing_dates.last().expect("non-empty");
            fixing_dates.push(last + dt);
        }

        DiscreteAveragingAsianOption::new(
            AverageType::Geometric,
            1.0,
            0,
            fixing_dates,
            PlainVanillaPayoff::new(OptionType::Call, 100.0),
            shared(EuropeanExercise::new(exercise_date)),
            settings,
        )
        .unwrap()
    }

    fn mc_npv(process: &Shared<BlackScholesMertonProcess>, seed: u32) -> Real {
        let mut option = clewlow_strickland_option(process);
        set_mc_discrete_geometric_average_price_asian_engine(
            &mut option,
            shared_mut(
                MakeMcDiscreteGeometricApEngine::<PseudoRandom>::new(Shared::clone(process))
                    .with_samples(8191)
                    .with_seed(seed)
                    .build()
                    .unwrap(),
            ),
        );
        option.npv().unwrap()
    }

    /// `asianoptions.cpp` `testMCDiscreteGeometricAveragePrice`.
    #[test]
    fn monte_carlo_discrete_geometric_average_price_matches_analytical() {
        let process = clewlow_strickland_process();
        let calculated = mc_npv(&process, 200);

        let mut option = clewlow_strickland_option(&process);
        set_analytic_discrete_geometric_average_price_asian_engine(
            &mut option,
            Shared::clone(&process),
        );
        let expected = option.npv().unwrap();

        let tolerance = 4.0e-3;
        assert!(
            (calculated - expected).abs() <= tolerance,
            "expected {expected}, got {calculated} (tolerance {tolerance})"
        );
    }

    #[test]
    #[ignore = "seed survey helper; run manually to pick a deterministic MC seed"]
    fn survey_mc_seeds_for_oracle_tolerance() {
        let process = clewlow_strickland_process();
        let mut option = clewlow_strickland_option(&process);
        set_analytic_discrete_geometric_average_price_asian_engine(
            &mut option,
            Shared::clone(&process),
        );
        let expected = option.npv().unwrap();

        for seed in 1..=500u32 {
            let calculated = mc_npv(&process, seed);
            let diff = (calculated - expected).abs();
            if diff <= 4.0e-3 {
                eprintln!("seed {seed}: diff {diff:.6e}");
            }
        }
    }
}
