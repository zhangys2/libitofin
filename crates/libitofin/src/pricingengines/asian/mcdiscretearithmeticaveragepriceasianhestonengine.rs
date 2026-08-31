//! Monte Carlo discrete arithmetic-average price Asian engine under Heston.
//!
//! Port of `ql/pricingengines/asian/mc_discr_arith_av_price_heston.{hpp,cpp}` and
//! the multi-factor branch of `mcdiscreteasianenginebase.hpp`.
//!
//! Control variate: analytic discrete geometric Heston Asian
//! ([`AnalyticDiscreteGeometricAveragePriceAsianHestonEngine`]) with geometric
//! Heston path pricer.

use std::any::Any;
use std::marker::PhantomData;

use crate::errors::QlResult;
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults,
    PlainVanillaPayoff, StrikedTypePayoff, TypePayoff,
};
use crate::math::randomnumbers::rngtraits::McRngTraits;
use crate::math::statistics::MeanStdDev;
use crate::math::timegrid::TimeGrid;
use crate::methods::montecarlo::{McSimulation, MultiPath, MultiPathGenerator, PathPricer};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::HestonProcess;
use crate::require;
use crate::shared::{Shared, SharedMut};
use crate::stochasticprocess::StochasticProcess;
use crate::types::{DiscountFactor, Real, Size, Time};

use super::{
    AnalyticDiscreteGeometricAveragePriceAsianHestonEngine, GeometricApoHestonPathPricer,
};

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Arithmetic average-price Asian path pricer under Heston
/// (`mc_discr_arith_av_price_heston.cpp`).
pub struct ArithmeticApoHestonPathPricer {
    payoff: PlainVanillaPayoff,
    discount: DiscountFactor,
    fixing_indices: Vec<Size>,
    running_sum: Real,
    past_fixings: Size,
}

impl ArithmeticApoHestonPathPricer {
    /// Builds the path pricer.
    pub fn new(
        option_type: OptionType,
        strike: Real,
        discount: DiscountFactor,
        fixing_indices: Vec<Size>,
        running_sum: Real,
        past_fixings: Size,
    ) -> QlResult<Self> {
        require!(strike >= 0.0, "strike less than zero not allowed");
        Ok(Self {
            payoff: PlainVanillaPayoff::new(option_type, strike),
            discount,
            fixing_indices,
            running_sum,
            past_fixings,
        })
    }
}

impl PathPricer<MultiPath> for ArithmeticApoHestonPathPricer {
    fn price(&self, multi_path: &MultiPath) -> Real {
        let path = &multi_path[0];
        let n = multi_path.path_size();
        if n == 0 {
            return 0.0;
        }

        let mut sum = self.running_sum;
        let fixings = self.past_fixings + self.fixing_indices.len();
        for &idx in &self.fixing_indices {
            sum += path[idx];
        }
        let average_price = sum / fixings as Real;
        self.discount * self.payoff.value(average_price)
    }
}

/// Monte Carlo engine for discrete arithmetic average-price Asians under Heston.
pub struct MCDiscreteArithmeticAveragePriceAsianHestonEngine<RNG> {
    base: EngineBase,
    process: Shared<HestonProcess>,
    time_steps: Option<Size>,
    time_steps_per_year: Option<Size>,
    required_samples: Option<Size>,
    max_samples: Option<Size>,
    required_tolerance: Option<Real>,
    antithetic_variate: bool,
    control_variate: bool,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MCDiscreteArithmeticAveragePriceAsianHestonEngine<RNG> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        process: Shared<HestonProcess>,
        antithetic_variate: bool,
        required_samples: Option<Size>,
        required_tolerance: Option<Real>,
        max_samples: Option<Size>,
        seed: u32,
        time_steps: Option<Size>,
        time_steps_per_year: Option<Size>,
        control_variate: bool,
    ) -> QlResult<Self> {
        require!(
            time_steps.is_none() || time_steps_per_year.is_none(),
            "both time steps and time steps per year were provided"
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
            antithetic_variate,
            control_variate,
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
        if fixing_times.is_empty() || (fixing_times.len() == 1 && fixing_times[0] == 0.0) {
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

    fn control_variate_value(&self) -> QlResult<Real> {
        let mut control =
            AnalyticDiscreteGeometricAveragePriceAsianHestonEngine::new(Shared::clone(
                &self.process,
            ))?;
        {
            let src = self.base.arguments();
            let Some(dst) = (control.arguments_mut() as &mut dyn Any)
                .downcast_mut::<DiscreteAveragingAsianArguments>()
            else {
                fail!("wrong argument type for control engine");
            };
            dst.average_type = src.average_type;
            dst.running_accumulator = src.running_accumulator;
            dst.past_fixings = src.past_fixings;
            dst.fixing_dates = src.fixing_dates.clone();
            dst.payoff = src.payoff;
            dst.exercise = src.exercise.clone();
        }
        control.calculate()?;
        let Some(results) = control.results().as_instrument_results() else {
            fail!("no results returned from control pricing engine");
        };
        let Some(value) = results.value else {
            fail!("engine does not provide control-variation price");
        };
        Ok(value)
    }
}

impl<RNG: McRngTraits> AsObservable for MCDiscreteArithmeticAveragePriceAsianHestonEngine<RNG> {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl<RNG: McRngTraits> PricingEngine for MCDiscreteArithmeticAveragePriceAsianHestonEngine<RNG> {
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
            args.average_type == Some(AverageType::Arithmetic),
            "not an arithmetic average option"
        );
        let payoff = args.payoff.expect("validated");
        let exercise = args.exercise.as_ref().expect("validated");
        let running_accumulator = args.running_accumulator.expect("validated");
        let past_fixings = args.past_fixings.expect("validated");

        let grid = self.time_grid()?;
        let fixing_indices: Vec<Size> = grid
            .mandatory_times()
            .iter()
            .map(|&t| grid.closest_index(t))
            .collect();

        let r_ts = self.process.risk_free_rate().current_link()?;
        let discount = r_ts.discount_date(exercise.last_date(), false)?;

        let dimensions = self.process.factors() * (grid.size() - 1);
        let generator = RNG::make_sequence_generator(dimensions, self.seed)?;
        let mpg = MultiPathGenerator::new(
            Shared::clone(&self.process) as Shared<dyn StochasticProcess>,
            grid,
            generator,
            false,
        )?;

        let path_pricer = ArithmeticApoHestonPathPricer::new(
            payoff.option_type(),
            payoff.strike(),
            discount,
            fixing_indices.clone(),
            running_accumulator,
            past_fixings,
        )?;

        let mut simulation = McSimulation::<
            MultiPathGenerator<RNG::RsgType>,
            ArithmeticApoHestonPathPricer,
        >::new(self.antithetic_variate, self.control_variate);

        if self.control_variate {
            // QL GeometricAPOHestonPathPricer defaults: runningProduct=1, pastFixings=0
            // (seasoned CV deferred until analytic seasoning is complete).
            let control_pricer = GeometricApoHestonPathPricer::new(
                payoff.option_type(),
                payoff.strike(),
                discount,
                fixing_indices,
                1.0,
                0,
            )?;
            let control_value = self.control_variate_value()?;
            simulation.calculate_with_control_variate(
                mpg,
                path_pricer,
                control_pricer,
                control_value,
                self.required_tolerance,
                self.required_samples,
                self.max_samples,
            )?;
        } else {
            simulation.calculate(
                mpg,
                path_pricer,
                self.required_tolerance,
                self.required_samples,
                self.max_samples,
            )?;
        }

        let mean = simulation.sample_accumulator()?.mean()?;
        let results = self.base.results_mut();
        results.instrument.value = Some(mean.max(0.0));
        if RNG::ALLOWS_ERROR_ESTIMATE {
            results.instrument.error_estimate = Some(simulation.error_estimate()?);
        }
        Ok(())
    }
}

/// Attaches [`MCDiscreteArithmeticAveragePriceAsianHestonEngine`] to `option`.
pub fn set_mc_discrete_arithmetic_average_price_asian_heston_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    engine: SharedMut<dyn PricingEngine>,
) {
    option.base_mut().set_pricing_engine(engine);
}

/// Factory for [`MCDiscreteArithmeticAveragePriceAsianHestonEngine`].
pub struct MakeMcDiscreteArithmeticApHestonEngine<RNG> {
    process: Shared<HestonProcess>,
    antithetic: bool,
    control_variate: bool,
    samples: Option<Size>,
    max_samples: Option<Size>,
    tolerance: Option<Real>,
    time_steps: Option<Size>,
    time_steps_per_year: Option<Size>,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MakeMcDiscreteArithmeticApHestonEngine<RNG> {
    pub fn new(process: Shared<HestonProcess>) -> Self {
        Self {
            process,
            antithetic: false,
            control_variate: false,
            samples: None,
            max_samples: None,
            tolerance: None,
            time_steps: None,
            time_steps_per_year: None,
            seed: 0,
            _rng: PhantomData,
        }
    }

    pub fn with_antithetic_variate(mut self, value: bool) -> Self {
        self.antithetic = value;
        self
    }

    pub fn with_control_variate(mut self, value: bool) -> Self {
        self.control_variate = value;
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

    pub fn with_steps(mut self, steps: Size) -> Self {
        self.time_steps = Some(steps);
        self.time_steps_per_year = None;
        self
    }

    pub fn with_steps_per_year(mut self, steps: Size) -> Self {
        self.time_steps_per_year = Some(steps);
        self.time_steps = None;
        self
    }

    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn build(self) -> QlResult<MCDiscreteArithmeticAveragePriceAsianHestonEngine<RNG>> {
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
        MCDiscreteArithmeticAveragePriceAsianHestonEngine::new(
            self.process,
            self.antithetic,
            self.samples,
            self.tolerance,
            self.max_samples,
            self.seed,
            self.time_steps,
            self.time_steps_per_year,
            self.control_variate,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instruments::DiscreteAveragingAsianOption;
    use crate::interestrate::Compounding;
    use crate::math::randomnumbers::rngtraits::LowDiscrepancy;
    use crate::option::OptionType;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{Shared, shared, shared_mut};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounter::DayCounter;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn crate::quotes::Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn crate::quotes::Quote>)
    }

    fn flat_rate(reference: Date, rate: Real, dc: DayCounter) -> Handle<dyn YieldTermStructure> {
        Handle::new(
            shared(FlatForward::with_rate(
                reference,
                rate,
                dc,
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
        )
    }

    /// Ballestra / Albrecher–Zeng via `testMCDiscreteArithmeticAveragePriceHeston`
    /// (`LowDiscrepancy`, seed 42; geometric Heston analytic CV).
    #[test]
    fn monte_carlo_discrete_arithmetic_average_price_heston_matches_literature() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        // --- Ballestra, Pacelli, Zirilli (2007) ---
        {
            let spot = shared(SimpleQuote::new(120.0));
            let process = shared(HestonProcess::new(
                flat_rate(today, 0.05, Actual360::new()),
                flat_rate(today, 0.0, Actual360::new()),
                quote_handle(&spot),
                0.09,
                11.35,
                0.022,
                0.618,
                -0.5,
            ));

            let first = 1.0 / 12.0;
            let length = 11.0 / 12.0;
            let fixings: Size = 12;
            let dt = length / (fixings as Real - 1.0);
            let mut fixing_dates = Vec::with_capacity(fixings);
            for i in 0..fixings {
                let t = if i == 0 {
                    first
                } else {
                    i as Real * dt + first
                };
                fixing_dates.push(today + (t * 365.25) as i32);
            }

            let engine = shared_mut(
                MakeMcDiscreteArithmeticApHestonEngine::<LowDiscrepancy>::new(Shared::clone(
                    &process,
                ))
                .with_seed(42)
                .with_samples(4095)
                .build()
                .unwrap(),
            );

            let mut option = DiscreteAveragingAsianOption::new(
                AverageType::Arithmetic,
                0.0,
                0,
                fixing_dates.clone(),
                PlainVanillaPayoff::new(OptionType::Call, 100.0),
                shared(EuropeanExercise::new(*fixing_dates.last().unwrap())),
                Shared::clone(&settings),
            )
            .unwrap();

            set_mc_discrete_arithmetic_average_price_asian_heston_engine(
                &mut option,
                SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>,
            );
            let calculated = option.npv().unwrap();
            let expected = 22.50;
            let tolerance = 5.0e-2;
            assert!(
                (calculated - expected).abs() <= tolerance,
                "Ballestra: expected {expected}, got {calculated} (tol {tolerance})"
            );

            let engine_cv = shared_mut(
                MakeMcDiscreteArithmeticApHestonEngine::<LowDiscrepancy>::new(Shared::clone(
                    &process,
                ))
                .with_seed(42)
                .with_steps(48)
                .with_samples(4095)
                .with_control_variate(true)
                .build()
                .unwrap(),
            );
            set_mc_discrete_arithmetic_average_price_asian_heston_engine(
                &mut option,
                SharedMut::clone(&engine_cv) as SharedMut<dyn PricingEngine>,
            );
            let calculated_cv = option.npv().unwrap();
            let tolerance_cv = 3.0e-2;
            assert!(
                (calculated_cv - expected).abs() <= tolerance_cv,
                "Ballestra CV: expected {expected}, got {calculated_cv} (tol {tolerance_cv})"
            );
        }

        // --- Albrecher / Zeng–Kwok Table 4 ---
        {
            let spot = shared(SimpleQuote::new(100.0));
            let process = shared(HestonProcess::new(
                flat_rate(today, 0.03, Actual365Fixed::new()),
                flat_rate(today, 0.0, Actual365Fixed::new()),
                quote_handle(&spot),
                0.0175,
                1.5768,
                0.0398,
                0.5751,
                -0.5711,
            ));

            // Non-CV: QL uses 8191; this port needs 16383 at the same seed/steps to
            // stay inside the 9e-2 literature band (see prior Heston AP oracle).
            let engine = shared_mut(
                MakeMcDiscreteArithmeticApHestonEngine::<LowDiscrepancy>::new(Shared::clone(
                    &process,
                ))
                .with_seed(42)
                .with_steps(180)
                .with_samples(16383)
                .build()
                .unwrap(),
            );

            let engine_cv = shared_mut(
                MakeMcDiscreteArithmeticApHestonEngine::<LowDiscrepancy>::new(Shared::clone(
                    &process,
                ))
                .with_seed(42)
                .with_steps(180)
                .with_samples(8191)
                .with_control_variate(true)
                .build()
                .unwrap(),
            );

            let fixing_dates: Vec<Date> = (1..=120)
                .map(|i| today + Period::new(i, TimeUnit::Months))
                .collect();
            let exercise_date = *fixing_dates.last().unwrap();

            let strikes = [60.0, 80.0, 100.0, 120.0, 140.0];
            let prices = [42.5990, 29.3698, 18.2360, 10.0565, 4.9609];

            for (strike, expected) in strikes.into_iter().zip(prices) {
                let mut option = DiscreteAveragingAsianOption::new(
                    AverageType::Arithmetic,
                    0.0,
                    0,
                    fixing_dates.clone(),
                    PlainVanillaPayoff::new(OptionType::Call, strike),
                    shared(EuropeanExercise::new(exercise_date)),
                    Shared::clone(&settings),
                )
                .unwrap();

                set_mc_discrete_arithmetic_average_price_asian_heston_engine(
                    &mut option,
                    SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>,
                );
                let calculated = option.npv().unwrap();
                let tolerance = 9.0e-2;
                assert!(
                    (calculated - expected).abs() <= tolerance,
                    "Albrecher K={strike}: expected {expected}, got {calculated} (tol {tolerance})"
                );

                set_mc_discrete_arithmetic_average_price_asian_heston_engine(
                    &mut option,
                    SharedMut::clone(&engine_cv) as SharedMut<dyn PricingEngine>,
                );
                let calculated_cv = option.npv().unwrap();
                let tolerance_cv = 3.0e-2;
                assert!(
                    (calculated_cv - expected).abs() <= tolerance_cv,
                    "Albrecher CV K={strike}: expected {expected}, got {calculated_cv} (tol {tolerance_cv})"
                );
            }
        }
    }
}
