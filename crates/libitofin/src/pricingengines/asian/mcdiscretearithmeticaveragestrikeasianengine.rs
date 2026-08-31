//! Monte Carlo discrete arithmetic average-strike Asian engine.
//!
//! Port of `ql/pricingengines/asian/mc_discr_arith_av_strike.{hpp,cpp}` and the
//! single-factor `includeExerciseDate` branch of `mcdiscreteasianenginebase.hpp`.

use std::marker::PhantomData;

use crate::errors::QlResult;
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults,
    PlainVanillaPayoff, TypePayoff,
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

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Arithmetic average-strike Asian path pricer (`mc_discr_arith_av_strike.cpp`).
pub struct ArithmeticAsoPathPricer {
    option_type: OptionType,
    discount: DiscountFactor,
    running_sum: Real,
    past_fixings: Size,
    /// When set, average only this many path points (exclude a trailing exercise
    /// node). `None` means use the full path length (`Null<Size>()` in QL).
    fixing_count: Option<Size>,
}

impl ArithmeticAsoPathPricer {
    /// Builds the path pricer.
    pub fn new(
        option_type: OptionType,
        discount: DiscountFactor,
        running_sum: Real,
        past_fixings: Size,
        fixing_count: Option<Size>,
    ) -> Self {
        Self {
            option_type,
            discount,
            running_sum,
            past_fixings,
            fixing_count,
        }
    }
}

impl PathPricer<Path> for ArithmeticAsoPathPricer {
    fn price(&self, path: &Path) -> Real {
        let n = path.length();
        if n <= 1 {
            return 0.0;
        }

        let n_fixings = self.fixing_count.unwrap_or(n);
        if n_fixings > n {
            return 0.0;
        }

        let values = path.values();
        let average_strike = if path
            .time_grid()
            .mandatory_times()
            .first()
            .is_some_and(|t| *t == 0.0)
        {
            let sum = self.running_sum + values[..n_fixings].iter().sum::<Real>();
            sum / (self.past_fixings + n_fixings) as Real
        } else {
            let sum = self.running_sum + values[1..n_fixings].iter().sum::<Real>();
            sum / (self.past_fixings + n_fixings - 1) as Real
        };

        self.discount * PlainVanillaPayoff::new(self.option_type, average_strike).value(path.back())
    }
}

/// Monte Carlo engine for discrete arithmetic average-strike Asians.
pub struct MCDiscreteArithmeticAverageStrikeAsianEngine<RNG> {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    required_samples: Option<Size>,
    max_samples: Option<Size>,
    required_tolerance: Option<Real>,
    brownian_bridge: bool,
    antithetic_variate: bool,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MCDiscreteArithmeticAverageStrikeAsianEngine<RNG> {
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        brownian_bridge: bool,
        antithetic_variate: bool,
        required_samples: Option<Size>,
        required_tolerance: Option<Real>,
        max_samples: Option<Size>,
        seed: u32,
    ) -> QlResult<Self> {
        let base = EngineBase::new(
            DiscreteAveragingAsianArguments::default(),
            DiscreteAveragingAsianResults::default(),
        );
        base.register_with(process.observable());

        Ok(Self {
            base,
            process,
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
        if fixing_times.is_empty() || (fixing_times.len() == 1 && fixing_times[0] == 0.0) {
            fail!("all fixings are in the past");
        }
        Ok(fixing_times)
    }

    /// Time grid with optional exercise date after the last fixing
    /// (`includeExerciseDate_ = true` in QL).
    fn time_grid(&self) -> QlResult<(TimeGrid, Option<Size>)> {
        let mut fixing_times = self.fixing_times()?;
        let exercise = self.base.arguments().exercise.as_ref().expect("validated");
        let exercise_time = self.process.time(&exercise.last_date())?;
        let last_fixing = *fixing_times.last().expect("non-empty");

        let mut fixing_count = None;
        if exercise_time > last_fixing {
            fixing_times.push(exercise_time);
        }

        let grid = TimeGrid::from_mandatory_times(&fixing_times)?;
        if exercise_time > last_fixing {
            fixing_count = Some(grid.size() - 1);
        }
        Ok((grid, fixing_count))
    }

    fn path_generator(&self, grid: TimeGrid) -> QlResult<PathGenerator<RNG::RsgType>> {
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

impl<RNG: McRngTraits> AsObservable for MCDiscreteArithmeticAverageStrikeAsianEngine<RNG> {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl<RNG: McRngTraits> PricingEngine for MCDiscreteArithmeticAverageStrikeAsianEngine<RNG> {
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

        let (grid, fixing_count) = self.time_grid()?;
        let r_ts = self.process.risk_free_rate().current_link()?;
        let discount = r_ts.discount_date(exercise.last_date(), false)?;
        let path_pricer = ArithmeticAsoPathPricer::new(
            payoff.option_type(),
            discount,
            running_accumulator,
            past_fixings,
            fixing_count,
        );

        let generator = self.path_generator(grid)?;
        let mut simulation: McSimulation<PathGenerator<RNG::RsgType>, ArithmeticAsoPathPricer> =
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
        results.instrument.value = Some(mean);
        if RNG::ALLOWS_ERROR_ESTIMATE {
            results.instrument.error_estimate = Some(simulation.error_estimate()?);
        }
        Ok(())
    }
}

/// Attaches [`MCDiscreteArithmeticAverageStrikeAsianEngine`] to `option`.
pub fn set_mc_discrete_arithmetic_average_strike_asian_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    engine: SharedMut<dyn PricingEngine>,
) {
    option.base_mut().set_pricing_engine(engine);
}

/// Factory for [`MCDiscreteArithmeticAverageStrikeAsianEngine`].
pub struct MakeMcDiscreteArithmeticAsEngine<RNG> {
    process: Shared<GeneralizedBlackScholesProcess>,
    brownian_bridge: bool,
    antithetic: bool,
    samples: Option<Size>,
    max_samples: Option<Size>,
    tolerance: Option<Real>,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MakeMcDiscreteArithmeticAsEngine<RNG> {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self {
            process,
            brownian_bridge: true,
            antithetic: false,
            samples: None,
            max_samples: None,
            tolerance: None,
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

    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn build(self) -> QlResult<MCDiscreteArithmeticAverageStrikeAsianEngine<RNG>> {
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
        MCDiscreteArithmeticAverageStrikeAsianEngine::new(
            self.process,
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
    use crate::math::randomnumbers::rngtraits::LowDiscrepancy;
    use crate::option::OptionType;
    use crate::pricingengines::vanilla::test_market::time_to_days;
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
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

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

    struct Case {
        first: Real,
        fixings: Size,
        result: Real,
    }

    /// Levy (1997) table via `asianoptions.cpp` `testMCDiscreteArithmeticAverageStrike`
    /// (`LowDiscrepancy`, seed 3456789, 1023 samples, tol 2e-2).
    #[test]
    fn monte_carlo_discrete_arithmetic_average_strike_matches_levy() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(90.0));
        let q_rate = shared(SimpleQuote::new(0.06));
        let r_rate = shared(SimpleQuote::new(0.025));
        let vol = shared(SimpleQuote::new(0.13));

        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        // cases5: Call, S=90, K=87 (unused by ASO pricer), q=0.06, r=0.025,
        // vol=0.13, length=11/12; first ∈ {0, 1/12, 3/12}.
        let cases = [
            Case {
                first: 0.0,
                fixings: 2,
                result: 1.51917595129,
            },
            Case {
                first: 0.0,
                fixings: 4,
                result: 1.67940165674,
            },
            Case {
                first: 0.0,
                fixings: 8,
                result: 1.75371215251,
            },
            Case {
                first: 0.0,
                fixings: 12,
                result: 1.77595318693,
            },
            Case {
                first: 0.0,
                fixings: 26,
                result: 1.81430536630,
            },
            Case {
                first: 0.0,
                fixings: 52,
                result: 1.82269246898,
            },
            Case {
                first: 0.0,
                fixings: 100,
                result: 1.83822402464,
            },
            Case {
                first: 0.0,
                fixings: 250,
                result: 1.83875059026,
            },
            Case {
                first: 0.0,
                fixings: 500,
                result: 1.83750703638,
            },
            Case {
                first: 0.0,
                fixings: 1000,
                result: 1.83887181884,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 2,
                result: 1.51154400089,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 4,
                result: 1.67103508506,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 8,
                result: 1.74529684070,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 12,
                result: 1.76667074564,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 26,
                result: 1.80528400613,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 52,
                result: 1.81400883891,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 100,
                result: 1.82922901451,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 250,
                result: 1.82937111773,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 500,
                result: 1.82826193186,
            },
            Case {
                first: 1.0 / 12.0,
                fixings: 1000,
                result: 1.82967846654,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 2,
                result: 1.49648170891,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 4,
                result: 1.65443100462,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 8,
                result: 1.72817806731,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 12,
                result: 1.74877367895,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 26,
                result: 1.78733801988,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 52,
                result: 1.79624826757,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 100,
                result: 1.81114186876,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 250,
                result: 1.81101152587,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 500,
                result: 1.81002311939,
            },
            Case {
                first: 3.0 / 12.0,
                fixings: 1000,
                result: 1.81145760308,
            },
        ];

        let engine = shared_mut(
            MakeMcDiscreteArithmeticAsEngine::<LowDiscrepancy>::new(Shared::clone(&process))
                .with_seed(3456789)
                .with_samples(1023)
                .build()
                .unwrap(),
        );

        let length = 11.0 / 12.0;
        let tolerance = 2.0e-2;

        for (i, case) in cases.iter().enumerate() {
            let dt = length / (case.fixings as Real - 1.0);
            let mut fixing_dates = Vec::with_capacity(case.fixings);
            fixing_dates.push(today + time_to_days(case.first));
            for j in 1..case.fixings {
                let t = j as Real * dt + case.first;
                fixing_dates.push(today + time_to_days(t));
            }

            let mut option = DiscreteAveragingAsianOption::new(
                AverageType::Arithmetic,
                0.0,
                0,
                fixing_dates.clone(),
                PlainVanillaPayoff::new(OptionType::Call, 87.0),
                shared(EuropeanExercise::new(*fixing_dates.last().unwrap())),
                Shared::clone(&settings),
            )
            .unwrap();

            set_mc_discrete_arithmetic_average_strike_asian_engine(
                &mut option,
                SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>,
            );
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - case.result).abs() <= tolerance,
                "case {i}: first={} fixings={} expected {}, got {} (tol {tolerance})",
                case.first,
                case.fixings,
                case.result,
                calculated
            );
        }
    }

    /// Issue #646 / `testMCDiscreteArithmeticAverageStrikeExerciseDate`:
    /// with r=q=0 and vol>0, later exercise must raise the ASO price.
    #[test]
    fn monte_carlo_discrete_arithmetic_average_strike_sensitive_to_exercise_date() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(90.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.0));
        let vol = shared(SimpleQuote::new(0.20));

        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        let engine = shared_mut(
            MakeMcDiscreteArithmeticAsEngine::<LowDiscrepancy>::new(Shared::clone(&process))
                .with_seed(42)
                .with_samples(8191)
                .build()
                .unwrap(),
        );

        let fixing_dates: Vec<Date> = (0..=6)
            .map(|i| today + Period::new(i, TimeUnit::Months))
            .collect();
        let payoff = PlainVanillaPayoff::new(OptionType::Call, 90.0);

        let mut option1 = DiscreteAveragingAsianOption::new(
            AverageType::Arithmetic,
            0.0,
            0,
            fixing_dates.clone(),
            payoff,
            shared(EuropeanExercise::new(*fixing_dates.last().unwrap())),
            Shared::clone(&settings),
        )
        .unwrap();
        set_mc_discrete_arithmetic_average_strike_asian_engine(
            &mut option1,
            SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>,
        );
        let price1 = option1.npv().unwrap();

        let mut option2 = DiscreteAveragingAsianOption::new(
            AverageType::Arithmetic,
            0.0,
            0,
            fixing_dates.clone(),
            payoff,
            shared(EuropeanExercise::new(
                *fixing_dates.last().unwrap() + Period::new(3, TimeUnit::Months),
            )),
            Shared::clone(&settings),
        )
        .unwrap();
        set_mc_discrete_arithmetic_average_strike_asian_engine(
            &mut option2,
            SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>,
        );
        let price2 = option2.npv().unwrap();

        assert!(
            price2 > price1,
            "average-strike Asian should be sensitive to exercise date: \
             exercise at last fixing = {price1}, exercise +3M = {price2}"
        );
    }
}
