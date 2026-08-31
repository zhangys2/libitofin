//! Monte Carlo discrete arithmetic-average price Asian engine.
//!
//! Port of `ql/pricingengines/asian/mc_discr_arith_av_price.{hpp,cpp}` and
//! `mcdiscreteasianenginebase.hpp` (single-factor branch with same-path CV).

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
use crate::methods::montecarlo::{McSimulation, Path, PathGenerator, PathPricer};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::asian::analyticdiscretegeometricaveragepriceasianengine::AnalyticDiscreteGeometricAveragePriceAsianEngine;
use crate::pricingengines::asian::mcdiscretegeometricaveragepriceasianengine::GeometricApoPathPricer;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{DiscountFactor, Real, Size, Time};

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Arithmetic average-price Asian path pricer (`mc_discr_arith_av_price.cpp`).
pub struct ArithmeticApoPathPricer {
    payoff: PlainVanillaPayoff,
    discount: DiscountFactor,
    running_sum: Real,
    past_fixings: Size,
}

impl ArithmeticApoPathPricer {
    /// Builds the path pricer.
    pub fn new(
        option_type: OptionType,
        strike: Real,
        discount: DiscountFactor,
        running_sum: Real,
        past_fixings: Size,
    ) -> QlResult<Self> {
        require!(strike >= 0.0, "strike less than zero not allowed");
        Ok(Self {
            payoff: PlainVanillaPayoff::new(option_type, strike),
            discount,
            running_sum,
            past_fixings,
        })
    }
}

impl PathPricer<Path> for ArithmeticApoPathPricer {
    fn price(&self, path: &Path) -> Real {
        let n = path.length();
        if n <= 1 {
            return 0.0;
        }

        let values = path.values();
        let (sum, fixings) = if path
            .time_grid()
            .mandatory_times()
            .first()
            .is_some_and(|t| *t == 0.0)
        {
            (
                self.running_sum + values.iter().sum::<Real>(),
                self.past_fixings + n,
            )
        } else {
            (
                self.running_sum + values[1..].iter().sum::<Real>(),
                self.past_fixings + n - 1,
            )
        };
        let average_price = sum / fixings as Real;
        self.discount * self.payoff.value(average_price)
    }
}

/// Monte Carlo engine for discrete arithmetic average-price Asians.
pub struct MCDiscreteArithmeticAveragePriceAsianEngine<RNG> {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    time_steps: Option<Size>,
    time_steps_per_year: Option<Size>,
    required_samples: Option<Size>,
    max_samples: Option<Size>,
    required_tolerance: Option<Real>,
    brownian_bridge: bool,
    antithetic_variate: bool,
    control_variate: bool,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MCDiscreteArithmeticAveragePriceAsianEngine<RNG> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        brownian_bridge: bool,
        antithetic_variate: bool,
        control_variate: bool,
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
            time_steps: None,
            time_steps_per_year: None,
            required_samples,
            max_samples,
            required_tolerance,
            brownian_bridge,
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

    fn control_variate_value(&self) -> QlResult<Real> {
        let mut control =
            AnalyticDiscreteGeometricAveragePriceAsianEngine::new(Shared::clone(&self.process));
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

impl<RNG: McRngTraits> AsObservable for MCDiscreteArithmeticAveragePriceAsianEngine<RNG> {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl<RNG: McRngTraits> PricingEngine for MCDiscreteArithmeticAveragePriceAsianEngine<RNG> {
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

        let r_ts = self.process.risk_free_rate().current_link()?;
        let discount = r_ts.discount_date(exercise.last_date(), false)?;
        let path_pricer = ArithmeticApoPathPricer::new(
            payoff.option_type(),
            payoff.strike(),
            discount,
            running_accumulator,
            past_fixings,
        )?;

        let generator = self.path_generator()?;
        let mut simulation: McSimulation<PathGenerator<RNG::RsgType>, ArithmeticApoPathPricer> =
            McSimulation::new(self.antithetic_variate, self.control_variate);

        if self.control_variate {
            let control_value = self.control_variate_value()?;
            let grid = self.time_grid()?;
            let t_back = grid.back().expect("non-empty grid");
            let cv_discount = r_ts.discount(t_back, false)?;
            let control_pricer = GeometricApoPathPricer::new(
                PlainVanillaPayoff::new(payoff.option_type(), payoff.strike()),
                cv_discount,
                1.0,
                0,
            )?;
            simulation.calculate_with_control_variate(
                generator,
                path_pricer,
                control_pricer,
                control_value,
                self.required_tolerance,
                self.required_samples,
                self.max_samples,
            )?;
        } else {
            simulation.calculate(
                generator,
                path_pricer,
                self.required_tolerance,
                self.required_samples,
                self.max_samples,
            )?;
        }

        let mean = simulation.sample_accumulator()?.mean()?;
        let results = self.base.results_mut();
        results.instrument.value = Some(if self.control_variate {
            mean.max(0.0)
        } else {
            mean
        });
        if RNG::ALLOWS_ERROR_ESTIMATE {
            results.instrument.error_estimate = Some(simulation.error_estimate()?);
        }
        Ok(())
    }
}

/// Attaches [`MCDiscreteArithmeticAveragePriceAsianEngine`] to `option`.
pub fn set_mc_discrete_arithmetic_average_price_asian_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    engine: SharedMut<dyn PricingEngine>,
) {
    option.base_mut().set_pricing_engine(engine);
}

/// Factory for [`MCDiscreteArithmeticAveragePriceAsianEngine`].
pub struct MakeMcDiscreteArithmeticApEngine<RNG> {
    process: Shared<GeneralizedBlackScholesProcess>,
    brownian_bridge: bool,
    antithetic: bool,
    control_variate: bool,
    samples: Option<Size>,
    max_samples: Option<Size>,
    tolerance: Option<Real>,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MakeMcDiscreteArithmeticApEngine<RNG> {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self {
            process,
            brownian_bridge: true,
            antithetic: false,
            control_variate: false,
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

    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn build(self) -> QlResult<MCDiscreteArithmeticAveragePriceAsianEngine<RNG>> {
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
        MCDiscreteArithmeticAveragePriceAsianEngine::new(
            self.process,
            self.brownian_bridge,
            self.antithetic,
            self.control_variate,
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

    struct Case {
        option_type: OptionType,
        underlying: Real,
        strike: Real,
        dividend_yield: Real,
        risk_free_rate: Real,
        first: Time,
        length: Time,
        fixings: Size,
        volatility: Real,
        control_variate: bool,
        result: Real,
    }

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

    /// Levy (1997) table via `asianoptions.cpp` `testMCDiscreteArithmeticAveragePrice`.
    #[test]
    fn monte_carlo_discrete_arithmetic_average_price_matches_levy() {
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

        // cases4 from asianoptions.cpp (controlVariate=true throughout).
        let cases = [
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 2,
                volatility: 0.13,
                control_variate: true,
                result: 1.3942835683,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 4,
                volatility: 0.13,
                control_variate: true,
                result: 1.5852442983,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 8,
                volatility: 0.13,
                control_variate: true,
                result: 1.66970673,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 12,
                volatility: 0.13,
                control_variate: true,
                result: 1.6980019214,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 26,
                volatility: 0.13,
                control_variate: true,
                result: 1.7255070456,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 52,
                volatility: 0.13,
                control_variate: true,
                result: 1.7401553533,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 100,
                volatility: 0.13,
                control_variate: true,
                result: 1.7478303712,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 250,
                volatility: 0.13,
                control_variate: true,
                result: 1.7490291943,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 500,
                volatility: 0.13,
                control_variate: true,
                result: 1.7515113291,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 0.0,
                length: 11.0 / 12.0,
                fixings: 1000,
                volatility: 0.13,
                control_variate: true,
                result: 1.7537344885,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 2,
                volatility: 0.13,
                control_variate: true,
                result: 1.8496053697,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 4,
                volatility: 0.13,
                control_variate: true,
                result: 2.0111495205,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 8,
                volatility: 0.13,
                control_variate: true,
                result: 2.0852138818,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 12,
                volatility: 0.13,
                control_variate: true,
                result: 2.1105094397,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 26,
                volatility: 0.13,
                control_variate: true,
                result: 2.1346526695,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 52,
                volatility: 0.13,
                control_variate: true,
                result: 2.147489651,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 100,
                volatility: 0.13,
                control_variate: true,
                result: 2.154728109,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 250,
                volatility: 0.13,
                control_variate: true,
                result: 2.1564276565,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 500,
                volatility: 0.13,
                control_variate: true,
                result: 2.1594238588,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 1.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 1000,
                volatility: 0.13,
                control_variate: true,
                result: 2.1595367326,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 2,
                volatility: 0.13,
                control_variate: true,
                result: 2.63315092584,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 4,
                volatility: 0.13,
                control_variate: true,
                result: 2.76723962361,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 8,
                volatility: 0.13,
                control_variate: true,
                result: 2.83124836881,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 12,
                volatility: 0.13,
                control_variate: true,
                result: 2.84290301412,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 26,
                volatility: 0.13,
                control_variate: true,
                result: 2.88179560417,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 52,
                volatility: 0.13,
                control_variate: true,
                result: 2.88447044543,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 100,
                volatility: 0.13,
                control_variate: true,
                result: 2.89985329603,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 250,
                volatility: 0.13,
                control_variate: true,
                result: 2.90047296063,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 500,
                volatility: 0.13,
                control_variate: true,
                result: 2.89813412160,
            },
            Case {
                option_type: OptionType::Put,
                underlying: 90.0,
                strike: 87.0,
                dividend_yield: 0.06,
                risk_free_rate: 0.025,
                first: 3.0 / 12.0,
                length: 11.0 / 12.0,
                fixings: 1000,
                volatility: 0.13,
                control_variate: true,
                result: 2.89703362437,
            },
        ];

        let tolerance = 2.0e-2;
        for (i, case) in cases.iter().enumerate() {
            spot.set_value(case.underlying);
            q_rate.set_value(case.dividend_yield);
            r_rate.set_value(case.risk_free_rate);
            vol.set_value(case.volatility);

            let dt = case.length / (case.fixings - 1) as Real;
            let mut fixing_dates = Vec::with_capacity(case.fixings);
            fixing_dates.push(today + time_to_days(case.first));
            for j in 1..case.fixings {
                let t = j as Real * dt + case.first;
                fixing_dates.push(today + time_to_days(t));
            }
            let exercise =
                shared(EuropeanExercise::new(*fixing_dates.last().expect("non-empty")));

            let mut option = DiscreteAveragingAsianOption::new(
                AverageType::Arithmetic,
                0.0,
                0,
                fixing_dates,
                PlainVanillaPayoff::new(case.option_type, case.strike),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();

            set_mc_discrete_arithmetic_average_price_asian_engine(
                &mut option,
                shared_mut(
                    MakeMcDiscreteArithmeticApEngine::<LowDiscrepancy>::new(Shared::clone(
                        &process,
                    ))
                    .with_samples(2047)
                    .with_control_variate(case.control_variate)
                    .build()
                    .unwrap(),
                ) as SharedMut<dyn PricingEngine>,
            );

            let calculated = option.npv().unwrap();
            assert!(
                (calculated - case.result).abs() <= tolerance,
                "case {i}: expected {}, got {calculated} (tol {tolerance})",
                case.result
            );
        }
    }
}
