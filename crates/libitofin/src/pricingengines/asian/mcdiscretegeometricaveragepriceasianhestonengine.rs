//! Monte Carlo discrete geometric-average price Asian engine under Heston.
//!
//! Port of `ql/pricingengines/asian/mc_discr_geom_av_price_heston.{hpp,cpp}` and
//! the multi-factor branch of `mcdiscreteasianenginebase.hpp`.

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

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Geometric average-price Asian path pricer under Heston
/// (`mc_discr_geom_av_price_heston.cpp`).
pub struct GeometricApoHestonPathPricer {
    payoff: PlainVanillaPayoff,
    discount: DiscountFactor,
    fixing_indices: Vec<Size>,
    running_product: Real,
    past_fixings: Size,
}

impl GeometricApoHestonPathPricer {
    /// Builds the path pricer.
    pub fn new(
        option_type: OptionType,
        strike: Real,
        discount: DiscountFactor,
        fixing_indices: Vec<Size>,
        running_product: Real,
        past_fixings: Size,
    ) -> QlResult<Self> {
        require!(strike >= 0.0, "strike less than zero not allowed");
        Ok(Self {
            payoff: PlainVanillaPayoff::new(option_type, strike),
            discount,
            fixing_indices,
            running_product,
            past_fixings,
        })
    }
}

impl PathPricer<MultiPath> for GeometricApoHestonPathPricer {
    fn price(&self, multi_path: &MultiPath) -> Real {
        let path = &multi_path[0];
        let n = multi_path.path_size();
        if n == 0 {
            return 0.0;
        }

        const MAX_VALUE: Real = Real::MAX;
        let mut average_price = 1.0;
        let mut product = self.running_product;
        let fixings = self.past_fixings + self.fixing_indices.len();

        for &idx in &self.fixing_indices {
            let price = path[idx];
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

/// Monte Carlo engine for discrete geometric average-price Asians under Heston.
pub struct MCDiscreteGeometricAveragePriceAsianHestonEngine<RNG> {
    base: EngineBase,
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

impl<RNG: McRngTraits> MCDiscreteGeometricAveragePriceAsianHestonEngine<RNG> {
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
}

impl<RNG: McRngTraits> AsObservable for MCDiscreteGeometricAveragePriceAsianHestonEngine<RNG> {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl<RNG: McRngTraits> PricingEngine for MCDiscreteGeometricAveragePriceAsianHestonEngine<RNG> {
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

        let path_pricer = GeometricApoHestonPathPricer::new(
            payoff.option_type(),
            payoff.strike(),
            discount,
            fixing_indices,
            running_accumulator,
            past_fixings,
        )?;

        let mut simulation = McSimulation::<
            MultiPathGenerator<RNG::RsgType>,
            GeometricApoHestonPathPricer,
        >::new(self.antithetic_variate, false);
        simulation.calculate(
            mpg,
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

/// Attaches [`MCDiscreteGeometricAveragePriceAsianHestonEngine`] to `option`.
pub fn set_mc_discrete_geometric_average_price_asian_heston_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    engine: SharedMut<dyn PricingEngine>,
) {
    option.base_mut().set_pricing_engine(engine);
}

/// Factory for [`MCDiscreteGeometricAveragePriceAsianHestonEngine`].
pub struct MakeMcDiscreteGeometricApHestonEngine<RNG> {
    process: Shared<HestonProcess>,
    antithetic: bool,
    samples: Option<Size>,
    max_samples: Option<Size>,
    tolerance: Option<Real>,
    time_steps: Option<Size>,
    time_steps_per_year: Option<Size>,
    seed: u32,
    _rng: PhantomData<RNG>,
}

impl<RNG: McRngTraits> MakeMcDiscreteGeometricApHestonEngine<RNG> {
    pub fn new(process: Shared<HestonProcess>) -> Self {
        Self {
            process,
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

    pub fn build(self) -> QlResult<MCDiscreteGeometricAveragePriceAsianHestonEngine<RNG>> {
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
        MCDiscreteGeometricAveragePriceAsianHestonEngine::new(
            self.process,
            self.antithetic,
            self.samples,
            self.tolerance,
            self.max_samples,
            self.seed,
            self.time_steps,
            self.time_steps_per_year,
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
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{Shared, shared, shared_mut};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn crate::quotes::Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn crate::quotes::Quote>)
    }

    fn flat_rate(reference: Date, rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(
            shared(FlatForward::with_rate(
                reference,
                rate,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
        )
    }

    /// Kim–Kim–Kim–Wee tables 1–3 via `testMCDiscreteGeometricAveragePriceHeston`
    /// (`LowDiscrepancy`, 8191 samples, seed 43).
    #[test]
    fn monte_carlo_discrete_geometric_average_price_heston_matches_kim() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(100.0));
        let process = shared(HestonProcess::new(
            flat_rate(today, 0.05),
            flat_rate(today, 0.0),
            quote_handle(&spot),
            0.09,
            1.15,
            0.0348,
            0.39,
            -0.64,
        ));

        let engine = shared_mut(
            MakeMcDiscreteGeometricApHestonEngine::<LowDiscrepancy>::new(Shared::clone(&process))
                .with_samples(8191)
                .with_seed(43)
                .build()
                .unwrap(),
        );

        // 30-day options need wider tolerance (weekly-fixing ambiguity).
        let days: [i32; 18] = [
            30, 91, 182, 365, 730, 1095, 30, 91, 182, 365, 730, 1095, 30, 91, 182, 365, 730,
            1095,
        ];
        let strikes: [Real; 18] = [
            90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 110.0,
            110.0, 110.0, 110.0, 110.0, 110.0,
        ];
        let prices: [Real; 18] = [
            10.2732, 10.9554, 11.9916, 13.6950, 16.1773, 18.0146, 2.4389, 3.7881, 5.2132, 7.2243,
            9.9948, 12.0639, 0.1012, 0.5949, 1.4444, 2.9479, 5.3531, 7.3315,
        ];
        let tol: [Real; 18] = [
            4.0e-2, 2.0e-2, 2.0e-2, 4.0e-2, 8.0e-2, 2.0e-1, 1.0e-1, 4.0e-2, 3.0e-2, 2.0e-2,
            9.0e-2, 2.0e-1, 2.0e-2, 1.0e-2, 2.0e-2, 2.0e-2, 7.0e-2, 2.0e-1,
        ];

        for i in 0..18 {
            let day = days[i];
            let strike = strikes[i];
            let expected = prices[i];
            let tolerance = tol[i];

            let future_fixings = (day as Real / 7.0).floor() as Size;
            let expiry_date = today + day;
            let mut fixing_dates = Vec::with_capacity(future_fixings);
            for j in 0..future_fixings {
                fixing_dates.push(expiry_date - (j as i32) * 7);
            }

            let mut option = DiscreteAveragingAsianOption::new(
                AverageType::Geometric,
                1.0,
                0,
                fixing_dates,
                PlainVanillaPayoff::new(OptionType::Call, strike),
                shared(EuropeanExercise::new(expiry_date)),
                Shared::clone(&settings),
            )
            .unwrap();

            set_mc_discrete_geometric_average_price_asian_heston_engine(
                &mut option,
                SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>,
            );
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - expected).abs() <= tolerance,
                "case {i}: K={strike} T={day}d expected {expected}, got {calculated} (tol {tolerance})"
            );
        }
    }
}
