//! Discrete-averaging Asian option instrument.
//!
//! Port of `ql/instruments/asianoption.hpp` / `asianoption.cpp` (discrete averaging).

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{AverageType, Greeks, PlainVanillaPayoff};
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::{Real, Size};

/// Arguments for discrete-averaging Asian engines.
#[derive(Default)]
pub struct DiscreteAveragingAsianArguments {
    pub average_type: Option<AverageType>,
    pub running_accumulator: Option<Real>,
    pub past_fixings: Option<Size>,
    pub fixing_dates: Vec<Date>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for DiscreteAveragingAsianArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.average_type.is_some(), "no average type given");
        require!(
            self.running_accumulator.is_some(),
            "no running accumulator given"
        );
        require!(self.past_fixings.is_some(), "no past fixings given");
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");

        let average_type = self.average_type.expect("checked");
        let running = self.running_accumulator.expect("checked");
        match average_type {
            AverageType::Arithmetic => {
                require!(
                    running >= 0.0,
                    "non negative running sum required: {running} not allowed"
                );
            }
            AverageType::Geometric => {
                require!(
                    running > 0.0,
                    "positive running product required: {running} not allowed"
                );
            }
        }
        Ok(())
    }
}

/// Engine results for discrete-averaging Asian options.
#[derive(Default)]
pub struct DiscreteAveragingAsianResults {
    pub instrument: InstrumentResults,
    pub greeks: Greeks,
}

impl Results for DiscreteAveragingAsianResults {
    fn reset(&mut self) {
        self.instrument.reset();
        self.greeks.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// European option on the discrete average of the underlying.
pub struct DiscreteAveragingAsianOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    average_type: AverageType,
    running_accumulator: Real,
    past_fixings: Size,
    fixing_dates: Vec<Date>,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
    greeks: Greeks,
    /// When true, [`setup_arguments`](Instrument::setup_arguments) derives the
    /// running accumulator / past-fixing count from [`all_past_fixings`] and
    /// dates before the evaluation date (`asianoption.cpp`).
    all_past_fixings_provided: bool,
    all_past_fixings: Vec<Real>,
}

impl DiscreteAveragingAsianOption {
    /// Classic ctor: running sum/product and past-fixing count; `fixing_dates`
    /// may be future-only (`asianoption.cpp`).
    pub fn new(
        average_type: AverageType,
        mut running_accumulator: Real,
        past_fixings: Size,
        mut fixing_dates: Vec<Date>,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        fixing_dates.sort();
        if past_fixings == 0 {
            running_accumulator = match average_type {
                AverageType::Geometric => 1.0,
                AverageType::Arithmetic => 0.0,
            };
        }

        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            average_type,
            running_accumulator,
            past_fixings,
            fixing_dates,
            payoff,
            exercise,
            greeks: Greeks::default(),
            all_past_fixings_provided: false,
            all_past_fixings: Vec::new(),
        })
    }

    /// Vector past-fixings ctor: expects *all* fixing dates (past and future,
    /// already sorted). Historic dates are matched against `all_past_fixings`
    /// at setup time (`asianoption.cpp`).
    pub fn with_all_past_fixings(
        average_type: AverageType,
        fixing_dates: Vec<Date>,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        all_past_fixings: Vec<Real>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            average_type,
            running_accumulator: 0.0,
            past_fixings: 0,
            fixing_dates,
            payoff,
            exercise,
            greeks: Greeks::default(),
            all_past_fixings_provided: true,
            all_past_fixings,
        })
    }

    fn greek(value: Option<Real>, description: &str) -> QlResult<Real> {
        let Some(value) = value else {
            fail!("{description} not provided");
        };
        Ok(value)
    }

    pub fn theta(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.theta, "theta")
    }

    pub fn delta(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.delta, "delta")
    }

    pub fn gamma(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.gamma, "gamma")
    }

    pub fn vega(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.vega, "vega")
    }

    pub fn rho(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.rho, "rho")
    }

    pub fn dividend_rho(&mut self) -> QlResult<Real> {
        self.calculate()?;
        Self::greek(self.greeks.dividend_rho, "dividend rho")
    }

    pub fn average_type(&self) -> AverageType {
        self.average_type
    }

    pub fn running_accumulator(&self) -> Real {
        self.running_accumulator
    }

    pub fn past_fixings(&self) -> Size {
        self.past_fixings
    }

    pub fn fixing_dates(&self) -> &[Date] {
        &self.fixing_dates
    }
}

impl Instrument for DiscreteAveragingAsianOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        crate::event::event_has_occurred(self.exercise.last_date(), &self.settings, None, None)
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) = (arguments as &mut dyn Any)
            .downcast_mut::<DiscreteAveragingAsianArguments>()
        else {
            fail!("wrong argument type");
        };

        let (running_accumulator, past_fixings, fixing_dates) = if self.all_past_fixings_provided
        {
            let Some(today) = self.settings.evaluation_date() else {
                fail!("no evaluation date set");
            };

            let mut past_fixings: Size = 0;
            let mut future_fixing_dates = Vec::new();
            for &fixing_date in &self.fixing_dates {
                if fixing_date < today {
                    past_fixings += 1;
                } else {
                    future_fixing_dates.push(fixing_date);
                }
            }

            require!(
                past_fixings <= self.all_past_fixings.len(),
                "Not enough past fixings have been provided for the required historical fixing dates"
            );

            let running_accumulator = match self.average_type {
                AverageType::Geometric => {
                    let mut product = 1.0;
                    for i in 0..past_fixings {
                        product *= self.all_past_fixings[i];
                    }
                    product
                }
                AverageType::Arithmetic => {
                    let mut sum = 0.0;
                    for i in 0..past_fixings {
                        sum += self.all_past_fixings[i];
                    }
                    sum
                }
            };
            (running_accumulator, past_fixings, future_fixing_dates)
        } else {
            (
                self.running_accumulator,
                self.past_fixings,
                self.fixing_dates.clone(),
            )
        };

        arguments.average_type = Some(self.average_type);
        arguments.running_accumulator = Some(running_accumulator);
        arguments.past_fixings = Some(past_fixings);
        arguments.fixing_dates = fixing_dates;
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<DiscreteAveragingAsianResults>()
        else {
            fail!("no greeks returned from pricing engine");
        };
        self.greeks = results.greeks;
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::math::comparison::close;
    use crate::math::randomnumbers::rngtraits::LowDiscrepancy;
    use crate::option::OptionType;
    use crate::pricingengines::asian::{
        MakeMcDiscreteArithmeticApEngine, MakeMcDiscreteArithmeticAsEngine,
        MakeMcDiscreteGeometricApEngine, set_analytic_discrete_geometric_average_price_asian_engine,
        set_mc_discrete_arithmetic_average_price_asian_engine,
        set_mc_discrete_arithmetic_average_strike_asian_engine,
        set_mc_discrete_geometric_average_price_asian_engine,
    };
    use crate::pricingengine::PricingEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::Month;
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

    /// `asianoptions.cpp` `testPastFixings`.
    #[test]
    fn past_fixings_affect_discrete_asian_prices() {
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

        let payoff = PlainVanillaPayoff::new(OptionType::Put, 100.0);
        let exercise: Shared<dyn Exercise> =
            shared(EuropeanExercise::new(today + Period::new(1, TimeUnit::Years)));

        let fixing_dates1: Vec<Date> = (0..=12)
            .map(|i| today + Period::new(i, TimeUnit::Months))
            .collect();
        let fixing_dates2: Vec<Date> = (-2..=12)
            .map(|i| today + Period::new(i, TimeUnit::Months))
            .collect();

        // --- MC arithmetic average-price ---
        let mut option1 = DiscreteAveragingAsianOption::new(
            AverageType::Arithmetic,
            0.0,
            0,
            fixing_dates1.clone(),
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        let past_fixings = 2;
        let running_sum = past_fixings as Real * spot.value().unwrap() * 0.8;
        let mut option2 = DiscreteAveragingAsianOption::new(
            AverageType::Arithmetic,
            running_sum,
            past_fixings,
            fixing_dates2.clone(),
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();

        let ap_engine = shared_mut(
            MakeMcDiscreteArithmeticApEngine::<LowDiscrepancy>::new(Shared::clone(&process))
                .with_samples(2047)
                .build()
                .unwrap(),
        );
        set_mc_discrete_arithmetic_average_price_asian_engine(
            &mut option1,
            SharedMut::clone(&ap_engine) as SharedMut<dyn PricingEngine>,
        );
        set_mc_discrete_arithmetic_average_price_asian_engine(
            &mut option2,
            SharedMut::clone(&ap_engine) as SharedMut<dyn PricingEngine>,
        );
        let price1 = option1.npv().unwrap();
        let price2 = option2.npv().unwrap();
        assert!(
            !close(price1, price2),
            "past fixings had no effect on arithmetic average-price option: \
             without={price1}, with={price2}"
        );

        // Vector past-fixings interface
        let all_past_fixings = vec![spot.value().unwrap() * 0.8, spot.value().unwrap() * 0.8];
        let mut option1a = DiscreteAveragingAsianOption::with_all_past_fixings(
            AverageType::Arithmetic,
            fixing_dates1.clone(),
            payoff,
            Shared::clone(&exercise),
            Vec::new(),
            Shared::clone(&settings),
        )
        .unwrap();
        let mut option2a = DiscreteAveragingAsianOption::with_all_past_fixings(
            AverageType::Arithmetic,
            fixing_dates2.clone(),
            payoff,
            Shared::clone(&exercise),
            all_past_fixings,
            Shared::clone(&settings),
        )
        .unwrap();
        set_mc_discrete_arithmetic_average_price_asian_engine(
            &mut option1a,
            SharedMut::clone(&ap_engine) as SharedMut<dyn PricingEngine>,
        );
        set_mc_discrete_arithmetic_average_price_asian_engine(
            &mut option2a,
            SharedMut::clone(&ap_engine) as SharedMut<dyn PricingEngine>,
        );
        let price1a = option1a.npv().unwrap();
        let price2a = option2a.npv().unwrap();
        assert!(
            (price1 - price1a).abs() <= 1e-8,
            "unseasoned AP prices differ: classic={price1}, vector={price1a}"
        );
        assert!(
            (price2 - price2a).abs() <= 1e-8,
            "seasoned AP prices differ: classic={price2}, vector={price2a}"
        );

        // --- MC arithmetic average-strike ---
        let as_engine = shared_mut(
            MakeMcDiscreteArithmeticAsEngine::<LowDiscrepancy>::new(Shared::clone(&process))
                .with_samples(2047)
                .build()
                .unwrap(),
        );
        set_mc_discrete_arithmetic_average_strike_asian_engine(
            &mut option1,
            SharedMut::clone(&as_engine) as SharedMut<dyn PricingEngine>,
        );
        set_mc_discrete_arithmetic_average_strike_asian_engine(
            &mut option2,
            SharedMut::clone(&as_engine) as SharedMut<dyn PricingEngine>,
        );
        let price1 = option1.npv().unwrap();
        let price2 = option2.npv().unwrap();
        assert!(
            !close(price1, price2),
            "past fixings had no effect on arithmetic average-strike option: \
             without={price1}, with={price2}"
        );

        // --- Analytic geometric average-price ---
        let mut option3 = DiscreteAveragingAsianOption::new(
            AverageType::Geometric,
            1.0,
            0,
            fixing_dates1.clone(),
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        let running_product = spot.value().unwrap() * spot.value().unwrap();
        let mut option4 = DiscreteAveragingAsianOption::new(
            AverageType::Geometric,
            running_product,
            2,
            fixing_dates2.clone(),
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_discrete_geometric_average_price_asian_engine(
            &mut option3,
            Shared::clone(&process),
        );
        set_analytic_discrete_geometric_average_price_asian_engine(
            &mut option4,
            Shared::clone(&process),
        );
        let price3 = option3.npv().unwrap();
        let price4 = option4.npv().unwrap();
        assert!(
            !close(price3, price4),
            "past fixings had no effect on geometric average-price option (analytic): \
             without={price3}, with={price4}"
        );

        // --- MC geometric average-price ---
        let geom_engine = shared_mut(
            MakeMcDiscreteGeometricApEngine::<LowDiscrepancy>::new(Shared::clone(&process))
                .with_samples(2047)
                .build()
                .unwrap(),
        );
        set_mc_discrete_geometric_average_price_asian_engine(
            &mut option3,
            SharedMut::clone(&geom_engine) as SharedMut<dyn PricingEngine>,
        );
        set_mc_discrete_geometric_average_price_asian_engine(
            &mut option4,
            SharedMut::clone(&geom_engine) as SharedMut<dyn PricingEngine>,
        );
        let price3 = option3.npv().unwrap();
        let price4 = option4.npv().unwrap();
        assert!(
            !close(price3, price4),
            "past fixings had no effect on geometric average-price option (MC): \
             without={price3}, with={price4}"
        );
    }
}
