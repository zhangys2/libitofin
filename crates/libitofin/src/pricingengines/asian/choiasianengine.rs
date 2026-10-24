//! Choi Asian engine (arithmetic discrete average via basket replication).
//!
//! Port of `ql/pricingengines/asian/choiasianengine.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::exercise::{EuropeanExercise, ExerciseType};
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageBasketPayoff, AverageType, BasketOption, DiscreteAveragingAsianArguments,
    DiscreteAveragingAsianResults, PlainVanillaPayoff, StrikedTypePayoff, TypePayoff,
};
use crate::math::array::Array;
use crate::math::matrix::Matrix;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::basket::ChoiBasketEngine;
use crate::pricingengines::blackformula::black_formula;
use crate::processes::{BlackProcess, GeneralizedBlackScholesProcess};
use crate::quotes::SimpleQuote;
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
use crate::termstructures::yields::FlatForward;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::types::{Real, Size};

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Choi (2018) engine for discrete arithmetic-average Asians.
pub struct ChoiAsianEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    lambda: Real,
    max_nr_integration_steps: Size,
    settings: Shared<Settings<Date>>,
}

impl ChoiAsianEngine {
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        Self::with_params(process, 15.0, Size::MAX, settings)
    }

    pub fn with_params(
        process: Shared<GeneralizedBlackScholesProcess>,
        lambda: Real,
        max_nr_integration_steps: Size,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = EngineBase::new(
            DiscreteAveragingAsianArguments::default(),
            DiscreteAveragingAsianResults::default(),
        );
        base.register_with(process.observable());
        Self {
            base,
            process,
            lambda,
            max_nr_integration_steps,
            settings,
        }
    }
}

impl AsObservable for ChoiAsianEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for ChoiAsianEngine {
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
            "must be Average::Type Arithmetic "
        );
        let exercise = args.exercise.as_ref().expect("validated");
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not a European Option"
        );
        let payoff = args.payoff.expect("validated");

        let mut fixing_dates = args.fixing_dates.clone();
        fixing_dates.sort_unstable();

        let mut future_fixings = fixing_dates.len();
        let mut past_fixings = args.past_fixings.expect("validated");
        let mut running_accumulator = args.running_accumulator.expect("validated");

        let exercise_date = exercise.last_date();
        let r_ts = self.process.risk_free_rate().current_link()?;

        if future_fixings > 0 && StochasticProcess1D::time(&*self.process, &fixing_dates[0])? == 0.0
        {
            fixing_dates.remove(0);
            future_fixings -= 1;
            past_fixings += 1;
            running_accumulator += self.process.state_variable().current_link()?.value()?;
        }

        if future_fixings == 0 {
            require!(past_fixings > 0, "no past fixings given");
            let value = payoff.value(running_accumulator / past_fixings as Real)
                * r_ts.discount_date(exercise_date, false)?;
            self.base.results_mut().instrument.value = Some(value);
            return Ok(());
        }

        require!(
            fixing_dates.last().copied().unwrap() <= exercise_date,
            "last fixing date must be before exercise date"
        );
        require!(
            StochasticProcess1D::time(&*self.process, &fixing_dates[0])? >= 0.0,
            "first fixing date is in the past"
        );
        require!(
            !fixing_dates.windows(2).any(|w| w[0] == w[1]),
            "two fixing dates are the same"
        );

        let accrued_average = if past_fixings != 0 {
            running_accumulator / (past_fixings + future_fixings) as Real
        } else {
            0.0
        };

        let strike = payoff.strike() - accrued_average;
        require!(strike >= 0.0, "effective strike should to be positive");

        let q_ts = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let vol_ref_date = vol_ts.reference_date()?;
        let vol_dc = vol_ts.day_counter().expect("vol day counter");
        let spot = self.process.state_variable().current_link()?.value()?;

        let value = if future_fixings > 1 {
            let mut fixing_times = Vec::with_capacity(future_fixings);
            let mut variances = Vec::with_capacity(future_fixings);
            for &fixing_date in &fixing_dates {
                fixing_times.push(vol_dc.year_fraction(vol_ref_date, fixing_date));
                variances.push(vol_ts.black_variance_date(fixing_date, strike, false)?);
            }

            let mut rho = Matrix::with_size(future_fixings, future_fixings);
            for i in 0..future_fixings {
                for j in i..future_fixings {
                    let corr = variances[i.min(j)] / (variances[i] * variances[j]).sqrt();
                    rho[(i, j)] = corr;
                    rho[(j, i)] = corr;
                }
            }

            let zero_ts = Handle::new(shared(FlatForward::with_rate(
                r_ts.reference_date()?,
                0.0,
                r_ts.day_counter().expect("risk-free day counter"),
                crate::interestrate::Compounding::Continuous,
                crate::time::frequency::Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>);

            let mut processes = Vec::with_capacity(future_fixings);
            for (i, &fixing_date) in fixing_dates.iter().enumerate() {
                let sig = vol_ts.black_vol_date(fixing_date, payoff.strike(), false)?
                    * (fixing_times[i] / fixing_times[future_fixings - 1]).sqrt();
                let forward_quote = shared(SimpleQuote::new(
                    spot * q_ts.discount_date(fixing_date, false)?
                        / r_ts.discount_date(fixing_date, false)?,
                ));
                let vol_quote = shared(SimpleQuote::new(sig));
                processes.push(shared(BlackProcess::new(
                    Handle::new(Shared::clone(&forward_quote) as Shared<dyn crate::quotes::Quote>),
                    Handle::clone(&zero_ts),
                    Handle::clone(&zero_ts),
                    Handle::new(shared(BlackConstantVol::with_quote(
                        vol_ref_date,
                        None,
                        Handle::new(Shared::clone(&vol_quote) as Shared<dyn crate::quotes::Quote>),
                        vol_dc.clone(),
                    )) as Shared<dyn BlackVolTermStructure>),
                )));
            }

            let weight = 1.0 / (future_fixings + past_fixings) as Real;
            let basket_payoff = AverageBasketPayoff::new(
                PlainVanillaPayoff::new(payoff.option_type(), strike),
                Array::filled(future_fixings, weight),
            );
            let basket_exercise: Shared<dyn crate::exercise::Exercise> =
                shared(EuropeanExercise::new(fixing_dates[future_fixings - 1]));

            let mut basket = BasketOption::new(
                basket_payoff,
                basket_exercise,
                Shared::clone(&self.settings),
            );
            basket
                .base_mut()
                .set_pricing_engine_silent(shared_mut(ChoiBasketEngine::with_params(
                    processes,
                    rho,
                    self.lambda,
                    self.max_nr_integration_steps,
                    false,
                    false,
                    Shared::clone(&self.settings),
                )) as SharedMut<dyn PricingEngine>);

            basket.npv()? * r_ts.discount_date(exercise_date, false)?
        } else {
            let fixing_date = fixing_dates[0];
            black_formula(
                payoff.option_type(),
                strike,
                spot / (past_fixings + future_fixings) as Real
                    * q_ts.discount_date(fixing_date, false)?
                    / r_ts.discount_date(fixing_date, false)?,
                vol_ts
                    .black_variance_date(fixing_date, strike, false)?
                    .sqrt(),
                r_ts.discount_date(exercise_date, false)?,
                0.0,
            )?
        };

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`ChoiAsianEngine`] to `option`.
pub fn set_choi_asian_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    process: Shared<GeneralizedBlackScholesProcess>,
    settings: Shared<Settings<Date>>,
) {
    let engine =
        shared_mut(ChoiAsianEngine::new(process, settings)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::DiscreteAveragingAsianOption;
    use crate::interestrate::Compounding;
    use crate::math::interpolations::linear::Linear;
    use crate::math::randomnumbers::rngtraits::LowDiscrepancy;
    use crate::option::OptionType;
    use crate::pricingengines::asian::{
        MakeMcDiscreteArithmeticApEngine, set_mc_discrete_arithmetic_average_price_asian_engine,
    };
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::shared::{Shared, shared};
    use crate::termstructures::volatility::{
        BlackConstantVol, BlackVarianceCurve, BlackVolTermStructure,
    };
    use crate::termstructures::yields::ZeroCurve;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;
    use crate::types::Volatility;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn crate::quotes::Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn crate::quotes::Quote>)
    }

    fn flat_rate(reference: Date, rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            reference,
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(reference: Date, vol: Volatility) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::new(
            reference,
            None,
            vol,
            Actual365Fixed::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    /// `asianoptions.cpp` `testChoiAsianEngineVsMC`.
    #[test]
    fn choi_asian_engine_vs_mc() {
        let settings = shared(Settings::new());
        let today = Date::new(5, Month::January, 2025);
        settings.set_evaluation_date(today);
        let maturity = today + Period::new(13, TimeUnit::Months);

        let mut fixing_dates = vec![today + Period::new(1, TimeUnit::Months)];
        while fixing_dates.last().copied().unwrap() < maturity - Period::new(1, TimeUnit::Months) {
            fixing_dates
                .push(fixing_dates.last().copied().unwrap() + Period::new(1, TimeUnit::Months));
        }

        let past_fixings_count = 2;
        let running_accumulator = past_fixings_count as Real * 97.0;
        let payoff = PlainVanillaPayoff::new(OptionType::Call, 110.0);
        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(maturity));

        let mut option = DiscreteAveragingAsianOption::new(
            AverageType::Arithmetic,
            running_accumulator,
            past_fixings_count,
            fixing_dates.clone(),
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();

        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0))),
            flat_rate(today, 0.1),
            flat_rate(today, 0.035),
            flat_vol(today, 0.5),
        ));

        let mc_engine = shared_mut(
            MakeMcDiscreteArithmeticApEngine::<LowDiscrepancy>::new(Shared::clone(&process))
                .with_samples(32000)
                .with_seed(43)
                .build()
                .unwrap(),
        );
        set_mc_discrete_arithmetic_average_price_asian_engine(&mut option, mc_engine);
        let expected = option.npv().unwrap();

        option
            .base_mut()
            .set_pricing_engine(shared_mut(ChoiAsianEngine::with_params(
                Shared::clone(&process),
                20.0,
                2 << 12,
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>);
        let calculated = option.npv().unwrap();
        let diff = (calculated - expected).abs();
        assert!(
            diff <= 0.01,
            "flat vol: expected={expected}, calculated={calculated}, diff={diff}"
        );

        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0))),
            Handle::new(shared(
                ZeroCurve::new(
                    vec![
                        today,
                        today + Period::new(3, TimeUnit::Months),
                        today + Period::new(13, TimeUnit::Months),
                    ],
                    vec![0.1, 0.0, 0.15],
                    Actual365Fixed::new(),
                    Linear,
                )
                .unwrap(),
            ) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(
                ZeroCurve::new(
                    vec![
                        today,
                        today + Period::new(3, TimeUnit::Months),
                        today + Period::new(13, TimeUnit::Months),
                    ],
                    vec![0.1, 0.2, 0.05],
                    Actual365Fixed::new(),
                    Linear,
                )
                .unwrap(),
            ) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(
                BlackVarianceCurve::new(
                    today,
                    &[
                        today + Period::new(1, TimeUnit::Days),
                        today + Period::new(100, TimeUnit::Days),
                        today + Period::new(13, TimeUnit::Months),
                    ],
                    &[0.25, 0.5, 0.4],
                    Actual365Fixed::new(),
                    false,
                )
                .unwrap(),
            ) as Shared<dyn BlackVolTermStructure>),
        ));

        let mc_engine = shared_mut(
            MakeMcDiscreteArithmeticApEngine::<LowDiscrepancy>::new(Shared::clone(&process))
                .with_samples(32000)
                .with_seed(43)
                .build()
                .unwrap(),
        );
        set_mc_discrete_arithmetic_average_price_asian_engine(&mut option, mc_engine);
        let expected = option.npv().unwrap();

        option
            .base_mut()
            .set_pricing_engine(shared_mut(ChoiAsianEngine::with_params(
                Shared::clone(&process),
                20.0,
                2 << 12,
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>);
        let calculated = option.npv().unwrap();
        let diff = (calculated - expected).abs();
        assert!(
            diff <= 0.01,
            "term structure: expected={expected}, calculated={calculated}, diff={diff}"
        );
    }

    /// `asianoptions.cpp` `testChoiAsianEngineSpecialCases`.
    #[test]
    fn choi_asian_engine_special_cases() {
        let settings = shared(Settings::new());
        let today = Date::new(5, Month::January, 2025);
        settings.set_evaluation_date(today);
        let maturity = today + Period::new(1, TimeUnit::Years);

        let past_fixings_count = 2;
        let running_accumulator = past_fixings_count as Real * 97.0;
        let mut fixing_dates = vec![today, today + Period::new(3, TimeUnit::Weeks)];

        let r_ts = flat_rate(today, 0.2);
        let q_ts = flat_rate(today, 0.075);
        let v_ts = flat_vol(today, 0.5);
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(100.0))),
            Handle::clone(&q_ts),
            Handle::clone(&r_ts),
            Handle::clone(&v_ts),
        ));

        let payoff = PlainVanillaPayoff::new(OptionType::Put, 103.0);
        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(maturity));

        let mut asian_option = DiscreteAveragingAsianOption::new(
            AverageType::Arithmetic,
            running_accumulator,
            past_fixings_count,
            fixing_dates.clone(),
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        asian_option
            .base_mut()
            .set_pricing_engine(shared_mut(ChoiAsianEngine::new(
                Shared::clone(&process),
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>);

        let calculated = asian_option.npv().unwrap();
        let r = r_ts.current_link().unwrap();
        let q = q_ts.current_link().unwrap();
        let v = v_ts.current_link().unwrap();
        let expected = black_formula(
            payoff.option_type(),
            payoff.strike()
                - (running_accumulator + 100.0) / (past_fixings_count + fixing_dates.len()) as Real,
            100.0 / (past_fixings_count + fixing_dates.len()) as Real
                * q.discount_date(fixing_dates[1], false).unwrap()
                / r.discount_date(fixing_dates[1], false).unwrap(),
            v.black_variance_date(fixing_dates[1], payoff.strike(), false)
                .unwrap()
                .sqrt(),
            r.discount_date(maturity, false).unwrap(),
            0.0,
        )
        .unwrap();

        let tol = 1000.0 * Real::EPSILON;
        assert!(
            (calculated - expected).abs() <= tol,
            "two future fixings: expected={expected}, calculated={calculated}"
        );

        fixing_dates = vec![today];
        let mut asian_option = DiscreteAveragingAsianOption::new(
            AverageType::Arithmetic,
            running_accumulator,
            past_fixings_count,
            fixing_dates.clone(),
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        asian_option
            .base_mut()
            .set_pricing_engine(shared_mut(ChoiAsianEngine::new(
                Shared::clone(&process),
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>);
        let calculated = asian_option.npv().unwrap();
        let expected = r.discount_date(maturity, false).unwrap()
            * payoff.value((running_accumulator + 100.0) / (past_fixings_count + 1) as Real);
        assert!(
            (calculated - expected).abs() <= tol,
            "one future fixing: expected={expected}, calculated={calculated}"
        );

        fixing_dates.clear();
        let mut asian_option = DiscreteAveragingAsianOption::new(
            AverageType::Arithmetic,
            running_accumulator,
            past_fixings_count,
            fixing_dates,
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        asian_option
            .base_mut()
            .set_pricing_engine(shared_mut(ChoiAsianEngine::new(
                Shared::clone(&process),
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>);
        let calculated = asian_option.npv().unwrap();
        let expected = r.discount_date(maturity, false).unwrap()
            * payoff.value(running_accumulator / past_fixings_count as Real);
        assert!(
            (calculated - expected).abs() <= tol,
            "no future fixings: expected={expected}, calculated={calculated}"
        );
    }
}
