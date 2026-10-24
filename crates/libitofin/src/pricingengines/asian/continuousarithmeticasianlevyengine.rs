//! Levy engine for continuous arithmetic-average Asian options.
//!
//! Port of `ql/pricingengines/asian/continuousarithmeticasianlevyengine.{hpp,cpp}`
//! (Haug, *Option Pricing Formulas*).

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, ContinuousAveragingAsianArguments, ContinuousAveragingAsianResults,
    StrikedTypePayoff, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::time::frequency::Frequency;
use crate::types::Real;

type EngineBase = GenericEngine<ContinuousAveragingAsianArguments, ContinuousAveragingAsianResults>;

/// Levy continuous arithmetic Asian engine.
pub struct ContinuousArithmeticAsianLevyEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    current_average: Handle<dyn Quote>,
}

impl ContinuousArithmeticAsianLevyEngine {
    /// `ContinuousArithmeticAsianLevyEngine(process, currentAverage)`.
    ///
    /// The averaging start date must be supplied on the option via
    /// [`ContinuousAveragingAsianOption::with_start_date`](crate::instruments::ContinuousAveragingAsianOption::with_start_date).
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        current_average: Handle<dyn Quote>,
    ) -> Self {
        let base = EngineBase::new(
            ContinuousAveragingAsianArguments::default(),
            ContinuousAveragingAsianResults::default(),
        );
        base.register_with(process.observable());
        current_average.register_observer(&base.observer());
        Self {
            base,
            process,
            current_average,
        }
    }
}

impl AsObservable for ContinuousArithmeticAsianLevyEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for ContinuousArithmeticAsianLevyEngine {
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
            "not an Arithmetic average option"
        );
        let exercise = Shared::clone(args.exercise.as_ref().expect("validated"));
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European Option"
        );

        let Some(start_date) = args.start_date else {
            fail!("start date not provided");
        };
        let payoff = args.payoff.expect("validated");

        let r_ts = self.process.risk_free_rate().current_link()?;
        let q_ts = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let reference_date = r_ts.reference_date()?;
        require!(
            start_date <= reference_date,
            "start date must be earlier than or equal to reference date"
        );

        let rfdc = r_ts.require_day_counter()?;
        let divdc = q_ts.require_day_counter()?;
        let spot = self.process.state_variable().current_link()?.value()?;

        let maturity = exercise.last_date();
        let t = rfdc.year_fraction(start_date, maturity);
        let t2 = rfdc.year_fraction(reference_date, maturity);
        let strike = payoff.strike();
        let volatility = vol_ts.black_vol_date(maturity, strike, false)?;

        let n = CumulativeNormalDistribution::standard();
        let risk_free_rate = r_ts
            .zero_rate_date(
                maturity,
                rfdc.clone(),
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let dividend_yield = q_ts
            .zero_rate_date(
                maturity,
                divdc,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let b = risk_free_rate - dividend_yield;

        let se = if b.abs() > 1000.0 * Real::EPSILON {
            (spot / (t * b)) * (((b - risk_free_rate) * t2).exp() - (-risk_free_rate * t2).exp())
        } else {
            spot * t2 / t * (-risk_free_rate * t2).exp()
        };

        let x = if t2 < t {
            require!(
                !self.current_average.is_empty() && self.current_average.current_link()?.is_valid(),
                "current average required for seasoned option"
            );
            strike - ((t - t2) / t) * self.current_average.current_link()?.value()?
        } else {
            strike
        };

        let m = if b.abs() > 1000.0 * Real::EPSILON {
            ((b * t2).exp() - 1.0) / b
        } else {
            t2
        };

        let big_m = (2.0 * spot * spot / (b + volatility * volatility))
            * ((((2.0 * b + volatility * volatility) * t2).exp() - 1.0)
                / (2.0 * b + volatility * volatility)
                - m);

        let d = big_m / (t * t);
        let v = d.ln() - 2.0 * (risk_free_rate * t2 + se.ln());
        let d1 = (1.0 / v.sqrt()) * (d.ln() / 2.0 - x.ln());
        let d2 = d1 - v.sqrt();

        let value = match payoff.option_type() {
            OptionType::Call => se * n.value(d1) - x * (-risk_free_rate * t2).exp() * n.value(d2),
            OptionType::Put => {
                se * n.value(d1) - x * (-risk_free_rate * t2).exp() * n.value(d2) - se
                    + x * (-risk_free_rate * t2).exp()
            }
        };

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`ContinuousArithmeticAsianLevyEngine`] to `option`.
pub fn set_continuous_arithmetic_asian_levy_engine(
    option: &mut crate::instruments::ContinuousAveragingAsianOption,
    process: Shared<GeneralizedBlackScholesProcess>,
    current_average: Handle<dyn Quote>,
) {
    let engine = shared_mut(ContinuousArithmeticAsianLevyEngine::new(
        process,
        current_average,
    )) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::instruments::{ContinuousAveragingAsianOption, PlainVanillaPayoff};
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::SerialNumber;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::types::{Natural, Volatility};

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn flat_rate(reference: Date, rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            reference,
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(reference: Date, vol: Volatility) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::new(
            reference,
            None,
            vol,
            Actual360::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    struct Case {
        option_type: OptionType,
        spot: Real,
        current_average: Real,
        strike: Real,
        dividend_yield: Real,
        risk_free_rate: Real,
        volatility: Real,
        length: Natural,
        elapsed: Natural,
        result: Real,
    }

    /// Haug p.99–100 via `asianoptions.cpp` `testLevyEngine` (tol 1e-4).
    #[test]
    fn continuous_arithmetic_asian_levy_matches_haug() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let cases = [
            Case {
                option_type: OptionType::Call,
                spot: 6.80,
                current_average: 6.80,
                strike: 6.90,
                dividend_yield: 0.09,
                risk_free_rate: 0.07,
                volatility: 0.14,
                length: 180,
                elapsed: 0,
                result: 0.0944,
            },
            Case {
                option_type: OptionType::Put,
                spot: 6.80,
                current_average: 6.80,
                strike: 6.90,
                dividend_yield: 0.09,
                risk_free_rate: 0.07,
                volatility: 0.14,
                length: 180,
                elapsed: 0,
                result: 0.2237,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 95.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.15,
                length: 270,
                elapsed: 0,
                result: 7.0544,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 95.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.15,
                length: 270,
                elapsed: 90,
                result: 5.6731,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 95.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.15,
                length: 270,
                elapsed: 180,
                result: 5.0806,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 95.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.35,
                length: 270,
                elapsed: 0,
                result: 10.1213,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 95.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.35,
                length: 270,
                elapsed: 90,
                result: 6.9705,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 95.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.35,
                length: 270,
                elapsed: 180,
                result: 5.1411,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 100.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.15,
                length: 270,
                elapsed: 0,
                result: 3.7845,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 100.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.15,
                length: 270,
                elapsed: 90,
                result: 1.9964,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 100.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.15,
                length: 270,
                elapsed: 180,
                result: 0.6722,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 100.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.35,
                length: 270,
                elapsed: 0,
                result: 7.5038,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 100.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.35,
                length: 270,
                elapsed: 90,
                result: 4.0687,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 100.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.35,
                length: 270,
                elapsed: 180,
                result: 1.4222,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 105.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.15,
                length: 270,
                elapsed: 0,
                result: 1.6729,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 105.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.15,
                length: 270,
                elapsed: 90,
                result: 0.3565,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 105.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.15,
                length: 270,
                elapsed: 180,
                result: 0.0004,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 105.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.35,
                length: 270,
                elapsed: 0,
                result: 5.4071,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 105.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.35,
                length: 270,
                elapsed: 90,
                result: 2.1359,
            },
            Case {
                option_type: OptionType::Call,
                spot: 100.0,
                current_average: 100.0,
                strike: 105.0,
                dividend_yield: 0.05,
                risk_free_rate: 0.1,
                volatility: 0.35,
                length: 270,
                elapsed: 180,
                result: 0.1552,
            },
        ];

        let tolerance = 1.0e-4;
        for (i, case) in cases.iter().enumerate() {
            let spot = shared(SimpleQuote::new(case.spot));
            let average = shared(SimpleQuote::new(case.current_average));
            let process = shared(BlackScholesMertonProcess::new(
                quote_handle(&spot),
                flat_rate(today, case.dividend_yield),
                flat_rate(today, case.risk_free_rate),
                flat_vol(today, case.volatility),
            ));

            let start_date = today - case.elapsed as SerialNumber;
            let maturity = start_date + case.length as SerialNumber;
            let mut option = ContinuousAveragingAsianOption::with_start_date(
                AverageType::Arithmetic,
                start_date,
                PlainVanillaPayoff::new(case.option_type, case.strike),
                shared(EuropeanExercise::new(maturity)),
                Shared::clone(&settings),
            )
            .unwrap();

            set_continuous_arithmetic_asian_levy_engine(
                &mut option,
                Shared::clone(&process),
                quote_handle(&average),
            );
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - case.result).abs() <= tolerance,
                "case {i}: expected {}, got {calculated}",
                case.result
            );
        }
    }

    /// `asianoptions.cpp` `testContinuousSeasonedAsianOptions`.
    #[test]
    fn continuous_seasoned_asian_options_ordering() {
        use crate::pricingengines::asian::set_analytic_continuous_geometric_average_price_asian_engine;
        use crate::time::calendars::Target;
        use crate::time::daycounters::actual365fixed::Actual365Fixed;

        let settings = shared(Settings::new());
        let today = Date::new(15, Month::November, 2025);
        settings.set_evaluation_date(today);

        let settlement = Date::new(17, Month::November, 2025);
        let maturity = Date::new(17, Month::November, 2026);
        let start_date = Date::new(17, Month::August, 2025);

        let spot = shared(SimpleQuote::new(100.0));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            Handle::new(shared(FlatForward::with_rate(
                settlement,
                0.03,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(FlatForward::with_rate(
                settlement,
                0.06,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(BlackConstantVol::new(
                settlement,
                Some(Target::new()),
                0.20,
                Actual365Fixed::new(),
            )) as Shared<dyn BlackVolTermStructure>),
        ));

        let payoff = PlainVanillaPayoff::new(OptionType::Put, 100.0);
        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(maturity));

        let zero_average = shared(SimpleQuote::new(0.0));
        let mut fresh = ContinuousAveragingAsianOption::with_start_date(
            AverageType::Arithmetic,
            settlement,
            payoff,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        set_continuous_arithmetic_asian_levy_engine(
            &mut fresh,
            Shared::clone(&process),
            quote_handle(&zero_average),
        );
        let fresh_npv = fresh.npv().unwrap();

        let low_average = shared(SimpleQuote::new(98.5));
        let mut seasoned = ContinuousAveragingAsianOption::with_start_date(
            AverageType::Arithmetic,
            start_date,
            PlainVanillaPayoff::new(OptionType::Put, 100.0),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        set_continuous_arithmetic_asian_levy_engine(
            &mut seasoned,
            Shared::clone(&process),
            quote_handle(&low_average),
        );
        let seasoned_npv = seasoned.npv().unwrap();
        assert!(
            seasoned_npv < fresh_npv,
            "seasoned put NPV ({seasoned_npv}) should be below fresh ({fresh_npv}) \
             when current average (98.5) is below strike (100)"
        );

        let high_average = shared(SimpleQuote::new(102.0));
        let mut seasoned_high = ContinuousAveragingAsianOption::with_start_date(
            AverageType::Arithmetic,
            start_date,
            PlainVanillaPayoff::new(OptionType::Put, 100.0),
            Shared::clone(&exercise),
            Shared::clone(&settings),
        )
        .unwrap();
        set_continuous_arithmetic_asian_levy_engine(
            &mut seasoned_high,
            Shared::clone(&process),
            quote_handle(&high_average),
        );
        let seasoned_high_npv = seasoned_high.npv().unwrap();
        assert!(
            seasoned_high_npv < seasoned_npv,
            "seasoned put with higher average ({seasoned_high_npv}) should be below \
             seasoned with lower average ({seasoned_npv})"
        );

        let mut seasoned_geometric = ContinuousAveragingAsianOption::with_start_date(
            AverageType::Geometric,
            start_date,
            PlainVanillaPayoff::new(OptionType::Put, 100.0),
            exercise,
            settings,
        )
        .unwrap();
        set_analytic_continuous_geometric_average_price_asian_engine(
            &mut seasoned_geometric,
            process,
        );
        let err = seasoned_geometric.npv().unwrap_err();
        assert!(
            err.message()
                .contains("seasoned continuous geometric Asian options not yet supported"),
            "unexpected error: {}",
            err.message()
        );
    }
}
