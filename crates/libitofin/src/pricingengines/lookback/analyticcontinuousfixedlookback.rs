//! Analytic continuous fixed-strike lookback engine.
//!
//! Port of `ql/pricingengines/lookback/analyticcontinuousfixedlookback.{hpp,cpp}`
//! (Haug, *Option Pricing Formulas*, pp.63–64).

use crate::errors::QlResult;
use crate::instrument::Instrument;
use crate::instruments::{
    ContinuousFixedLookbackArguments, ContinuousFixedLookbackResults, StrikedTypePayoff, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::{Real, Time, Volatility};

type EngineBase = GenericEngine<ContinuousFixedLookbackArguments, ContinuousFixedLookbackResults>;

/// Pricing engine for European continuous fixed-strike lookbacks.
pub struct AnalyticContinuousFixedLookbackEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    f: CumulativeNormalDistribution,
}

impl AnalyticContinuousFixedLookbackEngine {
    /// `AnalyticContinuousFixedLookbackEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            ContinuousFixedLookbackArguments::default(),
            ContinuousFixedLookbackResults::default(),
        );
        base.register_with(process.observable());
        Self {
            base,
            process,
            f: CumulativeNormalDistribution::standard(),
        }
    }

    fn underlying(&self) -> QlResult<Real> {
        self.process.x0()
    }

    fn strike(&self) -> Real {
        let args = self.base.arguments();
        args.payoff.expect("validated").strike()
    }

    fn residual_time(&self) -> QlResult<Time> {
        let exercise = self
            .base
            .arguments()
            .exercise
            .as_ref()
            .expect("validated");
        StochasticProcess1D::time(&*self.process, &exercise.last_date())
    }

    fn minmax(&self) -> Real {
        self.base.arguments().minmax.expect("validated")
    }

    fn volatility(&self) -> QlResult<Volatility> {
        let t = self.residual_time()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        vol_ts.black_vol(t, self.strike(), true)
    }

    fn std_deviation(&self) -> QlResult<Real> {
        Ok(self.volatility()? * self.residual_time()?.sqrt())
    }

    fn risk_free_rate(&self) -> QlResult<Real> {
        let t = self.residual_time()?;
        let r_ts = self.process.risk_free_rate().current_link()?;
        Ok(r_ts
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn risk_free_discount(&self) -> QlResult<Real> {
        let t = self.residual_time()?;
        let r_ts = self.process.risk_free_rate().current_link()?;
        r_ts.discount(t, false)
    }

    fn dividend_yield(&self) -> QlResult<Real> {
        let t = self.residual_time()?;
        let q_ts = self.process.dividend_yield().current_link()?;
        Ok(q_ts
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn dividend_discount(&self) -> QlResult<Real> {
        let t = self.residual_time()?;
        let q_ts = self.process.dividend_yield().current_link()?;
        q_ts.discount(t, false)
    }

    fn a(&self, eta: Real) -> QlResult<Real> {
        let vol = self.volatility()?;
        let lambda = 2.0 * (self.risk_free_rate()? - self.dividend_yield()?) / (vol * vol);
        let ss = self.underlying()? / self.minmax();
        let std_dev = self.std_deviation()?;
        let d1 = ss.ln() / std_dev + 0.5 * (lambda + 1.0) * std_dev;
        let n1 = self.f.value(eta * d1);
        let n2 = self.f.value(eta * (d1 - std_dev));
        let n3 = self.f.value(eta * (d1 - lambda * std_dev));
        let n4 = self.f.value(eta * d1);
        let pow_ss = ss.powf(-lambda);
        let underlying = self.underlying()?;
        let minmax = self.minmax();
        let rf_disc = self.risk_free_discount()?;
        let div_disc = self.dividend_discount()?;
        Ok(eta
            * (underlying * div_disc * n1
                - minmax * rf_disc * n2
                - underlying * rf_disc * (pow_ss * n3 - div_disc * n4 / rf_disc) / lambda))
    }

    fn b(&self, eta: Real) -> QlResult<Real> {
        let vol = self.volatility()?;
        let lambda = 2.0 * (self.risk_free_rate()? - self.dividend_yield()?) / (vol * vol);
        let ss = self.underlying()? / self.strike();
        let std_dev = self.std_deviation()?;
        let d1 = ss.ln() / std_dev + 0.5 * (lambda + 1.0) * std_dev;
        let n1 = self.f.value(eta * d1);
        let n2 = self.f.value(eta * (d1 - std_dev));
        let n3 = self.f.value(eta * (d1 - lambda * std_dev));
        let n4 = self.f.value(eta * d1);
        let pow_ss = ss.powf(-lambda);
        let underlying = self.underlying()?;
        let strike = self.strike();
        let rf_disc = self.risk_free_discount()?;
        let div_disc = self.dividend_discount()?;
        Ok(eta
            * (underlying * div_disc * n1
                - strike * rf_disc * n2
                - underlying * rf_disc * (pow_ss * n3 - div_disc * n4 / rf_disc) / lambda))
    }

    fn c(&self, eta: Real) -> QlResult<Real> {
        Ok(eta * self.risk_free_discount()? * (self.minmax() - self.strike()))
    }
}

impl AsObservable for AnalyticContinuousFixedLookbackEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticContinuousFixedLookbackEngine {
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
        let (option_type, strike) = {
            let args = self.base.arguments();
            let payoff = args.payoff.expect("validated");
            (payoff.option_type(), payoff.strike())
        };

        require!(self.process.x0()? > 0.0, "negative or null underlying");

        let minmax = self.minmax();
        let value = match option_type {
            OptionType::Call => {
                require!(strike >= 0.0, "Strike must be positive or null");
                if strike <= minmax {
                    self.a(1.0)? + self.c(1.0)?
                } else {
                    self.b(1.0)?
                }
            }
            OptionType::Put => {
                require!(strike > 0.0, "Strike must be positive");
                if strike >= minmax {
                    self.a(-1.0)? + self.c(-1.0)?
                } else {
                    self.b(-1.0)?
                }
            }
        };

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticContinuousFixedLookbackEngine`] to `option`.
pub fn set_analytic_continuous_fixed_lookback_engine(
    option: &mut crate::instruments::ContinuousFixedLookbackOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine =
        shared_mut(AnalyticContinuousFixedLookbackEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instruments::{ContinuousFixedLookbackOption, PlainVanillaPayoff};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::types::{Rate, Volatility};

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
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

    fn time_to_days(t: Time) -> i32 {
        (t * 360.0).round() as i32
    }

    struct Case {
        option_type: OptionType,
        strike: Real,
        minmax: Real,
        spot: Real,
        q: Rate,
        r: Rate,
        t: Time,
        v: Volatility,
        result: Real,
        tol: Real,
    }

    /// `lookbackoptions.cpp` `testAnalyticContinuousFixedLookback`.
    #[test]
    fn analytic_continuous_fixed_lookback_matches_haug() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let cases = [
            // Haug 1998 pp.63–64
            Case { option_type: OptionType::Call, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.10, result: 13.2687, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.20, result: 18.9263, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.30, result: 24.9857, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.10, result: 8.5126, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.20, result: 14.1702, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.30, result: 20.2296, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.10, result: 4.3908, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.20, result: 9.8905, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.30, result: 15.8512, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.10, result: 18.3241, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.20, result: 26.0731, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.30, result: 34.7116, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.10, result: 13.8000, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.20, result: 21.5489, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.30, result: 30.1874, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.10, result: 9.5445, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.20, result: 17.2965, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.30, result: 25.9002, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.10, result: 0.6899, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.20, result: 4.4448, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.30, result: 8.9213, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.10, result: 3.3917, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.20, result: 8.3177, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.30, result: 13.1579, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.10, result: 8.1478, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.20, result: 13.0739, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 0.50, v: 0.30, result: 17.9140, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.10, result: 1.0534, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.20, result: 6.2813, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 95.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.30, result: 12.2376, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.10, result: 3.8079, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.20, result: 10.1294, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 100.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.30, result: 16.3889, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.10, result: 8.3321, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.20, result: 14.6536, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 105.0, minmax: 100.0, spot: 100.0, q: 0.00, r: 0.10, t: 1.00, v: 0.30, result: 20.9130, tol: 1.0e-4 },
        ];

        let spot = shared(SimpleQuote::new(0.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.0));
        let vol = shared(SimpleQuote::new(0.0));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        for (i, case) in cases.iter().enumerate() {
            spot.set_value(case.spot);
            q_rate.set_value(case.q);
            r_rate.set_value(case.r);
            vol.set_value(case.v);

            let exercise = shared(EuropeanExercise::new(today + time_to_days(case.t)));
            let mut option = ContinuousFixedLookbackOption::new(
                case.minmax,
                PlainVanillaPayoff::new(case.option_type, case.strike),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_continuous_fixed_lookback_engine(
                &mut option,
                Shared::clone(&process),
            );
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - case.result).abs() <= case.tol,
                "case {i}: expected {}, got {calculated} (tol {})",
                case.result,
                case.tol
            );
        }
    }
}
