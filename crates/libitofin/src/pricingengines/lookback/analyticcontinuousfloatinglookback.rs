//! Analytic continuous floating-strike lookback engine.
//!
//! Port of `ql/pricingengines/lookback/analyticcontinuousfloatinglookback.{hpp,cpp}`
//! (Haug, *Option Pricing Formulas*, pp.61–62).

use crate::errors::QlResult;
use crate::instrument::Instrument;
use crate::instruments::{
    ContinuousFloatingLookbackArguments, ContinuousFloatingLookbackResults, TypePayoff,
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

type EngineBase =
    GenericEngine<ContinuousFloatingLookbackArguments, ContinuousFloatingLookbackResults>;

/// Pricing engine for European continuous floating-strike lookbacks.
pub struct AnalyticContinuousFloatingLookbackEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    f: CumulativeNormalDistribution,
}

impl AnalyticContinuousFloatingLookbackEngine {
    /// `AnalyticContinuousFloatingLookbackEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            ContinuousFloatingLookbackArguments::default(),
            ContinuousFloatingLookbackResults::default(),
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

    fn residual_time(&self) -> QlResult<Time> {
        let exercise = self.base.arguments().exercise.as_ref().expect("validated");
        StochasticProcess1D::time(&*self.process, &exercise.last_date())
    }

    fn minmax(&self) -> Real {
        self.base.arguments().minmax.expect("validated")
    }

    fn volatility(&self) -> QlResult<Volatility> {
        let t = self.residual_time()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        vol_ts.black_vol(t, self.minmax(), true)
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
        let s = self.underlying()? / self.minmax();
        let std_dev = self.std_deviation()?;
        let d1 = s.ln() / std_dev + 0.5 * (lambda + 1.0) * std_dev;
        let n1 = self.f.value(eta * d1);
        let n2 = self.f.value(eta * (d1 - std_dev));
        let n3 = self.f.value(eta * (-d1 + lambda * std_dev));
        let n4 = self.f.value(eta * -d1);
        let pow_s = s.powf(-lambda);
        let underlying = self.underlying()?;
        let minmax = self.minmax();
        let rf_disc = self.risk_free_discount()?;
        let div_disc = self.dividend_discount()?;
        Ok(eta
            * ((underlying * div_disc * n1 - minmax * rf_disc * n2)
                + (underlying * rf_disc * (pow_s * n3 - div_disc * n4 / rf_disc) / lambda)))
    }
}

impl AsObservable for AnalyticContinuousFloatingLookbackEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticContinuousFloatingLookbackEngine {
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
        let option_type = {
            let args = self.base.arguments();
            let payoff = args.payoff.expect("validated");
            payoff.option_type()
        };

        require!(self.process.x0()? > 0.0, "negative or null underlying");

        let value = match option_type {
            OptionType::Call => self.a(1.0)?,
            OptionType::Put => self.a(-1.0)?,
        };

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticContinuousFloatingLookbackEngine`] to `option`.
pub fn set_analytic_continuous_floating_lookback_engine(
    option: &mut crate::instruments::ContinuousFloatingLookbackOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticContinuousFloatingLookbackEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instruments::{ContinuousFloatingLookbackOption, FloatingTypePayoff};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::types::{Rate, Volatility};

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

    struct Case {
        option_type: OptionType,
        minmax: Real,
        spot: Real,
        q: Rate,
        r: Rate,
        t: Time,
        v: Volatility,
        result: Real,
        tol: Real,
    }

    /// `lookbackoptions.cpp` `testAnalyticContinuousFloatingLookback`.
    #[test]
    fn analytic_continuous_floating_lookback_matches_haug_broadie() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let cases = [
            // Haug 1998 pp.61–62
            Case {
                option_type: OptionType::Call,
                minmax: 100.0,
                spot: 120.0,
                q: 0.06,
                r: 0.10,
                t: 0.50,
                v: 0.30,
                result: 25.3533,
                tol: 1.0e-4,
            },
            // Broadie, Glasserman & Kou 1999 pp.70–74
            Case {
                option_type: OptionType::Call,
                minmax: 100.0,
                spot: 100.0,
                q: 0.00,
                r: 0.05,
                t: 1.00,
                v: 0.30,
                result: 23.7884,
                tol: 1.0e-4,
            },
            Case {
                option_type: OptionType::Call,
                minmax: 100.0,
                spot: 100.0,
                q: 0.00,
                r: 0.05,
                t: 0.20,
                v: 0.30,
                result: 10.7190,
                tol: 1.0e-4,
            },
            Case {
                option_type: OptionType::Call,
                minmax: 100.0,
                spot: 110.0,
                q: 0.00,
                r: 0.05,
                t: 0.20,
                v: 0.30,
                result: 14.4597,
                tol: 1.0e-4,
            },
            Case {
                option_type: OptionType::Put,
                minmax: 100.0,
                spot: 100.0,
                q: 0.00,
                r: 0.10,
                t: 0.50,
                v: 0.30,
                result: 15.3526,
                tol: 1.0e-4,
            },
            Case {
                option_type: OptionType::Put,
                minmax: 110.0,
                spot: 100.0,
                q: 0.00,
                r: 0.10,
                t: 0.50,
                v: 0.30,
                result: 16.8468,
                tol: 1.0e-4,
            },
            Case {
                option_type: OptionType::Put,
                minmax: 120.0,
                spot: 100.0,
                q: 0.00,
                r: 0.10,
                t: 0.50,
                v: 0.30,
                result: 21.0645,
                tol: 1.0e-4,
            },
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
            let mut option = ContinuousFloatingLookbackOption::new(
                case.minmax,
                FloatingTypePayoff::new(case.option_type),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_continuous_floating_lookback_engine(&mut option, Shared::clone(&process));
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
