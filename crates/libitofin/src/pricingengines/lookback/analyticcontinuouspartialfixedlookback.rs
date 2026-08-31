//! Analytic continuous partial-time fixed-strike lookback engine.
//!
//! Port of `ql/pricingengines/lookback/analyticcontinuouspartialfixedlookback.{hpp,cpp}`
//! (Haug, *Option Pricing Formulas*, 2nd ed., p.148).

use crate::errors::QlResult;
use crate::instrument::Instrument;
use crate::instruments::{
    ContinuousPartialFixedLookbackArguments, ContinuousPartialFixedLookbackResults,
    StrikedTypePayoff, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionWe04DP;
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

type EngineBase = GenericEngine<
    ContinuousPartialFixedLookbackArguments,
    ContinuousPartialFixedLookbackResults,
>;

/// Pricing engine for European continuous partial-time fixed-strike lookbacks.
pub struct AnalyticContinuousPartialFixedLookbackEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    f: CumulativeNormalDistribution,
}

impl AnalyticContinuousPartialFixedLookbackEngine {
    /// `AnalyticContinuousPartialFixedLookbackEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            ContinuousPartialFixedLookbackArguments::default(),
            ContinuousPartialFixedLookbackResults::default(),
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
        self.base
            .arguments()
            .payoff
            .expect("validated")
            .strike()
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

    fn lookback_period_start_time(&self) -> QlResult<Time> {
        let lookback_period_start = self
            .base
            .arguments()
            .lookback_period_start
            .expect("validated");
        StochasticProcess1D::time(&*self.process, &lookback_period_start)
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
        let residual_time = self.residual_time()?;
        let lookback_period_start_time = self.lookback_period_start_time()?;
        let different_start_of_lookback = lookback_period_start_time != residual_time;
        let carry = self.risk_free_rate()? - self.dividend_yield()?;
        let vol = self.volatility()?;
        let x = 2.0 * carry / (vol * vol);
        let s = self.underlying()? / self.strike();
        let std_dev = self.std_deviation()?;

        let ls = s.ln();
        let d1 = ls / std_dev + 0.5 * (x + 1.0) * std_dev;
        let d2 = d1 - std_dev;

        let (e1, e2) = if different_start_of_lookback {
            let dt = residual_time - lookback_period_start_time;
            let sqrt_dt = dt.sqrt();
            let e1 = (carry + vol * vol / 2.0) * dt / (vol * sqrt_dt);
            (e1, e1 - vol * sqrt_dt)
        } else {
            (0.0, 0.0)
        };

        let sqrt_t1 = lookback_period_start_time.sqrt();
        let f1 = (ls + (carry + vol * vol / 2.0) * lookback_period_start_time) / (vol * sqrt_t1);
        let f2 = f1 - vol * sqrt_t1;

        let n1 = self.f.value(eta * d1);
        let n2 = self.f.value(eta * d2);

        let (cnbn1, cnbn2, cnbn3) = if different_start_of_lookback {
            let t_ratio = lookback_period_start_time / residual_time;
            (
                BivariateCumulativeNormalDistributionWe04DP::new(-t_ratio.sqrt())?,
                BivariateCumulativeNormalDistributionWe04DP::new((1.0 - t_ratio).sqrt())?,
                BivariateCumulativeNormalDistributionWe04DP::new(-(1.0 - t_ratio).sqrt())?,
            )
        } else {
            (
                BivariateCumulativeNormalDistributionWe04DP::new(-1.0)?,
                BivariateCumulativeNormalDistributionWe04DP::new(0.0)?,
                BivariateCumulativeNormalDistributionWe04DP::new(0.0)?,
            )
        };

        let n3 = cnbn1.value(
            eta * (d1 - x * std_dev),
            eta * (-f1 + 2.0 * carry * sqrt_t1 / vol),
        );
        let n4 = cnbn2.value(eta * e1, eta * d1);
        let n5 = cnbn3.value(-eta * e1, eta * d1);
        let n6 = cnbn1.value(eta * f2, -eta * d2);
        let n7 = self.f.value(eta * f1);
        let n8 = self.f.value(-eta * e2);

        let pow_s = s.powf(-x);
        let carry_discount = (-carry * (residual_time - lookback_period_start_time)).exp();
        let underlying = self.underlying()?;
        let strike = self.strike();
        let rf_disc = self.risk_free_discount()?;
        let div_disc = self.dividend_discount()?;

        Ok(eta
            * (underlying * div_disc * n1
                - strike * rf_disc * n2
                + underlying * rf_disc / x * (-pow_s * n3 + div_disc / rf_disc * n4)
                - underlying * div_disc * n5
                - strike * rf_disc * n6
                + carry_discount
                    * div_disc
                    * (1.0 - 0.5 * vol * vol / carry)
                    * underlying
                    * n7
                    * n8))
    }
}

impl AsObservable for AnalyticContinuousPartialFixedLookbackEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticContinuousPartialFixedLookbackEngine {
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

        let value = match option_type {
            OptionType::Call => {
                require!(strike >= 0.0, "Strike must be positive or null");
                self.a(1.0)?
            }
            OptionType::Put => {
                require!(strike > 0.0, "Strike must be positive");
                self.a(-1.0)?
            }
        };

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticContinuousPartialFixedLookbackEngine`] to `option`.
pub fn set_analytic_continuous_partial_fixed_lookback_engine(
    option: &mut crate::instruments::ContinuousPartialFixedLookbackOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticContinuousPartialFixedLookbackEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instruments::{ContinuousPartialFixedLookbackOption, PlainVanillaPayoff};
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
        spot: Real,
        q: Rate,
        r: Rate,
        t: Time,
        v: Volatility,
        t1: Time,
        result: Real,
        tol: Real,
    }

    /// `lookbackoptions.cpp` `testAnalyticContinuousPartialFixedLookback`.
    #[test]
    fn analytic_continuous_partial_fixed_lookback_matches_haug() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let cases = [
            Case { option_type: OptionType::Call, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.25, result: 20.2845, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.5, result: 19.6239, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.75, result: 18.6244, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.25, result: 4.0432, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.5, result: 3.958, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.75, result: 3.7015, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.25, result: 27.5385, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.5, result: 25.8126, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.75, result: 23.4957, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.25, result: 11.4895, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.5, result: 10.8995, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.75, result: 9.8244, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.25, result: 35.4578, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.5, result: 32.7172, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.75, result: 29.1473, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.25, result: 19.725, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.5, result: 18.4025, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.75, result: 16.2976, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.25, result: 0.4973, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.5, result: 0.4632, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.75, result: 0.3863, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.25, result: 12.6978, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.5, result: 10.9492, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, t1: 0.75, result: 9.1555, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.25, result: 4.5863, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.5, result: 4.1925, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.75, result: 3.5831, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.25, result: 19.0255, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.5, result: 16.9433, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, t1: 0.75, result: 14.6505, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.25, result: 9.9348, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.5, result: 9.1111, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 90.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.75, result: 7.9267, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.25, result: 25.2112, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.5, result: 22.8217, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, strike: 110.0, spot: 100.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, t1: 0.75, result: 20.0566, tol: 1.0e-4 },
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
            let lookback_start = today + time_to_days(case.t1);
            let mut option = ContinuousPartialFixedLookbackOption::new(
                lookback_start,
                PlainVanillaPayoff::new(case.option_type, case.strike),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_continuous_partial_fixed_lookback_engine(
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
