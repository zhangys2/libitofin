//! Analytic continuous partial-time floating-strike lookback engine.
//!
//! Port of `ql/pricingengines/lookback/analyticcontinuouspartialfloatinglookback.{hpp,cpp}`
//! (Haug, *Option Pricing Formulas*, 2nd ed., p.146).

use crate::errors::QlResult;
use crate::instrument::Instrument;
use crate::instruments::{
    ContinuousPartialFloatingLookbackArguments, ContinuousPartialFloatingLookbackResults,
    TypePayoff,
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
    ContinuousPartialFloatingLookbackArguments,
    ContinuousPartialFloatingLookbackResults,
>;

/// Pricing engine for European continuous partial-time floating-strike lookbacks.
pub struct AnalyticContinuousPartialFloatingLookbackEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    f: CumulativeNormalDistribution,
}

impl AnalyticContinuousPartialFloatingLookbackEngine {
    /// `AnalyticContinuousPartialFloatingLookbackEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            ContinuousPartialFloatingLookbackArguments::default(),
            ContinuousPartialFloatingLookbackResults::default(),
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

    fn lambda(&self) -> Real {
        self.base.arguments().lambda.expect("validated")
    }

    fn lookback_period_end_time(&self) -> QlResult<Time> {
        let lookback_period_end = self
            .base
            .arguments()
            .lookback_period_end
            .expect("validated");
        StochasticProcess1D::time(&*self.process, &lookback_period_end)
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
        let residual_time = self.residual_time()?;
        let lookback_period_end_time = self.lookback_period_end_time()?;
        let full_lookback_period = lookback_period_end_time == residual_time;
        let carry = self.risk_free_rate()? - self.dividend_yield()?;
        let vol = self.volatility()?;
        let x = 2.0 * carry / (vol * vol);
        let s = self.underlying()? / self.minmax();
        let std_dev = self.std_deviation()?;

        let ls = s.ln();
        let d1 = ls / std_dev + 0.5 * (x + 1.0) * std_dev;
        let d2 = d1 - std_dev;

        let (e1, e2) = if full_lookback_period {
            (0.0, 0.0)
        } else {
            let dt = residual_time - lookback_period_end_time;
            let sqrt_dt = dt.sqrt();
            let e1 = (carry + vol * vol / 2.0) * dt / (vol * sqrt_dt);
            (e1, e1 - vol * sqrt_dt)
        };

        let sqrt_t1 = lookback_period_end_time.sqrt();
        let f1 = (ls + (carry + vol * vol / 2.0) * lookback_period_end_time) / (vol * sqrt_t1);
        let f2 = f1 - vol * sqrt_t1;

        let l1 = self.lambda().ln() / vol;
        let g1 = l1 / residual_time.sqrt();

        let n1 = self.f.value(eta * (d1 - g1));
        let n2 = self.f.value(eta * (d2 - g1));

        let (n3, n4, n5, n6, n7) = if full_lookback_period {
            let cnbn1 = BivariateCumulativeNormalDistributionWe04DP::new(1.0)?;
            let n3 = cnbn1.value(
                eta * (-f1 + 2.0 * carry * sqrt_t1 / vol),
                eta * (-d1 + x * std_dev - g1),
            );
            let n4 = self.f.value(-eta * (d1 + g1));
            (n3, n4, 0.0, 0.0, 0.0)
        } else {
            let t_ratio = lookback_period_end_time / residual_time;
            let cnbn1 =
                BivariateCumulativeNormalDistributionWe04DP::new(t_ratio.sqrt())?;
            let cnbn2 = BivariateCumulativeNormalDistributionWe04DP::new(
                -(1.0 - t_ratio).sqrt(),
            )?;
            let cnbn3 =
                BivariateCumulativeNormalDistributionWe04DP::new(-t_ratio.sqrt())?;

            let n3 = cnbn1.value(
                eta * (-f1 + 2.0 * carry * sqrt_t1 / vol),
                eta * (-d1 + x * std_dev - g1),
            );
            let g2 = l1 / (residual_time - lookback_period_end_time).sqrt();
            let n4 = cnbn2.value(-eta * (d1 + g1), eta * (e1 + g2));
            let n5 = cnbn2.value(-eta * (d1 - g1), eta * (e1 - g2));
            let n6 = cnbn3.value(eta * -f2, eta * (d2 - g1));
            let n7 = self.f.value(eta * (e2 - g2));
            (n3, n4, n5, n6, n7)
        };

        let n8 = self.f.value(-eta * f1);
        let pow_s = s.powf(-x);
        let pow_l = self.lambda().powf(x);
        let underlying = self.underlying()?;
        let minmax = self.minmax();
        let lambda = self.lambda();
        let rf_disc = self.risk_free_discount()?;
        let div_disc = self.dividend_discount()?;

        if full_lookback_period {
            Ok(eta
                * (underlying * div_disc * n1
                    - lambda * minmax * rf_disc * n2
                    + underlying * rf_disc * lambda / x
                        * (pow_s * n3 - div_disc / rf_disc * pow_l * n4)))
        } else {
            Ok(eta
                * (underlying * div_disc * n1
                    - lambda * minmax * rf_disc * n2
                    + underlying * rf_disc * lambda / x
                        * (pow_s * n3 - div_disc / rf_disc * pow_l * n4)
                    + underlying * div_disc * n5
                    + rf_disc * lambda * minmax * n6
                    - (-carry * (residual_time - lookback_period_end_time)).exp()
                        * div_disc
                        * (1.0 + 0.5 * vol * vol / carry)
                        * lambda
                        * underlying
                        * n7
                        * n8))
        }
    }
}

impl AsObservable for AnalyticContinuousPartialFloatingLookbackEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticContinuousPartialFloatingLookbackEngine {
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

/// Attaches [`AnalyticContinuousPartialFloatingLookbackEngine`] to `option`.
pub fn set_analytic_continuous_partial_floating_lookback_engine(
    option: &mut crate::instruments::ContinuousPartialFloatingLookbackOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticContinuousPartialFloatingLookbackEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instruments::{ContinuousPartialFloatingLookbackOption, FloatingTypePayoff};
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
        minmax: Real,
        spot: Real,
        q: Rate,
        r: Rate,
        t: Time,
        v: Volatility,
        lambda: Real,
        t1: Time,
        result: Real,
        tol: Real,
    }

    /// `lookbackoptions.cpp` `testAnalyticContinuousPartialFloatingLookback`.
    #[test]
    fn analytic_continuous_partial_floating_lookback_matches_haug() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let cases = [
            Case { option_type: OptionType::Call, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.25, result: 8.6524, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.5, result: 9.2128, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.75, result: 9.5567, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.25, result: 10.5751, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.5, result: 11.2601, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.75, result: 11.6804, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.25, result: 13.3402, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.5, result: 14.5121, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.75, result: 15.314, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.25, result: 16.3047, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.5, result: 17.737, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.75, result: 18.7171, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.25, result: 17.9831, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.5, result: 19.6618, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.75, result: 20.8493, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.25, result: 21.9793, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.5, result: 24.0311, tol: 1.0e-4 },
            Case { option_type: OptionType::Call, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.75, result: 25.4825, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.25, result: 2.7189, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.5, result: 3.4639, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.75, result: 4.1912, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.25, result: 3.3231, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.5, result: 4.2336, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.1, lambda: 1.0, t1: 0.75, result: 5.1226, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.25, result: 7.9153, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.5, result: 9.5825, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.75, result: 11.0362, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.25, result: 9.6743, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.5, result: 11.7119, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.2, lambda: 1.0, t1: 0.75, result: 13.4887, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.25, result: 13.4719, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.5, result: 16.1495, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 90.0, spot: 90.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.75, result: 18.4071, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.25, result: 16.4657, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.5, result: 19.7383, tol: 1.0e-4 },
            Case { option_type: OptionType::Put, minmax: 110.0, spot: 110.0, q: 0.0, r: 0.06, t: 1.0, v: 0.3, lambda: 1.0, t1: 0.75, result: 22.4976, tol: 1.0e-4 },
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
            let lookback_end = today + time_to_days(case.t1);
            let mut option = ContinuousPartialFloatingLookbackOption::new(
                case.minmax,
                case.lambda,
                lookback_end,
                FloatingTypePayoff::new(case.option_type),
                exercise,
                Shared::clone(&settings),
            )
            .unwrap();
            set_analytic_continuous_partial_floating_lookback_engine(
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
