//! Analytic European Margrabe engine (`analyticeuropeanmargrabeengine.{hpp,cpp}`).
//! NPV only; extra greeks (`delta1`/`delta2`/`gamma1`/`gamma2`/`theta`) follow-up.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instrument::Instrument;
use crate::instruments::{MargrabeArguments, MargrabeResults};
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::Real;

type EngineBase = GenericEngine<MargrabeArguments, MargrabeResults>;

/// Pricing engine for European exchange options.
pub struct AnalyticEuropeanMargrabeEngine {
    base: EngineBase,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
    f: CumulativeNormalDistribution,
}

impl AnalyticEuropeanMargrabeEngine {
    /// `AnalyticEuropeanMargrabeEngine(process1, process2, correlation)`.
    pub fn new(
        process1: Shared<GeneralizedBlackScholesProcess>,
        process2: Shared<GeneralizedBlackScholesProcess>,
        rho: Real,
    ) -> Self {
        let base = EngineBase::new(MargrabeArguments::default(), MargrabeResults::default());
        base.register_with(process1.observable());
        base.register_with(process2.observable());
        Self {
            base,
            process1,
            process2,
            rho,
            f: CumulativeNormalDistribution::standard(),
        }
    }
}

impl AsObservable for AnalyticEuropeanMargrabeEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticEuropeanMargrabeEngine {
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
        let arguments = self.base.arguments();
        let exercise = arguments.exercise.as_ref().expect("validated");
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European Option"
        );
        require!(arguments.payoff.is_some(), "non a Null Payoff type");
        let q1 = Real::from(arguments.q1.expect("validated"));
        let q2 = Real::from(arguments.q2.expect("validated"));
        let maturity = exercise.last_date();

        let s1 = self.process1.x0()?;
        let s2 = self.process2.x0()?;
        require!(s1 > 0.0, "negative or null underlying1");
        require!(s2 > 0.0, "negative or null underlying2");

        let vol1 = self.process1.black_volatility().current_link()?;
        let vol2 = self.process2.black_volatility().current_link()?;
        let variance1 = vol1.black_variance_date(maturity, s1, false)?;
        let variance2 = vol2.black_variance_date(maturity, s2, false)?;

        let rf = self.process1.risk_free_rate().current_link()?;
        let qy1 = self.process1.dividend_yield().current_link()?;
        let qy2 = self.process2.dividend_yield().current_link()?;
        let rf_disc = rf.discount_date(maturity, false)?;
        let q_disc1 = qy1.discount_date(maturity, false)?;
        let q_disc2 = qy2.discount_date(maturity, false)?;
        let forward1 = s1 * q_disc1 / rf_disc;
        let forward2 = s2 * q_disc2 / rf_disc;

        let variance = variance1 + variance2 - 2.0 * self.rho * variance1.sqrt() * variance2.sqrt();
        let std_dev = variance.sqrt();
        let d1 = (((q1 * forward1) / (q2 * forward2)).ln() + 0.5 * variance) / std_dev;
        let d2 = d1 - std_dev;
        let value = rf_disc * (q1 * forward1 * self.f.value(d1) - q2 * forward2 * self.f.value(d2));
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticEuropeanMargrabeEngine`] to `option`.
pub fn set_analytic_european_margrabe_engine(
    option: &mut crate::instruments::MargrabeOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
) {
    let engine = shared_mut(AnalyticEuropeanMargrabeEngine::new(process1, process2, rho))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::MargrabeOption;
    use crate::interestrate::Compounding;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::types::Integer;

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

    #[allow(clippy::too_many_arguments)]
    #[rustfmt::skip]
    fn price(s1: Real, s2: Real, q1: Integer, q2: Integer, div1: Real, div2: Real, r: Real, t: Real, v1: Real, v2: Real, rho: Real) -> Real {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);
        let p1 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(s1))),
            flat_rate(today, &shared(SimpleQuote::new(div1))),
            flat_rate(today, &shared(SimpleQuote::new(r))),
            flat_vol(today, &shared(SimpleQuote::new(v1))),
        ));
        let p2 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(s2))),
            flat_rate(today, &shared(SimpleQuote::new(div2))),
            flat_rate(today, &shared(SimpleQuote::new(r))),
            flat_vol(today, &shared(SimpleQuote::new(v2))),
        ));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + (t * 360.0).round() as i32));
        let mut option = MargrabeOption::new(q1, q2, exercise, settings);
        set_analytic_european_margrabe_engine(&mut option, p1, p2, rho);
        option.npv().unwrap()
    }

    /// `margrabeoption.cpp` `testEuroExchangeTwoAssets` NPV subset (Haug @ 1e-3).
    #[test]
    fn european_margrabe_haug_npv() {
        type Row = (
            Real,
            Real,
            Integer,
            Integer,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
        );
        #[rustfmt::skip]
        let rows: [Row; 6] = [
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.15, -0.50, 2.125),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.20,  0.00, 2.091),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.25,  0.50, 2.019),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20,  0.00, 2.650),
            (22.0, 10.0, 1, 2, 0.06, 0.04, 0.10, 0.50, 0.20, 0.15,  0.50, 2.138),
            (11.0, 20.0, 2, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20,  0.50, 2.231),
        ];
        for (s1, s2, q1, q2, d1, d2, r, t, v1, v2, rho, expected) in rows {
            let got = price(s1, s2, q1, q2, d1, d2, r, t, v1, v2, rho);
            assert!(
                (got - expected).abs() <= 1e-3,
                "s1={s1} s2={s2} Q=({q1},{q2}) t={t} ρ={rho}: expected {expected}, got {got}"
            );
        }
    }
}
