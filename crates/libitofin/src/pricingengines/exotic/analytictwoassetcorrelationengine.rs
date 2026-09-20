//! Analytic two-asset correlation engine (Zhang / Haug).
//!
//! Port of `ql/pricingengines/exotic/analytictwoassetcorrelationengine.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    StrikedTypePayoff, TwoAssetCorrelationArguments, TwoAssetCorrelationResults, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionDr78;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;

type EngineBase = GenericEngine<TwoAssetCorrelationArguments, TwoAssetCorrelationResults>;

/// Analytic engine for European two-asset correlation options.
pub struct AnalyticTwoAssetCorrelationEngine {
    base: EngineBase,
    p1: Shared<GeneralizedBlackScholesProcess>,
    p2: Shared<GeneralizedBlackScholesProcess>,
    correlation: Handle<dyn Quote>,
}

impl AnalyticTwoAssetCorrelationEngine {
    /// `AnalyticTwoAssetCorrelationEngine(p1, p2, correlation)`.
    pub fn new(
        p1: Shared<GeneralizedBlackScholesProcess>,
        p2: Shared<GeneralizedBlackScholesProcess>,
        correlation: Handle<dyn Quote>,
    ) -> Self {
        let base = EngineBase::new(
            TwoAssetCorrelationArguments::default(),
            TwoAssetCorrelationResults::default(),
        );
        base.register_with(p1.observable());
        base.register_with(p2.observable());
        correlation.register_observer(&base.observer());
        Self {
            base,
            p1,
            p2,
            correlation,
        }
    }
}

impl AsObservable for AnalyticTwoAssetCorrelationEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticTwoAssetCorrelationEngine {
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
        let a = self.base.arguments();
        let payoff = a.payoff.expect("validated");
        let strike = payoff.strike();
        require!(strike > 0.0, "strike must be positive");
        let last = a.exercise.as_ref().expect("validated").last_date();
        let s1 = self.p1.x0()?;
        let s2 = self.p2.x0()?;
        require!(s1 > 0.0, "negative or null underlying given");
        let x2 = a.x2.expect("validated");
        let t = self.p2.time(&last)?;
        let sqrt_t = t.sqrt();
        let vol1 = self.p1.black_volatility().current_link()?;
        let vol2 = self.p2.black_volatility().current_link()?;
        let sigma1 = vol1.black_vol(self.p1.time(&last)?, strike, true)?;
        let sigma2 = vol2.black_vol(t, strike, true)?;
        let q1 = self
            .p1
            .dividend_yield()
            .current_link()?
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate();
        let q2 = self
            .p2
            .dividend_yield()
            .current_link()?
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate();
        let r = self
            .p1
            .risk_free_rate()
            .current_link()?
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate();
        let rho = self.correlation.current_link()?.value()?;
        let b1 = r - q1;
        let b2 = r - q2;
        let y1 = ((s1 / strike).ln() + (b1 - sigma1 * sigma1 / 2.0) * t) / (sigma1 * sqrt_t);
        let y2 = ((s2 / x2).ln() + (b2 - sigma2 * sigma2 / 2.0) * t) / (sigma2 * sqrt_t);
        let m = BivariateCumulativeNormalDistributionDr78::new(rho)?;
        let df_s2 = ((b2 - r) * t).exp();
        let df_x2 = (-r * t).exp();
        let value = match payoff.option_type() {
            OptionType::Call => {
                s2 * df_s2 * m.value(y2 + sigma2 * sqrt_t, y1 + rho * sigma2 * sqrt_t)
                    - x2 * df_x2 * m.value(y2, y1)
            }
            OptionType::Put => {
                x2 * df_x2 * m.value(-y2, -y1)
                    - s2 * df_s2 * m.value(-y2 - sigma2 * sqrt_t, -y1 - rho * sigma2 * sqrt_t)
            }
        };
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticTwoAssetCorrelationEngine`] to `option`.
pub fn set_analytic_two_asset_correlation_engine(
    option: &mut crate::instruments::TwoAssetCorrelationOption,
    p1: Shared<GeneralizedBlackScholesProcess>,
    p2: Shared<GeneralizedBlackScholesProcess>,
    correlation: Handle<dyn Quote>,
) {
    let engine = shared_mut(AnalyticTwoAssetCorrelationEngine::new(p1, p2, correlation))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::instrument::Instrument;
    use crate::instruments::TwoAssetCorrelationOption;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::types::Real;

    fn quote_h(v: Real) -> Handle<dyn Quote> {
        Handle::new(shared(SimpleQuote::new(v)) as Shared<dyn Quote>)
    }

    fn flat_rate(today: Date, r: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            today,
            quote_h(r),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(today: Date, v: Real) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(
            today,
            None,
            quote_h(v),
            Actual360::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    /// `twoassetcorrelationoption.cpp` `testAnalyticEngine` (Haug @ 1e-4).
    #[test]
    fn two_asset_correlation_haug_npv() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);
        let r = flat_rate(today, 0.1);
        let p1 = shared(BlackScholesMertonProcess::new(
            quote_h(52.0),
            flat_rate(today, 0.0),
            Handle::clone(&r),
            flat_vol(today, 0.2),
        ));
        let p2 = shared(BlackScholesMertonProcess::new(
            quote_h(65.0),
            flat_rate(today, 0.0),
            r,
            flat_vol(today, 0.3),
        ));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 180));
        let mut option =
            TwoAssetCorrelationOption::new(OptionType::Call, 50.0, 70.0, exercise, settings);
        set_analytic_two_asset_correlation_engine(&mut option, p1, p2, quote_h(0.75));
        let got = option.npv().unwrap();
        assert!((got - 4.7073).abs() <= 1e-4, "expected 4.7073, got {got}");
    }
}
