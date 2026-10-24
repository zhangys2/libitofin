//! Analytic writer-extensible option engine (Haug).
//!
//! Port of `ql/pricingengines/exotic/analyticwriterextensibleoptionengine.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    StrikedTypePayoff, TypePayoff, WriterExtensibleArguments, WriterExtensibleResults,
};
use crate::interestrate::Compounding;
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionWe04DP;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::black_formula;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;

type EngineBase = GenericEngine<WriterExtensibleArguments, WriterExtensibleResults>;

/// Analytic engine for European writer-extensible options.
pub struct AnalyticWriterExtensibleOptionEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticWriterExtensibleOptionEngine {
    /// `AnalyticWriterExtensibleOptionEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            WriterExtensibleArguments::default(),
            WriterExtensibleResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for AnalyticWriterExtensibleOptionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticWriterExtensibleOptionEngine {
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
        let p1 = a.payoff.expect("validated");
        let p2 = a.payoff2.expect("validated");
        let last1 = a.exercise.as_ref().expect("validated").last_date();
        let last2 = a.exercise2.as_ref().expect("validated").last_date();
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");
        let t1 = self.process.time(&last1)?;
        let t2 = self.process.time(&last2)?;
        let zr = |ts: &Handle<dyn YieldTermStructure>| {
            ts.current_link()?
                .zero_rate(t1, Compounding::Continuous, Frequency::NoFrequency, false)
                .map(|z| z.rate())
        };
        let r = zr(&self.process.risk_free_rate())?;
        let q = zr(&self.process.dividend_yield())?;
        let b = r - q;
        let vol = self
            .process
            .black_volatility()
            .current_link()?
            .black_vol_date(last1, p1.strike(), true)?;
        let std_dev = vol * t1.sqrt();
        let discount = (-r * t1).exp();
        let black = black_formula(
            p1.option_type(),
            p1.strike(),
            spot * (b * t1).exp(),
            std_dev,
            discount,
            0.0,
        )?;
        let ro = (t1 / t2).sqrt();
        let biv = BivariateCumulativeNormalDistributionWe04DP::new(-ro)?;
        let z1 = ((spot / p2.strike()).ln() + (b + vol * vol / 2.0) * t2) / (vol * t2.sqrt());
        let z2 = ((spot / p1.strike()).ln() + (b + vol * vol / 2.0) * t1) / (vol * t1.sqrt());
        let df_s = ((b - r) * t2).exp();
        let df_x = (-r * t2).exp();
        let value = match p1.option_type() {
            OptionType::Call => {
                black + spot * df_s * biv.value(z1, -z2)
                    - p2.strike() * df_x * biv.value(z1 - vol * t2.sqrt(), -z2 + vol * t1.sqrt())
            }
            OptionType::Put => {
                black - spot * df_s * biv.value(-z1, z2)
                    + p2.strike() * df_x * biv.value(-z1 + vol * t2.sqrt(), z2 - vol * t1.sqrt())
            }
        };
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticWriterExtensibleOptionEngine`] to `option`.
pub fn set_analytic_writer_extensible_option_engine(
    option: &mut crate::instruments::WriterExtensibleOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticWriterExtensibleOptionEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{PlainVanillaPayoff, WriterExtensibleOption};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
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

    #[rustfmt::skip]
    fn flat_rate(today: Date, r: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(today, quote_h(r), Actual360::new(), Compounding::Continuous, Frequency::Annual)) as Shared<dyn YieldTermStructure>)
    }

    #[rustfmt::skip]
    fn flat_vol(today: Date, v: Real) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(today, None, quote_h(v), Actual360::new())) as Shared<dyn BlackVolTermStructure>)
    }

    #[rustfmt::skip]
    fn price(ty: OptionType, q: Real) -> Real {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);
        let r = flat_rate(today, 0.10);
        let p = shared(BlackScholesMertonProcess::new(
            quote_h(80.0), flat_rate(today, q), r, flat_vol(today, 0.30),
        ));
        let e1: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 180));
        let e2: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 270));
        let mut option = WriterExtensibleOption::new(
            PlainVanillaPayoff::new(ty, 90.0), e1,
            PlainVanillaPayoff::new(ty, 82.0), e2, settings,
        );
        set_analytic_writer_extensible_option_engine(&mut option, p);
        option.npv().unwrap()
    }

    /// `extensibleoptions.cpp` `testAnalyticWriterExtensibleOptionEngine` @ 1e-4.
    #[test]
    #[rustfmt::skip]
    fn writer_extensible_haug_npv() {
        let call = price(OptionType::Call, 0.0);
        assert!((call - 6.8238).abs() <= 1e-4, "Haug call 6.8238 vs {call}");
        let put = price(OptionType::Put, 0.0);
        assert!((put - 10.3105).abs() <= 1e-4, "put 10.3105 vs {put}");
        let q_call = price(OptionType::Call, 0.05);
        assert!((q_call - 5.8045).abs() <= 1e-4, "q≠0 call 5.8045 vs {q_call}");
    }
}
