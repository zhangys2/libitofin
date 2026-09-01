//! Analytic complex chooser option engine.
//!
//! Port of `ql/pricingengines/exotic/analyticcomplexchooserengine.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::instrument::Instrument;
use crate::instruments::{ComplexChooserArguments, ComplexChooserResults};
use crate::interestrate::Compounding;
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionDr78;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::{Rate, Real, Time, Volatility};

type EngineBase = GenericEngine<ComplexChooserArguments, ComplexChooserResults>;

/// Pricing engine for complex chooser options.
pub struct AnalyticComplexChooserEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticComplexChooserEngine {
    /// `AnalyticComplexChooserEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            ComplexChooserArguments::default(),
            ComplexChooserResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }

    fn choosing_time(&self) -> QlResult<Time> {
        let choosing = self.base.arguments().choosing_date.expect("validated");
        StochasticProcess1D::time(&*self.process, &choosing)
    }

    fn call_maturity(&self) -> QlResult<Time> {
        let exercise = self
            .base
            .arguments()
            .exercise_call
            .as_ref()
            .expect("validated");
        StochasticProcess1D::time(&*self.process, &exercise.last_date())
    }

    fn put_maturity(&self) -> QlResult<Time> {
        let exercise = self
            .base
            .arguments()
            .exercise_put
            .as_ref()
            .expect("validated");
        StochasticProcess1D::time(&*self.process, &exercise.last_date())
    }

    fn strike(&self, option_type: OptionType) -> Real {
        let args = self.base.arguments();
        match option_type {
            OptionType::Call => args.strike_call.expect("validated"),
            OptionType::Put => args.strike_put.expect("validated"),
        }
    }

    fn volatility(&self, t: Time) -> QlResult<Volatility> {
        let strike = self.strike(OptionType::Call);
        let vol_ts = self.process.black_volatility().current_link()?;
        vol_ts.black_vol(t, strike, true)
    }

    fn dividend_yield(&self, t: Time) -> QlResult<Rate> {
        let q_ts = self.process.dividend_yield().current_link()?;
        Ok(q_ts
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn dividend_discount(&self, t: Time) -> QlResult<Real> {
        let q_ts = self.process.dividend_yield().current_link()?;
        q_ts.discount(t, false)
    }

    fn risk_free_rate(&self, t: Time) -> QlResult<Rate> {
        let r_ts = self.process.risk_free_rate().current_link()?;
        Ok(r_ts
            .zero_rate(t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn risk_free_discount(&self, t: Time) -> QlResult<Real> {
        let r_ts = self.process.risk_free_rate().current_link()?;
        r_ts.discount(t, false)
    }

    fn bs_calculator(&self, spot: Real, option_type: OptionType) -> QlResult<BlackCalculator> {
        let t = self.choosing_time()?;
        let tau = if option_type == OptionType::Call {
            self.call_maturity()? - 2.0 * t
        } else {
            self.put_maturity()? - 2.0 * t
        };
        let vol = self.volatility(tau)?;
        let std_dev = vol * tau.sqrt();
        let growth = self.dividend_discount(tau)?;
        let discount = self.risk_free_discount(tau)?;
        let forward = spot * growth / discount;
        BlackCalculator::new(option_type, self.strike(option_type), forward, std_dev, discount)
    }

    fn critical_value(&self) -> QlResult<Real> {
        let mut sv = self.process.x0()?;
        let mut bs = self.bs_calculator(sv, OptionType::Call)?;
        let mut ci = bs.value();
        let mut dc = bs.delta(sv)?;

        bs = self.bs_calculator(sv, OptionType::Put)?;
        let mut pi = bs.value();
        let mut dp = bs.delta(sv)?;

        let mut yi = ci - pi;
        let mut di = dc - dp;
        let epsilon = 0.001;

        while yi.abs() > epsilon {
            sv -= yi / di;

            bs = self.bs_calculator(sv, OptionType::Call)?;
            ci = bs.value();
            dc = bs.delta(sv)?;

            bs = self.bs_calculator(sv, OptionType::Put)?;
            pi = bs.value();
            dp = bs.delta(sv)?;

            yi = ci - pi;
            di = dc - dp;
        }
        Ok(sv)
    }

    fn bivariate_m(&self, rho: Real, a: Real, b: Real) -> QlResult<Real> {
        Ok(BivariateCumulativeNormalDistributionDr78::new(rho)?.value(a, b))
    }
}

impl AsObservable for AnalyticComplexChooserEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticComplexChooserEngine {
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
        let s = self.process.x0()?;
        let xc = self.strike(OptionType::Call);
        let xp = self.strike(OptionType::Put);
        let t = self.choosing_time()?;
        let tc = self.call_maturity()? - t;
        let tp = self.put_maturity()? - t;
        let i = self.critical_value()?;

        let b = self.risk_free_rate(t)? - self.dividend_yield(t)?;
        let v = self.volatility(t)?;
        let d1 = ((s / i).ln() + (b + v * v / 2.0) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();

        let b_call = self.risk_free_rate(t + tc)? - self.dividend_yield(t + tc)?;
        let v_call = self.volatility(tc)?;
        let y1 = ((s / xc).ln() + (b_call + v_call * v_call / 2.0) * tc) / (v_call * tc.sqrt());

        let b_put = self.risk_free_rate(t + tp)? - self.dividend_yield(t + tp)?;
        let v_put = self.volatility(tp)?;
        let y2 = ((s / xp).ln() + (b_put + v_put * v_put / 2.0) * tp) / (v_put * tp.sqrt());

        let rho1 = (t / tc).sqrt();
        let rho2 = (t / tp).sqrt();

        let r_call = self.risk_free_rate(t + tc)?;
        let mut value = s * ((b_call - r_call) * tc).exp() * self.bivariate_m(rho1, d1, y1)?
            - xc * (-r_call * tc).exp()
                * self.bivariate_m(rho1, d2, y1 - v_call * tc.sqrt())?;

        let r_put = self.risk_free_rate(t + tp)?;
        value -= s * ((b_put - r_put) * tp).exp() * self.bivariate_m(rho2, -d1, -y2)?;
        value += xp * (-r_put * tp).exp() * self.bivariate_m(rho2, -d2, -y2 + v_put * tp.sqrt())?;

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticComplexChooserEngine`] to `option`.
pub fn set_analytic_complex_chooser_engine(
    option: &mut crate::instruments::ComplexChooserOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticComplexChooserEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::ComplexChooserOption;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;

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

    /// `chooseroption.cpp` `testAnalyticComplexChooserEngine`.
    #[test]
    fn complex_chooser_haug_matches_quantlib() {
        let settings = shared(Settings::new());
        let today = Date::new(8, Month::August, 2025);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(50.0));
        let q_rate = shared(SimpleQuote::new(0.05));
        let r_rate = shared(SimpleQuote::new(0.10));
        let vol = shared(SimpleQuote::new(0.35));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        let choosing_date = today + 90;
        let call_exercise: Shared<dyn Exercise> =
            shared(EuropeanExercise::new(choosing_date + 180));
        let put_exercise: Shared<dyn Exercise> =
            shared(EuropeanExercise::new(choosing_date + 210));

        let mut option = ComplexChooserOption::new(
            choosing_date,
            55.0,
            48.0,
            Shared::clone(&call_exercise),
            put_exercise,
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_complex_chooser_engine(&mut option, process);

        let calculated = option.npv().unwrap();
        assert!(
            (calculated - 6.0508).abs() <= 1e-4,
            "expected 6.0508, got {calculated}"
        );
    }
}
