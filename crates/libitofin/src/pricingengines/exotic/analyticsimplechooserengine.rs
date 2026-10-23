//! Analytic simple chooser option engine.
//!
//! Port of `ql/pricingengines/exotic/analyticsimplechooserengine.{hpp,cpp}`
//! (Haug, *The Complete Guide to Option Pricing Formulas*, pp.39–40).

use crate::errors::QlResult;
use crate::instrument::Instrument;
use crate::instruments::{SimpleChooserArguments, SimpleChooserResults, StrikedTypePayoff};
use crate::interestrate::Compounding;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::{Rate, Volatility};

type EngineBase = GenericEngine<SimpleChooserArguments, SimpleChooserResults>;

/// Pricing engine for simple chooser options.
pub struct AnalyticSimpleChooserEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    f: CumulativeNormalDistribution,
}

impl AnalyticSimpleChooserEngine {
    /// `AnalyticSimpleChooserEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            SimpleChooserArguments::default(),
            SimpleChooserResults::default(),
        );
        base.register_with(process.observable());
        Self {
            base,
            process,
            f: CumulativeNormalDistribution::standard(),
        }
    }
}

impl AsObservable for AnalyticSimpleChooserEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticSimpleChooserEngine {
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
        let payoff = arguments.payoff.expect("validated");
        let exercise = arguments.exercise.as_ref().expect("validated");
        let choosing_date = arguments.choosing_date.expect("validated");
        let strike = payoff.strike();

        let risk_free = self.process.risk_free_rate().current_link()?;
        let dividend = self.process.dividend_yield().current_link()?;
        let black_vol = self.process.black_volatility().current_link()?;

        let rfdc = risk_free.require_day_counter()?;
        let divdc = dividend.require_day_counter()?;
        let voldc = black_vol.require_day_counter()?;
        require!(
            rfdc.name() == divdc.name() && rfdc.name() == voldc.name(),
            "Risk-free rate, dividend yield and volatility must have the same day counter"
        );

        let spot = self.process.x0()?;
        let maturity = exercise.last_date();
        let today = risk_free.reference_date()?;
        let time_to_maturity = rfdc.year_fraction(today, maturity);
        let time_to_choosing = rfdc.year_fraction(today, choosing_date);

        let t_vol = black_vol.time_from_reference(maturity)?;
        let volatility: Volatility = black_vol.black_vol(t_vol, strike, true)?;

        let t_r = rfdc.year_fraction(risk_free.reference_date()?, maturity);
        let risk_free_rate: Rate = risk_free
            .zero_rate(t_r, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate();
        let t_q = divdc.year_fraction(dividend.reference_date()?, maturity);
        let dividend_rate: Rate = dividend
            .zero_rate(t_q, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate();

        require!(spot > 0.0, "negative or null spot value");
        require!(strike > 0.0, "negative or null strike value");
        require!(volatility > 0.0, "negative or null volatility");
        require!(
            time_to_choosing > 0.0,
            "choosing date earlier than or equal to evaluation date"
        );

        let d = ((spot / strike).ln()
            + ((risk_free_rate - dividend_rate) + volatility * volatility * 0.5)
                * time_to_maturity)
            / (volatility * time_to_maturity.sqrt());

        let y = ((spot / strike).ln()
            + (risk_free_rate - dividend_rate) * time_to_maturity
            + (volatility * volatility * time_to_choosing / 2.0))
            / (volatility * time_to_choosing.sqrt());

        let value = spot * (-dividend_rate * time_to_maturity).exp() * self.f.value(d)
            - strike
                * (-risk_free_rate * time_to_maturity).exp()
                * self.f.value(d - volatility * time_to_maturity.sqrt())
            - spot * (-dividend_rate * time_to_maturity).exp() * self.f.value(-y)
            + strike
                * (-risk_free_rate * time_to_maturity).exp()
                * self.f.value(-y + volatility * time_to_choosing.sqrt());

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticSimpleChooserEngine`] to `option`.
pub fn set_analytic_simple_chooser_engine(
    option: &mut crate::instruments::SimpleChooserOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine =
        shared_mut(AnalyticSimpleChooserEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::SimpleChooserOption;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;

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

    /// `chooseroption.cpp` `testAnalyticSimpleChooserEngine` (Haug pp.39–40).
    #[test]
    fn simple_chooser_haug_matches_quantlib() {
        let settings = shared(Settings::new());
        let today = Date::new(8, Month::August, 2025);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(50.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.08));
        let vol = shared(SimpleQuote::new(0.25));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        let strike = 50.0;
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 180));
        let choosing_date = today + 90;

        let mut option =
            SimpleChooserOption::new(choosing_date, strike, exercise, Shared::clone(&settings))
                .unwrap();
        set_analytic_simple_chooser_engine(&mut option, process);

        let calculated = option.npv().unwrap();
        assert!(
            (calculated - 6.1071).abs() <= 3e-5,
            "expected 6.1071, got {calculated}"
        );
    }
}
