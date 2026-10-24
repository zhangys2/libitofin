//! Analytic engine for American Margrabe option
//! (`ql/pricingengines/exotic/analyticamericanmargrabeengine.{hpp,cpp}`).

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    MargrabeArguments, MargrabeResults, OptionArguments, PlainVanillaPayoff, StrikedTypePayoff,
};
use crate::interestrate::Compounding;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::vanilla::BjerksundStenslandApproximationEngine;
use crate::processes::{BlackScholesMertonProcess, GeneralizedBlackScholesProcess};
use crate::quotes::{Quote, SimpleQuote};
use crate::require;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
use crate::termstructures::yields::FlatForward;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::daycounters::actual360::Actual360;
use crate::time::frequency::Frequency;
use crate::types::Real;

type EngineBase = GenericEngine<MargrabeArguments, MargrabeResults>;

/// Pricing engine for American Margrabe options via Bjerksund-Stensland reduction.
pub struct AnalyticAmericanMargrabeEngine {
    base: EngineBase,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
}

impl AnalyticAmericanMargrabeEngine {
    /// `AnalyticAmericanMargrabeEngine(process1, process2, correlation)`.
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
        }
    }
}

impl AsObservable for AnalyticAmericanMargrabeEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticAmericanMargrabeEngine {
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
        let Some(exercise) = &arguments.exercise else {
            return Ok(());
        };
        require!(
            exercise.exercise_type() == ExerciseType::American,
            "not an American option"
        );
        require!(arguments.payoff.is_some(), "non-null payoff given");

        let q1 = Real::from(arguments.q1.expect("validated"));
        let q2 = Real::from(arguments.q2.expect("validated"));
        let maturity = exercise.last_date();

        let s1 = self.process1.x0()?;
        let s2 = self.process2.x0()?;
        require!(s1 > 0.0, "negative or null underlying1");
        require!(s2 > 0.0, "negative or null underlying2");

        let rf_link = self.process1.risk_free_rate().current_link()?;
        let today = rf_link.reference_date()?;
        let rfdc = rf_link.day_counter().unwrap_or_else(Actual360::new);
        let t = rfdc.year_fraction(today, maturity);
        require!(t > 0.0, "maturity must be in the future");

        let q_disc1 = self
            .process1
            .dividend_yield()
            .current_link()?
            .discount_date(maturity, false)?;
        let q_disc2 = self
            .process2
            .dividend_yield()
            .current_link()?
            .discount_date(maturity, false)?;
        let rate_q1 = -q_disc1.ln() / t;
        let rate_q2 = -q_disc2.ln() / t;

        let vol1 = self.process1.black_volatility().current_link()?;
        let vol2 = self.process2.black_volatility().current_link()?;
        let variance1 = vol1.black_variance_date(maturity, s1, false)?;
        let variance2 = vol2.black_variance_date(maturity, s2, false)?;
        let variance = variance1 + variance2 - 2.0 * self.rho * variance1.sqrt() * variance2.sqrt();
        require!(variance >= 0.0, "variance must be non-negative");
        let volatility = (variance / t).sqrt();

        let spot = q1 * s1;
        let strike = q2 * s2;

        let spot_handle = Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>);
        let q_ts = Handle::new(shared(FlatForward::with_rate(
            today,
            rate_q1,
            rfdc.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let r_ts = Handle::new(shared(FlatForward::with_rate(
            today,
            rate_q2,
            rfdc.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let vol_ts = Handle::new(shared(BlackConstantVol::new(today, None, volatility, rfdc))
            as Shared<dyn BlackVolTermStructure>);

        let proc = shared(BlackScholesMertonProcess::new(
            spot_handle,
            q_ts,
            r_ts,
            vol_ts,
        ));
        let mut engine = BjerksundStenslandApproximationEngine::new(proc);
        let opt_args = (engine.arguments_mut() as &mut dyn Any)
            .downcast_mut::<OptionArguments>()
            .expect("option arguments");
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, strike));
        opt_args.payoff = Some(payoff);
        opt_args.exercise = Some(Shared::clone(exercise));
        engine.calculate()?;
        let value = engine
            .results()
            .as_instrument_results()
            .and_then(|r| r.value)
            .expect("calculated value");
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticAmericanMargrabeEngine`] to `option`.
pub fn set_analytic_american_margrabe_engine(
    option: &mut crate::instruments::MargrabeOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
) {
    let engine = shared_mut(AnalyticAmericanMargrabeEngine::new(process1, process2, rho))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::instruments::MargrabeOption;
    use crate::settings::Settings;
    use crate::time::date::{Date, Month};
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

    /// Full 18-row `margrabeoption.cpp` `testAmericanExchangeTwoAssets` oracle (Haug @ 1e-3).
    #[test]
    fn test_american_exchange_two_assets() {
        type Row = (
            Real,    // s1
            Real,    // s2
            Integer, // Q1
            Integer, // Q2
            Real,    // q1
            Real,    // q2
            Real,    // r
            Real,    // t
            Real,    // v1
            Real,    // v2
            Real,    // rho
            Real,    // result
        );
        #[rustfmt::skip]
        let rows: [Row; 21] = [
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.15, -0.50, 2.1357),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.20, -0.50, 2.2074),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.25, -0.50, 2.2902),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.15,  0.00, 2.0592),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.20,  0.00, 2.1032),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.25,  0.00, 2.1618),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.15,  0.50, 2.0001),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.20,  0.50, 2.0110),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.25,  0.50, 2.0359),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.15, -0.50, 2.8051),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20, -0.50, 3.0288),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.25, -0.50, 3.2664),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.15,  0.00, 2.5282),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20,  0.00, 2.6945),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.25,  0.00, 2.8893),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.15,  0.50, 2.2053),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20,  0.50, 2.2906),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.25,  0.50, 2.4261),

            // Quantity scaling tests: Q1*S1=22, Q2*S2=20 matches unit rows 16..18
            (22.0, 10.0, 1, 2, 0.06, 0.04, 0.10, 0.50, 0.20, 0.15,  0.50, 2.2053),
            (11.0, 20.0, 2, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20,  0.50, 2.2906),
            (11.0, 10.0, 2, 2, 0.06, 0.04, 0.10, 0.50, 0.20, 0.25,  0.50, 2.4261),
        ];

        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);

        for (s1, s2, q1, q2, d1, d2, r, t, v1, v2, rho, expected) in rows {
            let p1 = shared(BlackScholesMertonProcess::new(
                quote_handle(&shared(SimpleQuote::new(s1))),
                flat_rate(today, &shared(SimpleQuote::new(d1))),
                flat_rate(today, &shared(SimpleQuote::new(r))),
                flat_vol(today, &shared(SimpleQuote::new(v1))),
            ));
            let p2 = shared(BlackScholesMertonProcess::new(
                quote_handle(&shared(SimpleQuote::new(s2))),
                flat_rate(today, &shared(SimpleQuote::new(d2))),
                flat_rate(today, &shared(SimpleQuote::new(r))),
                flat_vol(today, &shared(SimpleQuote::new(v2))),
            ));
            let exercise_date = today + (t * 360.0).round() as i32;
            let exercise: Shared<dyn Exercise> =
                shared(AmericanExercise::new(today, exercise_date, false).unwrap());
            let mut option = MargrabeOption::new(q1, q2, exercise, Shared::clone(&settings));
            set_analytic_american_margrabe_engine(&mut option, p1, p2, rho);
            let got = option.npv().unwrap();
            assert!(
                (got - expected).abs() <= 1e-3,
                "s1={s1} s2={s2} t={t} rho={rho}: expected {expected}, got {got}"
            );
        }
    }

    #[test]
    fn test_american_margrabe_rejects_non_american_exercise() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);

        let p1 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(22.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.06))),
            flat_rate(today, &shared(SimpleQuote::new(0.10))),
            flat_vol(today, &shared(SimpleQuote::new(0.20))),
        ));
        let p2 = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(20.0))),
            flat_rate(today, &shared(SimpleQuote::new(0.04))),
            flat_rate(today, &shared(SimpleQuote::new(0.10))),
            flat_vol(today, &shared(SimpleQuote::new(0.15))),
        ));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 36));
        let mut option = MargrabeOption::new(1, 1, exercise, settings);
        set_analytic_american_margrabe_engine(&mut option, p1, p2, -0.50);
        assert!(option.npv().is_err());
    }
}
