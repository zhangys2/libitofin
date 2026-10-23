//! Analytic cliquet option engine.
//!
//! Port of `ql/pricingengines/cliquet/analyticcliquetengine.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instrument::Instrument;
use crate::instruments::{
    CliquetArguments, CliquetResults, PlainVanillaPayoff, StrikedTypePayoff, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;

type EngineBase = GenericEngine<CliquetArguments, CliquetResults>;

/// Pricing engine for uncapped European cliquet options.
pub struct AnalyticCliquetEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticCliquetEngine {
    /// `AnalyticCliquetEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(CliquetArguments::default(), CliquetResults::default());
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for AnalyticCliquetEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticCliquetEngine {
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
        let args = self.base.arguments();
        require!(
            args.accrued_coupon.is_none() && args.last_fixing.is_none(),
            "this engine cannot price options already started"
        );
        require!(
            args.local_cap.is_none()
                && args.local_floor.is_none()
                && args.global_cap.is_none()
                && args.global_floor.is_none(),
            "this engine cannot price capped/floored options"
        );
        let exercise = args.exercise.as_ref().expect("validated");
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European option"
        );
        let moneyness = args.payoff.expect("validated");

        let mut reset_dates = args.reset_dates.clone();
        reset_dates.push(exercise.last_date());

        let underlying = self.process.x0()?;
        require!(underlying > 0.0, "negative or null underlying");
        let strike = underlying * moneyness.strike();
        let payoff = PlainVanillaPayoff::new(moneyness.option_type(), strike);

        let r_ts = self.process.risk_free_rate().current_link()?;
        let q_ts = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let rfdc = r_ts.require_day_counter()?;
        let divdc = q_ts.require_day_counter()?;
        let voldc = vol_ts.require_day_counter()?;

        let mut value = 0.0;
        let mut delta = 0.0;
        let mut theta = 0.0;
        let mut rho = 0.0;
        let mut dividend_rho = 0.0;
        let mut vega = 0.0;

        for i in 1..reset_dates.len() {
            let weight = q_ts.discount_date(reset_dates[i - 1], false)?;
            let discount = r_ts.discount_date(reset_dates[i], false)?
                / r_ts.discount_date(reset_dates[i - 1], false)?;
            let q_discount = q_ts.discount_date(reset_dates[i], false)?
                / q_ts.discount_date(reset_dates[i - 1], false)?;
            let forward = underlying * q_discount / discount;
            let variance = vol_ts.black_forward_variance_dates(
                reset_dates[i - 1],
                reset_dates[i],
                strike,
                false,
            )?;

            let black = BlackCalculator::with_striked_payoff(
                &payoff as &dyn StrikedTypePayoff,
                forward,
                variance.sqrt(),
                discount,
            )?;

            value += weight * black.value();
            delta +=
                weight * (black.delta(underlying)? + moneyness.strike() * discount * black.beta());
            theta += q_ts
                .forward_rate_between(
                    reset_dates[i - 1],
                    reset_dates[i],
                    rfdc.clone(),
                    Compounding::Continuous,
                    Frequency::NoFrequency,
                    false,
                )?
                .rate()
                * weight
                * black.value();

            let dt = rfdc.year_fraction(reset_dates[i - 1], reset_dates[i]);
            rho += weight * black.rho(dt)?;

            let t = divdc.year_fraction(q_ts.reference_date()?, reset_dates[i - 1]);
            let dt_q = divdc.year_fraction(reset_dates[i - 1], reset_dates[i]);
            dividend_rho += weight * (black.dividend_rho(dt_q)? - t * black.value());

            let dt_v = voldc.year_fraction(reset_dates[i - 1], reset_dates[i]);
            vega += weight * black.vega(dt_v)?;
        }

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.greeks = crate::instruments::Greeks {
            delta: Some(delta),
            gamma: Some(0.0),
            theta: Some(theta),
            vega: Some(vega),
            rho: Some(rho),
            dividend_rho: Some(dividend_rho),
        };
        Ok(())
    }
}

/// Attaches [`AnalyticCliquetEngine`] to `option`.
pub fn set_analytic_cliquet_engine(
    option: &mut crate::instruments::CliquetOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticCliquetEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{CliquetOption, PercentageStrikePayoff};
    use crate::option::OptionType;
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

    /// `cliquetoption.cpp` `testValues` (Haug p.37).
    #[test]
    fn cliquet_haug_matches_quantlib() {
        let settings = shared(Settings::new());
        let today = Date::new(8, Month::August, 2025);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(60.0));
        let q_rate = shared(SimpleQuote::new(0.04));
        let r_rate = shared(SimpleQuote::new(0.08));
        let vol = shared(SimpleQuote::new(0.30));
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        let reset = vec![today + 90];
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 360));
        let mut option = CliquetOption::new(
            PercentageStrikePayoff::new(OptionType::Call, 1.1),
            exercise,
            reset,
            Shared::clone(&settings),
        )
        .unwrap();
        set_analytic_cliquet_engine(&mut option, process);

        let calculated = option.npv().unwrap();
        assert!(
            (calculated - 4.4064).abs() <= 1e-4,
            "expected 4.4064, got {calculated}"
        );
    }
}
