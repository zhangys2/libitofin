//! Analytic performance-option engine.
//!
//! Port of `ql/pricingengines/cliquet/analyticperformanceengine.{hpp,cpp}`.

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

/// Pricing engine for uncapped European performance options.
pub struct AnalyticPerformanceEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticPerformanceEngine {
    /// `AnalyticPerformanceEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(CliquetArguments::default(), CliquetResults::default());
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for AnalyticPerformanceEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticPerformanceEngine {
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
        let payoff = PlainVanillaPayoff::new(moneyness.option_type(), 1.0);

        let r_ts = self.process.risk_free_rate().current_link()?;
        let q_ts = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let rfdc = r_ts.require_day_counter()?;
        let divdc = q_ts.require_day_counter()?;
        let voldc = vol_ts.require_day_counter()?;

        let mut value = 0.0;
        let mut theta = 0.0;
        let mut rho = 0.0;
        let mut dividend_rho = 0.0;
        let mut vega = 0.0;

        for i in 1..reset_dates.len() {
            let discount = r_ts.discount_date(reset_dates[i - 1], false)?;
            let r_discount = r_ts.discount_date(reset_dates[i], false)?
                / r_ts.discount_date(reset_dates[i - 1], false)?;
            let q_discount = q_ts.discount_date(reset_dates[i], false)?
                / q_ts.discount_date(reset_dates[i - 1], false)?;
            let forward = (1.0 / moneyness.strike()) * q_discount / r_discount;
            let variance = vol_ts.black_forward_variance_dates(
                reset_dates[i - 1],
                reset_dates[i],
                underlying * moneyness.strike(),
                false,
            )?;

            let black =
                BlackCalculator::with_payoff(&payoff, forward, variance.sqrt(), r_discount)?;

            value += discount * moneyness.strike() * black.value();
            theta += r_ts
                .forward_rate_between(
                    reset_dates[i - 1],
                    reset_dates[i],
                    rfdc.clone(),
                    Compounding::Continuous,
                    Frequency::NoFrequency,
                    false,
                )?
                .rate()
                * discount
                * moneyness.strike()
                * black.value();

            let dt = rfdc.year_fraction(reset_dates[i - 1], reset_dates[i]);
            let t = rfdc.year_fraction(r_ts.reference_date()?, reset_dates[i - 1]);
            rho += discount * moneyness.strike() * (black.rho(dt)? - t * black.value());

            let dt_q = divdc.year_fraction(reset_dates[i - 1], reset_dates[i]);
            dividend_rho += discount * moneyness.strike() * black.dividend_rho(dt_q)?;

            let dt_v = voldc.year_fraction(reset_dates[i - 1], reset_dates[i]);
            vega += discount * moneyness.strike() * black.vega(dt_v)?;
        }

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.greeks = crate::instruments::Greeks {
            delta: Some(0.0),
            gamma: Some(0.0),
            theta: Some(theta),
            vega: Some(vega),
            rho: Some(rho),
            dividend_rho: Some(dividend_rho),
        };
        Ok(())
    }
}

/// Attaches [`AnalyticPerformanceEngine`] to `option`.
pub fn set_analytic_performance_engine(
    option: &mut crate::instruments::CliquetOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine =
        shared_mut(AnalyticPerformanceEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instruments::{CliquetOption, PercentageStrikePayoff};
    use crate::option::OptionType::{self, Call, Put};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::types::{Rate, Real, Volatility};

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

    struct Market {
        spot: Shared<SimpleQuote>,
        q_rate: Shared<SimpleQuote>,
        r_rate: Shared<SimpleQuote>,
        vol: Shared<SimpleQuote>,
        process: Shared<BlackScholesMertonProcess>,
        settings: Shared<Settings<Date>>,
        today: Date,
    }

    fn market() -> Market {
        let settings = shared(Settings::new());
        let today = Date::new(8, Month::August, 2025);
        settings.set_evaluation_date(today);
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
        Market {
            spot,
            q_rate,
            r_rate,
            vol,
            process,
            settings,
            today,
        }
    }

    fn option(market: &Market, option_type: OptionType, moneyness: Real) -> CliquetOption {
        let reset = vec![market.today + 90];
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(market.today + 360));
        let mut option = CliquetOption::new(
            PercentageStrikePayoff::new(option_type, moneyness),
            exercise,
            reset,
            Shared::clone(&market.settings),
        )
        .unwrap();
        set_analytic_performance_engine(
            &mut option,
            Shared::clone(&market.process) as Shared<GeneralizedBlackScholesProcess>,
        );
        option
    }

    /// `cliquetoption.cpp` `testPerformanceGreeks`: performance NPV is
    /// homogeneous of degree 0 in the spot, so δ and γ are identically zero
    /// (`analyticperformanceengine.cpp:75-76`).
    #[test]
    fn performance_delta_and_gamma_are_zero_and_npv_is_spot_homogeneous() {
        let market = market();
        market.spot.set_value(100.0);
        market.q_rate.set_value(0.04);
        market.r_rate.set_value(0.06);
        market.vol.set_value(0.30);
        let mut at_100 = option(&market, Call, 1.1);
        let value = at_100.npv().unwrap();
        assert!(
            value.abs() > 1.0e-4,
            "fixture must have a price, got {value}"
        );
        assert_eq!(at_100.delta().unwrap(), 0.0);
        assert_eq!(at_100.gamma().unwrap(), 0.0);
        market.spot.set_value(120.0);
        let mut at_120 = option(&market, Call, 1.1);
        assert!(
            (at_120.npv().unwrap() - value).abs() <= 1.0e-12,
            "performance NPV must be independent of spot"
        );
    }

    /// Compact arm of `testPerformanceGreeks`: ρ / divρ / ν vs central
    /// differences on a `q != r` fixture, both types. Skips the full
    /// moneyness × length × frequency grid.
    #[test]
    fn performance_greeks_match_central_differences() {
        let (moneyness, spot, q, r, vol): (Real, Real, Rate, Rate, Volatility) =
            (1.1, 100.0, 0.04, 0.06, 0.30);
        let tolerance: Real = 1.0e-5;
        let market = market();
        market.spot.set_value(spot);
        market.q_rate.set_value(q);
        market.r_rate.set_value(r);
        market.vol.set_value(vol);
        for option_type in [Call, Put] {
            let mut opt = option(&market, option_type, moneyness);
            let rho = opt.rho().unwrap();
            let dividend_rho = opt.dividend_rho().unwrap();
            let vega = opt.vega().unwrap();
            assert!(vega.abs() > 1.0e-4, "fixture must have a vega, got {vega}");

            let dr = r * 1.0e-4;
            market.r_rate.set_value(r + dr);
            let value_up = option(&market, option_type, moneyness).npv().unwrap();
            market.r_rate.set_value(r - dr);
            let value_down = option(&market, option_type, moneyness).npv().unwrap();
            market.r_rate.set_value(r);
            let fd_rho = (value_up - value_down) / (2.0 * dr);

            let dq = q * 1.0e-4;
            market.q_rate.set_value(q + dq);
            let value_up = option(&market, option_type, moneyness).npv().unwrap();
            market.q_rate.set_value(q - dq);
            let value_down = option(&market, option_type, moneyness).npv().unwrap();
            market.q_rate.set_value(q);
            let fd_div_rho = (value_up - value_down) / (2.0 * dq);

            let dv = vol * 1.0e-4;
            market.vol.set_value(vol + dv);
            let value_up = option(&market, option_type, moneyness).npv().unwrap();
            market.vol.set_value(vol - dv);
            let value_down = option(&market, option_type, moneyness).npv().unwrap();
            market.vol.set_value(vol);
            let fd_vega = (value_up - value_down) / (2.0 * dv);

            for (name, analytic, finite_difference) in [
                ("rho", rho, fd_rho),
                ("dividendRho", dividend_rho, fd_div_rho),
                ("vega", vega, fd_vega),
            ] {
                let error = (analytic - finite_difference).abs() / spot;
                assert!(
                    error <= tolerance,
                    "{name} of the {option_type:?} performance option: analytic {analytic} vs \
                     finite difference {finite_difference} (relative-to-spot error {error})"
                );
            }
        }
    }
}
