//! Analytic continuous geometric-average price Asian engine.
//!
//! Port of `ql/pricingengines/asian/analytic_cont_geom_av_price.{hpp,cpp}`:
//! Haug (1997) pp.96–97 via a scaled Black formula.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, ContinuousAveragingAsianArguments, ContinuousAveragingAsianResults, Greeks,
    StrikedTypePayoff, TypePayoff,
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

type EngineBase = GenericEngine<ContinuousAveragingAsianArguments, ContinuousAveragingAsianResults>;

/// Pricing engine for European continuous geometric average-price Asians.
pub struct AnalyticContinuousGeometricAveragePriceAsianEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticContinuousGeometricAveragePriceAsianEngine {
    /// `AnalyticContinuousGeometricAveragePriceAsianEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            ContinuousAveragingAsianArguments::default(),
            ContinuousAveragingAsianResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for AnalyticContinuousGeometricAveragePriceAsianEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticContinuousGeometricAveragePriceAsianEngine {
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
        let (average_type, payoff, exercise) = {
            let args = self.base.arguments();
            (
                args.average_type.expect("validated"),
                args.payoff.expect("validated"),
                args.exercise.as_ref().expect("validated"),
            )
        };

        require!(
            average_type == AverageType::Geometric,
            "not a geometric average option"
        );
        if exercise.exercise_type() != ExerciseType::European {
            fail!("not an European Option");
        }

        let exercise_date = exercise.last_date();
        let strike = payoff.strike();

        let vol_ts = self.process.black_volatility().current_link()?;
        let r_ts = self.process.risk_free_rate().current_link()?;
        let q_ts = self.process.dividend_yield().current_link()?;

        let volatility = vol_ts.black_vol_date(exercise_date, strike, false)?;
        let rfdc = r_ts.require_day_counter()?;
        let divdc = q_ts.require_day_counter()?;
        let voldc = vol_ts.require_day_counter()?;

        let r_rate = r_ts
            .zero_rate_date(
                exercise_date,
                rfdc.clone(),
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let q_rate = q_ts
            .zero_rate_date(
                exercise_date,
                divdc.clone(),
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let dividend_yield = 0.5 * (r_rate + q_rate + volatility * volatility / 6.0);

        let t_q = divdc.year_fraction(q_ts.reference_date()?, exercise_date);
        let dividend_discount = (-dividend_yield * t_q).exp();
        let risk_free_discount = r_ts.discount_date(exercise_date, false)?;

        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying");
        let forward = spot * dividend_discount / risk_free_discount;

        let t_v = voldc.year_fraction(vol_ts.reference_date()?, exercise_date);
        let variance = volatility * volatility * t_v;
        let std_dev = (variance / 3.0).sqrt();

        let black = BlackCalculator::new(
            payoff.option_type(),
            strike,
            forward,
            std_dev,
            risk_free_discount,
        )?;

        let t_r = rfdc.year_fraction(r_ts.reference_date()?, exercise_date);
        let div_rho = black.dividend_rho(t_q)?;
        let theta = black.theta(spot, t_v).ok();

        let results = self.base.results_mut();
        results.instrument.value = Some(black.value());
        results.greeks = Greeks {
            delta: Some(black.delta(spot)?),
            gamma: Some(black.gamma(spot)?),
            theta,
            vega: Some(black.vega(t_v)? / 3.0_f64.sqrt() + div_rho * volatility / 6.0),
            rho: Some(black.rho(t_r)? + 0.5 * div_rho),
            dividend_rho: Some(div_rho / 2.0),
        };
        Ok(())
    }
}

/// Attaches [`AnalyticContinuousGeometricAveragePriceAsianEngine`] to `option`.
pub fn set_analytic_continuous_geometric_average_price_asian_engine(
    option: &mut crate::instruments::ContinuousAveragingAsianOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticContinuousGeometricAveragePriceAsianEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{ContinuousAveragingAsianOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{Shared, shared};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn crate::quotes::Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn crate::quotes::Quote>)
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

    /// `asianoptions.cpp` `testAnalyticContinuousGeometricAveragePrice` (Haug p.96–97).
    #[test]
    fn haug_continuous_geometric_average_price_put() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(80.0));
        let q_rate = shared(SimpleQuote::new(-0.03));
        let r_rate = shared(SimpleQuote::new(0.05));
        let vol = shared(SimpleQuote::new(0.20));

        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        let payoff = PlainVanillaPayoff::new(OptionType::Put, 85.0);
        let exercise = shared(EuropeanExercise::new(today + 90));
        let mut option = ContinuousAveragingAsianOption::new(
            AverageType::Geometric,
            payoff,
            exercise,
            Shared::clone(&settings),
        )
        .unwrap();

        set_analytic_continuous_geometric_average_price_asian_engine(&mut option, process);

        let calculated = option.npv().unwrap();
        let expected = 4.6922;
        assert!(
            (calculated - expected).abs() <= 1.0e-4,
            "expected {expected}, got {calculated}"
        );
    }

    mod greeks {
        //! `asianoptions.cpp` `testAnalyticContinuousGeometricAveragePriceGreeks`.

        use super::*;
        use crate::pricingengine::PricingEngine;
        use crate::processes::GeneralizedBlackScholesProcess;
        use crate::shared::{SharedMut, shared_mut};
        use crate::termstructures::volatility::BlackVolTermStructure;
        use crate::termstructures::yieldtermstructure::YieldTermStructure;
        use crate::time::calendars::NullCalendar;
        use crate::time::date::Date;
        use crate::time::period::Period;
        use crate::time::timeunit::TimeUnit;
        use crate::types::Real;

        const TOLERANCE: Real = 1.0e-5;
        const UNDERLYING: Real = 100.0;

        struct MovingMarket {
            settings: Shared<Settings<Date>>,
            spot: Shared<SimpleQuote>,
            q_rate: Shared<SimpleQuote>,
            r_rate: Shared<SimpleQuote>,
            vol: Shared<SimpleQuote>,
            process: Shared<GeneralizedBlackScholesProcess>,
        }

        fn moving_market(today: Date) -> MovingMarket {
            let settings = shared(Settings::new());
            settings.set_evaluation_date(today);
            let spot = shared(SimpleQuote::new(0.0));
            let q_rate = shared(SimpleQuote::new(0.0));
            let r_rate = shared(SimpleQuote::new(0.0));
            let vol = shared(SimpleQuote::new(0.0));
            let flat = |quote: &Shared<SimpleQuote>| {
                shared(FlatForward::moving(
                    0,
                    NullCalendar::new(),
                    quote_handle(quote),
                    Actual360::new(),
                    Compounding::Continuous,
                    Frequency::Annual,
                    Shared::clone(&settings),
                )) as Shared<dyn YieldTermStructure>
            };
            let flat_vol = |quote: &Shared<SimpleQuote>| {
                shared(BlackConstantVol::moving_with_quote(
                    0,
                    NullCalendar::new(),
                    quote_handle(quote),
                    Actual360::new(),
                    Shared::clone(&settings),
                )) as Shared<dyn BlackVolTermStructure>
            };
            let process = shared(BlackScholesMertonProcess::new(
                quote_handle(&spot),
                Handle::new(flat(&q_rate)),
                Handle::new(flat(&r_rate)),
                Handle::new(flat_vol(&vol)),
            ));
            MovingMarket {
                settings,
                spot,
                q_rate,
                r_rate,
                vol,
                process,
            }
        }

        fn relative_error(x1: Real, x2: Real, reference: Real) -> Real {
            if reference != 0.0 {
                (x1 - x2).abs() / reference
            } else {
                (x1 - x2).abs()
            }
        }

        #[test]
        fn analytic_continuous_geometric_average_price_greeks_match_finite_differences() {
            let today = Date::new(15, Month::June, 2026);
            let market = moving_market(today);
            let types = [OptionType::Call, OptionType::Put];
            let strikes = [90.0, 100.0, 110.0];
            let q_rates = [0.04, 0.05, 0.06];
            let r_rates = [0.01, 0.05, 0.15];
            let lengths = [1, 2];
            let vols = [0.11, 0.50, 1.20];
            let dc = Actual360::new();

            for option_type in types {
                for strike in strikes {
                    for length in lengths {
                        let expiry = today + Period::new(length, TimeUnit::Years);
                        let payoff = PlainVanillaPayoff::new(option_type, strike);
                        let exercise = shared(EuropeanExercise::new(expiry));
                        let mut option = ContinuousAveragingAsianOption::new(
                            AverageType::Geometric,
                            payoff,
                            exercise,
                            Shared::clone(&market.settings),
                        )
                        .unwrap();
                        let engine = shared_mut(
                            AnalyticContinuousGeometricAveragePriceAsianEngine::new(
                                Shared::clone(&market.process),
                            ),
                        );
                        option
                            .base_mut()
                            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

                        for q in q_rates {
                            for r in r_rates {
                                for v in vols {
                                    let u = UNDERLYING;
                                    market.spot.set_value(u);
                                    market.q_rate.set_value(q);
                                    market.r_rate.set_value(r);
                                    market.vol.set_value(v);

                                    let value = option.npv().unwrap();
                                    if value <= u * 1.0e-5 {
                                        continue;
                                    }

                                    let delta = option.delta().unwrap();
                                    let gamma = option.gamma().unwrap();
                                    let theta = option.theta().unwrap();
                                    let rho = option.rho().unwrap();
                                    let dividend_rho = option.dividend_rho().unwrap();
                                    let vega = option.vega().unwrap();

                                    let du = u * 1.0e-4;
                                    market.spot.set_value(u + du);
                                    let value_p = option.npv().unwrap();
                                    let delta_p = option.delta().unwrap();
                                    market.spot.set_value(u - du);
                                    let value_m = option.npv().unwrap();
                                    let delta_m = option.delta().unwrap();
                                    market.spot.set_value(u);
                                    let expected_delta = (value_p - value_m) / (2.0 * du);
                                    let expected_gamma = (delta_p - delta_m) / (2.0 * du);

                                    let dr = r * 1.0e-4;
                                    market.r_rate.set_value(r + dr);
                                    let value_p = option.npv().unwrap();
                                    market.r_rate.set_value(r - dr);
                                    let value_m = option.npv().unwrap();
                                    market.r_rate.set_value(r);
                                    let expected_rho = (value_p - value_m) / (2.0 * dr);

                                    let dq = q * 1.0e-4;
                                    market.q_rate.set_value(q + dq);
                                    let value_p = option.npv().unwrap();
                                    market.q_rate.set_value(q - dq);
                                    let value_m = option.npv().unwrap();
                                    market.q_rate.set_value(q);
                                    let expected_dividend_rho = (value_p - value_m) / (2.0 * dq);

                                    let dv = v * 1.0e-4;
                                    market.vol.set_value(v + dv);
                                    let value_p = option.npv().unwrap();
                                    market.vol.set_value(v - dv);
                                    let value_m = option.npv().unwrap();
                                    market.vol.set_value(v);
                                    let expected_vega = (value_p - value_m) / (2.0 * dv);

                                    let dt = dc.year_fraction(today - 1, today + 1);
                                    market.settings.set_evaluation_date(today - 1);
                                    let value_m = option.npv().unwrap();
                                    market.settings.set_evaluation_date(today + 1);
                                    let value_p = option.npv().unwrap();
                                    market.settings.set_evaluation_date(today);
                                    let expected_theta = (value_p - value_m) / dt;

                                    for (name, expected, calculated) in [
                                        ("delta", expected_delta, delta),
                                        ("gamma", expected_gamma, gamma),
                                        ("theta", expected_theta, theta),
                                        ("rho", expected_rho, rho),
                                        ("divRho", expected_dividend_rho, dividend_rho),
                                        ("vega", expected_vega, vega),
                                    ] {
                                        let error = relative_error(expected, calculated, u);
                                        assert!(
                                            error <= TOLERANCE,
                                            "{name} of {option_type:?} K={strike} T={length}y \
                                             q={q} r={r} v={v}: analytic {calculated} vs FD \
                                             {expected} (rel err {error})"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
