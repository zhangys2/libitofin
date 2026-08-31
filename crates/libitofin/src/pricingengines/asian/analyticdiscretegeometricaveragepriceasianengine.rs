//! Analytic discrete geometric-average price Asian engine.
//!
//! Port of `ql/pricingengines/asian/analytic_discr_geom_av_price.{hpp,cpp}`:
//! Levy (1997) in Clewlow–Strickland.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults, StrikedTypePayoff,
    TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::distributions::normal::{CumulativeNormalDistribution, NormalDistribution};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::{BlackCalculator, black_scholes_theta};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::Real;

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Pricing engine for European discrete geometric average-price Asians.
pub struct AnalyticDiscreteGeometricAveragePriceAsianEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl AnalyticDiscreteGeometricAveragePriceAsianEngine {
    /// `AnalyticDiscreteGeometricAveragePriceAsianEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(
            DiscreteAveragingAsianArguments::default(),
            DiscreteAveragingAsianResults::default(),
        );
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for AnalyticDiscreteGeometricAveragePriceAsianEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticDiscreteGeometricAveragePriceAsianEngine {
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
        let average_type = args.average_type.expect("validated");
        let payoff = args.payoff.expect("validated");
        let exercise = args.exercise.as_ref().expect("validated");
        let running_accumulator = args.running_accumulator.expect("validated");
        let past_fixings = args.past_fixings.expect("validated");
        let fixing_dates = &args.fixing_dates;

        let (running_log, past_fixings) = if average_type == AverageType::Geometric {
            require!(
                running_accumulator > 0.0,
                "positive running product required: {running_accumulator} not allowed"
            );
            (running_accumulator.ln(), past_fixings)
        } else {
            (1.0, 0)
        };

        if exercise.exercise_type() != ExerciseType::European {
            fail!("not an European Option");
        }

        let vol_ts = self.process.black_volatility().current_link()?;
        let r_ts = self.process.risk_free_rate().current_link()?;
        let q_ts = self.process.dividend_yield().current_link()?;

        let reference_date = r_ts.reference_date()?;
        let rfdc = r_ts.require_day_counter()?;
        let divdc = q_ts.require_day_counter()?;
        let voldc = vol_ts.require_day_counter()?;

        let mut fixing_times = Vec::new();
        for fixing_date in fixing_dates {
            if *fixing_date >= reference_date {
                fixing_times.push(voldc.year_fraction(reference_date, *fixing_date));
            }
        }

        let remaining_fixings = fixing_times.len();
        let number_of_fixings = past_fixings + remaining_fixings;
        let n = number_of_fixings as Real;

        let past_weight = past_fixings as Real / n;
        let future_weight = 1.0 - past_weight;

        let time_sum: Real = fixing_times.iter().sum();

        let exercise_date = exercise.last_date();
        let strike = payoff.strike();
        let vola = vol_ts.black_vol_date(exercise_date, strike, false)?;

        let mut temp = 0.0;
        for i in (past_fixings + 1)..number_of_fixings {
            temp += fixing_times[i - past_fixings - 1] * (n - i as Real);
        }
        let variance = vola * vola / (n * n) * (time_sum + 2.0 * temp);
        let sig_g = vola * ((time_sum + 2.0 * temp) / (n * n)).sqrt();
        let dsig_g_dsig = ((time_sum + 2.0 * temp) / (n * n)).sqrt();
        let dmu_g_dsig = -(vola * time_sum) / n;

        let dividend_rate = q_ts
            .zero_rate_date(
                exercise_date,
                divdc.clone(),
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let risk_free_rate = r_ts
            .zero_rate_date(
                exercise_date,
                rfdc.clone(),
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let nu = risk_free_rate - dividend_rate - 0.5 * vola * vola;

        let spot = self.process.x0()?;
        require!(spot > 0.0, "positive underlying value required");

        let m = if past_fixings == 0 { 1 } else { past_fixings };
        let mu_g = past_weight * running_log / m as Real
            + future_weight * spot.ln()
            + nu * time_sum / n;
        let forward_price = (mu_g + variance / 2.0).exp();

        let risk_free_discount = r_ts.discount_date(exercise_date, false)?;

        let black = BlackCalculator::new(
            payoff.option_type(),
            strike,
            forward_price,
            variance.sqrt(),
            risk_free_discount,
        )?;

        let results = self.base.results_mut();
        let value = black.value();
        results.instrument.value = Some(value);

        // Greeks follow QuantLib; oracle for them is a follow-up slice.
        let (nx_1, nx_1_density) = if sig_g > Real::EPSILON {
            let x_1 = (mu_g - strike.ln() + variance) / sig_g;
            let cnd = CumulativeNormalDistribution::standard();
            let nd = NormalDistribution::standard();
            (cnd.value(x_1), nd.value(x_1))
        } else if mu_g > strike.ln() {
            (1.0, 0.0)
        } else {
            (0.0, 0.0)
        };

        let vega = forward_price
            * risk_free_discount
            * ((dmu_g_dsig + sig_g * dsig_g_dsig) * nx_1 + nx_1_density * dsig_g_dsig)
            - if payoff.option_type() == OptionType::Put {
                risk_free_discount
                    * forward_price
                    * (dmu_g_dsig + sig_g * dsig_g_dsig)
            } else {
                0.0
            };

        let t_rho = rfdc.year_fraction(r_ts.reference_date()?, exercise_date);
        let t_div = divdc.year_fraction(q_ts.reference_date()?, exercise_date);

        let delta =
            future_weight * black.delta(forward_price)? * forward_price / spot;
        let gamma = forward_price * future_weight / (spot * spot)
            * (black.gamma(forward_price)? * future_weight * forward_price
                - past_weight * black.delta(forward_price)?);

        results.greeks.delta = Some(delta);
        results.greeks.gamma = Some(gamma);
        results.greeks.vega = Some(vega);
        results.greeks.rho = Some(
            black.rho(t_rho)? * time_sum / (n * t_rho) - (t_rho - time_sum / n) * value,
        );
        results.greeks.dividend_rho =
            Some(black.dividend_rho(t_div)? * time_sum / (n * t_div));
        results.greeks.theta = Some(black_scholes_theta(
            &self.process,
            value,
            delta,
            gamma,
        )?);

        Ok(())
    }
}

/// Attaches [`AnalyticDiscreteGeometricAveragePriceAsianEngine`] to `option`.
pub fn set_analytic_discrete_geometric_average_price_asian_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine = shared_mut(AnalyticDiscreteGeometricAveragePriceAsianEngine::new(process))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{DiscreteAveragingAsianOption, PlainVanillaPayoff};
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

    /// `asianoptions.cpp` `testAnalyticDiscreteGeometricAveragePrice` (Levy 1997).
    #[test]
    fn levy_discrete_geometric_average_price_call() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.03));
        let r_rate = shared(SimpleQuote::new(0.06));
        let vol = shared(SimpleQuote::new(0.20));

        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat_rate(today, &q_rate),
            flat_rate(today, &r_rate),
            flat_vol(today, &vol),
        ));

        let future_fixings = 10;
        let exercise_date = today + 360;
        let dt = (360.0 / future_fixings as Real).round() as i32;
        let mut fixing_dates = Vec::with_capacity(future_fixings);
        fixing_dates.push(today + dt);
        for _ in 1..future_fixings {
            let last = *fixing_dates.last().expect("non-empty");
            fixing_dates.push(last + dt);
        }

        let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let exercise = shared(EuropeanExercise::new(exercise_date));
        let mut option = DiscreteAveragingAsianOption::new(
            AverageType::Geometric,
            1.0,
            0,
            fixing_dates,
            payoff,
            exercise,
            Shared::clone(&settings),
        )
        .unwrap();

        set_analytic_discrete_geometric_average_price_asian_engine(&mut option, process);

        let calculated = option.npv().unwrap();
        let expected = 5.3425606635;
        assert!(
            (calculated - expected).abs() <= 1.0e-10,
            "expected {expected}, got {calculated}"
        );
    }

    /// Theta from `blackScholesTheta(process, value, delta, gamma)` vs eval-date FD.
    #[test]
    fn discrete_geometric_average_price_theta_matches_finite_difference() {
        use crate::pricingengine::PricingEngine;
        use crate::shared::{SharedMut, shared_mut};
        use crate::time::calendars::NullCalendar;
        use crate::types::Real;

        const TOLERANCE: Real = 1.0e-5;
        const UNDERLYING: Real = 100.0;

        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let spot = shared(SimpleQuote::new(UNDERLYING));
        let q_rate = shared(SimpleQuote::new(0.03));
        let r_rate = shared(SimpleQuote::new(0.06));
        let vol = shared(SimpleQuote::new(0.20));
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

        let future_fixings = 10;
        let exercise_date = today + 360;
        let dt_fix = (360.0 / future_fixings as Real).round() as i32;
        let mut fixing_dates = Vec::with_capacity(future_fixings);
        fixing_dates.push(today + dt_fix);
        for _ in 1..future_fixings {
            let last = *fixing_dates.last().expect("non-empty");
            fixing_dates.push(last + dt_fix);
        }

        let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let exercise = shared(EuropeanExercise::new(exercise_date));
        let mut option = DiscreteAveragingAsianOption::new(
            AverageType::Geometric,
            1.0,
            0,
            fixing_dates,
            payoff,
            exercise,
            Shared::clone(&settings),
        )
        .unwrap();
        let engine = shared_mut(AnalyticDiscreteGeometricAveragePriceAsianEngine::new(
            Shared::clone(&process),
        ));
        option
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

        let value = option.npv().unwrap();
        assert!(value > UNDERLYING * 1.0e-5);
        let theta = option.theta().unwrap();

        let dc = Actual360::new();
        let dt = dc.year_fraction(today - 1, today + 1);
        settings.set_evaluation_date(today - 1);
        let value_m = option.npv().unwrap();
        settings.set_evaluation_date(today + 1);
        let value_p = option.npv().unwrap();
        settings.set_evaluation_date(today);
        let expected_theta = (value_p - value_m) / dt;

        let error = (expected_theta - theta).abs() / UNDERLYING;
        assert!(
            error <= TOLERANCE,
            "theta analytic {theta} vs FD {expected_theta} (rel err {error})"
        );
    }

    mod greeks {
        //! `asianoptions.cpp` `testAnalyticDiscreteGeometricAveragePriceGreeks`.

        use super::*;
        use crate::pricingengine::PricingEngine;
        use crate::processes::GeneralizedBlackScholesProcess;
        use crate::shared::{SharedMut, shared_mut};
        use crate::termstructures::volatility::BlackVolTermStructure;
        use crate::termstructures::yieldtermstructure::YieldTermStructure;
        use crate::time::calendars::NullCalendar;
        use crate::time::period::Period;
        use crate::time::timeunit::TimeUnit;
        use crate::types::Real;

        const TOLERANCE: Real = 1.0e-5;
        const UNDERLYING: Real = 100.0;
        const RUNNING_AVERAGE: Real = 120.0;
        const PAST_FIXINGS: usize = 1;

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

        fn fixing_dates(today: Date, expiry: Date) -> Vec<Date> {
            let mut dates = Vec::new();
            let mut d = today + Period::new(3, TimeUnit::Months);
            while d <= expiry {
                dates.push(d);
                d = d + Period::new(3, TimeUnit::Months);
            }
            dates
        }

        #[test]
        fn analytic_discrete_geometric_average_price_greeks_match_finite_differences() {
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
                        let mut option = DiscreteAveragingAsianOption::new(
                            AverageType::Geometric,
                            RUNNING_AVERAGE,
                            PAST_FIXINGS,
                            fixing_dates(today, expiry),
                            payoff,
                            exercise,
                            Shared::clone(&market.settings),
                        )
                        .unwrap();
                        let engine = shared_mut(
                            AnalyticDiscreteGeometricAveragePriceAsianEngine::new(
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
