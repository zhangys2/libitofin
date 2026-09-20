//! Forward (strike-resetting) vanilla pricing engines.
//!
//! Port of `ql/pricingengines/forward/` plus the quanto specialisation
//! `QuantoEngine<ForwardVanillaOption, ForwardVanillaEngine<AnalyticEuropeanEngine>>`.

mod analyticforwardperformancevanillaengine;
mod analyticforwardvanillaengine;
mod analytichestonforwardeuropeanengine;
mod mcforwardeuropeanbsengine;
mod mcforwardeuropeanhestonengine;
mod quantoforwardengine;
mod quantoforwardperformanceengine;

pub use analyticforwardperformancevanillaengine::{
    AnalyticForwardPerformanceVanillaEngine, set_analytic_forward_performance_vanilla_engine,
};
pub use analyticforwardvanillaengine::{
    AnalyticForwardVanillaEngine, BinomialForwardVanillaEngine, set_analytic_forward_vanilla_engine,
};
pub use analytichestonforwardeuropeanengine::AnalyticHestonForwardEuropeanEngine;
pub use mcforwardeuropeanbsengine::{
    ForwardEuropeanBsPathPricer, MakeMcForwardEuropeanBsEngine, McForwardEuropeanBsEngine,
    set_mc_forward_european_bs_engine,
};
pub use mcforwardeuropeanhestonengine::{
    ForwardEuropeanHestonPathPricer, MakeMcForwardEuropeanHestonEngine,
    McForwardEuropeanHestonEngine,
};
pub use quantoforwardengine::{QuantoForwardEuropeanEngine, set_quanto_forward_european_engine};
pub use quantoforwardperformanceengine::{
    QuantoForwardPerformanceEuropeanEngine, set_quanto_forward_performance_european_engine,
};

#[cfg(test)]
mod test_greeks {
    //! `forwardoption.cpp` `testGreeks` / `testPerformanceGreeks`: analytic
    //! forward (and performance) greeks vs central finite differences on
    //! moving curves (tol 1e-5 relative to spot).

    use super::{
        set_analytic_forward_performance_vanilla_engine, set_analytic_forward_vanilla_engine,
    };
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{ForwardVanillaOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengines::vanilla::test_market::{quote_handle, today};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{Shared, shared};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::NullCalendar;
    use crate::time::date::Date;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
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
        process: Shared<BlackScholesMertonProcess>,
    }

    fn moving_market() -> MovingMarket {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
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
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            Handle::new(flat(&q_rate)),
            Handle::new(flat(&r_rate)),
            Handle::new(shared(BlackConstantVol::moving_with_quote(
                0,
                NullCalendar::new(),
                quote_handle(&vol),
                Actual360::new(),
                Shared::clone(&settings),
            )) as Shared<dyn BlackVolTermStructure>),
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

    fn assert_greeks_match_finite_differences(
        attach: fn(&mut ForwardVanillaOption, Shared<BlackScholesMertonProcess>),
    ) {
        let market = moving_market();
        let types = [OptionType::Call, OptionType::Put];
        let moneyness = [0.9, 1.0, 1.1];
        let q_rates = [0.04, 0.05, 0.06];
        let r_rates = [0.01, 0.05, 0.15];
        let lengths = [1, 2];
        let start_months = [6, 9];
        let vols = [0.11, 0.50, 1.20];
        let day_counter = Actual360::new();

        for option_type in types {
            for m in moneyness {
                for length in lengths {
                    for start_month in start_months {
                        let expiry = today() + Period::new(length, TimeUnit::Years);
                        let reset = today() + Period::new(start_month, TimeUnit::Months);
                        let payoff = shared(PlainVanillaPayoff::new(option_type, 0.0))
                            as Shared<dyn crate::instruments::StrikedTypePayoff>;
                        let exercise = shared(EuropeanExercise::new(expiry));
                        let mut option = ForwardVanillaOption::new(
                            m,
                            reset,
                            payoff,
                            exercise,
                            Shared::clone(&market.settings),
                        );
                        attach(&mut option, Shared::clone(&market.process));

                        for u in [UNDERLYING] {
                            for q in q_rates {
                                for r in r_rates {
                                    for v in vols {
                                        market.spot.set_value(u);
                                        market.q_rate.set_value(q);
                                        market.r_rate.set_value(r);
                                        market.vol.set_value(v);

                                        let value = option.npv().unwrap();
                                        let delta = option.delta().unwrap();
                                        let gamma = option.gamma().unwrap();
                                        let theta = option.theta().unwrap();
                                        let rho = option.rho().unwrap();
                                        let div_rho = option.dividend_rho().unwrap();
                                        let vega = option.vega().unwrap();

                                        if value <= u * 1.0e-5 {
                                            continue;
                                        }

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
                                        let expected_div_rho = (value_p - value_m) / (2.0 * dq);

                                        let dv = v * 1.0e-4;
                                        market.vol.set_value(v + dv);
                                        let value_p = option.npv().unwrap();
                                        market.vol.set_value(v - dv);
                                        let value_m = option.npv().unwrap();
                                        market.vol.set_value(v);
                                        let expected_vega = (value_p - value_m) / (2.0 * dv);

                                        let dt =
                                            day_counter.year_fraction(today() - 1, today() + 1);
                                        market.settings.set_evaluation_date(today() - 1);
                                        let value_m = option.npv().unwrap();
                                        market.settings.set_evaluation_date(today() + 1);
                                        let value_p = option.npv().unwrap();
                                        market.settings.set_evaluation_date(today());
                                        let expected_theta = (value_p - value_m) / dt;

                                        for (name, expected, calculated) in [
                                            ("delta", expected_delta, delta),
                                            ("gamma", expected_gamma, gamma),
                                            ("theta", expected_theta, theta),
                                            ("rho", expected_rho, rho),
                                            ("divRho", expected_div_rho, div_rho),
                                            ("vega", expected_vega, vega),
                                        ] {
                                            let error = relative_error(expected, calculated, u);
                                            assert!(
                                                error <= TOLERANCE,
                                                "{name} of {option_type:?} moneyness={m} \
                                                 length={length} start={start_month} q={q} r={r} \
                                                 v={v}: analytic {calculated} vs FD {expected} \
                                                 (relative error {error})"
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

    #[test]
    fn analytic_forward_greeks_match_finite_differences() {
        assert_greeks_match_finite_differences(set_analytic_forward_vanilla_engine);
    }

    #[test]
    fn analytic_forward_performance_greeks_match_finite_differences() {
        assert_greeks_match_finite_differences(set_analytic_forward_performance_vanilla_engine);
    }
}
