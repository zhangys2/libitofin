//! Analytic quanto vanilla engine.
//!
//! Port of `ql/pricingengines/quanto/quantoengine.hpp` specialised to
//! `QuantoEngine<VanillaOption, AnalyticEuropeanEngine>`: the inner Black
//! engine is run on a process whose dividend curve is a
//! [`QuantoTermStructure`](crate::termstructures::yields::QuantoTermStructure),
//! and the quanto greeks are recovered from the inner dividend-rho as in
//! `quantoengine.hpp:130-168`.

use std::any::Any;

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instruments::{OneAssetOptionEngine, OneAssetOptionResults, OptionArguments};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::shared::{Shared, shared};
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yields::QuantoTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::Real;

/// `QuantoEngine<VanillaOption, AnalyticEuropeanEngine>`.
pub struct QuantoEuropeanEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
}

impl QuantoEuropeanEngine {
    /// `QuantoEngine(process, foreignRiskFreeRate, exchangeRateVolatility, correlation)`.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
        exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
        correlation: Handle<dyn Quote>,
    ) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        let observer = base.observer();
        foreign_risk_free_rate.register_observer(&observer);
        exchange_rate_volatility.register_observer(&observer);
        correlation.register_observer(&observer);
        Self {
            base,
            process,
            foreign_risk_free_rate,
            exchange_rate_volatility,
            correlation,
        }
    }
}

impl AsObservable for QuantoEuropeanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for QuantoEuropeanEngine {
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
        let (payoff, exercise, strike) = {
            let arguments = self.base.arguments();
            let Some(payoff) = arguments.payoff.as_ref() else {
                fail!("no payoff given");
            };
            let Some(exercise) = arguments.exercise.as_ref() else {
                fail!("no exercise given");
            };
            (
                Shared::clone(payoff),
                Shared::clone(exercise),
                payoff.strike(),
            )
        };
        let spot = self.process.state_variable().current_link()?.value()?;
        if spot.is_nan() || spot <= 0.0 {
            fail!("negative or null underlying");
        }

        const EXCHANGE_RATE_ATM: Real = 1.0;
        let correlation = self.correlation.current_link()?.value()?;
        let dividend_yield = Handle::new(shared(QuantoTermStructure::new(
            self.process.dividend_yield(),
            self.process.risk_free_rate(),
            self.foreign_risk_free_rate.clone(),
            self.process.black_volatility(),
            strike,
            self.exchange_rate_volatility.clone(),
            EXCHANGE_RATE_ATM,
            correlation,
        )) as Shared<dyn YieldTermStructure>);
        let quanto_process = shared(GeneralizedBlackScholesProcess::new(
            self.process.state_variable(),
            dividend_yield,
            self.process.risk_free_rate(),
            self.process.black_volatility(),
        ));

        let (value, mut greeks, more_greeks) = {
            let mut original = AnalyticEuropeanEngine::new(quanto_process);
            let original_results = original
                .calculate_from_arguments(Shared::clone(&payoff), Shared::clone(&exercise))?;
            (
                original_results.instrument.value,
                original_results.greeks,
                original_results.more_greeks,
            )
        };

        let maturity = exercise.last_date();
        let fx_flat_vol = self
            .exchange_rate_volatility
            .current_link()?
            .black_vol_date(maturity, EXCHANGE_RATE_ATM, false)?;
        let original_div_rho = greeks.dividend_rho;

        // `quantoengine.hpp:136-143`: both rho and dividendRho must be present
        // or both are cleared. The original dividendRho is kept for vega / q*.
        if let (Some(rho), Some(div_rho)) = (greeks.rho, original_div_rho) {
            greeks.rho = Some(rho + div_rho);
            greeks.dividend_rho = Some(div_rho);
        } else {
            greeks.rho = None;
            greeks.dividend_rho = None;
        }
        if let (Some(vega), Some(div_rho)) = (greeks.vega, original_div_rho) {
            greeks.vega = Some(vega + correlation * fx_flat_vol * div_rho);
        }

        let (qvega, qrho, qlambda) = if let Some(div_rho) = original_div_rho {
            let eq_vol = self
                .process
                .black_volatility()
                .current_link()?
                .black_vol_date(maturity, spot, false)?;
            (
                Some(correlation * eq_vol * div_rho),
                Some(-div_rho),
                Some(fx_flat_vol * eq_vol * div_rho),
            )
        } else {
            (None, None, None)
        };

        let results = self.base.results_mut();
        results.instrument.value = value;
        results.greeks = greeks;
        results.more_greeks = more_greeks;
        let extras = &mut results.instrument.additional_results;
        let mut extra = |tag: &str, value: Option<Real>| {
            if let Some(value) = value {
                extras.insert(tag.to_string(), shared(value) as Shared<dyn Any>);
            }
        };
        extra("qvega", qvega);
        extra("qrho", qrho);
        extra("qlambda", qlambda);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::instrument::Instrument;
    use crate::instruments::{PlainVanillaPayoff, StrikedTypePayoff, VanillaOption};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared_mut};
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::types::{Rate, Time, Volatility};

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn time_to_days(t: Time) -> i32 {
        (t * 360.0).round() as i32
    }

    fn flat_rate(rate: Rate) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(vol: Volatility) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(
            shared(BlackConstantVol::new(today(), None, vol, Actual360::new()))
                as Shared<dyn BlackVolTermStructure>,
        )
    }

    fn haug_market() -> (
        Real,
        Real,
        Rate,
        Rate,
        Time,
        Volatility,
        Rate,
        Volatility,
        Real,
    ) {
        (100.0, 105.0, 0.04, 0.08, 0.5, 0.2, 0.05, 0.10, 0.3)
    }

    #[allow(clippy::too_many_arguments)]
    fn quanto_option(
        settings: Shared<Settings<Date>>,
        option_type: OptionType,
        s: Real,
        k: Real,
        q: Rate,
        r: Rate,
        t: Time,
        vol: Volatility,
        fxr: Rate,
        fxv: Volatility,
        corr: Real,
    ) -> VanillaOption {
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(s)) as Shared<dyn Quote>),
            flat_rate(q),
            flat_rate(r),
            flat_vol(vol),
        ));
        let engine = shared_mut(QuantoEuropeanEngine::new(
            process,
            flat_rate(fxr),
            flat_vol(fxv),
            Handle::new(shared(SimpleQuote::new(corr)) as Shared<dyn Quote>),
        ));
        let payoff: Shared<dyn StrikedTypePayoff> = shared(PlainVanillaPayoff::new(option_type, k));
        let exercise = shared(EuropeanExercise::new(today() + time_to_days(t)));
        let mut option = VanillaOption::new(payoff, exercise, settings);
        option
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        option
    }

    /// `quantooption.cpp` `testValues`: Haug p.105–106 / VBA @ 1e-4.
    #[test]
    fn haug_quanto_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let (s, k, q, r, t, vol, fxr, fxv, corr) = haug_market();
        let cases = [(OptionType::Call, 5.3280 / 1.5), (OptionType::Put, 8.1636)];
        for (option_type, expected) in cases {
            let mut option = quanto_option(
                Shared::clone(&settings),
                option_type,
                s,
                k,
                q,
                r,
                t,
                vol,
                fxr,
                fxv,
                corr,
            );
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - expected).abs() <= 1.0e-4,
                "{option_type:?}: {calculated} vs Haug {expected}"
            );
        }
    }

    /// Quanto call equals Black with `q + (r_d − r_f + ρ σ σ_fx)`.
    #[test]
    fn quanto_call_matches_adjusted_black() {
        use crate::pricingengines::vanilla::AnalyticEuropeanEngine;

        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let (s, k, q, r, t, vol, fxr, fxv, corr) = haug_market();
        let q_adj = q + (r - fxr + corr * vol * fxv);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, k));
        let exercise = shared(EuropeanExercise::new(today() + time_to_days(t)));

        let mut quanto = quanto_option(
            Shared::clone(&settings),
            OptionType::Call,
            s,
            k,
            q,
            r,
            t,
            vol,
            fxr,
            fxv,
            corr,
        );

        let adj_process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(s)) as Shared<dyn Quote>),
            flat_rate(q_adj),
            flat_rate(r),
            flat_vol(vol),
        ));
        let mut black = VanillaOption::new(payoff, exercise, settings);
        black
            .base_mut()
            .set_pricing_engine(shared_mut(AnalyticEuropeanEngine::new(adj_process))
                as SharedMut<dyn PricingEngine>);

        let q_npv = quanto.npv().unwrap();
        let b_npv = black.npv().unwrap();
        assert!(
            (q_npv - b_npv).abs() < 1e-12,
            "quanto {q_npv} vs adjusted Black {b_npv}"
        );
        assert!(
            (quanto.delta().unwrap() - black.delta().unwrap()).abs() < 1e-12,
            "delta should match the inner Black engine"
        );
        assert!(
            (quanto.gamma().unwrap() - black.gamma().unwrap()).abs() < 1e-12,
            "gamma should match the inner Black engine"
        );

        let div_rho = black.dividend_rho().unwrap();
        assert!(
            (quanto.rho().unwrap() - (black.rho().unwrap() + div_rho)).abs() < 1e-12,
            "quanto rho = inner rho + dividend rho"
        );
        assert!(
            (quanto.vega().unwrap() - (black.vega().unwrap() + corr * fxv * div_rho)).abs() < 1e-12,
            "quanto vega = inner vega + ρ σ_fx dividend rho"
        );
        let qrho: Real = quanto.result("qrho").unwrap();
        let qvega: Real = quanto.result("qvega").unwrap();
        let qlambda: Real = quanto.result("qlambda").unwrap();
        assert!((qrho + div_rho).abs() < 1e-12, "qrho = −dividend rho");
        assert!(
            (qvega - corr * vol * div_rho).abs() < 1e-12,
            "qvega = ρ σ_eq dividend rho"
        );
        assert!(
            (qlambda - fxv * vol * div_rho).abs() < 1e-12,
            "qlambda = σ_fx σ_eq dividend rho"
        );
    }
}

#[cfg(test)]
mod test_greeks {
    //! `quantooption.cpp` `testGreeks`: analytic quanto greeks vs central
    //! finite differences on moving curves (tol 1e-5 relative to spot).

    use super::QuantoEuropeanEngine;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{PlainVanillaPayoff, VanillaOption};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::vanilla::test_market::{quote_handle, today};
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{Shared, SharedMut, shared, shared_mut};
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

    struct MovingQuantoMarket {
        settings: Shared<Settings<Date>>,
        spot: Shared<SimpleQuote>,
        q_rate: Shared<SimpleQuote>,
        r_rate: Shared<SimpleQuote>,
        vol: Shared<SimpleQuote>,
        fx_rate: Shared<SimpleQuote>,
        fx_vol: Shared<SimpleQuote>,
        correlation: Shared<SimpleQuote>,
        process: Shared<BlackScholesMertonProcess>,
        fx_r_ts: Handle<dyn YieldTermStructure>,
        fx_vol_ts: Handle<dyn BlackVolTermStructure>,
        corr_h: Handle<dyn Quote>,
    }

    fn moving_quanto_market() -> MovingQuantoMarket {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let spot = shared(SimpleQuote::new(0.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.0));
        let vol = shared(SimpleQuote::new(0.0));
        let fx_rate = shared(SimpleQuote::new(0.0));
        let fx_vol = shared(SimpleQuote::new(0.0));
        let correlation = shared(SimpleQuote::new(0.0));
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
        let fx_r_ts = Handle::new(flat(&fx_rate));
        let fx_vol_ts = Handle::new(flat_vol(&fx_vol));
        let corr_h = quote_handle(&correlation);
        MovingQuantoMarket {
            settings,
            spot,
            q_rate,
            r_rate,
            vol,
            fx_rate,
            fx_vol,
            correlation,
            process,
            fx_r_ts,
            fx_vol_ts,
            corr_h,
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
    fn analytic_quanto_greeks_match_finite_differences() {
        let market = moving_quanto_market();
        let types = [OptionType::Call, OptionType::Put];
        let strikes = [50.0, 99.5, 100.0, 100.5, 150.0];
        let q_rates = [0.04, 0.05];
        let r_rates = [0.01, 0.05, 0.15];
        let lengths = [2];
        let vols = [0.11, 1.20];
        let correlations = [0.10, 0.90];
        let day_counter = Actual360::new();

        for option_type in types {
            for strike in strikes {
                for length in lengths {
                    let expiry = today() + Period::new(length, TimeUnit::Years);
                    let payoff = shared(PlainVanillaPayoff::new(option_type, strike));
                    let exercise = shared(EuropeanExercise::new(expiry));
                    let mut option =
                        VanillaOption::new(payoff, exercise, Shared::clone(&market.settings));
                    let engine = shared_mut(QuantoEuropeanEngine::new(
                        Shared::clone(&market.process),
                        market.fx_r_ts.clone(),
                        market.fx_vol_ts.clone(),
                        market.corr_h.clone(),
                    ));
                    option
                        .base_mut()
                        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

                    for u in [UNDERLYING] {
                        for q in q_rates {
                            for r in r_rates {
                                for v in vols {
                                    for fxr in r_rates {
                                        for fxv in vols {
                                            for corr in correlations {
                                                market.spot.set_value(u);
                                                market.q_rate.set_value(q);
                                                market.r_rate.set_value(r);
                                                market.vol.set_value(v);
                                                market.fx_rate.set_value(fxr);
                                                market.fx_vol.set_value(fxv);
                                                market.correlation.set_value(corr);

                                                let value = option.npv().unwrap();
                                                let delta = option.delta().unwrap();
                                                let gamma = option.gamma().unwrap();
                                                let theta = option.theta().unwrap();
                                                let rho = option.rho().unwrap();
                                                let div_rho = option.dividend_rho().unwrap();
                                                let vega = option.vega().unwrap();
                                                let qrho: Real = option.result("qrho").unwrap();
                                                let qvega: Real = option.result("qvega").unwrap();
                                                let qlambda: Real =
                                                    option.result("qlambda").unwrap();

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
                                                let expected_delta =
                                                    (value_p - value_m) / (2.0 * du);
                                                let expected_gamma =
                                                    (delta_p - delta_m) / (2.0 * du);

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
                                                let expected_div_rho =
                                                    (value_p - value_m) / (2.0 * dq);

                                                let dv = v * 1.0e-4;
                                                market.vol.set_value(v + dv);
                                                let value_p = option.npv().unwrap();
                                                market.vol.set_value(v - dv);
                                                let value_m = option.npv().unwrap();
                                                market.vol.set_value(v);
                                                let expected_vega =
                                                    (value_p - value_m) / (2.0 * dv);

                                                let dfxr = fxr * 1.0e-4;
                                                market.fx_rate.set_value(fxr + dfxr);
                                                let value_p = option.npv().unwrap();
                                                market.fx_rate.set_value(fxr - dfxr);
                                                let value_m = option.npv().unwrap();
                                                market.fx_rate.set_value(fxr);
                                                let expected_qrho =
                                                    (value_p - value_m) / (2.0 * dfxr);

                                                let dfxv = fxv * 1.0e-4;
                                                market.fx_vol.set_value(fxv + dfxv);
                                                let value_p = option.npv().unwrap();
                                                market.fx_vol.set_value(fxv - dfxv);
                                                let value_m = option.npv().unwrap();
                                                market.fx_vol.set_value(fxv);
                                                let expected_qvega =
                                                    (value_p - value_m) / (2.0 * dfxv);

                                                let dcorr = corr * 1.0e-4;
                                                market.correlation.set_value(corr + dcorr);
                                                let value_p = option.npv().unwrap();
                                                market.correlation.set_value(corr - dcorr);
                                                let value_m = option.npv().unwrap();
                                                market.correlation.set_value(corr);
                                                let expected_qlambda =
                                                    (value_p - value_m) / (2.0 * dcorr);

                                                let dt = day_counter
                                                    .year_fraction(today() - 1, today() + 1);
                                                market.settings.set_evaluation_date(today() - 1);
                                                let value_m = option.npv().unwrap();
                                                market.settings.set_evaluation_date(today() + 1);
                                                let value_p = option.npv().unwrap();
                                                market.settings.set_evaluation_date(today());
                                                let expected_theta = (value_p - value_m) / dt;

                                                let checks = [
                                                    ("delta", expected_delta, delta),
                                                    ("gamma", expected_gamma, gamma),
                                                    ("theta", expected_theta, theta),
                                                    ("rho", expected_rho, rho),
                                                    ("divRho", expected_div_rho, div_rho),
                                                    ("vega", expected_vega, vega),
                                                    ("qrho", expected_qrho, qrho),
                                                    ("qvega", expected_qvega, qvega),
                                                    ("qlambda", expected_qlambda, qlambda),
                                                ];
                                                for (name, expected, calculated) in checks {
                                                    let error =
                                                        relative_error(expected, calculated, u);
                                                    assert!(
                                                        error <= TOLERANCE,
                                                        "{name} of {option_type:?} K={strike} \
                                                         length={length} q={q} r={r} v={v} \
                                                         fxr={fxr} fxv={fxv} corr={corr}: \
                                                         analytic {calculated} vs FD {expected} \
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
        }
    }
}
