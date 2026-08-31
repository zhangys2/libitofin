//! Analytic quanto forward vanilla engine.
//!
//! Port of `ql/pricingengines/quanto/quantoengine.hpp` specialised to
//! `QuantoEngine<ForwardVanillaOption, ForwardVanillaEngine<AnalyticEuropeanEngine>>`:
//! the inner forward engine is run on a process whose dividend curve is a
//! [`QuantoTermStructure`](crate::termstructures::yields::QuantoTermStructure).

use std::any::Any;

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{ForwardOptionArguments, OneAssetOptionResults};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::forward::AnalyticForwardVanillaEngine;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::shared::{Shared, shared};
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yields::QuantoTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::Real;

type ForwardEngineBase = GenericEngine<ForwardOptionArguments, OneAssetOptionResults>;

/// `QuantoEngine<ForwardVanillaOption, ForwardVanillaEngine<AnalyticEuropeanEngine>>`.
pub struct QuantoForwardEuropeanEngine {
    base: ForwardEngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
}

impl QuantoForwardEuropeanEngine {
    /// `QuantoEngine(process, foreignRiskFreeRate, exchangeRateVolatility, correlation)`.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
        exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
        correlation: Handle<dyn Quote>,
    ) -> Self {
        let base = ForwardEngineBase::new(
            ForwardOptionArguments::default(),
            OneAssetOptionResults::default(),
        );
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

impl AsObservable for QuantoForwardEuropeanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for QuantoForwardEuropeanEngine {
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
        let (strike, maturity) = {
            let args = self.base.arguments();
            let Some(payoff) = args.payoff.as_ref() else {
                fail!("no payoff given");
            };
            let Some(exercise) = args.exercise.as_ref() else {
                fail!("no exercise given");
            };
            (payoff.strike(), exercise.last_date())
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
            let mut original = AnalyticForwardVanillaEngine::new(quanto_process);
            let original_results = original.calculate_from_arguments(self.base.arguments())?;
            (
                original_results.instrument.value,
                original_results.greeks,
                original_results.more_greeks,
            )
        };

        let fx_flat_vol = self
            .exchange_rate_volatility
            .current_link()?
            .black_vol_date(maturity, EXCHANGE_RATE_ATM, false)?;
        let original_div_rho = greeks.dividend_rho;

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

/// Attach a [`QuantoForwardEuropeanEngine`] to a forward vanilla option.
pub fn set_quanto_forward_european_engine(
    option: &mut crate::instruments::ForwardVanillaOption,
    process: Shared<GeneralizedBlackScholesProcess>,
    foreign_risk_free_rate: Handle<dyn YieldTermStructure>,
    exchange_rate_volatility: Handle<dyn BlackVolTermStructure>,
    correlation: Handle<dyn Quote>,
) {
    use crate::shared::{SharedMut, shared_mut};
    let engine = shared_mut(QuantoForwardEuropeanEngine::new(
        process,
        foreign_risk_free_rate,
        exchange_rate_volatility,
        correlation,
    )) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::instrument::Instrument;
    use crate::instruments::{ForwardVanillaOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengines::forward::set_analytic_forward_vanilla_engine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
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

    /// One row of `quantooption.cpp` `testForwardValues` (tol 1e-4).
    struct ForwardRow {
        option_type: OptionType,
        moneyness: Real,
        spot: Real,
        q: Rate,
        r: Rate,
        start: Time,
        t: Time,
        vol: Volatility,
        fxr: Rate,
        fxv: Volatility,
        corr: Real,
        result: Real,
    }

    #[rustfmt::skip]
    const HAUG_QUANTO_FORWARD: &[ForwardRow] = &[
        // reset=0.0, quanto (not-forward) options — Haug
        ForwardRow {
            option_type: OptionType::Call, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, start: 0.0, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 5.3280 / 1.5,
        },
        ForwardRow {
            option_type: OptionType::Put, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, start: 0.0, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 8.1636,
        },
        // reset!=0.0, quanto-forward (FinCAD 7)
        ForwardRow {
            option_type: OptionType::Call, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, start: 0.25, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 2.0171,
        },
        ForwardRow {
            option_type: OptionType::Put, moneyness: 1.05, spot: 100.0,
            q: 0.04, r: 0.08, start: 0.25, t: 0.5, vol: 0.20,
            fxr: 0.05, fxv: 0.10, corr: 0.3, result: 6.7296,
        },
    ];

    #[test]
    fn haug_quanto_forward_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        const TOL: Real = 1.0e-4;

        for row in HAUG_QUANTO_FORWARD {
            let process = shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(row.spot)) as Shared<dyn Quote>),
                flat_rate(row.q),
                flat_rate(row.r),
                flat_vol(row.vol),
            ));
            let payoff = shared(PlainVanillaPayoff::new(row.option_type, 0.0))
                as Shared<dyn crate::instruments::StrikedTypePayoff>;
            let exercise = shared(EuropeanExercise::new(today() + time_to_days(row.t)));
            let reset = today() + time_to_days(row.start);
            let mut option = ForwardVanillaOption::new(
                row.moneyness,
                reset,
                payoff,
                exercise,
                Shared::clone(&settings),
            );
            set_quanto_forward_european_engine(
                &mut option,
                process,
                flat_rate(row.fxr),
                flat_vol(row.fxv),
                Handle::new(shared(SimpleQuote::new(row.corr)) as Shared<dyn Quote>),
            );
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - row.result).abs() <= TOL,
                "{:?} start={}: {calculated} vs {} (tol {TOL})",
                row.option_type,
                row.start,
                row.result
            );
        }
    }

    /// reset=0 quanto-forward NPV ≡ quanto European with K = moneyness × S.
    #[test]
    fn reset_zero_matches_quanto_european() {
        use crate::instruments::VanillaOption;
        use crate::pricingengines::vanilla::QuantoEuropeanEngine;
        use crate::shared::shared_mut;

        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let (s, m, q, r, t, vol, fxr, fxv, corr) =
            (100.0, 1.05, 0.04, 0.08, 0.5, 0.20, 0.05, 0.10, 0.3);
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(s)) as Shared<dyn Quote>),
            flat_rate(q),
            flat_rate(r),
            flat_vol(vol),
        ));
        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(today() + time_to_days(t)));

        let mut fwd = ForwardVanillaOption::new(
            m,
            today(),
            shared(PlainVanillaPayoff::new(OptionType::Call, 0.0))
                as Shared<dyn crate::instruments::StrikedTypePayoff>,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        set_quanto_forward_european_engine(
            &mut fwd,
            Shared::clone(&process),
            flat_rate(fxr),
            flat_vol(fxv),
            Handle::new(shared(SimpleQuote::new(corr)) as Shared<dyn Quote>),
        );

        let mut euro = VanillaOption::new(
            shared(PlainVanillaPayoff::new(OptionType::Call, m * s))
                as Shared<dyn crate::instruments::StrikedTypePayoff>,
            exercise,
            settings,
        );
        let engine = shared_mut(QuantoEuropeanEngine::new(
            process,
            flat_rate(fxr),
            flat_vol(fxv),
            Handle::new(shared(SimpleQuote::new(corr)) as Shared<dyn Quote>),
        )) as crate::shared::SharedMut<dyn PricingEngine>;
        euro.base_mut().set_pricing_engine(engine);

        let f = fwd.npv().unwrap();
        let e = euro.npv().unwrap();
        assert!(
            (f - e).abs() < 1e-12,
            "reset-0 forward quanto {f} vs european quanto {e}"
        );
    }

    /// Plain (non-quanto) forward with reset=0 ≡ AnalyticEuropeanEngine.
    #[test]
    fn plain_forward_reset_zero_matches_european() {
        use crate::instruments::VanillaOption;
        use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
        use crate::shared::shared_mut;

        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let (s, m, q, r, t, vol) = (100.0, 1.05, 0.04, 0.08, 0.5, 0.20);
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(s)) as Shared<dyn Quote>),
            flat_rate(q),
            flat_rate(r),
            flat_vol(vol),
        ));
        let exercise: Shared<dyn crate::exercise::Exercise> =
            shared(EuropeanExercise::new(today() + time_to_days(t)));

        let mut fwd = ForwardVanillaOption::new(
            m,
            today(),
            shared(PlainVanillaPayoff::new(OptionType::Call, 0.0))
                as Shared<dyn crate::instruments::StrikedTypePayoff>,
            Shared::clone(&exercise),
            Shared::clone(&settings),
        );
        set_analytic_forward_vanilla_engine(&mut fwd, Shared::clone(&process));

        let mut euro = VanillaOption::new(
            shared(PlainVanillaPayoff::new(OptionType::Call, m * s))
                as Shared<dyn crate::instruments::StrikedTypePayoff>,
            exercise,
            settings,
        );
        let engine = shared_mut(AnalyticEuropeanEngine::new(process))
            as crate::shared::SharedMut<dyn PricingEngine>;
        euro.base_mut().set_pricing_engine(engine);

        let f = fwd.npv().unwrap();
        let e = euro.npv().unwrap();
        assert!((f - e).abs() < 1e-12, "reset-0 forward {f} vs european {e}");
    }
}

#[cfg(test)]
mod test_greeks {
    //! `quantooption.cpp` `testForwardGreeks`: analytic quanto-forward greeks vs
    //! central finite differences on moving curves (tol 1e-5 relative to spot).

    use super::QuantoForwardEuropeanEngine;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{ForwardVanillaOption, PlainVanillaPayoff};
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
    fn analytic_quanto_forward_greeks_match_finite_differences() {
        let market = moving_quanto_market();
        let types = [OptionType::Call, OptionType::Put];
        let moneyness = [0.9, 1.0, 1.1];
        let q_rates = [0.04, 0.05];
        let r_rates = [0.01, 0.05, 0.15];
        let lengths = [2];
        let start_months = [6, 9];
        let vols = [0.11, 1.20];
        let correlations = [0.10, 0.90];
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
                        let engine = shared_mut(QuantoForwardEuropeanEngine::new(
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
                                                    let qvega: Real =
                                                        option.result("qvega").unwrap();
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
                                                    let expected_rho =
                                                        (value_p - value_m) / (2.0 * dr);

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
                                                    market
                                                        .settings
                                                        .set_evaluation_date(today() - 1);
                                                    let value_m = option.npv().unwrap();
                                                    market
                                                        .settings
                                                        .set_evaluation_date(today() + 1);
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
                                                            "{name} of {option_type:?} \
                                                             m={m} reset={start_month}mo \
                                                             length={length}y q={q} r={r} v={v} \
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
}
