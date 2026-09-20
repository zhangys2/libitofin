//! Barone-Adesi–Whaley American approximation (`baroneadesiwhaleyengine`).

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
    StrikedTypePayoff,
};
use crate::math::comparison::close_n;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::{BlackCalculator, black_formula};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::Shared;
use crate::types::Real;

/// American vanilla engine using the Barone-Adesi and Whaley (1987) formula.
pub struct BaroneAdesiWhaleyApproximationEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl BaroneAdesiWhaleyApproximationEngine {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self { base, process }
    }

    /// Critical commodity price `Si` (Newton–Raphson; QL tolerance 1e-6).
    pub fn critical_price(
        payoff: &dyn StrikedTypePayoff,
        rf_disc: Real,
        q_disc: Real,
        variance: Real,
        tolerance: Real,
    ) -> QlResult<Real> {
        require!(
            rf_disc <= 1.0,
            "the Barone-Adesi-Whaley approximation is not applicable \
             with negative interest rates (risk-free discount factor: {rf_disc})"
        );
        let strike = payoff.strike();
        let n = 2.0 * (q_disc / rf_disc).ln() / variance;
        let m = -2.0 * rf_disc.ln() / variance;
        let b_t = (q_disc / rf_disc).ln();
        let std_dev = variance.sqrt();
        let n1 = n - 1.0;
        let (qu, h_num, call) = match payoff.option_type() {
            OptionType::Call => (
                (-n1 + (n1 * n1 + 4.0 * m).sqrt()) / 2.0,
                -(b_t + 2.0 * std_dev),
                true,
            ),
            OptionType::Put => (
                (-n1 - (n1 * n1 + 4.0 * m).sqrt()) / 2.0,
                b_t - 2.0 * std_dev,
                false,
            ),
        };
        let su = strike / (1.0 - 1.0 / qu);
        let mut si = if call {
            let h = h_num * strike / (su - strike);
            strike + (su - strike) * (1.0 - h.exp())
        } else {
            let h = h_num * strike / (strike - su);
            su + (strike - su) * h.exp()
        };
        let n_cdf = CumulativeNormalDistribution::standard();
        let k = if !close_n(rf_disc, 1.0, 1000) {
            -2.0 * rf_disc.ln() / (variance * (1.0 - rf_disc))
        } else {
            2.0 / variance
        };
        let q = if call {
            (-n1 + (n1 * n1 + 4.0 * k).sqrt()) / 2.0
        } else {
            (-n1 - (n1 * n1 + 4.0 * k).sqrt()) / 2.0
        };
        loop {
            let fwd = si * q_disc / rf_disc;
            let d1 = (fwd / strike).ln() / std_dev + 0.5 * std_dev;
            let temp = black_formula(payoff.option_type(), strike, fwd, std_dev, rf_disc, 0.0)?;
            let (lhs, rhs, bi) = if call {
                let nd1 = n_cdf.value(d1);
                (
                    si - strike,
                    temp + (1.0 - q_disc * nd1) * si / q,
                    q_disc * nd1 * (1.0 - 1.0 / q)
                        + (1.0 - q_disc * n_cdf.derivative(d1) / std_dev) / q,
                )
            } else {
                let nd1 = n_cdf.value(-d1);
                (
                    strike - si,
                    temp - (1.0 - q_disc * nd1) * si / q,
                    -q_disc * nd1 * (1.0 - 1.0 / q)
                        - (1.0 + q_disc * n_cdf.derivative(-d1) / std_dev) / q,
                )
            };
            if (lhs - rhs).abs() / strike <= tolerance {
                return Ok(si);
            }
            si = if call {
                (strike + rhs - bi * si) / (1.0 - bi)
            } else {
                (strike - rhs + bi * si) / (1.0 + bi)
            };
        }
    }
}

impl AsObservable for BaroneAdesiWhaleyApproximationEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BaroneAdesiWhaleyApproximationEngine {
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
        let Some(exercise) = args.exercise.as_ref() else {
            fail!("no exercise given");
        };
        require!(
            exercise.exercise_type() == ExerciseType::American,
            "not an American Option"
        );
        require!(!exercise.payoff_at_expiry(), "payoff at expiry not handled");
        let Some(payoff) = args.payoff.as_ref() else {
            fail!("non-striked payoff given");
        };
        let last = exercise.last_date();
        let vol = self.process.black_volatility().current_link()?;
        let variance = vol.black_variance_date(last, payoff.strike(), false)?;
        let q_disc = self
            .process
            .dividend_yield()
            .current_link()?
            .discount_date(last, false)?;
        let rf_disc = self
            .process
            .risk_free_rate()
            .current_link()?
            .discount_date(last, false)?;
        let spot = self.process.state_variable().current_link()?.value()?;
        require!(spot > 0.0, "negative or null underlying given");
        let forward = spot * q_disc / rf_disc;
        let std_dev = variance.sqrt();
        let black = BlackCalculator::with_striked_payoff(&**payoff, forward, std_dev, rf_disc)?;
        if q_disc >= 1.0 && payoff.option_type() == OptionType::Call {
            let rf = self.process.risk_free_rate().current_link()?;
            let qts = self.process.dividend_yield().current_link()?;
            let t_rf = rf
                .require_day_counter()?
                .year_fraction(rf.reference_date()?, last);
            let t_q = qts
                .require_day_counter()?
                .year_fraction(qts.reference_date()?, last);
            let t_vol = vol.time_from_reference(last)?;
            let theta = black.theta(spot, t_vol).ok();
            let results = self.base.results_mut();
            results.instrument.value = Some(black.value());
            results.greeks = Greeks {
                delta: Some(black.delta(spot)?),
                gamma: Some(black.gamma(spot)?),
                theta,
                vega: Some(black.vega(t_vol)?),
                rho: Some(black.rho(t_rf)?),
                dividend_rho: Some(black.dividend_rho(t_q)?),
            };
            results.more_greeks = MoreGreeks {
                itm_cash_probability: Some(black.itm_cash_probability()),
                delta_forward: Some(black.delta_forward()),
                elasticity: Some(black.elasticity(spot)?),
                theta_per_day: theta.map(|th| th / 365.0),
                strike_sensitivity: Some(black.strike_sensitivity()),
            };
            return Ok(());
        }
        let sk = Self::critical_price(&**payoff, rf_disc, q_disc, variance, 1e-6)?;
        let fwd_sk = sk * q_disc / rf_disc;
        let d1 = (fwd_sk / payoff.strike()).ln() / std_dev + 0.5 * std_dev;
        let n = 2.0 * (q_disc / rf_disc).ln() / variance;
        let n1 = n - 1.0;
        let k = if !close_n(rf_disc, 1.0, 1000) {
            -2.0 * rf_disc.ln() / (variance * (1.0 - rf_disc))
        } else {
            2.0 / variance
        };
        let n_cdf = CumulativeNormalDistribution::standard();
        let value = match payoff.option_type() {
            OptionType::Call => {
                let q = (-n1 + (n1 * n1 + 4.0 * k).sqrt()) / 2.0;
                let a = (sk / q) * (1.0 - q_disc * n_cdf.value(d1));
                if spot < sk {
                    black.value() + a * (spot / sk).powf(q)
                } else {
                    spot - payoff.strike()
                }
            }
            OptionType::Put => {
                let q = (-n1 - (n1 * n1 + 4.0 * k).sqrt()) / 2.0;
                let a = -(sk / q) * (1.0 - q_disc * n_cdf.value(-d1));
                if spot > sk {
                    black.value() + a * (spot / sk).powf(q)
                } else {
                    payoff.strike() - spot
                }
            }
        };
        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::AmericanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{PlainVanillaPayoff, VanillaOption};
    use crate::interestrate::Compounding;
    use crate::option::OptionType::{Call, Put};
    use crate::pricingengine::PricingEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }
    fn process(s: Real, q: Real, r: Real, v: Real) -> Shared<BlackScholesMertonProcess> {
        let yts = |rate: Real| {
            Handle::new(shared(FlatForward::with_rate(
                today(),
                rate,
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(s)) as Shared<dyn Quote>),
            yts(q),
            yts(r),
            Handle::new(
                shared(BlackConstantVol::new(today(), None, v, Actual360::new()))
                    as Shared<dyn BlackVolTermStructure>,
            ),
        ))
    }

    /// Haug p.24 as in `americanoption.cpp` `testBaroneAdesiWhaleyValues`.
    type Row = (OptionType, Real, Real, Real, Real, Real, Real, Real);
    #[rustfmt::skip]
    const ROWS: &[Row] = &[
        (Call, 100.00,  90.00, 0.10, 0.10, 0.10, 0.15,  0.0206),
        (Call, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8771),
        (Call, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 10.0089),
        (Call, 100.00,  90.00, 0.10, 0.10, 0.10, 0.25,  0.3159),
        (Call, 100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1280),
        (Call, 100.00, 110.00, 0.10, 0.10, 0.10, 0.25, 10.3919),
        (Call, 100.00,  90.00, 0.10, 0.10, 0.10, 0.35,  0.9495),
        (Call, 100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3777),
        (Call, 100.00, 110.00, 0.10, 0.10, 0.10, 0.35, 11.1679),
        (Call, 100.00,  90.00, 0.10, 0.10, 0.50, 0.15,  0.8208),
        (Call, 100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0842),
        (Call, 100.00, 110.00, 0.10, 0.10, 0.50, 0.15, 10.8087),
        (Call, 100.00,  90.00, 0.10, 0.10, 0.50, 0.25,  2.7437),
        (Call, 100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.8015),
        (Call, 100.00, 110.00, 0.10, 0.10, 0.50, 0.25, 13.0170),
        (Call, 100.00,  90.00, 0.10, 0.10, 0.50, 0.35,  5.0063),
        (Call, 100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.5106),
        (Call, 100.00, 110.00, 0.10, 0.10, 0.50, 0.35, 15.5689),
        (Put,  100.00,  90.00, 0.10, 0.10, 0.10, 0.15, 10.0000),
        (Put,  100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8770),
        (Put,  100.00, 110.00, 0.10, 0.10, 0.10, 0.15,  0.0410),
        (Put,  100.00,  90.00, 0.10, 0.10, 0.10, 0.25, 10.2533),
        (Put,  100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1277),
        (Put,  100.00, 110.00, 0.10, 0.10, 0.10, 0.25,  0.4562),
        (Put,  100.00,  90.00, 0.10, 0.10, 0.10, 0.35, 10.8787),
        (Put,  100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3777),
        (Put,  100.00, 110.00, 0.10, 0.10, 0.10, 0.35,  1.2402),
        (Put,  100.00,  90.00, 0.10, 0.10, 0.50, 0.15, 10.5595),
        (Put,  100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0842),
        (Put,  100.00, 110.00, 0.10, 0.10, 0.50, 0.15,  1.0822),
        (Put,  100.00,  90.00, 0.10, 0.10, 0.50, 0.25, 12.4419),
        (Put,  100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.8014),
        (Put,  100.00, 110.00, 0.10, 0.10, 0.50, 0.25,  3.3226),
        (Put,  100.00,  90.00, 0.10, 0.10, 0.50, 0.35, 14.6945),
        (Put,  100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.5104),
        (Put,  100.00, 110.00, 0.10, 0.10, 0.50, 0.35,  5.8823),
        (Put,  100.00, 100.00, 0.00, 0.00, 0.50, 0.15,  4.2294),
    ];

    #[test]
    fn haug_barone_adesi_whaley_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        for &(ty, k, s, q, r, t, v, expected) in ROWS {
            let process = process(s, q, r, v);
            let expiry = today() + (t * 360.0).round() as i32;
            let mut option = VanillaOption::new(
                shared(PlainVanillaPayoff::new(ty, k)),
                shared(AmericanExercise::over(today(), expiry).unwrap()),
                Shared::clone(&settings),
            );
            option.base_mut().set_pricing_engine(shared_mut(
                BaroneAdesiWhaleyApproximationEngine::new(process),
            ) as SharedMut<dyn PricingEngine>);
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - expected).abs() <= 3.0e-3,
                "{ty:?} K={k} S={s} q={q} r={r} t={t} v={v}: {calculated} vs Haug {expected}"
            );
        }
    }

    #[test]
    fn negative_rates_are_rejected() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let process = process(36.0, 0.0, -0.012, 0.20);
        let mut put = VanillaOption::new(
            shared(PlainVanillaPayoff::new(Put, 40.0)),
            shared(AmericanExercise::over(today(), today() + 360).unwrap()),
            Shared::clone(&settings),
        );
        put.base_mut()
            .set_pricing_engine(shared_mut(BaroneAdesiWhaleyApproximationEngine::new(
                Shared::clone(&process),
            )) as SharedMut<dyn PricingEngine>);
        assert!(
            put.npv()
                .unwrap_err()
                .message()
                .contains("negative interest rates")
        );
    }

    #[test]
    fn zero_dividend_call_matches_european() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let process = process(100.0, 0.0, 0.10, 0.15);
        let mut am = VanillaOption::new(
            shared(PlainVanillaPayoff::new(Call, 100.0)),
            shared(AmericanExercise::over(today(), today() + 180).unwrap()),
            settings,
        );
        am.base_mut()
            .set_pricing_engine(
                shared_mut(BaroneAdesiWhaleyApproximationEngine::new(process))
                    as SharedMut<dyn PricingEngine>,
            );
        let df = (-0.05_f64).exp();
        let black = BlackCalculator::with_payoff(
            &PlainVanillaPayoff::new(Call, 100.0),
            100.0 / df,
            0.15 * 0.5_f64.sqrt(),
            df,
        )
        .unwrap();
        assert!((am.npv().unwrap() - black.value()).abs() <= 1.0e-12);
        assert!((am.delta().unwrap() - black.delta(100.0).unwrap()).abs() <= 1.0e-12);
    }
}
