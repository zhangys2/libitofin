//! Ju quadratic (1999) American option approximation engine.
//!
//! Port of `ql/pricingengines/vanilla/juquadraticengine.{hpp,cpp}`:
//! N. Ju, "An Approximate Formula for Pricing American Options",
//! Journal of Derivatives, Winter 1999.

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
};
use crate::math::distributions::normal::{CumulativeNormalDistribution, NormalDistribution};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::pricingengines::blackformula::black_formula;
use crate::pricingengines::vanilla::BaroneAdesiWhaleyApproximationEngine;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::Shared;

/// Pricing engine for American options with Ju quadratic approximation.
///
/// Reference:
/// N. Ju, "An Approximate Formula for Pricing American Options",
/// Journal of Derivatives, Winter 1999.
pub struct JuQuadraticApproximationEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl JuQuadraticApproximationEngine {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self { base, process }
    }
}

impl AsObservable for JuQuadraticApproximationEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for JuQuadraticApproximationEngine {
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

        let last_date = exercise.last_date();
        let vol = self.process.black_volatility().current_link()?;
        let variance = vol.black_variance_date(last_date, payoff.strike(), false)?;
        let q_disc = self
            .process
            .dividend_yield()
            .current_link()?
            .discount_date(last_date, false)?;
        let rf_disc = self
            .process
            .risk_free_rate()
            .current_link()?
            .discount_date(last_date, false)?;
        let spot = self.process.state_variable().current_link()?.value()?;
        require!(spot > 0.0, "negative or null underlying given");

        let forward = spot * q_disc / rf_disc;
        let std_dev = variance.sqrt();
        let black = BlackCalculator::with_striked_payoff(&**payoff, forward, std_dev, rf_disc)?;

        if q_disc >= 1.0 && payoff.option_type() == OptionType::Call {
            // early exercise never optimal
            let rf = self.process.risk_free_rate().current_link()?;
            let qts = self.process.dividend_yield().current_link()?;
            let t_rf = rf
                .require_day_counter()?
                .year_fraction(rf.reference_date()?, last_date);
            let t_q = qts
                .require_day_counter()?
                .year_fraction(qts.reference_date()?, last_date);
            let t_vol = vol.time_from_reference(last_date)?;
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

        // early exercise can be optimal
        let cum_normal = CumulativeNormalDistribution::standard();
        let normal = NormalDistribution::standard();

        let tolerance = 1e-6;
        let sk = BaroneAdesiWhaleyApproximationEngine::critical_price(
            &**payoff, rf_disc, q_disc, variance, tolerance,
        )?;

        let forward_sk = sk * q_disc / rf_disc;
        let alpha = -2.0 * rf_disc.ln() / variance;
        let beta = 2.0 * (q_disc / rf_disc).ln() / variance;
        let h = 1.0 - rf_disc;
        let phi = match payoff.option_type() {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        };

        let temp_root = ((beta - 1.0) * (beta - 1.0) + (4.0 * alpha) / h).sqrt();
        let lambda = (-(beta - 1.0) + phi * temp_root) / 2.0;
        let lambda_prime = -phi * alpha / (h * h * temp_root);

        let black_sk = black_formula(
            payoff.option_type(),
            payoff.strike(),
            forward_sk,
            std_dev,
            rf_disc,
            0.0,
        )?;
        let h_a = phi * (sk - payoff.strike()) - black_sk;

        let d1_sk = ((forward_sk / payoff.strike()).ln() + 0.5 * variance) / std_dev;
        let d2_sk = d1_sk - std_dev;
        let part1 = forward_sk * normal.value(d1_sk) / (alpha * std_dev);
        let part2 = -phi * forward_sk * cum_normal.value(phi * d1_sk) * q_disc.ln() / rf_disc.ln();
        let part3 = phi * payoff.strike() * cum_normal.value(phi * d2_sk);
        let v_e_h = part1 + part2 + part3;

        let denom = 2.0 * (2.0 * lambda + beta - 1.0);
        let b = (1.0 - h) * alpha * lambda_prime / denom;
        let c = -((1.0 - h) * alpha / (2.0 * lambda + beta - 1.0))
            * (v_e_h / h_a + 1.0 / h + lambda_prime / (2.0 * lambda + beta - 1.0));
        let temp_spot_ratio = (spot / sk).ln();
        let chi = temp_spot_ratio * (b * temp_spot_ratio + c);

        let (value, delta, gamma) = if phi * (sk - spot) > 0.0 {
            let val = black.value() + h_a * (spot / sk).powf(lambda) / (1.0 - chi);
            let temp_chi_prime = (2.0 * b / spot) * (spot / sk).ln();
            let chi_prime = temp_chi_prime + c / spot;
            let chi_double_prime =
                2.0 * b / (spot * spot) - temp_chi_prime / spot - c / (spot * spot);
            let d1_s = ((forward / payoff.strike()).ln() + 0.5 * variance) / std_dev;

            let d = phi * q_disc * cum_normal.value(phi * d1_s)
                + (lambda / (spot * (1.0 - chi)) + chi_prime / ((1.0 - chi) * (1.0 - chi)))
                    * h_a
                    * (spot / sk).powf(lambda);

            let g = q_disc * normal.value(phi * d1_s) / (spot * std_dev)
                + (2.0 * lambda * chi_prime / (spot * (1.0 - chi) * (1.0 - chi))
                    + 2.0 * chi_prime * chi_prime / ((1.0 - chi) * (1.0 - chi) * (1.0 - chi))
                    + chi_double_prime / ((1.0 - chi) * (1.0 - chi))
                    + lambda * (lambda - 1.0) / (spot * spot * (1.0 - chi)))
                    * h_a
                    * (spot / sk).powf(lambda);

            (val, d, g)
        } else {
            (phi * (spot - payoff.strike()), phi, 0.0)
        };

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.greeks = Greeks {
            delta: Some(delta),
            gamma: Some(gamma),
            theta: None,
            vega: None,
            rho: None,
            dividend_rho: None,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::AmericanExercise;
    use crate::instrument::Instrument;
    use crate::instruments::{PlainVanillaPayoff, VanillaOption};
    use crate::option::OptionType::{Call, Put};
    use crate::pricingengines::vanilla::test_market::{market, time_to_days, today};
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::types::{Rate, Real, Volatility};
    type Row = (OptionType, Real, Real, Rate, Rate, Real, Volatility, Real);

    /// Complete 47-row table from QuantLib `americanoption.cpp` `testJuValues`.
    /// Data from N. Ju (1999), Exhibit 3 (short-dated puts) and Exhibit 6 (long-dated calls with dividends).
    #[rustfmt::skip]
    const JU_VALUES: &[Row] = &[
        // Exhibit 3 - Short dated Put Options
        (Put, 35.00, 40.00, 0.0, 0.0488, 0.0833, 0.2, 0.006),
        (Put, 35.00, 40.00, 0.0, 0.0488, 0.3333, 0.2, 0.201),
        (Put, 35.00, 40.00, 0.0, 0.0488, 0.5833, 0.2, 0.433),
        (Put, 40.00, 40.00, 0.0, 0.0488, 0.0833, 0.2, 0.851),
        (Put, 40.00, 40.00, 0.0, 0.0488, 0.3333, 0.2, 1.576),
        (Put, 40.00, 40.00, 0.0, 0.0488, 0.5833, 0.2, 1.984),
        (Put, 45.00, 40.00, 0.0, 0.0488, 0.0833, 0.2, 5.000),
        (Put, 45.00, 40.00, 0.0, 0.0488, 0.3333, 0.2, 5.084),
        (Put, 45.00, 40.00, 0.0, 0.0488, 0.5833, 0.2, 5.260),

        (Put, 35.00, 40.00, 0.0, 0.0488, 0.0833, 0.3, 0.078),
        (Put, 35.00, 40.00, 0.0, 0.0488, 0.3333, 0.3, 0.697),
        (Put, 35.00, 40.00, 0.0, 0.0488, 0.5833, 0.3, 1.218),
        (Put, 40.00, 40.00, 0.0, 0.0488, 0.0833, 0.3, 1.309),
        (Put, 40.00, 40.00, 0.0, 0.0488, 0.3333, 0.3, 2.477),
        (Put, 40.00, 40.00, 0.0, 0.0488, 0.5833, 0.3, 3.161),
        (Put, 45.00, 40.00, 0.0, 0.0488, 0.0833, 0.3, 5.059),
        (Put, 45.00, 40.00, 0.0, 0.0488, 0.3333, 0.3, 5.699),
        (Put, 45.00, 40.00, 0.0, 0.0488, 0.5833, 0.3, 6.231),

        (Put, 35.00, 40.00, 0.0, 0.0488, 0.0833, 0.4, 0.247),
        (Put, 35.00, 40.00, 0.0, 0.0488, 0.3333, 0.4, 1.344),
        (Put, 35.00, 40.00, 0.0, 0.0488, 0.5833, 0.4, 2.150),
        (Put, 40.00, 40.00, 0.0, 0.0488, 0.0833, 0.4, 1.767),
        (Put, 40.00, 40.00, 0.0, 0.0488, 0.3333, 0.4, 3.381),
        (Put, 40.00, 40.00, 0.0, 0.0488, 0.5833, 0.4, 4.342),
        (Put, 45.00, 40.00, 0.0, 0.0488, 0.0833, 0.4, 5.288),
        (Put, 45.00, 40.00, 0.0, 0.0488, 0.3333, 0.4, 6.501),
        (Put, 45.00, 40.00, 0.0, 0.0488, 0.5833, 0.4, 7.367),

        // Exhibit 6 - Long dated Call Options with dividends
        (Call, 100.00,  80.00, 0.07, 0.03,    3.0, 0.2,  2.605),
        (Call, 100.00,  90.00, 0.07, 0.03,    3.0, 0.2,  5.182),
        (Call, 100.00, 100.00, 0.07, 0.03,    3.0, 0.2,  9.065),
        (Call, 100.00, 110.00, 0.07, 0.03,    3.0, 0.2, 14.430),
        (Call, 100.00, 120.00, 0.07, 0.03,    3.0, 0.2, 21.398),

        (Call, 100.00,  80.00, 0.07, 0.03,    3.0, 0.4, 11.336),
        (Call, 100.00,  90.00, 0.07, 0.03,    3.0, 0.4, 15.711),
        (Call, 100.00, 100.00, 0.07, 0.03,    3.0, 0.4, 20.760),
        (Call, 100.00, 110.00, 0.07, 0.03,    3.0, 0.4, 26.440),
        (Call, 100.00, 120.00, 0.07, 0.03,    3.0, 0.4, 32.709),

        (Call, 100.00,  80.00, 0.07, 0.00001, 3.0, 0.3,  5.552),
        (Call, 100.00,  90.00, 0.07, 0.00001, 3.0, 0.3,  8.868),
        (Call, 100.00, 100.00, 0.07, 0.00001, 3.0, 0.3, 13.158),
        (Call, 100.00, 110.00, 0.07, 0.00001, 3.0, 0.3, 18.458),
        (Call, 100.00, 120.00, 0.07, 0.00001, 3.0, 0.3, 24.786),

        (Call, 100.00,  80.00, 0.03, 0.07,    3.0, 0.3, 12.177),
        (Call, 100.00,  90.00, 0.03, 0.07,    3.0, 0.3, 17.411),
        (Call, 100.00, 100.00, 0.03, 0.07,    3.0, 0.3, 23.402),
        (Call, 100.00, 110.00, 0.03, 0.07,    3.0, 0.3, 30.028),
        (Call, 100.00, 120.00, 0.03, 0.07,    3.0, 0.3, 37.177),
    ];

    #[test]
    fn test_ju_values() {
        let m = market();
        let tolerance = 1.0e-3;

        for &(option_type, strike, spot, q, r, t, vol, expected) in JU_VALUES {
            m.set(spot, q, r, vol);
            let ex_date = today() + time_to_days(t);
            let exercise = shared(AmericanExercise::over(today(), ex_date).unwrap());
            let payoff = shared(PlainVanillaPayoff::new(option_type, strike));
            let mut option = VanillaOption::new(payoff, exercise, Shared::clone(&m.settings));
            option
                .base_mut()
                .set_pricing_engine(
                    shared_mut(JuQuadraticApproximationEngine::new(Shared::clone(
                        &m.process,
                    ))) as SharedMut<dyn PricingEngine>,
                );

            let calculated = option.npv().unwrap();
            let err = (calculated - expected).abs();
            assert!(
                err <= tolerance,
                "Ju value failed for {:?} K={} S={} q={} r={} t={}: calculated {} vs expected {} (err={}, tol={})",
                option_type,
                strike,
                spot,
                q,
                r,
                t,
                calculated,
                expected,
                err,
                tolerance
            );
        }
    }

    #[test]
    fn test_ju_greeks() {
        let m = market();
        // Test an ATM put and call option for delta & gamma
        m.set(100.0, 0.05, 0.05, 0.25);
        let ex_date = today() + time_to_days(1.0);

        for opt_type in [Call, Put] {
            let exercise = shared(AmericanExercise::over(today(), ex_date).unwrap());
            let payoff = shared(PlainVanillaPayoff::new(opt_type, 100.0));
            let mut option = VanillaOption::new(payoff, exercise, Shared::clone(&m.settings));
            option
                .base_mut()
                .set_pricing_engine(
                    shared_mut(JuQuadraticApproximationEngine::new(Shared::clone(
                        &m.process,
                    ))) as SharedMut<dyn PricingEngine>,
                );

            let delta = option.delta().unwrap();
            let gamma = option.gamma().unwrap();

            // Check against central finite difference
            let eps = 1e-4;
            m.spot.set_value(100.0 + eps);
            let p_up = option.npv().unwrap();
            m.spot.set_value(100.0 - eps);
            let p_down = option.npv().unwrap();
            m.spot.set_value(100.0);
            let npv = option.npv().unwrap();

            let num_delta = (p_up - p_down) / (2.0 * eps);
            let num_gamma = (p_up - 2.0 * npv + p_down) / (eps * eps);

            assert!(
                (delta - num_delta).abs() <= 1e-3,
                "Delta error for {:?}: {} vs {}",
                opt_type,
                delta,
                num_delta
            );
            assert!(
                (gamma - num_gamma).abs() <= 1e-3,
                "Gamma error for {:?}: {} vs {}",
                opt_type,
                gamma,
                num_gamma
            );
        }
    }

    #[test]
    fn test_zero_dividend_call_matches_european() {
        let m = market();
        m.set(100.0, 0.0, 0.10, 0.15);
        let ex_date = today() + time_to_days(0.5);

        let mut euro = m.option(Call, 100.0, ex_date);

        let payoff = shared(PlainVanillaPayoff::new(Call, 100.0));
        let exercise = shared(AmericanExercise::over(today(), ex_date).unwrap());
        let mut am = VanillaOption::new(payoff, exercise, Shared::clone(&m.settings));
        am.base_mut()
            .set_pricing_engine(
                shared_mut(JuQuadraticApproximationEngine::new(Shared::clone(
                    &m.process,
                ))) as SharedMut<dyn PricingEngine>,
            );

        for (a, e) in [
            (am.npv(), euro.npv()),
            (am.delta(), euro.delta()),
            (am.gamma(), euro.gamma()),
            (am.theta(), euro.theta()),
            (am.vega(), euro.vega()),
            (am.rho(), euro.rho()),
            (am.dividend_rho(), euro.dividend_rho()),
            (am.strike_sensitivity(), euro.strike_sensitivity()),
        ] {
            assert!((a.unwrap() - e.unwrap()).abs() <= 1e-12);
        }
    }

    #[test]
    fn test_negative_rates_are_rejected() {
        let m = market();
        m.set(36.0, 0.0, -0.012, 0.20);
        let ex_date = today() + time_to_days(1.0);
        let payoff = shared(PlainVanillaPayoff::new(Put, 40.0));
        let exercise = shared(AmericanExercise::over(today(), ex_date).unwrap());
        let mut put = VanillaOption::new(payoff, exercise, Shared::clone(&m.settings));
        put.base_mut()
            .set_pricing_engine(
                shared_mut(JuQuadraticApproximationEngine::new(Shared::clone(
                    &m.process,
                ))) as SharedMut<dyn PricingEngine>,
            );
        let err = put.npv().unwrap_err();
        assert!(
            err.message().contains("negative interest rates"),
            "unexpected error message: {}",
            err.message()
        );
    }
}
