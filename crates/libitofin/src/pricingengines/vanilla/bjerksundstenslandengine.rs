//! Bjerksund and Stensland American option approximation (`bjerksundstenslandengine`).
//!
//! Port of `ql/pricingengines/vanilla/bjerksundstenslandengine.{hpp,cpp}`:
//! Bjerksund and Stensland (1993) closed-form approximation for American
//! options, with put-call symmetry for puts, early exercise boundary checks,
//! and analytic Greeks.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
    PlainVanillaPayoff,
};
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::math::errorfunction::erfc;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::BlackCalculator;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, shared};
use crate::time::date::Date;
use crate::types::Real;

fn squared(x: Real) -> Real {
    x * x
}

const M_SQRT2: Real = std::f64::consts::SQRT_2;
const M_SQRTPI: Real = 1.77245385090551602729816748334;
const M_PI: Real = std::f64::consts::PI;

#[allow(clippy::too_many_arguments)]
fn phi(s: Real, gamma: Real, h: Real, i: Real, r_t: Real, b_t: Real, variance: Real) -> Real {
    let cum_normal_dist = CumulativeNormalDistribution::standard();
    let lambda = -r_t + gamma * b_t + 0.5 * gamma * (gamma - 1.0) * variance;
    let d = -((s / h).ln() + (b_t + (gamma - 0.5) * variance)) / variance.sqrt();
    let kappa = 2.0 * b_t / variance + (2.0 * gamma - 1.0);
    lambda.exp()
        * (cum_normal_dist.value(d)
            - (i / s).powf(kappa) * cum_normal_dist.value(d - 2.0 * (i / s).ln() / variance.sqrt()))
}

fn phi_s(s: Real, gamma: Real, h: Real, i: Real, r_t: Real, b_t: Real, v: Real) -> Real {
    let lsh = (s / h).ln();
    let lis = (i / s).ln();
    let sv = v.sqrt();

    (b_t * gamma - r_t + ((-1.0 + gamma) * gamma * v) / 2.0).exp()
        * ((-((i / s).powf(2.0 * (gamma + b_t / v))
            / ((squared(2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh) / (8.0 * v))
                .exp()
                * i))
            - 1.0 / ((squared(2.0 * b_t - v + 2.0 * gamma * v + 2.0 * lsh) / (8.0 * v)).exp() * s))
            / (M_SQRT2 * M_SQRTPI * sv)
            + ((i / s).powf(2.0 * (gamma + b_t / v))
                * (2.0 * b_t + (-1.0 + 2.0 * gamma) * v)
                * erfc(
                    (2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh)
                        / (2.0 * M_SQRT2 * sv),
                ))
                / (2.0 * i * v))
}

fn phi_ss(s: Real, gamma: Real, h: Real, i: Real, r_t: Real, b_t: Real, v: Real) -> Real {
    let lsh = (s / h).ln();
    let lis = (i / s).ln();
    let sv = v.sqrt();
    let ex = (squared(2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh) / (8.0 * v)).exp();
    let ey = (squared(2.0 * b_t + (-1.0 + 2.0 * gamma) * v + 2.0 * lsh) / (8.0 * v)).exp();

    ((b_t * gamma - r_t + ((-1.0 + gamma) * gamma * v) / 2.0).exp()
        * ((M_SQRT2 * i * v * sv) / ey
            + (2.0
                * M_SQRT2
                * (i / s).powf(2.0 * (gamma + b_t / v))
                * s
                * sv
                * (2.0 * b_t + (-1.0 + 2.0 * gamma) * v))
                / ex
            - 2.0
                * M_PI.sqrt()
                * (i / s).powf(2.0 * (gamma + b_t / v))
                * s
                * (b_t + gamma * v)
                * (2.0 * b_t + (-1.0 + 2.0 * gamma) * v)
                * erfc(
                    (2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh)
                        / (2.0 * M_SQRT2 * sv),
                )
            + (M_SQRT2 * i * sv * (b_t + (-0.5 + gamma) * v + lsh)) / ey
            - ((i / s).powf(2.0 * (gamma + b_t / v))
                * s
                * sv
                * (2.0 * b_t - 3.0 * v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh))
                / (M_SQRT2 * ex)))
        / (2.0 * i * M_SQRTPI * squared(s * v))
}

fn phi_gamma(s: Real, gamma: Real, h: Real, i: Real, r_t: Real, b_t: Real, v: Real) -> Real {
    let lsh = (s / h).ln();
    let lis = (i / s).ln();
    let sv = v.sqrt();

    (b_t * gamma - r_t + ((-1.0 + gamma) * gamma * v) / 2.0).exp()
        * (((-(-(squared(2.0 * b_t - v + 2.0 * gamma * v + 2.0 * lsh) / (8.0 * v))).exp()
            + (i / s).powf(-1.0 + 2.0 * gamma + (2.0 * b_t) / v)
                / (squared(2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh) / (8.0 * v))
                    .exp())
            * sv)
            / (M_SQRT2 * M_SQRTPI)
            + ((2.0 * b_t + (-1.0 + 2.0 * gamma) * v)
                * erfc((2.0 * b_t + (-1.0 + 2.0 * gamma) * v + 2.0 * lsh) / (2.0 * M_SQRT2 * sv)))
                / 4.0
            - ((i / s).powf(-1.0 + 2.0 * gamma + (2.0 * b_t) / v)
                * erfc(
                    (2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh)
                        / (2.0 * M_SQRT2 * sv),
                )
                * (2.0 * b_t + (-1.0 + 2.0 * gamma) * v + 4.0 * lis))
                / 4.0)
}

fn phi_h(s: Real, gamma: Real, h: Real, i: Real, r_t: Real, b_t: Real, v: Real) -> Real {
    let lsh = (s / h).ln();
    let lis = (i / s).ln();
    let sv = v.sqrt();

    ((b_t * gamma - r_t + ((-1.0 + gamma) * gamma * v) / 2.0).exp()
        * (i / (squared(2.0 * b_t - v + 2.0 * gamma * v + 2.0 * lsh) / (8.0 * v)).exp()
            - ((i / s).powf(2.0 * (gamma + b_t / v)) * s)
                / (squared(2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh) / (8.0 * v))
                    .exp()))
        / (h * i * (2.0 * M_PI).sqrt() * sv)
}

fn phi_i(s: Real, gamma: Real, _h: Real, i: Real, r_t: Real, b_t: Real, v: Real) -> Real {
    let lis = (i / s).ln();
    let lsh = (s / _h).ln();
    let sv = v.sqrt();

    ((b_t * gamma - r_t + ((-1.0 + gamma) * gamma * v) / 2.0).exp()
        * (i / s).powf(2.0 * (gamma + b_t / v))
        * s
        * ((2.0 * (2.0 / M_PI).sqrt())
            / ((squared(2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh) / (8.0 * v))
                .exp()
                * sv)
            + (1.0 - 2.0 * gamma - (2.0 * b_t) / v)
                * erfc(
                    (2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh)
                        / (2.0 * M_SQRT2 * sv),
                )))
        / (2.0 * i * i)
}

fn phi_rt(s: Real, gamma: Real, h: Real, i: Real, r_t: Real, b_t: Real, v: Real) -> Real {
    let lsh = (s / h).ln();
    let lis = (i / s).ln();
    let sv = v.sqrt();

    ((b_t * gamma - r_t + ((-1.0 + gamma) * gamma * v) / 2.0).exp()
        * (-(i * erfc((2.0 * b_t - v + 2.0 * gamma * v + 2.0 * lsh) / (2.0 * M_SQRT2 * sv)))
            + (i / s).powf(2.0 * (gamma + b_t / v))
                * s
                * erfc(
                    (2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh)
                        / (2.0 * M_SQRT2 * sv),
                )))
        / (2.0 * i)
}

fn phi_bt(s: Real, gamma: Real, h: Real, i: Real, r_t: Real, b_t: Real, v: Real) -> Real {
    let lsh = (s / h).ln();
    let lis = (i / s).ln();
    let sv = v.sqrt();

    ((b_t * gamma - r_t + ((-1.0 + gamma) * gamma * v) / 2.0).exp()
        * (M_SQRT2
            * (-(i / (squared(2.0 * b_t - v + 2.0 * gamma * v + 2.0 * lsh) / (8.0 * v)).exp())
                + ((i / s).powf(2.0 * (gamma + b_t / v)) * s)
                    / (squared(2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh)
                        / (8.0 * v))
                        .exp())
            * sv
            + gamma
                * i
                * M_PI.sqrt()
                * v
                * erfc((2.0 * b_t - v + 2.0 * gamma * v + 2.0 * lsh) / (2.0 * M_SQRT2 * sv))
            - M_SQRTPI
                * (i / s).powf(2.0 * (gamma + b_t / v))
                * s
                * erfc(
                    (2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh)
                        / (2.0 * M_SQRT2 * sv),
                )
                * (gamma * v + 2.0 * lis)))
        / (2.0 * i * M_SQRTPI * v)
}

fn phi_v(s: Real, gamma: Real, h: Real, i: Real, r_t: Real, b_t: Real, v: Real) -> Real {
    let lsh = (s / h).ln();
    let lis = (i / s).ln();
    let sv = v.sqrt();
    let er = erfc((2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh) / (2.0 * M_SQRT2 * sv));

    ((b_t * gamma - r_t + ((-1.0 + gamma) * gamma * v) / 2.0).exp()
        * ((((-1.0 + gamma)
            * gamma
            * (i * erfc((2.0 * b_t - v + 2.0 * gamma * v + 2.0 * lsh) / (2.0 * M_SQRT2 * sv))
                - (i / s).powf(2.0 * (gamma + b_t / v)) * s * er))
            / (2.0 * i))
            + (2.0 * b_t * (i / s).powf(-1.0 + 2.0 * gamma + (2.0 * b_t) / v) * er * lis)
                / (v * v)
            + (2.0 * b_t + v - 2.0 * gamma * v + 2.0 * lsh)
                / (2.0
                    * (squared(2.0 * b_t + (-1.0 + 2.0 * gamma) * v + 2.0 * lsh) / (8.0 * v))
                        .exp()
                    * M_SQRT2
                    * M_SQRTPI
                    * v
                    * sv)
            - ((i / s).powf(-1.0 + 2.0 * gamma + (2.0 * b_t) / v)
                * (2.0 * b_t + v - 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh))
                / (2.0
                    * (squared(2.0 * b_t - v + 2.0 * gamma * v + 4.0 * lis + 2.0 * lsh)
                        / (8.0 * v))
                        .exp()
                    * M_SQRT2
                    * M_SQRTPI
                    * v
                    * sv)))
        / 2.0
}

#[derive(Clone, Debug, Default)]
struct BjerksundResults {
    value: Real,
    delta: Real,
    gamma: Real,
    theta: Real,
    theta_per_day: Real,
    vega: Real,
    rho: Real,
    dividend_rho: Real,
    strike_sensitivity: Real,
    strike_gamma: Real,
    exercise_type: &'static str,
}

/// American vanilla engine using the Bjerksund and Stensland (1993) approximation.
pub struct BjerksundStenslandApproximationEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
}

impl BjerksundStenslandApproximationEngine {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self { base, process }
    }

    fn european_call_results(
        &self,
        s: Real,
        x: Real,
        rf_d: Real,
        d_d: Real,
        variance: Real,
        last_date: Date,
    ) -> QlResult<BjerksundResults> {
        let forward_price = s * d_d / rf_d;
        let std_dev = variance.sqrt();
        let black = BlackCalculator::with_striked_payoff(
            &PlainVanillaPayoff::new(OptionType::Call, x),
            forward_price,
            std_dev,
            rf_d,
        )?;

        let risk_free = self.process.risk_free_rate().current_link()?;
        let dividend = self.process.dividend_yield().current_link()?;
        let black_vol = self.process.black_volatility().current_link()?;

        let rfdc = risk_free.require_day_counter()?;
        let divdc = dividend.require_day_counter()?;
        let voldc = black_vol.require_day_counter()?;

        let t_rf = rfdc.year_fraction(risk_free.reference_date()?, last_date);
        let t_div = divdc.year_fraction(dividend.reference_date()?, last_date);
        let t_vol = voldc.year_fraction(black_vol.reference_date()?, last_date);

        let value = black.value();
        let delta = black.delta(s)?;
        let gamma = black.gamma(s)?;
        let rho = black.rho(t_rf)?;
        let dividend_rho = black.dividend_rho(t_div)?;
        let vega = black.vega(t_vol)?;
        let theta = black.theta(s, t_vol).unwrap_or(0.0);
        let theta_per_day = black.theta_per_day(s, t_vol).unwrap_or(0.0);
        let strike_sensitivity = black.strike_sensitivity();
        let strike_gamma = gamma * squared(s / x);

        Ok(BjerksundResults {
            value,
            delta,
            gamma,
            theta,
            theta_per_day,
            vega,
            rho,
            dividend_rho,
            strike_sensitivity,
            strike_gamma,
            exercise_type: "European",
        })
    }

    fn immediate_exercise(s: Real, x: Real) -> BjerksundResults {
        let value = (s - x).max(0.0);
        let delta = if s >= x { 1.0 } else { 0.0 };
        BjerksundResults {
            value,
            delta,
            gamma: 0.0,
            theta: 0.0,
            theta_per_day: 0.0,
            vega: 0.0,
            rho: 0.0,
            dividend_rho: 0.0,
            strike_sensitivity: -delta,
            strike_gamma: 0.0,
            exercise_type: "Immediate",
        }
    }

    fn american_call_approximation(
        &self,
        s: Real,
        x: Real,
        rf_d: Real,
        d_d: Real,
        variance: Real,
        last_date: Date,
    ) -> QlResult<BjerksundResults> {
        let european_results = self.european_call_results(s, x, rf_d, d_d, variance, last_date)?;

        let b_t = (d_d / rf_d).ln();
        let r_t = (1.0 / rf_d).ln();

        let beta =
            (0.5 - b_t / variance) + ((b_t / variance - 0.5).powi(2) + 2.0 * r_t / variance).sqrt();

        let b_infinity = beta / (beta - 1.0) * x;
        let b0 = if b_t == r_t {
            x
        } else {
            x.max(r_t / (r_t - b_t) * x)
        };
        let ht = -(b_t + 2.0 * variance.sqrt()) * b0 / (b_infinity - b0);

        let i = b0 + (b_infinity - b0) * (1.0 - ht.exp());

        let fwd = s * d_d / rf_d;
        let q = (i / fwd).ln() / variance.sqrt();

        let mut results = if s >= i {
            Self::immediate_exercise(s, x)
        } else if q > 12.5 {
            european_results.clone()
        } else {
            let phi_s_beta_i_i_rt_bt_v = phi(s, beta, i, i, r_t, b_t, variance);
            let phi_s_1_i_i_rt_bt_v = phi(s, 1.0, i, i, r_t, b_t, variance);
            let phi_s_1_x_i_rt_bt_v = phi(s, 1.0, x, i, r_t, b_t, variance);

            let value = (i - x) * (s / i).powf(beta) * (1.0 - phi_s_beta_i_i_rt_bt_v)
                + s * phi_s_1_i_i_rt_bt_v
                - s * phi_s_1_x_i_rt_bt_v
                - x * phi(s, 0.0, i, i, r_t, b_t, variance)
                + x * phi(s, 0.0, x, i, r_t, b_t, variance);

            let phi_s_s_beta_i_i_rt_bt_v = phi_s(s, beta, i, i, r_t, b_t, variance);
            let phi_s_s_1_i_i_rt_bt_v = phi_s(s, 1.0, i, i, r_t, b_t, variance);
            let phi_s_s_1_x_i_rt_bt_v = phi_s(s, 1.0, x, i, r_t, b_t, variance);

            let delta = (i - x) * (s / i).powf(beta - 1.0) * beta / i
                * (1.0 - phi_s_beta_i_i_rt_bt_v)
                - (i - x) * (s / i).powf(beta) * phi_s_s_beta_i_i_rt_bt_v
                + phi_s_1_i_i_rt_bt_v
                + s * phi_s_s_1_i_i_rt_bt_v
                - phi_s_1_x_i_rt_bt_v
                - s * phi_s_s_1_x_i_rt_bt_v
                - x * phi_s(s, 0.0, i, i, r_t, b_t, variance)
                + x * phi_s(s, 0.0, x, i, r_t, b_t, variance);

            let risk_free = self.process.risk_free_rate().current_link()?;
            let dividend = self.process.dividend_yield().current_link()?;
            let black_vol = self.process.black_volatility().current_link()?;

            let ref_date = risk_free.reference_date()?;
            let qdc = dividend.require_day_counter()?;
            let tq = qdc.year_fraction(ref_date, last_date);

            let beta_dq = tq
                * (1.0 / variance
                    - 1.0 / (2.0 * ((b_t / variance - 0.5).powi(2) + 2.0 * r_t / variance).sqrt())
                        * 2.0
                        * (b_t / variance - 0.5)
                        / variance);
            let b_infinity_dq = -x / (beta - 1.0).powi(2) * beta_dq;
            let b0_dq = if d_d <= rf_d {
                0.0
            } else {
                x * rf_d.ln() / (d_d.ln().powi(2)) * tq
            };

            let ht_dq = tq * b0 / (b_infinity - b0)
                - (b_t + 2.0 * variance.sqrt())
                    * (b0_dq * (b_infinity - b0) - b0 * (b_infinity_dq - b0_dq))
                    / (b_infinity - b0).powi(2);
            let i_dq = b0_dq + (b_infinity_dq - b0_dq) * (1.0 - ht.exp())
                - (b_infinity - b0) * ht.exp() * ht_dq;

            let phi_h_s_beta_i_i_rt_bt_v = phi_h(s, beta, i, i, r_t, b_t, variance);
            let phi_i_s_beta_i_i_rt_bt_v = phi_i(s, beta, i, i, r_t, b_t, variance);
            let phi_gamma_s_beta_i_i_rt_bt_v = phi_gamma(s, beta, i, i, r_t, b_t, variance);
            let phi_bt_s_beta_i_i_rt_bt_v = phi_bt(s, beta, i, i, r_t, b_t, variance);
            let phi_h_s_1_i_i_rt_bt_v = phi_h(s, 1.0, i, i, r_t, b_t, variance);
            let phi_i_s_1_i_i_rt_bt_v = phi_i(s, 1.0, i, i, r_t, b_t, variance);
            let phi_bt_s_1_i_i_rt_bt_v = phi_bt(s, 1.0, i, i, r_t, b_t, variance);
            let phi_i_s_1_x_i_rt_bt_v = phi_i(s, 1.0, x, i, r_t, b_t, variance);
            let phi_bt_s_1_x_i_rt_bt_v = phi_bt(s, 1.0, x, i, r_t, b_t, variance);
            let phi_h_s_0_i_i_rt_bt_v = phi_h(s, 0.0, i, i, r_t, b_t, variance);
            let phi_i_s_0_i_i_rt_bt_v = phi_i(s, 0.0, i, i, r_t, b_t, variance);
            let phi_bt_s_0_i_i_rt_bt_v = phi_bt(s, 0.0, i, i, r_t, b_t, variance);
            let phi_i_s_0_x_i_rt_bt_v = phi_i(s, 0.0, x, i, r_t, b_t, variance);
            let phi_bt_s_0_x_i_rt_bt_v = phi_bt(s, 0.0, x, i, r_t, b_t, variance);

            let dividend_rho = (i_dq * (s / i).powf(beta)
                + (i - x) * (s / i).powf(beta) * (beta_dq * (s / i).ln() - beta / i * i_dq))
                * (1.0 - phi_s_beta_i_i_rt_bt_v)
                - (i - x)
                    * (s / i).powf(beta)
                    * (phi_h_s_beta_i_i_rt_bt_v * i_dq
                        + phi_i_s_beta_i_i_rt_bt_v * i_dq
                        + phi_gamma_s_beta_i_i_rt_bt_v * beta_dq
                        - phi_bt_s_beta_i_i_rt_bt_v * tq)
                + s * (phi_h_s_1_i_i_rt_bt_v * i_dq + phi_i_s_1_i_i_rt_bt_v * i_dq
                    - phi_bt_s_1_i_i_rt_bt_v * tq)
                - s * (phi_i_s_1_x_i_rt_bt_v * i_dq - phi_bt_s_1_x_i_rt_bt_v * tq)
                - x * (phi_h_s_0_i_i_rt_bt_v * i_dq + phi_i_s_0_i_i_rt_bt_v * i_dq
                    - phi_bt_s_0_i_i_rt_bt_v * tq)
                + x * (phi_i_s_0_x_i_rt_bt_v * i_dq - phi_bt_s_0_x_i_rt_bt_v * tq);

            let rdc = risk_free.require_day_counter()?;
            let tr = rdc.year_fraction(ref_date, last_date);

            let beta_dr = tr
                * (-1.0 / variance
                    + 1.0 / (2.0 * ((b_t / variance - 0.5).powi(2) + 2.0 * r_t / variance).sqrt())
                        * 2.0
                        * ((b_t / variance - 0.5) / variance + 1.0 / variance));
            let b_infinity_dr = -x / (beta - 1.0).powi(2) * beta_dr;
            let b0_dr = if d_d <= rf_d { 0.0 } else { -x * tr / d_d.ln() };
            let ht_dr = -tr * b0 / (b_infinity - b0)
                - (b_t + 2.0 * variance.sqrt())
                    * (b0_dr * (b_infinity - b0) - b0 * (b_infinity_dr - b0_dr))
                    / (b_infinity - b0).powi(2);
            let i_dr = b0_dr + (b_infinity_dr - b0_dr) * (1.0 - ht.exp())
                - (b_infinity - b0) * ht.exp() * ht_dr;

            let phi_rt_s_beta_i_i_rt_bt_v = phi_rt(s, beta, i, i, r_t, b_t, variance);
            let phi_rt_s_1_i_i_rt_bt_v = phi_rt(s, 1.0, i, i, r_t, b_t, variance);
            let phi_rt_s_1_x_i_rt_bt_v = phi_rt(s, 1.0, x, i, r_t, b_t, variance);
            let phi_rt_s_0_i_i_rt_bt_v = phi_rt(s, 0.0, i, i, r_t, b_t, variance);
            let phi_rt_s_0_x_i_rt_bt_v = phi_rt(s, 0.0, x, i, r_t, b_t, variance);

            let rho = (i_dr * (s / i).powf(beta)
                + (i - x) * (s / i).powf(beta) * (beta_dr * (s / i).ln() - beta / i * i_dr))
                * (1.0 - phi_s_beta_i_i_rt_bt_v)
                - (i - x)
                    * (s / i).powf(beta)
                    * (phi_h_s_beta_i_i_rt_bt_v * i_dr
                        + phi_i_s_beta_i_i_rt_bt_v * i_dr
                        + phi_gamma_s_beta_i_i_rt_bt_v * beta_dr
                        + phi_rt_s_beta_i_i_rt_bt_v * tr
                        + phi_bt_s_beta_i_i_rt_bt_v * tr)
                + s * (phi_h_s_1_i_i_rt_bt_v * i_dr
                    + phi_i_s_1_i_i_rt_bt_v * i_dr
                    + phi_rt_s_1_i_i_rt_bt_v * tr
                    + phi_bt_s_1_i_i_rt_bt_v * tr)
                - s * (phi_i_s_1_x_i_rt_bt_v * i_dr
                    + phi_rt_s_1_x_i_rt_bt_v * tr
                    + phi_bt_s_1_x_i_rt_bt_v * tr)
                - x * (phi_h_s_0_i_i_rt_bt_v * i_dr
                    + phi_i_s_0_i_i_rt_bt_v * i_dr
                    + phi_rt_s_0_i_i_rt_bt_v * tr
                    + phi_bt_s_0_i_i_rt_bt_v * tr)
                + x * (phi_i_s_0_x_i_rt_bt_v * i_dr
                    + phi_rt_s_0_x_i_rt_bt_v * tr
                    + phi_bt_s_0_x_i_rt_bt_v * tr);

            let vdc = black_vol.require_day_counter()?;
            let tv = vdc.year_fraction(ref_date, last_date);
            let variance_dv = 2.0 * (variance * tv).sqrt();

            let beta_dv = b_t / variance.powi(2) * variance_dv
                - 1.0 / (2.0 * ((b_t / variance - 0.5).powi(2) + 2.0 * r_t / variance).sqrt())
                    * (2.0 * (b_t / variance - 0.5) * b_t * variance_dv / variance.powi(2)
                        + 2.0 * r_t / variance.powi(2) * variance_dv);
            let b_infinity_dv = -x / (beta - 1.0).powi(2) * beta_dv;
            let ht_dv = -1.0 / variance.sqrt() * variance_dv * b0 / (b_infinity - b0)
                + (b_t + 2.0 * variance.sqrt()) * b0 / (b_infinity - b0).powi(2) * b_infinity_dv;

            let i_dv = b_infinity_dv * (1.0 - ht.exp()) - (b_infinity - b0) * ht.exp() * ht_dv;

            let phi_v_s_beta_i_i_rt_bt_v = phi_v(s, beta, i, i, r_t, b_t, variance);
            let phi_v_s_1_i_i_rt_bt_v = phi_v(s, 1.0, i, i, r_t, b_t, variance);
            let phi_v_s_1_x_i_rt_bt_v = phi_v(s, 1.0, x, i, r_t, b_t, variance);
            let phi_v_s_0_i_i_rt_bt_v = phi_v(s, 0.0, i, i, r_t, b_t, variance);
            let phi_v_s_0_x_i_rt_bt_v = phi_v(s, 0.0, x, i, r_t, b_t, variance);

            let vega = (i_dv * (s / i).powf(beta)
                + (i - x) * (s / i).powf(beta) * (beta_dv * (s / i).ln() - beta / i * i_dv))
                * (1.0 - phi_s_beta_i_i_rt_bt_v)
                - (i - x)
                    * (s / i).powf(beta)
                    * (phi_h_s_beta_i_i_rt_bt_v * i_dv
                        + phi_i_s_beta_i_i_rt_bt_v * i_dv
                        + phi_gamma_s_beta_i_i_rt_bt_v * beta_dv
                        + phi_v_s_beta_i_i_rt_bt_v * variance_dv)
                + s * (phi_h_s_1_i_i_rt_bt_v * i_dv
                    + phi_i_s_1_i_i_rt_bt_v * i_dv
                    + phi_v_s_1_i_i_rt_bt_v * variance_dv)
                - s * (phi_i_s_1_x_i_rt_bt_v * i_dv + phi_v_s_1_x_i_rt_bt_v * variance_dv)
                - x * (phi_h_s_0_i_i_rt_bt_v * i_dv
                    + phi_i_s_0_i_i_rt_bt_v * i_dv
                    + phi_v_s_0_i_i_rt_bt_v * variance_dv)
                + x * (phi_i_s_0_x_i_rt_bt_v * i_dv + phi_v_s_0_x_i_rt_bt_v * variance_dv);

            let phi_ss_s_beta_i_i_rt_bt_v = phi_ss(s, beta, i, i, r_t, b_t, variance);
            let phi_ss_s_1_i_i_rt_bt_v = phi_ss(s, 1.0, i, i, r_t, b_t, variance);
            let phi_ss_s_1_x_i_rt_bt_v = phi_ss(s, 1.0, x, i, r_t, b_t, variance);
            let phi_ss_s_0_i_i_rt_bt_v = phi_ss(s, 0.0, i, i, r_t, b_t, variance);
            let phi_ss_s_0_x_i_rt_bt_v = phi_ss(s, 0.0, x, i, r_t, b_t, variance);

            let gamma = (i - x) * (s / i).powf(beta - 2.0) * beta * (beta - 1.0) / i.powi(2)
                * (1.0 - phi_s_beta_i_i_rt_bt_v)
                - 2.0 * (i - x) * (s / i).powf(beta - 1.0) * beta / i * phi_s_s_beta_i_i_rt_bt_v
                - (i - x) * (s / i).powf(beta) * phi_ss_s_beta_i_i_rt_bt_v
                + 2.0 * phi_s_s_1_i_i_rt_bt_v
                + s * phi_ss_s_1_i_i_rt_bt_v
                - 2.0 * phi_s_s_1_x_i_rt_bt_v
                - s * phi_ss_s_1_x_i_rt_bt_v
                - x * phi_ss_s_0_i_i_rt_bt_v
                + x * phi_ss_s_0_x_i_rt_bt_v;

            let vol = (variance / tv).sqrt();
            let tomorrow = ref_date + 1;
            let dtq =
                qdc.year_fraction(ref_date, last_date) - qdc.year_fraction(tomorrow, last_date);
            let dtr =
                rdc.year_fraction(ref_date, last_date) - rdc.year_fraction(tomorrow, last_date);
            let dtv =
                vdc.year_fraction(ref_date, last_date) - vdc.year_fraction(tomorrow, last_date);

            let theta_per_day = -(0.5 * vega * vol / tv * dtv
                + rho * r_t / (tr * tr) * dtr
                + dividend_rho * (r_t - b_t) / (tq * tq) * dtq);
            let theta = 365.0 * theta_per_day;

            let strike_sensitivity = value / x - s / x * delta;
            let strike_gamma = gamma * squared(s / x);

            BjerksundResults {
                value,
                delta,
                gamma,
                theta,
                theta_per_day,
                vega,
                rho,
                dividend_rho,
                strike_sensitivity,
                strike_gamma,
                exercise_type: "American",
            }
        };

        if results.value < european_results.value {
            results = european_results;
        }

        Ok(results)
    }
}

impl AsObservable for BjerksundStenslandApproximationEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BjerksundStenslandApproximationEngine {
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

        let vol_structure = self.process.black_volatility().current_link()?;
        let variance = vol_structure.black_variance_date(last_date, payoff.strike(), false)?;

        let mut dividend_discount = self
            .process
            .dividend_yield()
            .current_link()?
            .discount_date(last_date, false)?;
        let mut risk_free_discount = self
            .process
            .risk_free_rate()
            .current_link()?
            .discount_date(last_date, false)?;
        let mut spot = self.process.state_variable().current_link()?.value()?;
        require!(spot > 0.0, "negative or null underlying given");
        let mut strike = payoff.strike();
        let is_put = payoff.option_type() == OptionType::Put;

        if is_put {
            // use put-call symmetry
            std::mem::swap(&mut spot, &mut strike);
            std::mem::swap(&mut risk_free_discount, &mut dividend_discount);
        }

        require!(
            !(dividend_discount > 1.0 && risk_free_discount > dividend_discount),
            "double-boundary case r<q<0 for a call given"
        );

        let mut res = if dividend_discount >= 1.0 && dividend_discount >= risk_free_discount {
            self.european_call_results(
                spot,
                strike,
                risk_free_discount,
                dividend_discount,
                variance,
                last_date,
            )?
        } else {
            self.american_call_approximation(
                spot,
                strike,
                risk_free_discount,
                dividend_discount,
                variance,
                last_date,
            )?
        };

        // check if immediate exercise gives higher NPV
        if res.value < (spot - strike) * (1.0 + 10.0 * Real::EPSILON) {
            res = Self::immediate_exercise(spot, strike);
        }

        if is_put {
            std::mem::swap(&mut res.delta, &mut res.strike_sensitivity);
            std::mem::swap(&mut res.gamma, &mut res.strike_gamma);
            std::mem::swap(&mut res.rho, &mut res.dividend_rho);

            let risk_free = self.process.risk_free_rate().current_link()?;
            let dividend = self.process.dividend_yield().current_link()?;
            let tr = risk_free
                .require_day_counter()?
                .year_fraction(risk_free.reference_date()?, last_date);
            let tq = dividend
                .require_day_counter()?
                .year_fraction(dividend.reference_date()?, last_date);

            res.rho *= tr / tq;
            res.dividend_rho *= tq / tr;
        }

        let results = self.base.results_mut();
        results.instrument.value = Some(res.value);
        results.greeks = Greeks {
            delta: Some(res.delta),
            gamma: Some(res.gamma),
            theta: Some(res.theta),
            vega: Some(res.vega),
            rho: Some(res.rho),
            dividend_rho: Some(res.dividend_rho),
        };
        results.more_greeks = MoreGreeks {
            itm_cash_probability: None,
            delta_forward: None,
            elasticity: None,
            theta_per_day: Some(res.theta_per_day),
            strike_sensitivity: Some(res.strike_sensitivity),
        };
        let extras = &mut results.instrument.additional_results;
        extras.insert(
            "strikeGamma".to_string(),
            shared(res.strike_gamma) as Shared<dyn Any>,
        );
        extras.insert(
            "exerciseType".to_string(),
            shared(res.exercise_type.to_string()) as Shared<dyn Any>,
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{PlainVanillaPayoff, VanillaOption};
    use crate::interestrate::Compounding;
    use crate::option::OptionType::{Call, Put};
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn make_process_actual360(
        s: Real,
        q: Real,
        r: Real,
        v: Real,
    ) -> Shared<BlackScholesMertonProcess> {
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

    /// Tests from QuantLib `americanoption.cpp` `testBjerksundStenslandValues`.
    ///
    /// Tolerance is 5e-5 as in QuantLib.
    #[test]
    fn test_bjerksund_stensland_values() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());

        type Row = (OptionType, Real, Real, Real, Real, Real, Real, Real);
        let values: &[Row] = &[
            // from "Option pricing formulas", Haug, McGraw-Hill 1998, pag 27
            (Call, 40.00, 42.00, 0.08, 0.04, 0.75, 0.35, 5.2704),
            // from "Option pricing formulas", Haug, McGraw-Hill 1998, VBA code
            (Put, 40.00, 36.00, 0.00, 0.06, 1.00, 0.20, 4.4531),
            // ATM option with very small volatility, reference value taken from R
            (Call, 100.0, 100.0, 0.05, 0.05, 1.0, 0.0021, 0.08032314),
            // ATM option with very small volatility, reference from Barone-Adesi and Whaley
            (Call, 100.0, 100.0, 0.05, 0.05, 1.0, 0.0001, 0.003860656),
            (Call, 100.0, 99.99, 0.05, 0.05, 1.0, 0.0001, 0.00081),
            // ITM option with a very small volatility
            (Call, 100.0, 110.0, 0.05, 0.05, 1.0, 0.0001, 10.0),
            (Put, 110.0, 100.0, 0.05, 0.05, 1.0, 0.0001, 10.0),
            // ATM option with a very large volatility
            (Put, 100.0, 110.0, 0.05, 0.05, 1.0, 10.0, 95.12289),
        ];

        let tolerance = 5.0e-5;

        for &(ty, strike, spot, q, r, t, vol, expected) in values {
            let process = make_process_actual360(spot, q, r, vol);
            let ex_date = today() + (t * 360.0).round() as i32;
            let mut option = VanillaOption::new(
                shared(PlainVanillaPayoff::new(ty, strike)),
                shared(AmericanExercise::over(today(), ex_date).unwrap()),
                Shared::clone(&settings),
            );
            option.base_mut().set_pricing_engine(shared_mut(
                BjerksundStenslandApproximationEngine::new(process),
            ) as SharedMut<dyn PricingEngine>);

            let calculated = option.npv().unwrap();
            let error = (calculated - expected).abs();
            assert!(
                error <= tolerance,
                "{ty:?} strike={strike} spot={spot} q={q} r={r} t={t} vol={vol}: \
                 calc={calculated} expected={expected} error={error} tol={tolerance}"
            );
        }
    }

    /// Tests from QuantLib `americanoption.cpp` `testBjerksundStenslandEuropeanGreeks`:
    /// when early exercise is not optimal, American option prices and Greeks match European.
    #[test]
    fn test_bjerksund_stensland_european_greeks() {
        let today = Date::new(5, Month::November, 2022);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);

        let spot = 100.0;
        let k = 105.0;
        let sigma = 0.40;
        let maturity = today + 724;

        let test_cases = [(Put, -0.05, 0.02), (Call, 0.05, -0.025)];

        for (opt_type, r, q) in test_cases {
            let q_ts = Handle::new(shared(FlatForward::with_rate(
                today,
                q,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>);
            let r_ts = Handle::new(shared(FlatForward::with_rate(
                today,
                r,
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>);
            let vol_ts = Handle::new(shared(BlackConstantVol::new(
                today,
                None,
                sigma,
                Thirty360::with_convention(Convention::European),
            )) as Shared<dyn BlackVolTermStructure>);

            let process = shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
                q_ts,
                r_ts,
                vol_ts,
            ));

            let mut american_opt = VanillaOption::new(
                shared(PlainVanillaPayoff::new(opt_type, k)),
                shared(AmericanExercise::over(today, maturity).unwrap()),
                Shared::clone(&settings),
            );
            american_opt
                .base_mut()
                .set_pricing_engine(shared_mut(BjerksundStenslandApproximationEngine::new(
                    Shared::clone(&process),
                )) as SharedMut<dyn PricingEngine>);

            let mut european_opt = VanillaOption::new(
                shared(PlainVanillaPayoff::new(opt_type, k)),
                shared(EuropeanExercise::new(maturity)),
                Shared::clone(&settings),
            );
            european_opt
                .base_mut()
                .set_pricing_engine(shared_mut(AnalyticEuropeanEngine::new(process))
                    as SharedMut<dyn PricingEngine>);

            let tol = 1000.0 * Real::EPSILON;

            let am_npv = american_opt.npv().unwrap();
            let eu_npv = european_opt.npv().unwrap();
            assert!(
                (am_npv - eu_npv).abs() <= tol,
                "NPV mismatch: {am_npv} vs {eu_npv}"
            );

            let am_delta = american_opt.delta().unwrap();
            let eu_delta = european_opt.delta().unwrap();
            assert!(
                (am_delta - eu_delta).abs() <= tol,
                "Delta mismatch: {am_delta} vs {eu_delta}"
            );

            let am_strike_sens = american_opt.strike_sensitivity().unwrap();
            let eu_strike_sens = european_opt.strike_sensitivity().unwrap();
            assert!(
                (am_strike_sens - eu_strike_sens).abs() <= tol,
                "StrikeSensitivity mismatch: {am_strike_sens} vs {eu_strike_sens}"
            );

            let am_gamma = american_opt.gamma().unwrap();
            let eu_gamma = european_opt.gamma().unwrap();
            assert!(
                (am_gamma - eu_gamma).abs() <= tol,
                "Gamma mismatch: {am_gamma} vs {eu_gamma}"
            );

            let am_vega = american_opt.vega().unwrap();
            let eu_vega = european_opt.vega().unwrap();
            assert!(
                (am_vega - eu_vega).abs() <= tol,
                "Vega mismatch: {am_vega} vs {eu_vega}"
            );

            let am_theta = american_opt.theta().unwrap();
            let eu_theta = european_opt.theta().unwrap();
            assert!(
                (am_theta - eu_theta).abs() <= tol,
                "Theta mismatch: {am_theta} vs {eu_theta}"
            );

            let am_theta_per_day = american_opt.theta_per_day().unwrap();
            let eu_theta_per_day = european_opt.theta_per_day().unwrap();
            assert!(
                (am_theta_per_day - eu_theta_per_day).abs() <= tol,
                "ThetaPerDay mismatch: {am_theta_per_day} vs {eu_theta_per_day}"
            );

            let am_rho = american_opt.rho().unwrap();
            let eu_rho = european_opt.rho().unwrap();
            assert!(
                (am_rho - eu_rho).abs() <= tol,
                "Rho mismatch: {am_rho} vs {eu_rho}"
            );

            let am_div_rho = american_opt.dividend_rho().unwrap();
            let eu_div_rho = european_opt.dividend_rho().unwrap();
            assert!(
                (am_div_rho - eu_div_rho).abs() <= tol,
                "DividendRho mismatch: {am_div_rho} vs {eu_div_rho}"
            );
        }
    }

    #[test]
    fn double_boundary_rejected() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        // r < q < 0: riskFreeDiscount > dividendDiscount > 1.0
        let process = make_process_actual360(100.0, -0.02, -0.05, 0.20);
        let mut opt = VanillaOption::new(
            shared(PlainVanillaPayoff::new(Call, 100.0)),
            shared(AmericanExercise::over(today(), today() + 360).unwrap()),
            settings,
        );
        opt.base_mut()
            .set_pricing_engine(
                shared_mut(BjerksundStenslandApproximationEngine::new(process))
                    as SharedMut<dyn PricingEngine>,
            );
        let err = opt.npv().unwrap_err();
        assert!(err.message().contains("double-boundary case"));
    }

    /// Single Greeks test from QuantLib `americanoption.cpp` `testSingleBjerksundStenslandGreeks`.
    #[test]
    fn test_single_bjerksund_stensland_greeks() {
        let today = Date::new(20, Month::January, 2023);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);

        let s = 100.0;
        let v = 0.3;
        let q = 0.04;
        let r = 0.07;

        let q_ts = Handle::new(shared(FlatForward::with_rate(
            today,
            q,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let r_ts = Handle::new(shared(FlatForward::with_rate(
            today,
            r,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let vol_ts =
            Handle::new(
                shared(BlackConstantVol::new(today, None, v, Actual365Fixed::new()))
                    as Shared<dyn BlackVolTermStructure>,
            );

        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(s)) as Shared<dyn Quote>),
            q_ts,
            r_ts,
            vol_ts,
        ));

        let maturity = Date::new(20, Month::January, 2025);
        let mut option = VanillaOption::new(
            shared(PlainVanillaPayoff::new(Call, 100.0)),
            shared(AmericanExercise::over(today, maturity).unwrap()),
            settings,
        );
        option.base_mut().set_pricing_engine(
            shared_mut(BjerksundStenslandApproximationEngine::new(process))
                as SharedMut<dyn PricingEngine>,
        );

        let expected_npv = 17.9251834488399169;
        let expected_delta = 0.590801845261082592;
        let expected_gamma = 0.00825347110063545664;
        let expected_strike_sensitivity = -0.411550010772683383;
        let expected_div_rho = -114.137818682236826;
        let expected_rho = 80.4900013901554416;
        let expected_vega = 49.2906331545933227;
        let expected_theta = -4.22540293840206704;

        let tol = 1e6 * Real::EPSILON;

        assert!((option.npv().unwrap() - expected_npv).abs() <= tol);
        assert!((option.delta().unwrap() - expected_delta).abs() <= tol);
        assert!((option.gamma().unwrap() - expected_gamma).abs() <= tol);
        assert!((option.strike_sensitivity().unwrap() - expected_strike_sensitivity).abs() <= tol);
        assert!((option.dividend_rho().unwrap() - expected_div_rho).abs() <= tol);
        assert!((option.rho().unwrap() - expected_rho).abs() <= tol);
        assert!((option.vega().unwrap() - expected_vega).abs() <= tol);
        assert!((option.theta().unwrap() - expected_theta).abs() <= tol);
        assert!((option.theta_per_day().unwrap() - expected_theta / 365.0).abs() <= tol);
        assert_eq!(option.result::<String>("exerciseType").unwrap(), "American");
    }

    /// American Greeks FD test from QuantLib `americanoption.cpp` `testBjerksundStenslandAmericanGreeks`.
    /// Tests both Call and Put with mixed day counters (Actual360, Actual365Fixed, Thirty360 ISDA)
    /// against numerical finite differences, validating the put-call symmetry transformation.
    #[test]
    fn test_bjerksund_stensland_american_greeks() {
        let today = Date::new(5, Month::December, 2022);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);

        let s = 99.9;
        let v = 0.2;
        let q = 0.08;
        let r = 0.06;
        let strike = 100.0;
        let maturity = today + 182;

        let spot_quote = shared(SimpleQuote::new(s));
        let vol_quote = shared(SimpleQuote::new(v));
        let q_quote = shared(SimpleQuote::new(q));
        let r_quote = shared(SimpleQuote::new(r));

        let q_ts = Handle::new(shared(FlatForward::new(
            today,
            Handle::new(q_quote.clone() as Shared<dyn Quote>),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let r_ts = Handle::new(shared(FlatForward::new(
            today,
            Handle::new(r_quote.clone() as Shared<dyn Quote>),
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let vol_ts = Handle::new(shared(BlackConstantVol::with_quote(
            today,
            None,
            Handle::new(vol_quote.clone() as Shared<dyn Quote>),
            Thirty360::with_convention(Convention::ISDA),
        )) as Shared<dyn BlackVolTermStructure>);

        let bs_process = shared(BlackScholesMertonProcess::new(
            Handle::new(spot_quote.clone() as Shared<dyn Quote>),
            q_ts,
            r_ts,
            vol_ts,
        ));

        for opt_type in [Call, Put] {
            let make_opt = |payoff: Shared<PlainVanillaPayoff>, mat: Date| {
                let mut opt = VanillaOption::new(
                    payoff,
                    shared(AmericanExercise::over(today, mat).unwrap()),
                    Shared::clone(&settings),
                );
                opt.base_mut()
                    .set_pricing_engine(shared_mut(BjerksundStenslandApproximationEngine::new(
                        Shared::clone(&bs_process),
                    )) as SharedMut<dyn PricingEngine>);
                opt
            };

            let f_d = 1e-5;
            let f_g = 5e-5;
            let f_q = 1e-6;

            let mut option = make_opt(shared(PlainVanillaPayoff::new(opt_type, strike)), maturity);
            let mut strike_up = make_opt(
                shared(PlainVanillaPayoff::new(opt_type, strike * (1.0 + f_d))),
                maturity,
            );
            let mut strike_down = make_opt(
                shared(PlainVanillaPayoff::new(opt_type, strike * (1.0 - f_d))),
                maturity,
            );
            let mut day_up = make_opt(
                shared(PlainVanillaPayoff::new(opt_type, strike)),
                maturity + 1,
            );
            let mut day_down = make_opt(
                shared(PlainVanillaPayoff::new(opt_type, strike)),
                maturity - 1,
            );

            // Base greeks
            spot_quote.set_value(s);
            vol_quote.set_value(v);
            q_quote.set_value(q);
            r_quote.set_value(r);

            let npv = option.npv().unwrap();
            let delta = option.delta().unwrap();
            let gamma = option.gamma().unwrap();
            let strike_sens = option.strike_sensitivity().unwrap();
            let div_rho = option.dividend_rho().unwrap();
            let rho = option.rho().unwrap();
            let vega = option.vega().unwrap();
            let theta = option.theta().unwrap();
            let theta_per_day = option.theta_per_day().unwrap();
            let exercise_type = option.result::<String>("exerciseType").unwrap();
            assert_eq!(exercise_type, "American");

            // Delta FD
            spot_quote.set_value(s * (1.0 + f_d));
            let f2 = option.npv().unwrap();
            spot_quote.set_value(s * (1.0 - f_d));
            let f1 = option.npv().unwrap();
            spot_quote.set_value(s);
            let num_delta = (f2 - f1) / (2.0 * f_d * s);
            assert!(
                (delta - num_delta).abs() <= 5e-6,
                "delta error for {opt_type:?}: {delta} vs {num_delta}"
            );

            // Gamma FD (5-point stencil)
            spot_quote.set_value(s * (1.0 + 2.0 * f_g));
            let gp2 = option.npv().unwrap();
            spot_quote.set_value(s * (1.0 + f_g));
            let gp1 = option.npv().unwrap();
            spot_quote.set_value(s * (1.0 - f_g));
            let gm1 = option.npv().unwrap();
            spot_quote.set_value(s * (1.0 - 2.0 * f_g));
            let gm2 = option.npv().unwrap();
            spot_quote.set_value(s);
            let num_gamma =
                (-gp2 + 16.0 * gp1 - 30.0 * npv + 16.0 * gm1 - gm2) / (12.0 * (f_g * s).powi(2));
            assert!(
                (gamma - num_gamma).abs() <= 1e-4,
                "gamma error for {opt_type:?}: {gamma} vs {num_gamma}"
            );

            // Strike sensitivity FD
            let k2 = strike_up.npv().unwrap();
            let k1 = strike_down.npv().unwrap();
            let num_strike_sens = (k2 - k1) / (2.0 * f_d * strike);
            assert!(
                (strike_sens - num_strike_sens).abs() <= 5e-6,
                "strike_sens error for {opt_type:?}: {strike_sens} vs {num_strike_sens}"
            );

            // Dividend rho FD
            q_quote.set_value(q + f_q);
            let q2 = option.npv().unwrap();
            q_quote.set_value(q - f_q);
            let q1 = option.npv().unwrap();
            q_quote.set_value(q);
            let num_div_rho = (q2 - q1) / (2.0 * f_q);
            assert!(
                (div_rho - num_div_rho).abs() <= 3e-2,
                "div_rho error for {opt_type:?}: {div_rho} vs {num_div_rho}"
            );

            // Rho FD
            r_quote.set_value(r + f_q);
            let r2 = option.npv().unwrap();
            r_quote.set_value(r - f_q);
            let r1 = option.npv().unwrap();
            r_quote.set_value(r);
            let num_rho = (r2 - r1) / (2.0 * f_q);
            assert!(
                (rho - num_rho).abs() <= 3e-2,
                "rho error for {opt_type:?}: {rho} vs {num_rho}"
            );

            // Vega FD
            vol_quote.set_value(v + f_d);
            let v2 = option.npv().unwrap();
            vol_quote.set_value(v - f_d);
            let v1 = option.npv().unwrap();
            vol_quote.set_value(v);
            let num_vega = (v2 - v1) / (2.0 * f_d);
            assert!(
                (vega - num_vega).abs() <= 5e-4,
                "vega error for {opt_type:?}: {vega} vs {num_vega}"
            );

            // Theta FD
            let t2 = day_up.npv().unwrap();
            let t1 = day_down.npv().unwrap();
            let num_theta_per_day = (t1 - t2) / 2.0;
            let num_theta = 365.0 * num_theta_per_day;
            assert!(
                (theta_per_day - num_theta_per_day).abs() <= 5e-4 / 365.0,
                "theta_per_day error for {opt_type:?}: {theta_per_day} vs {num_theta_per_day}"
            );
            assert!(
                (theta - num_theta).abs() <= 5e-4,
                "theta error for {opt_type:?}: {theta} vs {num_theta}"
            );
        }
    }
}
