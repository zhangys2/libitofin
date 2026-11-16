//! Closed-form Hagan SABR implied-volatility formula.
//!
//! Port of the free functions in `ql/termstructures/volatility/sabr.{hpp,cpp}`:
//! the shift-zero lognormal and Normal (Bachelier) SABR implied volatilities
//! ([`sabr_volatility`]), their unvalidated core ([`unsafe_sabr_volatility`]),
//! and the parameter validator ([`validate_sabr_parameters`]). These are stateless functions; calibration
//! and the smile-section layer that consume them land in later tickets.
//!
//! "Unsafe" in [`unsafe_sabr_volatility`] means *unvalidated inputs*, not
//! Rust's `unsafe`: there is no `unsafe` block anywhere here. The function
//! skips the strike/forward/expiry guards and [`validate_sabr_parameters`] so a
//! caller that has already validated once (for example a smile section pricing
//! a whole strike row) does not repeat the checks per strike.
//!
//! ## Divergences from QuantLib
//!
//! - `validateSabrParameters` checks `beta >= 0.0 && beta <= 1.0` (inclusive,
//!   `sabr.cpp:155`) but its C++ error text reads `"beta must be in (0.0,
//!   1.0)"`. The exclusive-looking message is misleading; this port keeps the
//!   inclusive check and states the interval as `[0.0, 1.0]` in the message.
//! - The C++ guard failures (`QL_REQUIRE`) become `Err` values per D4.
//!
//! ## Deferred to #586 (visible, no stubs)
//!
//! - `shiftedSabrVolatility` (`sabr.hpp:84`), the displaced (shift != 0)
//!   variant.
//! - `sabrFlochKennedyVolatility` (`sabr.hpp:94`).
//! - `sabrGuess` (`sabr.hpp:122`), the calibration initial guess.

use crate::errors::QlResult;
use crate::math::comparison::close;
use crate::require;
use crate::termstructures::volatility::VolatilityType;
use crate::types::{Rate, Real, Time};

/// Lognormal Hagan SABR implied volatility at shift zero.
///
/// Verbatim port of `unsafeSabrLogNormalVolatility` (`sabr.cpp:37-76`). Assumes
/// its inputs are already valid; [`sabr_volatility`] performs the guards.
fn unsafe_sabr_lognormal_volatility(
    strike: Rate,
    forward: Rate,
    expiry_time: Time,
    alpha: Real,
    beta: Real,
    nu: Real,
    rho: Real,
) -> Real {
    let one_minus_beta = 1.0 - beta;
    let a = (forward * strike).powf(one_minus_beta);
    let sqrt_a = a.sqrt();
    let log_m = if !close(forward, strike) {
        (forward / strike).ln()
    } else {
        let epsilon = (forward - strike) / strike;
        epsilon - 0.5 * epsilon * epsilon
    };
    let z = (nu / alpha) * sqrt_a * log_m;
    let b = 1.0 - 2.0 * rho * z + z * z;
    let c = one_minus_beta * one_minus_beta * log_m * log_m;
    let tmp = (b.sqrt() + z - rho) / (1.0 - rho);
    let xx = tmp.ln();
    let denominator = sqrt_a * (1.0 + c / 24.0 + c * c / 1920.0);
    let time_factor = 1.0
        + expiry_time
            * (one_minus_beta * one_minus_beta * alpha * alpha / (24.0 * a)
                + 0.25 * rho * beta * nu * alpha / sqrt_a
                + (2.0 - 3.0 * rho * rho) * (nu * nu / 24.0));

    const M: Real = 10.0;
    let multiplier = if (z * z).abs() > Real::EPSILON * M {
        z / xx
    } else {
        1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0
    };
    (alpha / denominator) * multiplier * time_factor
}

/// Normal (Bachelier) Hagan SABR implied volatility at shift zero.
///
/// Verbatim port of `unsafeSabrNormalVolatility` (`sabr.cpp:94-132`). Assumes
/// its inputs are already valid; [`sabr_volatility`] performs the guards.
fn unsafe_sabr_normal_volatility(
    strike: Rate,
    forward: Rate,
    expiry_time: Time,
    alpha: Real,
    beta: Real,
    nu: Real,
    rho: Real,
) -> Real {
    let one_minus_beta = 1.0 - beta;
    let minus_beta = -beta;
    let a = (forward * strike).powf(one_minus_beta);
    let sqrt_a = a.sqrt();
    let log_m = if !close(forward, strike) {
        (forward / strike).ln()
    } else {
        let epsilon = (forward - strike) / strike;
        epsilon - 0.5 * epsilon * epsilon
    };
    let z = (nu / alpha) * sqrt_a * log_m;
    let b = 1.0 - 2.0 * rho * z + z * z;
    let c = one_minus_beta * one_minus_beta * log_m * log_m;
    let d = log_m * log_m;
    let tmp = (b.sqrt() + z - rho) / (1.0 - rho);
    let xx = tmp.ln();
    let e_1 = 1.0 + d / 24.0 + d * d / 1920.0;
    let e_2 = 1.0 + c / 24.0 + c * c / 1920.0;
    let e = e_1 / e_2;
    let time_factor = 1.0
        + expiry_time
            * (minus_beta * (2.0 - beta) * alpha * alpha / (24.0 * a)
                + 0.25 * rho * beta * nu * alpha / sqrt_a
                + (2.0 - 3.0 * rho * rho) * (nu * nu / 24.0));

    const M: Real = 10.0;
    let multiplier = if (z * z).abs() > Real::EPSILON * M {
        z / xx
    } else {
        1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0
    };
    let f = alpha * (forward * strike).powf(beta / 2.0);
    f * e * multiplier * time_factor
}

/// SABR implied volatility without the input or parameter guards.
///
/// Port of `unsafeSabrVolatility` (`sabr.cpp:134`). "Unsafe" here means the
/// inputs are trusted, not that any Rust `unsafe` is involved. Callers that
/// have already validated (for instance a smile section repricing a strike row)
/// use this to avoid re-running [`validate_sabr_parameters`] per strike.
///
/// Both branches are infallible today; the `QlResult` return is kept so the
/// public signature does not change.
#[allow(clippy::too_many_arguments)]
pub fn unsafe_sabr_volatility(
    strike: Rate,
    forward: Rate,
    expiry_time: Time,
    alpha: Real,
    beta: Real,
    nu: Real,
    rho: Real,
    volatility_type: VolatilityType,
) -> QlResult<Real> {
    match volatility_type {
        VolatilityType::ShiftedLognormal => Ok(unsafe_sabr_lognormal_volatility(
            strike,
            forward,
            expiry_time,
            alpha,
            beta,
            nu,
            rho,
        )),
        VolatilityType::Normal => Ok(unsafe_sabr_normal_volatility(
            strike,
            forward,
            expiry_time,
            alpha,
            beta,
            nu,
            rho,
        )),
    }
}

/// Validate the four SABR parameters.
///
/// Port of `validateSabrParameters` (`sabr.cpp:149`): `alpha > 0`,
/// `0 <= beta <= 1` (inclusive), `nu >= 0`, `rho^2 < 1`.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn validate_sabr_parameters(alpha: Real, beta: Real, nu: Real, rho: Real) -> QlResult<()> {
    require!(alpha > 0.0, "alpha must be positive: {alpha} not allowed");
    require!(
        (0.0..=1.0).contains(&beta),
        "beta must be in [0.0, 1.0]: {beta} not allowed"
    );
    require!(nu >= 0.0, "nu must be non negative: {nu} not allowed");
    require!(
        rho * rho < 1.0,
        "rho square must be less than one: {rho} not allowed"
    );
    Ok(())
}

/// Hagan SABR implied volatility.
///
/// Port of `sabrVolatility` (`sabr.cpp:163`): guards `strike > 0`,
/// `forward > 0`, `expiry_time >= 0`, then [`validate_sabr_parameters`], then
/// the unvalidated core. `volatility_type` selects the lognormal or the Normal
/// (Bachelier) formula.
#[allow(clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]
pub fn sabr_volatility(
    strike: Rate,
    forward: Rate,
    expiry_time: Time,
    alpha: Real,
    beta: Real,
    nu: Real,
    rho: Real,
    volatility_type: VolatilityType,
) -> QlResult<Real> {
    require!(
        strike > 0.0,
        "strike must be positive: {strike} not allowed"
    );
    require!(
        forward > 0.0,
        "at the money forward rate must be positive: {forward} not allowed"
    );
    require!(
        expiry_time >= 0.0,
        "expiry time must be non-negative: {expiry_time} not allowed"
    );
    validate_sabr_parameters(alpha, beta, nu, rho)?;
    unsafe_sabr_volatility(
        strike,
        forward,
        expiry_time,
        alpha,
        beta,
        nu,
        rho,
        volatility_type,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORWARD: Real = 0.039;
    const EXPIRY: Real = 1.0;
    const ALPHA: Real = 0.3;
    const BETA: Real = 0.6;
    const NU: Real = 0.02;
    const RHO: Real = 0.01;

    /// Oracle: `interpolations.cpp` `testSabrInterpolation` first arm
    /// (`:1375-1421`). Strikes and expected vols are transcribed verbatim from
    /// the C++ `strikes[i]`/`volatilities[i]` assignments; `sabrVolatility`
    /// must reproduce them to `1e-12`.
    #[test]
    fn reproduces_the_cpp_31_strike_vol_table() {
        let strikes: [Real; 31] = [
            0.03, 0.032, 0.034, 0.036, 0.038, 0.04, 0.042, 0.044, 0.046, 0.048, 0.05, 0.052, 0.054,
            0.056, 0.058, 0.06, 0.062, 0.064, 0.066, 0.068, 0.07, 0.072, 0.074, 0.076, 0.078, 0.08,
            0.082, 0.084, 0.086, 0.088, 0.09,
        ];
        let expected: [Real; 31] = [
            1.16725837321531,
            1.15226075991385,
            1.13829711098834,
            1.12524190877505,
            1.11299079244474,
            1.10145609357162,
            1.09056348513411,
            1.08024942745106,
            1.07045919457758,
            1.06114533019077,
            1.05226642581503,
            1.04378614411707,
            1.03567243073732,
            1.0278968727451,
            1.02043417226345,
            1.01326171139321,
            1.00635919013311,
            0.999708323124949,
            0.993292584155381,
            0.987096989695393,
            0.98110791455717,
            0.975312934134512,
            0.969700688771689,
            0.964260766651027,
            0.958983602256592,
            0.953860388001395,
            0.948882997029509,
            0.944043915545469,
            0.939336183299237,
            0.934753341079515,
            0.930289384251337,
        ];
        for (strike, expected_vol) in strikes.into_iter().zip(expected) {
            let vol = sabr_volatility(
                strike,
                FORWARD,
                EXPIRY,
                ALPHA,
                BETA,
                NU,
                RHO,
                VolatilityType::ShiftedLognormal,
            )
            .unwrap();
            assert!(
                (vol - expected_vol).abs() <= 1e-12,
                "strike {strike}: expected {expected_vol}, got {vol}"
            );
        }
    }

    /// At-the-money (`strike == forward`) is the only coverage of the
    /// `close(forward, strike)` epsilon branch (`sabr.cpp:49-54`): the forward
    /// is not among the 31 table strikes. The value must be finite and
    /// positive, and it must join continuously to the log branch a hair away,
    /// so a broken epsilon branch (which would jump) is caught.
    #[test]
    fn atm_hits_the_close_epsilon_branch_continuously() {
        let atm = sabr_volatility(
            FORWARD,
            FORWARD,
            EXPIRY,
            ALPHA,
            BETA,
            NU,
            RHO,
            VolatilityType::ShiftedLognormal,
        )
        .unwrap();
        assert!(
            atm.is_finite() && atm > 0.0,
            "atm vol not finite positive: {atm}"
        );

        let just_off = sabr_volatility(
            FORWARD + 1e-13,
            FORWARD,
            EXPIRY,
            ALPHA,
            BETA,
            NU,
            RHO,
            VolatilityType::ShiftedLognormal,
        )
        .unwrap();
        assert!(
            (atm - just_off).abs() < 1e-9,
            "epsilon branch discontinuous: atm {atm} vs just_off {just_off}"
        );
    }

    #[test]
    fn validate_accepts_the_inclusive_beta_boundaries() {
        assert!(validate_sabr_parameters(ALPHA, 0.0, NU, RHO).is_ok());
        assert!(validate_sabr_parameters(ALPHA, 1.0, NU, RHO).is_ok());
    }

    #[test]
    fn validate_rejects_out_of_range_parameters() {
        assert!(validate_sabr_parameters(0.0, BETA, NU, RHO).is_err());
        assert!(validate_sabr_parameters(-0.1, BETA, NU, RHO).is_err());
        assert!(validate_sabr_parameters(ALPHA, -0.001, NU, RHO).is_err());
        assert!(validate_sabr_parameters(ALPHA, 1.001, NU, RHO).is_err());
        assert!(validate_sabr_parameters(ALPHA, BETA, -0.001, RHO).is_err());
        assert!(validate_sabr_parameters(ALPHA, BETA, NU, 1.0).is_err());
        assert!(validate_sabr_parameters(ALPHA, BETA, NU, -1.0).is_err());
    }

    #[test]
    fn sabr_volatility_rejects_bad_inputs() {
        for (strike, forward, expiry) in [
            (0.0, FORWARD, EXPIRY),
            (-0.01, FORWARD, EXPIRY),
            (0.04, 0.0, EXPIRY),
            (0.04, -0.01, EXPIRY),
            (0.04, FORWARD, -1.0),
        ] {
            assert!(
                sabr_volatility(
                    strike,
                    forward,
                    expiry,
                    ALPHA,
                    BETA,
                    NU,
                    RHO,
                    VolatilityType::ShiftedLognormal,
                )
                .is_err(),
                "expected Err for strike {strike}, forward {forward}, expiry {expiry}"
            );
        }
    }

    /// Oracle: QuantLib-Python 1.44 `sabrVolatility(..., ql.Normal)`
    /// (`tests/fixtures/sabr_normal/generate.py`). Rows 6-7 hit the
    /// `close(forward, strike)` branch (`sabr.cpp:101-106`), rows 8-9 the
    /// `z * z <= 10 eps` expansion (`:124-128`) at a non-zero `z`, and row 10 a
    /// zero expiry. Every row is within one ulp of QuantLib except row 4
    /// (`beta = 1`, `rho = -0.5`), where `sqrt(B) + z - rho` cancels and the
    /// result lands 24 ulps (4.4e-15 rel) away on macOS arm64; that row's
    /// `pow`/`log` chain is libm- and contraction-dependent, so it is held to
    /// 1e-12 and the rest to 1e-14.
    #[test]
    fn normal_volatility_matches_quantlib_python() {
        let rows: [[Real; 8]; 10] = [
            [0.03, 0.04, 1.0, 0.02, 0.5, 0.4, -0.2, 0.004623798467824781],
            [0.05, 0.04, 2.5, 0.02, 0.5, 0.4, -0.2, 0.004551741184353164],
            [0.01, 0.035, 5.0, 0.01, 0.0, 0.3, 0.3, 0.010076137974416257],
            [
                0.06,
                0.035,
                0.5,
                0.05,
                1.0,
                0.6,
                -0.5,
                0.0047552190399932385,
            ],
            [
                0.025,
                0.03,
                10.0,
                0.015,
                0.3,
                0.25,
                0.1,
                0.005330869164134957,
            ],
            [0.04, 0.04, 1.0, 0.02, 0.5, 0.4, -0.2, 0.004044883333333333],
            [0.035, 0.035, 3.0, 0.01, 0.0, 0.3, 0.3, 0.010194624999999999],
            [
                0.03,
                0.04,
                1.0,
                0.02,
                0.5,
                1e-10,
                -0.2,
                0.003730704113398029,
            ],
            [0.045, 0.04, 2.0, 0.02, 0.7, 1e-9, 0.4, 0.002190362796104491],
            [0.05, 0.04, 0.0, 0.02, 0.5, 0.4, -0.2, 0.0044266002735856656],
        ];
        for (row, [strike, forward, expiry, alpha, beta, nu, rho, expected]) in
            rows.into_iter().enumerate()
        {
            let tolerance = if row == 3 { 1e-12 } else { 1e-14 };
            let vol = sabr_volatility(
                strike,
                forward,
                expiry,
                alpha,
                beta,
                nu,
                rho,
                VolatilityType::Normal,
            )
            .unwrap();
            let rel = ((vol - expected) / expected).abs();
            assert!(
                rel <= tolerance,
                "strike {strike}, forward {forward}: expected {expected}, got {vol}, rel {rel:e}"
            );
        }
    }
}
