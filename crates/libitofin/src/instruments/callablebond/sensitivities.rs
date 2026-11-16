//! Centered conventional-OAS sensitivities normalized by the dirty center price,
//! as QuantLib v1.44 does (`callablebond.cpp:362-369,397-404`).

use super::{CallableFixedRateBond, CallableZeroCouponBond};
use crate::errors::QlResult;
use crate::interestrate::Compounding;
use crate::pricingengines::bond::TreeCallableFixedRateBondEngine;
use crate::require;
use crate::time::daycounter::DayCounter;
use crate::time::frequency::Frequency;
use crate::types::{Real, Spread};

fn sensitivity(
    spread: Spread,
    bump: Real,
    convexity: bool,
    mut price: impl FnMut(Spread) -> QlResult<Real>,
    accrued: impl FnOnce() -> QlResult<Real>,
) -> QlResult<Real> {
    require!(spread.is_finite(), "OAS must be finite");
    require!(
        bump.is_finite() && bump > 0.0,
        "bump must be finite and positive"
    );
    let upper = spread + bump;
    let lower = spread - bump;
    require!(
        upper.is_finite() && lower.is_finite() && upper > spread && lower < spread,
        "OAS bump must be finite and representable"
    );
    let center = price(spread)?;
    let up = price(upper)?;
    let down = price(lower)?;
    require!(
        [center, up, down].iter().all(|value| value.is_finite()),
        "nonfinite sensitivity price"
    );
    let dirty = center + accrued()?;
    require!(dirty.is_finite(), "nonfinite sensitivity price");
    if dirty == 0.0 {
        return Ok(0.0);
    }
    let (numerator, denominator) = if convexity {
        (up + down - 2.0 * center, bump * bump * dirty)
    } else {
        (down - up, 2.0 * dirty * bump)
    };
    require!(
        numerator.is_finite() && denominator.is_finite() && denominator != 0.0,
        "sensitivity arithmetic overflow or underflow"
    );
    let result = numerator / denominator;
    require!(result.is_finite(), "nonfinite sensitivity result");
    Ok(result)
}

impl CallableFixedRateBond {
    /// Effective duration from centered conventional-OAS clean-price bumps.
    ///
    /// Uses `(P(s-h) - P(s+h)) / (2 D h)` with `D = P(s) + accrued`, the dirty
    /// center price, as QuantLib v1.44 does. Accrued is read after the three
    /// prices, so pricing errors come first. A zero `D` returns zero after
    /// evaluating both bumped prices. `None` selects a bump of `2e-4`.
    /// Uses the explicit immutable Hull-White engine without replacing the
    /// installed pricing engine or changing cached instrument results.
    ///
    /// # Errors
    ///
    /// Rejects nonfinite, nonpositive or unrepresentable bumps, pricing and
    /// spread-convention errors, and nonfinite or underflowed arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub fn effective_duration(
        &self,
        spread: Spread,
        engine: &TreeCallableFixedRateBondEngine,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: Option<Real>,
    ) -> QlResult<Real> {
        sensitivity(
            spread,
            bump.unwrap_or(2e-4),
            false,
            |value| {
                self.clean_price_oas(
                    value,
                    engine,
                    day_counter.clone(),
                    compounding,
                    frequency,
                    None,
                )
            },
            || self.bond().accrued_amount(None),
        )
    }
    /// Effective convexity from centered conventional-OAS clean-price bumps.
    ///
    /// Uses `(P(s+h) + P(s-h) - 2 P(s)) / (D h²)` with `D = P(s) + accrued`, the
    /// dirty center price, as QuantLib v1.44 does. Accrued is read after the
    /// three prices, so pricing errors come first. A zero `D` returns zero after
    /// evaluating both bumped prices. `None` selects a bump of `2e-4`.
    /// Uses the explicit immutable Hull-White engine without replacing the
    /// installed pricing engine or changing cached instrument results.
    ///
    /// # Errors
    ///
    /// Rejects nonfinite, nonpositive or unrepresentable bumps, pricing and
    /// spread-convention errors, and nonfinite or underflowed arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub fn effective_convexity(
        &self,
        spread: Spread,
        engine: &TreeCallableFixedRateBondEngine,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: Option<Real>,
    ) -> QlResult<Real> {
        sensitivity(
            spread,
            bump.unwrap_or(2e-4),
            true,
            |value| {
                self.clean_price_oas(
                    value,
                    engine,
                    day_counter.clone(),
                    compounding,
                    frequency,
                    None,
                )
            },
            || self.bond().accrued_amount(None),
        )
    }
}

impl CallableZeroCouponBond {
    /// Effective duration from centered conventional-OAS clean-price bumps.
    ///
    /// Uses `(P(s-h) - P(s+h)) / (2 D h)` with `D = P(s) + accrued`, the dirty
    /// center price, as QuantLib v1.44 does. Accrued is read after the three
    /// prices, so pricing errors come first. A zero `D` returns zero after
    /// evaluating both bumped prices. `None` selects a bump of `2e-4`.
    /// Uses the explicit immutable Hull-White engine without replacing the
    /// installed pricing engine or changing cached instrument results.
    ///
    /// # Errors
    ///
    /// Rejects nonfinite, nonpositive or unrepresentable bumps, pricing and
    /// spread-convention errors, and nonfinite or underflowed arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub fn effective_duration(
        &self,
        spread: Spread,
        engine: &TreeCallableFixedRateBondEngine,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: Option<Real>,
    ) -> QlResult<Real> {
        sensitivity(
            spread,
            bump.unwrap_or(2e-4),
            false,
            |value| {
                self.clean_price_oas(
                    value,
                    engine,
                    day_counter.clone(),
                    compounding,
                    frequency,
                    None,
                )
            },
            || self.bond().accrued_amount(None),
        )
    }
    /// Effective convexity from centered conventional-OAS clean-price bumps.
    ///
    /// Uses `(P(s+h) + P(s-h) - 2 P(s)) / (D h²)` with `D = P(s) + accrued`, the
    /// dirty center price, as QuantLib v1.44 does. Accrued is read after the
    /// three prices, so pricing errors come first. A zero `D` returns zero after
    /// evaluating both bumped prices. `None` selects a bump of `2e-4`.
    /// Uses the explicit immutable Hull-White engine without replacing the
    /// installed pricing engine or changing cached instrument results.
    ///
    /// # Errors
    ///
    /// Rejects nonfinite, nonpositive or unrepresentable bumps, pricing and
    /// spread-convention errors, and nonfinite or underflowed arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub fn effective_convexity(
        &self,
        spread: Spread,
        engine: &TreeCallableFixedRateBondEngine,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: Option<Real>,
    ) -> QlResult<Real> {
        sensitivity(
            spread,
            bump.unwrap_or(2e-4),
            true,
            |value| {
                self.clean_price_oas(
                    value,
                    engine,
                    day_counter.clone(),
                    compounding,
                    frequency,
                    None,
                )
            },
            || self.bond().accrued_amount(None),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fail;

    #[test]
    fn zero_center_still_evaluates_both_bumps() {
        let mut probes = Vec::new();
        assert_eq!(
            sensitivity(
                0.0,
                0.01,
                false,
                |s| {
                    probes.push(s);
                    Ok(s)
                },
                || Ok(0.0)
            )
            .unwrap(),
            0.0
        );
        assert_eq!(probes, vec![0.0, 0.01, -0.01]);
        assert!(
            sensitivity(
                0.0,
                0.01,
                true,
                |s| {
                    require!(s.is_finite() && s <= 0.0, "failed upper price");
                    Ok(0.0)
                },
                || Ok(0.0)
            )
            .is_err()
        );
    }

    #[test]
    fn dirty_center_normalizes_and_accrued_is_read_after_prices() {
        let price = |s: Spread| Ok(100.0 - 100.0 * s);
        let duration = sensitivity(0.0, 0.01, false, price, || Ok(2.0)).unwrap();
        assert_eq!(duration, 2.0 / (2.0 * 102.0 * 0.01));
        let convexity = sensitivity(
            0.0,
            0.01,
            true,
            |s: Spread| Ok(100.0 + s * s * 1e4),
            || Ok(2.0),
        )
        .unwrap();
        assert_eq!(convexity, 2.0 / (0.01 * 0.01 * 102.0));
        assert_eq!(
            sensitivity(0.0, 0.01, false, |_| Ok(-1.0), || Ok(1.0)).unwrap(),
            0.0
        );
        assert!(sensitivity(0.0, 0.01, false, |_| Ok(0.0), || Ok(1.0)).is_ok());
        assert!(
            sensitivity(
                0.0,
                0.01,
                false,
                |_| Ok(1.0),
                || -> QlResult<Real> { fail!("accrued") }
            )
            .is_err()
        );
        let error = sensitivity(
            0.0,
            0.01,
            false,
            |_| -> QlResult<Real> { fail!("price") },
            || -> QlResult<Real> { panic!("accrued read before prices") },
        );
        assert!(error.is_err());
        assert!(sensitivity(0.0, 0.01, false, |_| Ok(1.0), || Ok(f64::INFINITY)).is_err());
    }

    #[test]
    fn rejects_nonfinite_prices_and_arithmetic() {
        for convexity in [false, true] {
            for price in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert!(sensitivity(0.0, 0.01, convexity, |_| Ok(price), || Ok(0.0)).is_err());
            }
            assert!(
                sensitivity(
                    0.0,
                    f64::MIN_POSITIVE,
                    convexity,
                    |_| Ok(f64::MIN_POSITIVE),
                    || Ok(0.0)
                )
                .is_err()
            );
        }
        assert!(sensitivity(0.0, 0.01, true, |_| Ok(f64::MAX), || Ok(0.0)).is_err());
        assert!(
            sensitivity(
                0.0,
                0.01,
                false,
                |s| Ok(if s == 0.0 {
                    f64::MIN_POSITIVE
                } else {
                    s.signum()
                }),
                || Ok(0.0)
            )
            .is_err()
        );
    }
}
