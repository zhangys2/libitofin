//! Geometric Asian option (continuous average) analytic price.
//!
//! Kemna–Vorst via the Haug continuous-geometric formula implemented in
//! [`AnalyticContinuousGeometricAveragePriceAsianEngine`]. This helper uses
//! flat curves and a single day-count for all time fractions.

use crate::errors::QlResult;
use crate::option::OptionType;
use crate::pricingengines::BlackCalculator;
use crate::types::Real;

/// Continuous geometric-average price Asian (Haug 1997, flat curves).
pub fn geometric_average_price_asian(
    option_type: OptionType,
    spot: Real,
    strike: Real,
    r: Real,
    q: Real,
    vol: Real,
    t: Real,
) -> QlResult<Real> {
    let dividend_yield = 0.5 * (r + q + vol * vol / 6.0);
    let dividend_discount = (-dividend_yield * t).exp();
    let risk_free_discount = (-r * t).exp();
    let forward = spot * dividend_discount / risk_free_discount;
    let std_dev = (vol * vol * t / 3.0).sqrt();
    let black = BlackCalculator::new(option_type, strike, forward, std_dev, risk_free_discount)?;
    Ok(black.value())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::distributions::normal::CumulativeNormalDistribution;

    #[test]
    fn geometric_asian_call_is_below_vanilla_black_scholes() {
        let spot = 100.0;
        let strike = 100.0;
        let r = 0.05;
        let q = 0.0;
        let vol = 0.20;
        let t = 1.0;
        let asian =
            geometric_average_price_asian(OptionType::Call, spot, strike, r, q, vol, t).unwrap();
        let std = vol * t.sqrt();
        let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / std;
        let d2 = d1 - std;
        let n = CumulativeNormalDistribution::standard();
        let vanilla = spot * (-q * t).exp() * n.value(d1) - strike * (-r * t).exp() * n.value(d2);
        assert!(asian.is_finite() && asian > 0.0);
        assert!(asian < vanilla, "asian {asian} >= vanilla {vanilla}");
    }
}
