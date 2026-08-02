//! Geometric Asian option (continuous average) analytic price.
//!
//! First exotic Asian slice: Kemna-Vorst continuous geometric-average European
//! option. Discrete arithmetic Asians remain follow-up.

use crate::errors::QlResult;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::types::Real;

/// Continuous geometric-average price Asian (Kemna–Vorst).
pub fn geometric_average_price_asian(
    option_type: OptionType,
    spot: Real,
    strike: Real,
    r: Real,
    q: Real,
    vol: Real,
    t: Real,
) -> QlResult<Real> {
    // Adjust drift/vol for the continuous geometric average.
    let vol_a = vol / 3.0_f64.sqrt();
    let b = 0.5 * (r + q + vol * vol / 6.0);
    let std = vol_a * t.sqrt();
    let d1 = ((spot / strike).ln() + (b + 0.5 * vol_a * vol_a) * t) / std;
    let d2 = d1 - std;
    let n = CumulativeNormalDistribution::standard();
    let df_r = (-r * t).exp();
    let forward_factor = ((b - r) * t).exp();
    Ok(match option_type {
        OptionType::Call => df_r * (spot * forward_factor * n.value(d1) - strike * n.value(d2)),
        OptionType::Put => df_r * (strike * n.value(-d2) - spot * forward_factor * n.value(-d1)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
