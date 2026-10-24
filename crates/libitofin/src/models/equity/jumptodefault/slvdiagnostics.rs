//! Cross-asset Greeks of an SLV-JD option under common random numbers.
//!
//! The leverage surface stays fixed. Spot is bumped by one percent on each
//! side, leverage by one volatility point, and the jump intensity by ten
//! basis points.

use crate::errors::QlResult;
use crate::option::OptionType;
use crate::require;

use super::{SlvJtdCalibrationResult, defaultable_option_value, simulate_fixed_leverage};
use crate::types::{Real, Time};

/// Price and bumps of one European option on the defaultable stock.
#[derive(Clone, Debug)]
pub struct CrossAssetGreeks {
    /// Unbumped price.
    pub price: Real,
    /// Central spot delta.
    pub delta: Real,
    /// Central spot gamma.
    pub gamma: Real,
    /// Price change for a parallel `+0.01` leverage shift.
    pub vega_1pct: Real,
    /// Price change for a parallel `+10` basis-point intensity shift.
    pub cs01_10bps: Real,
    /// `Payoff at immediate default − price`. A call defaults to 0; a put to `K D(T)`.
    pub jtd_pnl: Real,
}

/// Delta, gamma, vega, CS01, and jump-to-default exposure of one option.
///
/// # Errors
///
/// Returns an error when the expiry is off the calibration grid or a bumped
/// path cannot be priced.
pub fn compute_cross_asset_greeks(
    calibration: &SlvJtdCalibrationResult,
    option_type: OptionType,
    strike: Real,
    expiry: Time,
) -> QlResult<CrossAssetGreeks> {
    require!(
        strike.is_finite() && strike > 0.0,
        "greek strike must be positive, got {strike}"
    );
    let spot = calibration.process.slv().s0().current_link()?.value()?;
    let bump = 0.01 * spot;
    let (time, spots, weights) = simulate_fixed_leverage(calibration, expiry, spot, 0.0, 0.0)?;
    let (_, spots_up, weights_up) =
        simulate_fixed_leverage(calibration, expiry, spot + bump, 0.0, 0.0)?;
    let (_, spots_down, weights_down) =
        simulate_fixed_leverage(calibration, expiry, spot - bump, 0.0, 0.0)?;
    let (_, spots_vol, weights_vol) =
        simulate_fixed_leverage(calibration, expiry, spot, 0.0, 0.01)?;
    let (_, spots_credit, weights_credit) =
        simulate_fixed_leverage(calibration, expiry, spot, 0.001, 0.0)?;
    let discount = calibration
        .process
        .slv()
        .risk_free_rate()
        .current_link()?
        .discount(time, true)?;
    let price = defaultable_option_value(option_type, strike, discount, &spots, &weights)?.0;
    let up = defaultable_option_value(option_type, strike, discount, &spots_up, &weights_up)?.0;
    let down =
        defaultable_option_value(option_type, strike, discount, &spots_down, &weights_down)?.0;
    let vol = defaultable_option_value(option_type, strike, discount, &spots_vol, &weights_vol)?.0;
    let credit = defaultable_option_value(
        option_type,
        strike,
        discount,
        &spots_credit,
        &weights_credit,
    )?
    .0;
    let jtd_payoff = match option_type {
        OptionType::Call => 0.0,
        OptionType::Put => strike * discount,
    };
    Ok(CrossAssetGreeks {
        price,
        delta: (up - down) / (2.0 * bump),
        gamma: (up - 2.0 * price + down) / (bump * bump),
        vega_1pct: vol - price,
        cs01_10bps: credit - price,
        jtd_pnl: jtd_payoff - price,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::models::equity::jumptodefault::{
        MomentGrid, SlvJtdParticleConfig, calibrate_slv_jump_to_default,
    };
    use crate::processes::HestonProcess;
    use crate::quotes::make_quote_handle;
    use crate::shared::{Shared, shared};
    use crate::termstructures::credit::defaulttermstructure::DefaultProbabilityTermStructure;
    use crate::termstructures::credit::flathazardrate::FlatHazardRate;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    #[test]
    fn deep_otm_put_greeks_have_the_structural_signs() {
        let today = Date::new(15, Month::June, 2026);
        let curve = |rate| {
            Handle::new(shared(FlatForward::with_rate(
                today,
                rate,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        let black = Handle::new(shared(BlackConstantVol::new(
            today,
            None,
            0.25,
            Actual365Fixed::new(),
        )) as Shared<dyn BlackVolTermStructure>);
        let heston = shared(HestonProcess::new(
            curve(0.03),
            curve(0.01),
            make_quote_handle(100.0).handle(),
            1.0,
            2.2,
            1.0,
            0.45,
            -0.55,
        ));
        let survival = Handle::new(shared(FlatHazardRate::with_rate(
            today,
            0.04,
            Actual365Fixed::new(),
        )) as Shared<dyn DefaultProbabilityTermStructure>);
        let config = SlvJtdParticleConfig {
            n_particles: 4_000,
            time_steps_per_year: 40,
            n_strike_knots: 21,
            moment_grid: MomentGrid::new(8.0, 16).unwrap(),
            ..SlvJtdParticleConfig::default()
        };
        let calibration =
            calibrate_slv_jump_to_default(black, heston, survival, -1.25, 1.0, 1.0, &config)
                .unwrap();
        let greeks = compute_cross_asset_greeks(&calibration, OptionType::Put, 70.0, 1.0).unwrap();
        assert!(greeks.price.is_finite() && greeks.price > 0.0);
        assert!(greeks.delta < 0.0, "delta {}", greeks.delta);
        assert!(greeks.gamma > 0.0, "gamma {}", greeks.gamma);
        assert!(greeks.vega_1pct > 0.0, "vega {}", greeks.vega_1pct);
        assert!(greeks.cs01_10bps > 0.0, "cs01 {}", greeks.cs01_10bps);
        assert!(greeks.jtd_pnl > 0.0, "jtd {}", greeks.jtd_pnl);
    }
}
