//! Black–Scholes delta calculator for FX conventions.
//!
//! Port of `ql/pricingengines/blackdeltacalculator.{hpp,cpp}` (subset used by
//! Vanna/Volga engines: spot and forward delta, ATM strike helpers).

use crate::errors::QlResult;
use crate::fail;
use crate::math::distributions::normal::InverseCumulativeNormal;
use crate::option::OptionType;
use crate::quotes::{AtmType, DeltaType};
use crate::require;
use crate::types::{DiscountFactor, Real};

/// Black delta calculator (`blackdeltacalculator.hpp`).
#[derive(Clone, Copy, Debug)]
pub struct BlackDeltaCalculator {
    delta_type: DeltaType,
    f_discount: DiscountFactor,
    std_dev: Real,
    spot: Real,
    forward: Real,
    phi: Real,
    f_exp_pos: Real,
    f_exp_neg: Real,
}

impl BlackDeltaCalculator {
    /// `std_dev` is `vol * sqrt(T)`, not annualized volatility.
    pub fn new(
        option_type: OptionType,
        delta_type: DeltaType,
        spot: Real,
        d_discount: DiscountFactor,
        f_discount: DiscountFactor,
        std_dev: Real,
    ) -> QlResult<Self> {
        require!(spot > 0.0, "positive spot value required");
        require!(
            d_discount > 0.0,
            "positive domestic discount factor required"
        );
        require!(
            f_discount > 0.0,
            "positive foreign discount factor required"
        );
        require!(std_dev >= 0.0, "non-negative standard deviation required");
        let phi = match option_type {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        };
        let forward = spot * f_discount / d_discount;
        Ok(Self {
            delta_type,
            f_discount,
            std_dev,
            spot,
            forward,
            phi,
            f_exp_pos: forward * (0.5 * std_dev * std_dev).exp(),
            f_exp_neg: forward * (-0.5 * std_dev * std_dev).exp(),
        })
    }

    /// Inverts the Black delta to a strike (`strikeFromDelta`).
    pub fn strike_from_delta(&self, delta: Real) -> QlResult<Real> {
        require!(
            delta * self.phi >= 0.0,
            "option type and delta are incoherent"
        );
        match self.delta_type {
            DeltaType::Spot => {
                require!(delta.abs() <= self.f_discount, "spot delta out of range");
                let arg = -self.phi
                    * InverseCumulativeNormal::standard_value(self.phi * delta / self.f_discount)?
                    * self.std_dev
                    + 0.5 * self.std_dev * self.std_dev;
                Ok(self.forward * arg.exp())
            }
            DeltaType::Fwd => {
                require!(delta.abs() <= 1.0, "forward delta out of range");
                let arg = -self.phi
                    * InverseCumulativeNormal::standard_value(self.phi * delta)?
                    * self.std_dev
                    + 0.5 * self.std_dev * self.std_dev;
                Ok(self.forward * arg.exp())
            }
            DeltaType::PaSpot | DeltaType::PaFwd => {
                fail!("premium-adjusted delta inversion not ported")
            }
        }
    }

    /// ATM strike for the given convention (`atmStrike`).
    pub fn atm_strike(&self, atm_type: AtmType) -> QlResult<Real> {
        match atm_type {
            AtmType::Spot => Ok(self.spot),
            AtmType::Fwd => Ok(self.forward),
            AtmType::DeltaNeutral => Ok(
                if matches!(self.delta_type, DeltaType::Spot | DeltaType::Fwd) {
                    self.f_exp_pos
                } else {
                    self.f_exp_neg
                },
            ),
            AtmType::GammaMax | AtmType::VegaMax | AtmType::PutCall50 => Ok(self.f_exp_pos),
            AtmType::Null => fail!("invalid atm type"),
        }
    }
}
