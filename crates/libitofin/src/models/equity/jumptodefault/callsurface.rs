//! Market call prices and strike derivatives for the Carr–Madan calibration.
//!
//! `C(K, T)` is the price of a call on the defaultable stock. `C_KK` is the
//! defective density of the surviving stock (working-paper text under eq. 4).
//! On a Black surface it is `D n(d2) g(k) / (K √w)`, with the same `g`
//! [`LocalVolSurface`](crate::termstructures::volatility::LocalVolSurface)
//! uses for Dupire local variance `∂_T w / g`.

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::pricingengines::black_formula;
use crate::quotes::Quote;
use crate::require;
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::{Real, Time};

/// Discounted calls and the strike derivatives the power moment needs.
pub trait CallSurface {
    /// Risk-free discount `D(t)`.
    fn discount(&self, t: Time) -> QlResult<Real>;

    /// Forward used to center the moment integral, in the coordinates of the
    /// density this surface quotes.
    fn forward(&self, t: Time) -> QlResult<Real>;

    /// Total Black variance at the forward. Sets the width of the moment grid.
    fn atm_total_variance(&self, t: Time) -> QlResult<Real>;

    /// Call price `C(K, t)`.
    fn call(&self, strike: Real, t: Time) -> QlResult<Real>;

    /// Strike derivative `C_K`.
    fn call_dk(&self, strike: Real, t: Time) -> QlResult<Real>;

    /// Second strike derivative `C_KK`, the discounted density.
    fn call_dkk(&self, strike: Real, t: Time) -> QlResult<Real>;
}

/// Calls on a Black volatility surface.
///
/// Strike derivatives are analytic in total variance `w(k, T)` and its
/// log-moneyness derivatives. Those derivatives use the same step as
/// [`LocalVolSurface`](crate::termstructures::volatility::LocalVolSurface).
/// Every variance query passes `extrapolate = true`, so strikes outside the
/// quoted grid follow that surface's own extrapolation: flat for
/// [`BlackConstantVol`](crate::termstructures::volatility::BlackConstantVol),
/// and the surface's strike mode for a variance surface. For `p < 0` the
/// power moment `M_p(0, T)` is dominated by that low-strike extrapolation.
pub struct BlackVolCallSurface {
    black: Handle<dyn BlackVolTermStructure>,
    risk_free: Handle<dyn YieldTermStructure>,
    dividend: Handle<dyn YieldTermStructure>,
    spot: Handle<dyn Quote>,
}

impl BlackVolCallSurface {
    /// Surface over a Black vol, the two yield curves, and the spot.
    pub fn new(
        black: Handle<dyn BlackVolTermStructure>,
        risk_free: Handle<dyn YieldTermStructure>,
        dividend: Handle<dyn YieldTermStructure>,
        spot: Handle<dyn Quote>,
    ) -> Self {
        Self {
            black,
            risk_free,
            dividend,
            spot,
        }
    }

    fn inputs(&self, strike: Real, t: Time) -> QlResult<CallInputs> {
        require!(
            strike.is_finite() && strike > 0.0,
            "call strike must be positive and finite, got {strike}"
        );
        require!(
            t.is_finite() && t > 0.0,
            "call maturity must be positive and finite, got {t}"
        );
        let discount = self.risk_free.current_link()?.discount(t, true)?;
        let dividend_discount = self.dividend.current_link()?.discount(t, true)?;
        require!(
            discount > 0.0,
            "call discount must be positive, got {discount}"
        );
        let spot = self.spot.current_link()?.value()?;
        require!(spot > 0.0, "call spot must be positive, got {spot}");
        let forward = spot * dividend_discount / discount;
        let black = self.black.current_link()?;
        let w = black.black_variance(t, strike, true)?;
        require!(
            w.is_finite() && w >= 0.0,
            "total variance must be non-negative at strike {strike} and time {t}, got {w}"
        );
        let y = (strike / forward).ln();
        // A vanished total variance is the intrinsic call. Far wings of an
        // inverted smile hit this when the price underflows, and the density
        // there is zero.
        let (wy, wyy) = if w == 0.0 {
            (0.0, 0.0)
        } else {
            log_moneyness_derivatives(&*black, strike, y, t, w)?
        };
        Ok(CallInputs {
            discount,
            forward,
            variance: w,
            log_moneyness: y,
            variance_y: wy,
            variance_yy: wyy,
        })
    }
}

impl CallSurface for BlackVolCallSurface {
    fn discount(&self, t: Time) -> QlResult<Real> {
        self.risk_free.current_link()?.discount(t, true)
    }

    fn forward(&self, t: Time) -> QlResult<Real> {
        let discount = self.discount(t)?;
        let dividend = self.dividend.current_link()?.discount(t, true)?;
        let spot = self.spot.current_link()?.value()?;
        require!(
            discount > 0.0 && spot > 0.0,
            "forward needs a positive spot and discount"
        );
        Ok(spot * dividend / discount)
    }

    fn atm_total_variance(&self, t: Time) -> QlResult<Real> {
        let forward = self.forward(t)?;
        self.black.current_link()?.black_variance(t, forward, true)
    }

    fn call(&self, strike: Real, t: Time) -> QlResult<Real> {
        let inputs = self.inputs(strike, t)?;
        black_call(&inputs, strike)
    }

    fn call_dk(&self, strike: Real, t: Time) -> QlResult<Real> {
        let inputs = self.inputs(strike, t)?;
        Ok(call_strike_derivative(&inputs))
    }

    fn call_dkk(&self, strike: Real, t: Time) -> QlResult<Real> {
        let inputs = self.inputs(strike, t)?;
        call_density(&inputs, strike)
    }
}

/// Closed-form calls of the constant-hazard, flat-volatility model.
///
/// `C = D(T) e^{-h T} Black(F e^{h T}, K, σ² T)` with `F = S e^{(r-q)T}`.
/// This is the market surface of Carr–Madan when `p = 0` and `f = h`. It is
/// the oracle for the moment and calibration tests, not a market adapter.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub struct ConstantHazardCallSurface {
    spot: Real,
    rate: Real,
    dividend: Real,
    volatility: Real,
    hazard: Real,
}

#[cfg(test)]
impl ConstantHazardCallSurface {
    /// Flat rates, dividend, volatility, and hazard `h`.
    ///
    /// # Errors
    ///
    /// Returns an error unless spot and volatility are positive and every
    /// input is finite. `hazard == 0` is the pure Black surface.
    pub fn new(
        spot: Real,
        rate: Real,
        dividend: Real,
        volatility: Real,
        hazard: Real,
    ) -> QlResult<Self> {
        require!(
            spot.is_finite() && spot > 0.0,
            "constant-hazard surface needs a positive spot, got {spot}"
        );
        require!(
            rate.is_finite() && dividend.is_finite() && hazard.is_finite(),
            "constant-hazard surface rates must be finite"
        );
        require!(
            volatility.is_finite() && volatility > 0.0,
            "constant-hazard surface needs a positive volatility, got {volatility}"
        );
        require!(
            hazard >= 0.0,
            "constant-hazard surface hazard must be non-negative, got {hazard}"
        );
        Ok(Self {
            spot,
            rate,
            dividend,
            volatility,
            hazard,
        })
    }

    fn inputs(&self, strike: Real, t: Time) -> QlResult<CallInputs> {
        require!(
            strike.is_finite() && strike > 0.0,
            "call strike must be positive and finite, got {strike}"
        );
        require!(
            t.is_finite() && t > 0.0,
            "call maturity must be positive and finite, got {t}"
        );
        let discount = (-self.rate * t).exp();
        let survival = (-self.hazard * t).exp();
        // The Black forward is the survival-conditional forward `F e^{h T}`.
        // Survival multiplies the price, so it rides in the discount slot of
        // the Black formula and cancels out of `d1`/`d2`.
        let forward = self.spot * ((self.rate - self.dividend + self.hazard) * t).exp();
        Ok(CallInputs {
            discount: discount * survival,
            forward,
            variance: self.volatility * self.volatility * t,
            log_moneyness: (strike / forward).ln(),
            variance_y: 0.0,
            variance_yy: 0.0,
        })
    }
}

#[cfg(test)]
impl CallSurface for ConstantHazardCallSurface {
    fn discount(&self, t: Time) -> QlResult<Real> {
        require!(
            t.is_finite() && t >= 0.0,
            "discount time must be non-negative, got {t}"
        );
        Ok((-self.rate * t).exp())
    }

    fn forward(&self, t: Time) -> QlResult<Real> {
        require!(
            t.is_finite() && t >= 0.0,
            "forward time must be non-negative, got {t}"
        );
        Ok(self.spot * ((self.rate - self.dividend + self.hazard) * t).exp())
    }

    fn atm_total_variance(&self, t: Time) -> QlResult<Real> {
        require!(
            t.is_finite() && t > 0.0,
            "variance time must be positive, got {t}"
        );
        Ok(self.volatility * self.volatility * t)
    }

    fn call(&self, strike: Real, t: Time) -> QlResult<Real> {
        black_call(&self.inputs(strike, t)?, strike)
    }

    fn call_dk(&self, strike: Real, t: Time) -> QlResult<Real> {
        Ok(call_strike_derivative(&self.inputs(strike, t)?))
    }

    fn call_dkk(&self, strike: Real, t: Time) -> QlResult<Real> {
        call_density(&self.inputs(strike, t)?, strike)
    }
}

/// Pieces of one Black call, shared by the market surface and the oracle.
struct CallInputs {
    discount: Real,
    forward: Real,
    variance: Real,
    log_moneyness: Real,
    variance_y: Real,
    variance_yy: Real,
}

fn black_call(inputs: &CallInputs, strike: Real) -> QlResult<Real> {
    black_formula(
        crate::option::OptionType::Call,
        strike,
        inputs.forward,
        inputs.variance.sqrt(),
        inputs.discount,
        0.0,
    )
}

/// `C_K = D (−N(d2) + n(d2) w_y / (2 √w))`, total derivative through the smile.
fn call_strike_derivative(inputs: &CallInputs) -> Real {
    if inputs.variance == 0.0 {
        return if inputs.log_moneyness < 0.0 {
            -inputs.discount
        } else {
            0.0
        };
    }
    let sqrt_w = inputs.variance.sqrt();
    let d2 = -inputs.log_moneyness / sqrt_w - 0.5 * sqrt_w;
    let phi = CumulativeNormalDistribution::standard();
    let skew = phi.derivative(d2) * inputs.variance_y / (2.0 * sqrt_w);
    inputs.discount * (-phi.value(d2) + skew)
}

/// Discounted density `D n(d2) g(y) / (K √w)`.
///
/// `g` is the Dupire denominator [`LocalVolSurface`](crate::termstructures::volatility::LocalVolSurface)
/// divides `∂_T w` by, not the squared form in Gatheral's notes. The two
/// agree for a flat smile. Using this `g` keeps `C_KK` consistent with that
/// surface, which is the `σ²_Dupire` the Carr–Madan correction subtracts from.
fn call_density(inputs: &CallInputs, strike: Real) -> QlResult<Real> {
    if inputs.variance == 0.0 {
        return Ok(0.0);
    }
    let sqrt_w = inputs.variance.sqrt();
    let d2 = -inputs.log_moneyness / sqrt_w - 0.5 * sqrt_w;
    let density = CumulativeNormalDistribution::standard().derivative(d2);
    let w = inputs.variance;
    let y = inputs.log_moneyness;
    let wy = inputs.variance_y;
    let g = 1.0 - y / w * wy
        + 0.25 * (-0.25 - 1.0 / w + y * y / (w * w)) * wy * wy
        + 0.5 * inputs.variance_yy;
    let value = inputs.discount * density * g / (strike * sqrt_w);
    require!(
        value.is_finite(),
        "call density is not finite at strike {strike}"
    );
    Ok(value)
}

/// Symmetric step in `y = ln(K/F)`, copied from `LocalVolSurface`.
fn log_moneyness_derivatives(
    black: &dyn BlackVolTermStructure,
    strike: Real,
    y: Real,
    t: Time,
    w: Real,
) -> QlResult<(Real, Real)> {
    let dy = if y.abs() > 0.001 {
        y * 0.0001
    } else {
        0.000001
    };
    let strikep = strike * dy.exp();
    let strikem = strike / dy.exp();
    let wp = black.black_variance(t, strikep, true)?;
    let wm = black.black_variance(t, strikem, true)?;
    let wy = (wp - wm) / (2.0 * dy);
    let wyy = (wp - 2.0 * w + wm) / (dy * dy);
    Ok((wy, wyy))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::interestrate::Compounding;
    use crate::quotes::SimpleQuote;
    use crate::shared::{Shared, shared};
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn assert_rel(actual: Real, expected: Real, tol: Real) {
        let scale = expected.abs().max(1e-8);
        let error = (actual - expected).abs() / scale;
        assert!(
            error <= tol,
            "{actual} vs {expected} (rel {error}, tol {tol})"
        );
    }

    fn reference() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn black_surface(vol: Real) -> BlackVolCallSurface {
        let dc = Actual365Fixed::new();
        let today = reference();
        let flat = |rate, day_counter| {
            Handle::new(shared(FlatForward::with_rate(
                today,
                rate,
                day_counter,
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        BlackVolCallSurface::new(
            Handle::new(shared(BlackConstantVol::new(today, None, vol, dc.clone()))
                as Shared<dyn BlackVolTermStructure>),
            flat(0.03, dc.clone()),
            flat(0.01, dc),
            Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
        )
    }

    #[test]
    fn black_constant_vol_matches_the_zero_hazard_oracle() {
        let vol = 0.2;
        let market = black_surface(vol);
        let oracle = ConstantHazardCallSurface::new(100.0, 0.03, 0.01, vol, 0.0).unwrap();
        let t = 1.25;
        assert_rel(
            market.discount(t).unwrap(),
            oracle.discount(t).unwrap(),
            1e-12,
        );
        assert_rel(
            market.forward(t).unwrap(),
            oracle.forward(t).unwrap(),
            1e-12,
        );
        for moneyness in [0.5, 0.8, 1.0, 1.3, 2.0] {
            let strike = 100.0 * moneyness;
            assert_rel(
                market.call(strike, t).unwrap(),
                oracle.call(strike, t).unwrap(),
                1e-8,
            );
            assert_rel(
                market.call_dk(strike, t).unwrap(),
                oracle.call_dk(strike, t).unwrap(),
                1e-8,
            );
            assert_rel(
                market.call_dkk(strike, t).unwrap(),
                oracle.call_dkk(strike, t).unwrap(),
                1e-8,
            );
        }
    }
}
