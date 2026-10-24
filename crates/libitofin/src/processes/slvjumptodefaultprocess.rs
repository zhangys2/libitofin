//! Heston SLV diffusion with a Carr–Madan jump to default.
//!
//! Prior to default the spot is the [`HestonSLVProcess`] Andersen
//! quadratic-exponential step with the compensator `λ(t, S)` added to the
//! log-drift. The survival factor of the step is the Rao–Blackwellized Cox
//! weight `exp(-λ Δt)`, with `λ = f(t) max(S, S_min)^p`.

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::models::equity::jumptodefault::PowerHazard;
use crate::processes::HestonSLVProcess;
use crate::require;
use crate::shared::Shared;
use crate::stochasticprocess::StochasticProcess;
use crate::types::{Real, Time};

/// Stochastic-local-volatility diffusion that can jump to zero.
pub struct SlvJumpToDefaultProcess {
    slv: Shared<HestonSLVProcess>,
    hazard: Shared<PowerHazard>,
    spot_floor: Real,
}

impl SlvJumpToDefaultProcess {
    /// Couples `slv` with the power hazard and a positive spot floor.
    ///
    /// # Errors
    ///
    /// Returns an error when `spot_floor` is non-positive or non-finite.
    pub fn new(
        slv: Shared<HestonSLVProcess>,
        hazard: Shared<PowerHazard>,
        spot_floor: Real,
    ) -> QlResult<Self> {
        require!(
            spot_floor.is_finite() && spot_floor > 0.0,
            "jump-to-default spot floor must be positive and finite, got {spot_floor}"
        );
        Ok(Self {
            slv,
            hazard,
            spot_floor,
        })
    }

    /// The stochastic-local-volatility diffusion.
    pub fn slv(&self) -> &Shared<HestonSLVProcess> {
        &self.slv
    }

    /// The power hazard `f(t) S^p`.
    pub fn hazard(&self) -> &Shared<PowerHazard> {
        &self.hazard
    }

    /// Floor applied to the spot inside the intensity.
    pub fn spot_floor(&self) -> Real {
        self.spot_floor
    }

    /// Intensity `f(t) max(spot, spot_floor)^p`.
    ///
    /// # Errors
    ///
    /// Returns an error when `spot` is non-positive or non-finite.
    pub fn intensity(&self, t: Time, spot: Real) -> QlResult<Real> {
        require!(
            spot.is_finite() && spot > 0.0,
            "jump-to-default intensity needs a positive finite spot, got {spot}"
        );
        self.hazard.intensity(t, spot.max(self.spot_floor))
    }

    /// One particle step: default-compensated spot, QE variance, survival factor.
    ///
    /// The variance and the diffusion piece of the spot are
    /// [`HestonSLVProcess::evolve`](StochasticProcess::evolve). Multiplying the
    /// resulting spot by `exp(λ Δt)` is the compensator. The third component is
    /// `exp(-λ Δt)`. A zero scale therefore returns survival `1` and the
    /// uncompensated SLV step.
    ///
    /// # Errors
    ///
    /// Returns an error when the state or the time step is not usable, or when
    /// the underlying SLV step fails.
    pub fn evolve_with_survival(
        &self,
        t0: Time,
        spot0: Real,
        var0: Real,
        dt: Time,
        dw_spot: Real,
        dw_var: Real,
    ) -> QlResult<(Real, Real, Real)> {
        require!(
            t0.is_finite() && t0 >= 0.0 && dt.is_finite() && dt > 0.0,
            "jump-to-default step needs a non-negative time and a positive step, got t={t0} dt={dt}"
        );
        require!(
            spot0.is_finite()
                && spot0 > 0.0
                && var0.is_finite()
                && dw_spot.is_finite()
                && dw_var.is_finite(),
            "jump-to-default step state must be finite"
        );
        let lambda = self.intensity(t0, spot0)?;
        let (next_spot, next_var) = if self.slv.mixing_factor() == 0.0 || self.slv.sigma() == 0.0 {
            // Andersen QE divides by the mixed vol-of-vol. The zero-vol-of-vol
            // limit is a deterministic variance and a one-factor log-Euler spot.
            self.evolve_without_vol_of_vol(t0, spot0, var0, dt, dw_spot)?
        } else {
            let evolved = self.slv.evolve(
                t0,
                &Array::from([spot0, var0]),
                dt,
                &Array::from([dw_spot, dw_var]),
            )?;
            (evolved[0], evolved[1])
        };
        require!(
            next_spot.is_finite() && next_spot > 0.0 && next_var.is_finite(),
            "jump-to-default step produced a non-finite state"
        );
        let survival = (-lambda * dt).exp();
        let next_spot = next_spot * (lambda * dt).exp();
        Ok((next_spot, next_var, survival))
    }

    /// Deterministic variance and a full spot shock. `dw_var` is unused: there
    /// is no variance Brownian when the mixed vol-of-vol is zero.
    fn evolve_without_vol_of_vol(
        &self,
        t0: Time,
        spot0: Real,
        var0: Real,
        dt: Time,
        dw_spot: Real,
    ) -> QlResult<(Real, Real)> {
        let kappa = self.slv.kappa();
        let theta = self.slv.theta();
        let next_var = if kappa == 0.0 {
            var0
        } else {
            theta + (var0 - theta) * (-kappa * dt).exp()
        };
        let leverage = self.slv.leverage_fct().local_vol(t0, spot0, true)?;
        let variance = 0.5 * (var0 + next_var) * leverage * leverage;
        let mu = self.carry(t0, dt)?;
        let next_spot = spot0
            * (mu * dt - 0.5 * variance * dt + (variance * dt).max(0.0).sqrt() * dw_spot).exp();
        Ok((next_spot, next_var))
    }

    fn carry(&self, t0: Time, dt: Time) -> QlResult<Real> {
        let risk_free = self.slv.risk_free_rate().current_link()?;
        let dividend = self.slv.dividend_yield().current_link()?;
        let dr0 = risk_free.discount(t0, true)?;
        let dr1 = risk_free.discount(t0 + dt, true)?;
        let dq0 = dividend.discount(t0, true)?;
        let dq1 = dividend.discount(t0 + dt, true)?;
        require!(
            dr0 > 0.0 && dr1 > 0.0 && dq0 > 0.0 && dq1 > 0.0,
            "jump-to-default step discounts must be positive"
        );
        Ok(-(dr1 / dr0).ln() / dt + (dq1 / dq0).ln() / dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::models::equity::jumptodefault::PowerHazard;
    use crate::processes::HestonProcess;
    use crate::quotes::make_quote_handle;
    use crate::shared::shared;
    use crate::stochasticprocess::StochasticProcess;
    use crate::termstructures::volatility::LocalConstantVol;
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn ref_date() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn flat_yield(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            ref_date(),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn process(scale: Real, exponent: Real, floor: Real) -> SlvJumpToDefaultProcess {
        let heston = shared(HestonProcess::new(
            flat_yield(0.03),
            flat_yield(0.01),
            make_quote_handle(100.0).handle(),
            0.04,
            1.2,
            0.04,
            0.3,
            -0.5,
        ));
        let leverage = shared(LocalConstantVol::new(
            ref_date(),
            1.0,
            Actual365Fixed::new(),
        ));
        let slv = shared(HestonSLVProcess::new(heston, leverage, 1.0));
        SlvJumpToDefaultProcess::new(
            slv,
            shared(PowerHazard::constant(scale, exponent).unwrap()),
            floor,
        )
        .unwrap()
    }

    #[test]
    fn rejects_a_non_positive_spot_floor() {
        let heston = shared(HestonProcess::new(
            flat_yield(0.03),
            flat_yield(0.01),
            make_quote_handle(100.0).handle(),
            0.04,
            1.0,
            0.04,
            0.2,
            0.0,
        ));
        let leverage = shared(LocalConstantVol::new(
            ref_date(),
            0.2,
            Actual365Fixed::new(),
        ));
        let slv = shared(HestonSLVProcess::new(heston, leverage, 1.0));
        let hazard = shared(PowerHazard::constant(0.0, 0.0).unwrap());
        assert!(
            SlvJumpToDefaultProcess::new(Shared::clone(&slv), Shared::clone(&hazard), 0.0).is_err()
        );
        assert!(SlvJumpToDefaultProcess::new(slv, hazard, -1.0).is_err());
    }

    #[test]
    fn zero_scale_matches_the_slv_step_with_unit_survival() {
        let process = process(0.0, -1.0, 5.0);
        let t0 = 0.4;
        let spot0 = 100.0;
        let var0 = 0.04;
        let dt = 0.1;
        let dw_spot = 0.15;
        let dw_var = -0.25;
        let (spot, variance, survival) = process
            .evolve_with_survival(t0, spot0, var0, dt, dw_spot, dw_var)
            .unwrap();
        let plain = process
            .slv()
            .evolve(
                t0,
                &Array::from([spot0, var0]),
                dt,
                &Array::from([dw_spot, dw_var]),
            )
            .unwrap();
        assert!((survival - 1.0).abs() <= 1e-14);
        assert!((spot - plain[0]).abs() <= 1e-14);
        assert!((variance - plain[1]).abs() <= 1e-14);
        assert!(process.intensity(t0, 0.0).is_err());
    }

    #[test]
    fn positive_scale_compensates_the_spot_and_the_survival_weight() {
        let floor = 10.0;
        let scale = 0.25;
        let exponent = -1.0;
        let process = process(scale, exponent, floor);
        let spot0 = 4.0;
        let dt = 0.2;
        let lambda = scale * floor.powf(exponent);
        assert!((process.intensity(0.3, spot0).unwrap() - lambda).abs() <= 1e-14);
        let (spot, _, survival) = process
            .evolve_with_survival(0.3, spot0, 0.04, dt, 0.1, -0.2)
            .unwrap();
        let plain = process
            .slv()
            .evolve(
                0.3,
                &Array::from([spot0, 0.04]),
                dt,
                &Array::from([0.1, -0.2]),
            )
            .unwrap();
        let expected_survival = (-lambda * dt).exp();
        assert!((survival - expected_survival).abs() <= 1e-14);
        assert!((spot - plain[0] * (lambda * dt).exp()).abs() <= 1e-12 * spot.abs().max(1.0));
    }

    #[test]
    fn zero_mixing_is_finite_and_the_variance_is_deterministic() {
        let heston = shared(HestonProcess::new(
            flat_yield(0.03),
            flat_yield(0.01),
            make_quote_handle(100.0).handle(),
            0.09,
            1.2,
            0.04,
            0.3,
            -0.5,
        ));
        let leverage = shared(LocalConstantVol::new(
            ref_date(),
            1.0,
            Actual365Fixed::new(),
        ));
        let slv = shared(HestonSLVProcess::new(heston, leverage, 0.0));
        let process = SlvJumpToDefaultProcess::new(
            slv,
            shared(PowerHazard::constant(0.0, 0.0).unwrap()),
            1.0,
        )
        .unwrap();
        let dt = 0.25;
        let var0 = 0.09;
        let (spot, variance, survival) = process
            .evolve_with_survival(0.2, 100.0, var0, dt, 0.3, -0.4)
            .unwrap();
        let expected = 0.04 + (var0 - 0.04) * (-1.2 * dt).exp();
        assert!(spot.is_finite() && spot > 0.0, "spot {spot}");
        assert!(
            (variance - expected).abs() <= 1e-12,
            "{variance} vs {expected}"
        );
        assert!((survival - 1.0).abs() <= 1e-14);
    }
}
