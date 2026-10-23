//! Survival-weighted truncated power prices.
//!
//! `M_p(K, T) = ∫_K^∞ s^p C_KK(s, T) ds`, computed by parts so only `C_K` is
//! sampled (working-paper eq. 9, up to the survival factor `V(T)` which this
//! quantity keeps). The integral is truncated to `ln F ± n_sd √w_ATM`. For
//! `p < 0` the `K → 0` moment is dominated by the low-strike end of that
//! window and by the surface's extrapolation there.

use crate::errors::QlResult;
use crate::math::integrals::gaussianquadratures::GaussianQuadrature;
use crate::require;
use crate::types::{Real, Time};

use super::callsurface::CallSurface;

/// Log-strike window and quadrature size for [`truncated_power_price`].
#[derive(Clone, Copy, Debug)]
pub struct MomentGrid {
    n_std_dev: Real,
    points: usize,
}

impl MomentGrid {
    /// Window of `n_std_dev` ATM standard deviations and `points` quadrature nodes.
    ///
    /// # Errors
    ///
    /// Returns an error unless `n_std_dev` is finite and positive and `points`
    /// is at least 4.
    pub fn new(n_std_dev: Real, points: usize) -> QlResult<Self> {
        require!(
            n_std_dev.is_finite() && n_std_dev > 0.0,
            "moment grid width must be a positive finite number of standard deviations, got {n_std_dev}"
        );
        require!(
            points >= 4,
            "moment grid needs at least 4 quadrature points, got {points}"
        );
        Ok(Self { n_std_dev, points })
    }

    /// Ten standard deviations and 64 nodes.
    ///
    /// # Errors
    ///
    /// Returns an error only if the default arguments fail [`new`](Self::new),
    /// which they do not.
    pub fn standard() -> QlResult<Self> {
        Self::new(10.0, 64)
    }

    fn window(&self, surface: &dyn CallSurface, t: Time) -> QlResult<(Real, Real)> {
        let forward = surface.forward(t)?;
        let variance = surface.atm_total_variance(t)?;
        require!(
            forward > 0.0 && variance.is_finite() && variance > 0.0,
            "moment grid needs a positive forward and total variance at time {t}"
        );
        let width = self.n_std_dev * variance.sqrt();
        let center = forward.ln();
        Ok((center - width, center + width))
    }
}

/// `M_p(K, T)` on the grid. A strike below the window starts at the window.
pub fn truncated_power_price(
    surface: &dyn CallSurface,
    exponent: Real,
    strike: Real,
    t: Time,
    grid: &MomentGrid,
) -> QlResult<Real> {
    require!(
        exponent.is_finite() && exponent <= 0.0,
        "power moment exponent must be finite and <= 0, got {exponent}"
    );
    require!(
        strike.is_finite() && strike >= 0.0,
        "power moment strike must be non-negative, got {strike}"
    );
    let (y_lo, y_hi) = grid.window(surface, t)?;
    let y_start = if strike == 0.0 {
        y_lo
    } else {
        strike.ln().max(y_lo)
    };
    integrate_power(surface, exponent, y_start, y_hi, t, grid.points)
}

/// `M_p(0, T)`, the power moment from the bottom of the grid.
pub fn power_price(
    surface: &dyn CallSurface,
    exponent: Real,
    t: Time,
    grid: &MomentGrid,
) -> QlResult<Real> {
    truncated_power_price(surface, exponent, 0.0, t, grid)
}

/// `∫_{e^{y0}}^{e^{y1}} s^p C_KK ds` by parts, including both boundary terms.
fn integrate_power(
    surface: &dyn CallSurface,
    exponent: Real,
    y0: Real,
    y1: Real,
    t: Time,
    points: usize,
) -> QlResult<Real> {
    if y0 >= y1 {
        return Ok(0.0);
    }
    let s0 = y0.exp();
    let s1 = y1.exp();
    let lower = s0.powf(exponent) * surface.call_dk(s0, t)?;
    let upper = s1.powf(exponent) * surface.call_dk(s1, t)?;
    let integral = composite_gauss(
        |y| {
            let weight = (exponent * y).exp();
            Ok(weight * surface.call_dk(y.exp(), t)?)
        },
        y0,
        y1,
        points,
    )?;
    Ok(upper - lower - exponent * integral)
}

/// Composite 16-point Gauss–Legendre on `[a, b]`. `points` is spread across panels.
fn composite_gauss<F>(mut f: F, a: Real, b: Real, points: usize) -> QlResult<Real>
where
    F: FnMut(Real) -> QlResult<Real>,
{
    const ORDER: usize = 16;
    let panels = (points / ORDER).max(1);
    let quad = GaussianQuadrature::legendre(ORDER)?;
    let width = (b - a) / panels as Real;
    let mut sum = 0.0;
    for panel in 0..panels {
        let left = a + panel as Real * width;
        let right = left + width;
        sum += gauss_panel(&quad, left, right, &mut f)?;
    }
    Ok(sum)
}

fn gauss_panel<F>(quad: &GaussianQuadrature, a: Real, b: Real, f: &mut F) -> QlResult<Real>
where
    F: FnMut(Real) -> QlResult<Real>,
{
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut sum = 0.0;
    let xs = quad.abscissas();
    let ws = quad.weights();
    for i in 0..quad.order() {
        sum += ws[i] * f(mid + half * xs[i])?;
    }
    Ok(sum * half)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::math::distributions::normal::CumulativeNormalDistribution;
    use crate::types::Real;

    use super::super::ConstantHazardCallSurface;

    fn assert_rel(actual: Real, expected: Real, tol: Real) {
        let scale = expected.abs().max(1e-12);
        let error = (actual - expected).abs() / scale;
        assert!(
            error <= tol,
            "{actual} vs {expected} (rel {error}, tol {tol})"
        );
    }

    fn oracle() -> ConstantHazardCallSurface {
        ConstantHazardCallSurface::new(100.0, 0.03, 0.01, 0.25, 0.0).unwrap()
    }

    /// `D F^p exp(½ p (p-1) σ² T) N(d2 + p σ √T)`.
    fn closed_form(
        surface: &ConstantHazardCallSurface,
        exponent: Real,
        strike: Real,
        t: Time,
    ) -> Real {
        let forward = surface.forward(t).unwrap();
        let variance = surface.atm_total_variance(t).unwrap();
        let sqrt_w = variance.sqrt();
        let d2 = (forward / strike).ln() / sqrt_w - 0.5 * sqrt_w;
        let discount = surface.discount(t).unwrap();
        let adjust = (0.5 * exponent * (exponent - 1.0) * variance).exp();
        let cdf = CumulativeNormalDistribution::standard().value(d2 + exponent * sqrt_w);
        discount * forward.powf(exponent) * adjust * cdf
    }

    #[test]
    fn power_moments_match_the_lognormal_formula() {
        let surface = oracle();
        let grid = MomentGrid::new(10.0, 128).unwrap();
        let t = 1.2;
        let forward = surface.forward(t).unwrap();
        for exponent in [0.0, -0.5, -1.5] {
            let from_zero = power_price(&surface, exponent, t, &grid).unwrap();
            let limit = {
                let variance = surface.atm_total_variance(t).unwrap();
                let adjust = (0.5 * exponent * (exponent - 1.0) * variance).exp();
                surface.discount(t).unwrap() * forward.powf(exponent) * adjust
            };
            assert_rel(from_zero, limit, 1e-6);
            for moneyness in [0.7, 1.0, 1.4] {
                let strike = forward * moneyness;
                let priced = truncated_power_price(&surface, exponent, strike, t, &grid).unwrap();
                assert_rel(priced, closed_form(&surface, exponent, strike, t), 1e-6);
            }
        }
    }

    /// The 6-standard-deviation window still moves the `p = -1.5` moment by
    /// about `2e-8` relative; from 8 onward the change is far below `1e-8`.
    #[test]
    fn widening_the_window_changes_the_moment_by_less_than_1e_8() {
        let surface = oracle();
        let t = 1.0;
        let moments: Vec<Real> = [6.0, 8.0, 10.0, 12.0]
            .into_iter()
            .map(|n_std_dev| {
                let grid = MomentGrid::new(n_std_dev, 192).unwrap();
                power_price(&surface, -1.5, t, &grid).unwrap()
            })
            .collect();
        let change = |i: usize| (moments[i + 1] - moments[i]).abs() / moments[i + 1].abs();
        let from_6 = change(0);
        let from_8 = change(1);
        let from_10 = change(2);
        assert!(
            from_6 < 5e-8
                && from_8 <= 1e-8
                && from_10 <= 1e-8
                && from_6 > from_8
                && from_8 > from_10,
            "relative changes 6→8 {from_6}, 8→10 {from_8}, 10→12 {from_10}, moments {moments:?}"
        );
    }
}
