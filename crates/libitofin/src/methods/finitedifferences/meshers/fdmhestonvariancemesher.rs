//! 1-D mesher for the Heston variance coordinate.
//!
//! Port of `ql/methods/finitedifferences/meshers/fdmhestonvariancemesher.{hpp,cpp}`:
//! [`FdmHestonVarianceMesher`] builds a CIR grid from the inverse non-central
//! chi-square distribution of the variance, averaged over a handful of times
//! up to maturity, then estimates a scalar Black volatility from the
//! probability-weighted square root of that grid.
//!
//! `FdmHestonLocalVolatilityVarianceMesher` (the leverage-function branch) is
//! omitted with the rest of the local-vol Heston work: a null leverage function
//! makes that C++ class identical to this one, which is the path
//! [`FdHestonBarrierEngine`](crate::pricingengines::barrier::FdHestonBarrierEngine)
//! takes.

use crate::errors::QlResult;
use crate::math::distributions::noncentralchisquare::{
    InverseNonCentralCumulativeChiSquareDistribution, NonCentralCumulativeChiSquareDistribution,
};
use crate::math::distributions::{Probability, Quantile};
use crate::math::integrals::Integrator;
use crate::math::integrals::lobatto::GaussLobattoIntegral;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::linear::LinearInterpolation;
use crate::processes::HestonProcess;
use crate::types::{Real, Size, Time};
use crate::utilities::null::Null;

use super::fdm1dmesher::Fdm1dMesher;

/// Variance-direction mesher plus the Black-vol estimate the equity mesher
/// is built from (`fdmhestonvariancemesher.hpp:38`).
#[derive(Debug)]
pub struct FdmHestonVarianceMesher {
    mesher: Fdm1dMesher,
    vola_estimate: Real,
}

impl FdmHestonVarianceMesher {
    /// `FdmHestonVarianceMesher(size, process, maturity, tAvgSteps, epsilon, mixingFactor)`
    /// (`fdmhestonvariancemesher.cpp:52-149`).
    ///
    /// QuantLib defaults: `tAvgSteps = 10`, `epsilon = 0.0001`, `mixingFactor = 1`.
    ///
    /// Inverse-chi-square construction failures fall back to a uniform grid
    /// around `θ ± 4 vol`, matching C++'s `catch (const Error&)`.
    ///
    /// # Errors
    ///
    /// Fails if `size < 2`, `maturity` is not positive, the inverse CDF cannot
    /// be built on the fallback path, or the Gauss–Lobatto estimate fails.
    #[allow(
        clippy::too_many_arguments,
        clippy::float_cmp,
        clippy::neg_cmp_op_on_partial_ord
    )]
    pub fn new(
        size: Size,
        process: &HestonProcess,
        maturity: Time,
        t_avg_steps: Size,
        epsilon: Real,
        mixing_factor: Real,
    ) -> QlResult<Self> {
        crate::require!(size >= 2, "at least two variance points required");
        crate::require!(maturity > 0.0, "maturity must be positive");
        crate::require!(t_avg_steps >= 1, "tAvgSteps must be positive");

        let mixed_sigma = process.sigma() * mixing_factor;
        let (mut v_grid, mut p_grid) =
            match chi_square_grid(size, process, maturity, t_avg_steps, epsilon, mixed_sigma) {
                Ok(grids) => grids,
                Err(_) => fallback_grid(size, process, mixed_sigma),
            };

        let skew_hint = if process.kappa() != 0.0 {
            Real::max(1.0, mixed_sigma / process.kappa())
        } else {
            1.0
        };

        p_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let variance =
            LinearInterpolation::new(p_grid.clone(), v_grid.clone())?.with_extrapolation(true);
        let vola_estimate = GaussLobattoIntegral::new(100_000, 1e-4)?.integrate(
            |x| variance.value(x).map(|v| v.max(0.0).sqrt()).unwrap_or(0.0),
            p_grid[0],
            p_grid[p_grid.len() - 1],
        )? * skew_hint.powf(1.5);

        snap_v0(&mut v_grid, process.v0());

        let mut dplus = vec![0.0; size];
        let mut dminus = vec![0.0; size];
        for i in 0..size - 1 {
            let dx = v_grid[i + 1] - v_grid[i];
            dplus[i] = dx;
            dminus[i + 1] = dx;
        }
        dplus[size - 1] = Real::null();
        dminus[0] = Real::null();

        Ok(FdmHestonVarianceMesher {
            mesher: Fdm1dMesher::new(v_grid, dplus, dminus),
            vola_estimate,
        })
    }

    /// The 1-D variance grid.
    pub fn mesher(&self) -> &Fdm1dMesher {
        &self.mesher
    }

    /// Consumes the wrapper and returns the 1-D grid, for composite construction.
    pub fn into_mesher(self) -> Fdm1dMesher {
        self.mesher
    }

    /// Scalar Black-vol estimate `volaEstimate()` (`hpp:45`).
    pub fn vola_estimate(&self) -> Real {
        self.vola_estimate
    }
}

fn inverse_nccs(df: Real, ncp: Real, p: Real) -> QlResult<Real> {
    let inv = InverseNonCentralCumulativeChiSquareDistribution::new(df, ncp)?;
    inv.quantile(Probability::try_from(p)?)
}

fn chi_square_grid(
    size: Size,
    process: &HestonProcess,
    maturity: Time,
    t_avg_steps: Size,
    epsilon: Real,
    mixed_sigma: Real,
) -> QlResult<(Vec<Real>, Vec<Real>)> {
    let df = 4.0 * process.theta() * process.kappa() / (mixed_sigma * mixed_sigma);
    let mut grid: Vec<(Real, Real)> = Vec::with_capacity(size * t_avg_steps);

    for l in 1..=t_avg_steps {
        let t = (maturity * l as Real) / t_avg_steps as Real;
        let exp_kt = (-process.kappa() * t).exp();
        let ncp = 4.0 * process.kappa() * exp_kt / (mixed_sigma * mixed_sigma * (1.0 - exp_kt))
            * process.v0();
        let k = mixed_sigma * mixed_sigma * (1.0 - exp_kt) / (4.0 * process.kappa());

        let q_min = 0.0;
        let q_max = process.v0().max(k * inverse_nccs(df, ncp, 1.0 - epsilon)?);
        let min_v_step = (q_max - q_min) / (50.0 * size as Real);

        let mut p = 0.0;
        let mut v_tmp = q_min;
        grid.push((q_min, epsilon));

        for i in 1..size {
            let ps = (1.0 - epsilon - p) / (size - i) as Real;
            p += ps;
            let tmp = k * inverse_nccs(df, ncp, p)?;
            let vx = (v_tmp + min_v_step).max(tmp);
            p = NonCentralCumulativeChiSquareDistribution::new(df, ncp)?.value(vx / k);
            v_tmp = vx;
            grid.push((vx, p));
        }
    }

    crate::require!(
        grid.len() == size * t_avg_steps,
        "something wrong with the grid size"
    );

    grid.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut v_grid = vec![0.0; size];
    let mut p_grid = vec![0.0; size];
    for i in 0..size {
        let b = (i * grid.len()) / size;
        let e = ((i + 1) * grid.len()) / size;
        let n = (e - b) as Real;
        for pair in grid.iter().take(e).skip(b) {
            v_grid[i] += pair.0 / n;
            p_grid[i] += pair.1 / n;
        }
    }
    Ok((v_grid, p_grid))
}

fn fallback_grid(size: Size, process: &HestonProcess, mixed_sigma: Real) -> (Vec<Real>, Vec<Real>) {
    let vol = mixed_sigma * (process.theta() / (2.0 * process.kappa())).sqrt();
    let mean = process.theta();
    let upper_bound = (process.v0() + 4.0 * vol).max(mean + 4.0 * vol);
    let lower_bound = 0.0_f64.max((process.v0() - 4.0 * vol).min(mean - 4.0 * vol));

    let mut p_grid = vec![0.0; size];
    let mut v_grid = vec![0.0; size];
    for i in 0..size {
        p_grid[i] = i as Real / (size - 1) as Real;
        v_grid[i] = lower_bound + i as Real * (upper_bound - lower_bound) / (size - 1) as Real;
    }
    (v_grid, p_grid)
}

fn snap_v0(v_grid: &mut [Real], v0: Real) {
    for i in 1..v_grid.len() {
        if v_grid[i - 1] <= v0 && v_grid[i] >= v0 {
            if (v_grid[i - 1] - v0).abs() < (v_grid[i] - v0).abs() {
                v_grid[i - 1] = v0;
            } else {
                v_grid[i] = v0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::shared::{Shared, shared};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn process() -> HestonProcess {
        let dc = Actual365Fixed::new();
        let today = Date::new(5, Month::July, 2002);
        let r = Handle::new(shared(FlatForward::with_rate(
            today,
            0.04,
            dc.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let q = Handle::new(shared(FlatForward::with_rate(
            today,
            0.0,
            dc,
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        HestonProcess::new(
            r,
            q,
            Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
            0.04,
            1.0,
            0.04,
            0.3,
            -0.5,
        )
    }

    #[test]
    fn grid_contains_v0_and_a_finite_vol_estimate() {
        let p = process();
        let mesher = FdmHestonVarianceMesher::new(21, &p, 1.0, 10, 0.0001, 1.0).unwrap();
        assert_eq!(mesher.mesher().size(), 21);
        assert!(mesher.vola_estimate().is_finite() && mesher.vola_estimate() > 0.0);
        let locs = mesher.mesher().locations();
        assert!(locs.windows(2).all(|w| w[1] > w[0]), "locations={locs:?}");
        assert!(
            locs.iter().any(|&v| (v - p.v0()).abs() < 1e-14),
            "v0={} not on grid {locs:?}",
            p.v0()
        );
        assert_eq!(locs[0], 0.0);
        assert!(mesher.mesher().dminus(0).is_null());
        assert!(mesher.mesher().dplus(20).is_null());
    }

    #[test]
    fn size_must_be_at_least_two() {
        let err = FdmHestonVarianceMesher::new(1, &process(), 1.0, 5, 0.0001, 1.0).unwrap_err();
        assert!(err.message().contains("at least two"));
    }
}
