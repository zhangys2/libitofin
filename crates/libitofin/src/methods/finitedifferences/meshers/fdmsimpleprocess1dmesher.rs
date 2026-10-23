//! 1-D mesher from the quantile range of a 1-D stochastic process.
//!
//! Port of `ql/methods/finitedifferences/meshers/fdmsimpleprocess1dmesher.{hpp,cpp}`:
//! for each averaging horizon `t = maturity · ℓ / tAvgSteps`, the grid endpoints
//! are the process evolve of the `ε` / `1−ε` normal quantiles (clamped through
//! `x0` and an optional mandatory point), and interior nodes are the evolve of
//! the equally spaced probabilities in `(ε, 1−ε)`. Averaging over the horizons
//! yields the final locations.
//!
//! QuantLib's `FdG2SwaptionEngine` builds one of these per G2 factor with
//! `t_avg_steps = 1`.

use crate::errors::QlResult;
use crate::math::distributions::normal::InverseCumulativeNormal;
use crate::require;
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size, Time};
use crate::utilities::null::Null;

use super::fdm1dmesher::Fdm1dMesher;

/// Grid from the process quantile range over `[0, maturity]`.
///
/// Divergences from C++ at the API boundary only:
///
/// - the C++ `Fdm1dMesher` subclass constructor becomes a constructor function,
///   as for [`uniform_1d_mesher`](super::uniform_1d_mesher);
/// - `mandatory_point` is [`Option`] rather than a `Null<Real>` sentinel,
///   matching [`concentrating_1d_mesher`](super::concentrating_1d_mesher);
/// - the process is borrowed: the mesher only reads it while building the grid.
///
/// Defaults in C++ (`hpp:42-45`): `tAvgSteps = 10`, `epsilon = 0.0001`,
/// `mandatoryPoint = Null`.
///
/// # Errors
///
/// Fails unless `size >= 2`, `t_avg_steps >= 1`, and `0 < eps < 0.5`. Propagates
/// process / inverse-CDF failures.
pub fn fdm_simple_process_1d_mesher(
    size: Size,
    process: &dyn StochasticProcess1D,
    maturity: Time,
    t_avg_steps: Size,
    eps: Real,
    mandatory_point: Option<Real>,
) -> QlResult<Fdm1dMesher> {
    require!(size >= 2, "size must be at least 2");
    require!(t_avg_steps >= 1, "tAvgSteps must be at least 1");
    require!(eps > 0.0 && eps < 0.5, "eps must lie in (0, 1/2)");

    let x0 = process.x0()?;
    let mut locations = vec![0.0; size];

    for l in 1..=t_avg_steps {
        let t = (maturity * l as Real) / t_avg_steps as Real;
        let mp = mandatory_point.unwrap_or(x0);

        let q_min = mp.min(x0).min(process.evolve(
            0.0,
            x0,
            t,
            InverseCumulativeNormal::standard_value(eps)?,
        )?);
        let q_max = mp.max(x0).max(process.evolve(
            0.0,
            x0,
            t,
            InverseCumulativeNormal::standard_value(1.0 - eps)?,
        )?);

        let dp = (1.0 - 2.0 * eps) / (size - 1) as Real;
        let mut p = eps;
        locations[0] += q_min;

        for location in locations.iter_mut().take(size - 1).skip(1) {
            p += dp;
            *location += process.evolve(0.0, x0, t, InverseCumulativeNormal::standard_value(p)?)?;
        }
        locations[size - 1] += q_max;
    }

    for location in &mut locations {
        *location /= t_avg_steps as Real;
    }

    let mut dplus = vec![0.0; size];
    let mut dminus = vec![0.0; size];
    for i in 0..size - 1 {
        let gap = locations[i + 1] - locations[i];
        dplus[i] = gap;
        dminus[i + 1] = gap;
    }
    dplus[size - 1] = Real::null();
    dminus[0] = Real::null();

    Ok(Fdm1dMesher::new(locations, dplus, dminus))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::finitedifferences::meshers::{FdmMesher, FdmMesherComposite};
    use crate::processes::OrnsteinUhlenbeckProcess;

    fn ou(speed: Real, vol: Real) -> OrnsteinUhlenbeckProcess {
        OrnsteinUhlenbeckProcess::new(speed, vol, 0.0, 0.0).unwrap()
    }

    #[test]
    fn endpoints_match_quantile_evolve_when_t_avg_steps_is_one() {
        let process = ou(0.1, 0.01);
        let size = 11;
        let maturity = 5.0;
        let eps = 1e-4;
        let mesher = fdm_simple_process_1d_mesher(size, &process, maturity, 1, eps, None).unwrap();

        let x0 = process.x0().unwrap();
        let expected_min = process
            .evolve(
                0.0,
                x0,
                maturity,
                InverseCumulativeNormal::standard_value(eps).unwrap(),
            )
            .unwrap()
            .min(x0);
        let expected_max = process
            .evolve(
                0.0,
                x0,
                maturity,
                InverseCumulativeNormal::standard_value(1.0 - eps).unwrap(),
            )
            .unwrap()
            .max(x0);

        assert!((mesher.location(0) - expected_min).abs() < 1e-14);
        assert!((mesher.location(size - 1) - expected_max).abs() < 1e-14);
        assert!(mesher.dminus(0).is_null());
        assert!(mesher.dplus(size - 1).is_null());
    }

    #[test]
    fn locations_are_strictly_increasing_with_consistent_gaps() {
        let process = ou(0.2, 0.008);
        let size = 21;
        let mesher = fdm_simple_process_1d_mesher(size, &process, 10.0, 1, 1e-4, None).unwrap();

        for i in 0..size - 1 {
            let gap = mesher.location(i + 1) - mesher.location(i);
            assert!(gap > 0.0, "gap at {i} is not positive: {gap}");
            assert!((mesher.dplus(i) - gap).abs() < 1e-14);
            assert!((mesher.dminus(i + 1) - gap).abs() < 1e-14);
        }
    }

    #[test]
    fn averaging_over_horizons_matches_mean_of_per_t_grids() {
        let process = ou(0.1, 0.01);
        let size = 7;
        let maturity = 3.0;
        let t_avg = 4;
        let eps = 1e-3;

        let averaged =
            fdm_simple_process_1d_mesher(size, &process, maturity, t_avg, eps, None).unwrap();

        let mut mean = vec![0.0; size];
        for l in 1..=t_avg {
            let t = (maturity * l as Real) / t_avg as Real;
            let single = fdm_simple_process_1d_mesher(size, &process, t, 1, eps, None).unwrap();
            for (i, slot) in mean.iter_mut().enumerate() {
                *slot += single.location(i);
            }
        }
        for slot in &mut mean {
            *slot /= t_avg as Real;
        }

        for (i, expected) in mean.iter().enumerate() {
            assert!(
                (averaged.location(i) - expected).abs() < 1e-12,
                "node {i}: {} vs {expected}",
                averaged.location(i)
            );
        }
    }

    #[test]
    fn mandatory_point_extends_the_range() {
        let process = ou(0.1, 0.01);
        let size = 5;
        let maturity = 1.0;
        let eps = 1e-4;

        let plain = fdm_simple_process_1d_mesher(size, &process, maturity, 1, eps, None).unwrap();
        // Far outside the natural quantile range.
        let forced =
            fdm_simple_process_1d_mesher(size, &process, maturity, 1, eps, Some(0.5)).unwrap();

        assert!(forced.location(size - 1) >= 0.5 - 1e-14);
        assert!(forced.location(size - 1) > plain.location(size - 1));
    }

    #[test]
    fn two_ou_meshers_compose_into_fdg2_shaped_layout() {
        let x = fdm_simple_process_1d_mesher(9, &ou(0.1, 0.01), 5.0, 1, 1e-4, None).unwrap();
        let y = fdm_simple_process_1d_mesher(11, &ou(0.2, 0.008), 5.0, 1, 1e-4, None).unwrap();
        let composite = FdmMesherComposite::new(vec![x, y]);
        assert_eq!(composite.layout().dim(), &[9, 11]);
        assert_eq!(composite.layout().size(), 9 * 11);
    }

    #[test]
    fn rejects_degenerate_size_and_eps() {
        let process = ou(0.1, 0.01);
        assert!(fdm_simple_process_1d_mesher(1, &process, 1.0, 1, 1e-4, None).is_err());
        assert!(fdm_simple_process_1d_mesher(5, &process, 1.0, 0, 1e-4, None).is_err());
        assert!(fdm_simple_process_1d_mesher(5, &process, 1.0, 1, 0.0, None).is_err());
        assert!(fdm_simple_process_1d_mesher(5, &process, 1.0, 1, 0.5, None).is_err());
    }
}
