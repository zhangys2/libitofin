//! 1-D mesher for the Heston variance coordinate.
//!
//! Port of `ql/methods/finitedifferences/meshers/fdmhestonvariancemesher.{hpp,cpp}`:
//! [`FdmHestonVarianceMesher`] builds a CIR grid from the inverse non-central
//! chi-square distribution of the variance, averaged over a handful of times
//! up to maturity, then estimates a scalar Black volatility from the
//! probability-weighted square root of that grid.
//!
//! [`FdmHestonLocalVolatilityVarianceMesher`] keeps that CIR v-grid and
//! multiplies `volaEstimate` by a path-averaged leverage when an SLV
//! `LocalVolTermStructure` is supplied (`cpp:148-211`).

use crate::errors::QlResult;
use crate::math::distributions::noncentralchisquare::{
    InverseNonCentralCumulativeChiSquareDistribution, NonCentralCumulativeChiSquareDistribution,
};
use crate::math::distributions::normal::InverseCumulativeNormal;
use crate::math::distributions::{Probability, Quantile};
use crate::math::integrals::Integrator;
use crate::math::integrals::lobatto::GaussLobattoIntegral;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::linear::LinearInterpolation;
use crate::processes::HestonProcess;
use crate::shared::Shared;
use crate::termstructures::volatility::LocalVolTermStructure;
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

/// CIR variance grid with a leverage-scaled Black-vol estimate
/// (`fdmhestonvariancemesher.hpp:51`).
///
/// The v-locations are identical to [`FdmHestonVarianceMesher`]. When
/// `leverage` is `None` the vol estimate is too; otherwise it is multiplied
/// by the running mean of `L(0,S0)` and the Gauss–Lobatto averages of
/// `L(t, F exp(x σ √t))` over a 50-point normal quantile grid (`cpp:168-210`).
#[derive(Debug)]
pub struct FdmHestonLocalVolatilityVarianceMesher {
    mesher: Fdm1dMesher,
    vola_estimate: Real,
}

impl FdmHestonLocalVolatilityVarianceMesher {
    /// `FdmHestonLocalVolatilityVarianceMesher(size, process, leverageFct, maturity, tAvgSteps, epsilon, mixingFactor)`.
    ///
    /// # Errors
    ///
    /// Propagates construction of the CIR mesher, the leverage queries, and
    /// the Gauss–Lobatto averages.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        size: Size,
        process: &HestonProcess,
        leverage: Option<Shared<dyn LocalVolTermStructure>>,
        maturity: Time,
        t_avg_steps: Size,
        epsilon: Real,
        mixing_factor: Real,
    ) -> QlResult<Self> {
        let inner = FdmHestonVarianceMesher::new(
            size,
            process,
            maturity,
            t_avg_steps,
            epsilon,
            mixing_factor,
        )?;
        let mut vola_estimate = inner.vola_estimate();
        if let Some(leverage) = leverage {
            vola_estimate *= path_averaged_leverage(
                process,
                leverage.as_ref(),
                maturity,
                t_avg_steps,
                epsilon,
                vola_estimate,
            )?;
        }
        Ok(Self {
            mesher: inner.into_mesher(),
            vola_estimate,
        })
    }

    /// The 1-D variance grid (same locations as the CIR mesher).
    pub fn mesher(&self) -> &Fdm1dMesher {
        &self.mesher
    }

    /// Consumes the wrapper and returns the 1-D grid.
    pub fn into_mesher(self) -> Fdm1dMesher {
        self.mesher
    }

    /// Scalar Black-vol estimate `volaEstimate()` (`hpp:60`).
    pub fn vola_estimate(&self) -> Real {
        self.vola_estimate
    }
}

/// Running-mean leverage that scales `volaEstimate` (`cpp:168-210`).
fn path_averaged_leverage(
    process: &HestonProcess,
    leverage: &dyn LocalVolTermStructure,
    maturity: Time,
    t_avg_steps: Size,
    epsilon: Real,
    vola_estimate: Real,
) -> QlResult<Real> {
    let s0 = process.s0().current_link()?.value()?;
    let mut acc_sum = leverage.local_vol(0.0, s0, true)?;
    let mut acc_n = 1.0;
    let r_ts = process.risk_free_rate().current_link()?;
    let q_ts = process.dividend_yield().current_link()?;
    let inv_n = InverseCumulativeNormal::standard();
    const S_AVG_STEPS: Size = 50;

    for l in 1..=t_avg_steps {
        let t = (maturity * l as Real) / t_avg_steps as Real;
        let vol = vola_estimate * (acc_sum / acc_n);
        let fwd = s0 * q_ts.discount(t, false)? / r_ts.discount(t, false)?;
        let mut u = vec![0.0; S_AVG_STEPS];
        let mut sig = vec![0.0; S_AVG_STEPS];
        for i in 0..S_AVG_STEPS {
            u[i] = epsilon + ((1.0 - 2.0 * epsilon) / (S_AVG_STEPS - 1) as Real) * i as Real;
            let x = inv_n.value(u[i])?;
            let f = fwd * (x * vol * t.sqrt()).exp();
            let lv = leverage.local_vol(t, f, true)?;
            sig[i] = lv * lv;
        }
        let interp = LinearInterpolation::new(u.clone(), sig)?.with_extrapolation(true);
        let leverage_avg = GaussLobattoIntegral::new(10_000, 1e-4)?.integrate(
            |x| interp.value(x).map(|v| v.max(0.0).sqrt()).unwrap_or(0.0),
            u[0],
            u[S_AVG_STEPS - 1],
        )? / (1.0 - 2.0 * epsilon);
        acc_sum += leverage_avg;
        acc_n += 1.0;
    }
    Ok(acc_sum / acc_n)
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

    /// `fdheston.cpp` `testFdmHestonVarianceMesher`: cached CIR locations.
    fn oracle_process() -> HestonProcess {
        let dc = Actual365Fixed::new();
        let today = Date::new(22, Month::February, 2018);
        let flat = |rate: Real| {
            Handle::new(shared(FlatForward::with_rate(
                today,
                rate,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        HestonProcess::new(
            flat(0.02),
            flat(0.02),
            Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
            0.09,
            1.0,
            0.09,
            0.2,
            -0.5,
        )
    }

    #[test]
    fn cir_locations_match_the_cached_heston_mesh() {
        let mesher =
            FdmHestonVarianceMesher::new(5, &oracle_process(), 1.0, 10, 0.0001, 1.0).unwrap();
        let expected = [0.0, 6.652314e-02, 9.000000e-02, 1.095781e-01, 2.563610e-01];
        for (i, &exp) in expected.iter().enumerate() {
            let loc = mesher.mesher().location(i);
            assert!((loc - exp).abs() < 1e-6, "location[{i}]={loc} vs {exp}");
        }
    }

    #[test]
    fn constant_leverage_scales_the_vol_estimate_exactly() {
        use crate::termstructures::volatility::{LocalConstantVol, LocalVolTermStructure};

        let process = oracle_process();
        let cir = FdmHestonVarianceMesher::new(5, &process, 1.0, 10, 0.0001, 1.0).unwrap();
        let lvol: Shared<dyn LocalVolTermStructure> = shared(LocalConstantVol::new(
            Date::new(22, Month::February, 2018),
            2.5,
            Actual365Fixed::new(),
        ));
        let slv = FdmHestonLocalVolatilityVarianceMesher::new(
            5,
            &process,
            Some(lvol),
            1.0,
            10,
            0.0001,
            1.0,
        )
        .unwrap();
        let expected = 2.5 * cir.vola_estimate();
        assert!(
            (slv.vola_estimate() - expected).abs() < 1e-6,
            "const SLV vol {} vs {expected}",
            slv.vola_estimate()
        );
        assert_eq!(slv.mesher().locations(), cir.mesher().locations());
    }

    /// C++ `ParableLocalVolatility`: `α ((S0 − S)² + 25)`.
    struct ParableLocalVolatility {
        base: crate::termstructures::TermStructureBase,
        s0: Real,
        alpha: Real,
    }

    impl ParableLocalVolatility {
        fn new(
            reference: Date,
            s0: Real,
            alpha: Real,
            dc: crate::time::daycounter::DayCounter,
        ) -> Self {
            Self {
                base: crate::termstructures::TermStructureBase::with_reference_date(
                    reference,
                    None,
                    Some(dc),
                ),
                s0,
                alpha,
            }
        }
    }

    impl crate::patterns::observable::AsObservable for ParableLocalVolatility {
        fn observable(&self) -> &crate::patterns::observable::Observable {
            self.base.observable()
        }
    }

    impl crate::termstructures::TermStructure for ParableLocalVolatility {
        fn base(&self) -> &crate::termstructures::TermStructureBase {
            &self.base
        }

        fn max_date(&self) -> Date {
            Date::max_date()
        }
    }

    impl crate::termstructures::volatility::VolatilityTermStructure for ParableLocalVolatility {
        fn business_day_convention(
            &self,
        ) -> crate::time::businessdayconvention::BusinessDayConvention {
            crate::time::businessdayconvention::BusinessDayConvention::Following
        }

        fn min_strike(&self) -> Real {
            0.0
        }

        fn max_strike(&self) -> Real {
            Real::MAX
        }
    }

    impl crate::termstructures::volatility::LocalVolTermStructure for ParableLocalVolatility {
        fn local_vol_impl(&self, _t: Time, s: Real) -> QlResult<crate::types::Volatility> {
            Ok(self.alpha * ((self.s0 - s).powi(2) + 25.0))
        }
    }

    #[test]
    fn parable_leverage_matches_the_cached_path_average() {
        use crate::termstructures::volatility::LocalVolTermStructure;

        let today = Date::new(22, Month::February, 2018);
        let process = oracle_process();
        let alpha = 0.01;
        let leverage: Shared<dyn LocalVolTermStructure> = shared(ParableLocalVolatility::new(
            today,
            100.0,
            alpha,
            Actual365Fixed::new(),
        ));
        let slv = FdmHestonLocalVolatilityVarianceMesher::new(
            5,
            &process,
            Some(Shared::clone(&leverage)),
            0.5,
            1,
            0.01,
            1.0,
        )
        .unwrap();
        let initial = FdmHestonVarianceMesher::new(5, &process, 0.5, 1, 0.01, 1.0)
            .unwrap()
            .vola_estimate();
        let leverage_avg = 0.455881 / (1.0 - 0.02);
        let l0 = leverage.local_vol(0.0, 100.0, true).unwrap();
        let expected = 0.5 * (leverage_avg + l0) * initial;
        assert!(
            (slv.vola_estimate() - expected).abs() < 0.001,
            "parable SLV vol {} vs {expected}",
            slv.vola_estimate()
        );
    }
}
