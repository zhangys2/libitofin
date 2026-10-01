//! Python facade for the standardized-coordinate cubic mid-IV smile.

use crate::PyQlError;
use libitofin::termstructures::volatility::{
    CubicSmileSection, DEFAULT_STD_DEV_POINTS, SmileSection,
};
use pyo3::prelude::*;
#[allow(unused_imports)]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A curvature-regularized natural cubic fit of one expiry's mid implied volatilities.
///
/// Source strikes are mapped to signed standard-deviation log-moneyness using
/// the supplied forward, expiry time, and fixed ATM volatility. The default
/// nine spline knots are `[-3, -1.5, -1, -0.6, 0, 0.6, 1, 1.5, 3]`.
#[gen_stub_pyclass]
#[pyclass(
    name = "CubicSmileSection",
    unsendable,
    module = "itofin.termstructures"
)]
pub struct PyCubicSmileSection {
    inner: CubicSmileSection,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCubicSmileSection {
    /// Fit a natural cubic smile to paired strike/mid-IV observations.
    ///
    /// Args:
    ///     strikes (list[float]): Positive strikes, in any order.
    ///     mid_ivs (list[float]): Annualized decimal Black mid implied
    ///         volatilities paired with `strikes`. Duplicate strikes must be
    ///         consolidated by the caller.
    ///     forward (float): Positive expiry forward used as the ATM level.
    ///     exercise_time (float): Positive expiry time in years.
    ///     atm_vol (float): Positive annualized ATM volatility used to scale
    ///         log-moneyness into standard-deviation units.
    ///     std_dev_points (list[float] | None): Fixed spline-knot locations.
    ///         Defaults to `[-3, -1.5, -1, -0.6, 0, 0.6, 1, 1.5, 3]`;
    ///         input locations are sorted before fitting.
    ///     extrapolate (bool): Extend the end cubic segments for out-of-range
    ///         queries and samples. Defaults to False.
    ///     smoothing (float): Nonnegative weight on integrated squared curvature.
    ///         Defaults to 0.01. The data term is mean squared IV error. Positive
    ///         smoothing needs at least two distinct in-range observations;
    ///         zero disables regularization and requires full observation rank.
    ///         Unsampled wings are model-dependent, not measured quote IVs.
    ///
    /// Raises:
    ///     ItofinError: If inputs are invalid, source points are duplicated,
    ///         or the in-range data and penalty do not determine a full-rank fit.
    ///         Direct out-of-domain queries fail unless extrapolation is enabled.
    #[new]
    #[allow(clippy::too_many_arguments)] // Preserve the existing Python arguments plus smoothing.
    #[pyo3(signature = (strikes, mid_ivs, forward, exercise_time, atm_vol, std_dev_points=None, extrapolate=false, smoothing=0.01))]
    fn new(
        strikes: Vec<f64>,
        mid_ivs: Vec<f64>,
        forward: f64,
        exercise_time: f64,
        atm_vol: f64,
        std_dev_points: Option<Vec<f64>>,
        extrapolate: bool,
        smoothing: f64,
    ) -> PyResult<Self> {
        let mut inner = CubicSmileSection::with_smoothing(
            strikes,
            mid_ivs,
            forward,
            exercise_time,
            atm_vol,
            std_dev_points.unwrap_or_else(|| DEFAULT_STD_DEV_POINTS.to_vec()),
            smoothing,
        )
        .map_err(PyQlError::from)?;
        inner = inner.with_extrapolation(extrapolate);
        Ok(Self { inner })
    }

    /// Return the fitted volatility at a strike.
    fn volatility(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.volatility(strike).map_err(PyQlError::from)?)
    }

    /// Return the Black variance at a strike.
    ///
    /// Args:
    ///     strike (float): The strike the variance is read at.
    ///
    /// Returns:
    ///     float: The squared volatility times the exercise time.
    ///
    /// Raises:
    ///     ItofinError: On the same conditions volatility() reports.
    fn variance(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.variance(strike).map_err(PyQlError::from)?)
    }

    /// Return the fitted volatility at a signed standard-deviation point.
    fn volatility_at_std_dev(&self, point: f64) -> PyResult<f64> {
        Ok(self
            .inner
            .volatility_at_std_dev(point)
            .map_err(PyQlError::from)?)
    }

    /// Map a signed standard-deviation point back to its strike.
    fn strike_at_std_dev(&self, point: f64) -> PyResult<f64> {
        Ok(self
            .inner
            .strike_at_std_dev(point)
            .map_err(PyQlError::from)?)
    }

    /// Fitted volatilities at each configured knot location.
    #[getter]
    fn sampled_mid_ivs(&self) -> PyResult<Vec<Option<f64>>> {
        Ok(self.inner.sampled_mid_ivs().map_err(PyQlError::from)?)
    }

    /// Fixed spline knots, sorted in increasing order.
    #[getter]
    fn std_dev_points(&self) -> Vec<f64> {
        self.inner.std_dev_points().to_vec()
    }

    /// Fixed knot coordinates used by the fitted spline.
    #[getter]
    fn node_std_dev_points(&self) -> Vec<f64> {
        self.inner.node_std_dev_points().to_vec()
    }

    /// Fitted IV ordinates paired with `node_std_dev_points`.
    #[getter]
    fn node_mid_ivs(&self) -> Vec<f64> {
        self.inner.node_mid_ivs().to_vec()
    }

    /// Sorted source market strikes.
    #[getter]
    fn observed_strikes(&self) -> Vec<f64> {
        self.inner.observed_strikes().to_vec()
    }

    /// Standard-deviation coordinates of source market observations.
    #[getter]
    fn observed_std_dev_points(&self) -> Vec<f64> {
        self.inner.observed_std_dev_points().to_vec()
    }

    /// Source mid-IVs paired with `observed_strikes`.
    #[getter]
    fn observed_mid_ivs(&self) -> Vec<f64> {
        self.inner.observed_mid_ivs().to_vec()
    }

    /// Fitted-minus-observed IVs at source observations. Entries outside the
    /// fixed knot range are `None` because those observations are not fitted.
    #[getter]
    fn observation_residuals(&self) -> PyResult<Vec<Option<f64>>> {
        Ok(self
            .inner
            .observation_residuals()
            .map_err(PyQlError::from)?)
    }

    /// Per-segment `[a, b, c]` coefficients in the local polynomial basis.
    ///
    /// Entry `i` applies on `[x_i, x_{i+1}]`:
    /// `sigma(x) = sigma_i + a*dx + b*dx**2 + c*dx**3`, where `dx = x - x_i`.
    #[getter]
    fn segment_coefficients(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .segment_coefficients()
            .into_iter()
            .map(|[a, b, c]| (a, b, c))
            .collect()
    }

    /// Fitted-minus-ordinate residuals at the fixed knot locations.
    ///
    /// These are zero up to floating-point rounding. Use `observation_residuals`
    /// to inspect fit errors against source market quotes.
    #[getter]
    fn node_residuals(&self) -> PyResult<Vec<f64>> {
        Ok(self.inner.node_residuals().map_err(PyQlError::from)?)
    }

    /// Forward/ATM level used to standardize strikes.
    #[getter]
    fn forward(&self) -> f64 {
        self.inner.forward()
    }

    /// Fixed ATM volatility used to standardize strikes.
    #[getter]
    fn atm_vol(&self) -> f64 {
        self.inner.atm_vol()
    }

    /// Nonnegative curvature penalty weight; zero disables regularization.
    #[getter]
    fn smoothing(&self) -> f64 {
        self.inner.smoothing()
    }

    /// Expiry time in years.
    #[getter]
    fn exercise_time(&self) -> f64 {
        self.inner.exercise_time()
    }

    /// Strike mapped from the lower fitted knot-domain boundary.
    #[getter]
    fn min_strike(&self) -> f64 {
        self.inner.min_strike()
    }

    /// Strike mapped from the upper fitted knot-domain boundary.
    #[getter]
    fn max_strike(&self) -> f64 {
        self.inner.max_strike()
    }
}
