//! Python facade for the standardized-coordinate cubic mid-IV smile.

use crate::PyQlError;
use libitofin::termstructures::volatility::{CubicSmileSection, SmileSection};
use pyo3::prelude::*;
#[allow(unused_imports)]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A natural cubic interpolation of one expiry's mid implied volatilities.
///
/// Source strikes are mapped to signed standard-deviation log-moneyness using
/// the supplied forward, expiry time, and fixed ATM volatility. The default
/// sample grid is `[-3, -1.5, -1, -0.6, 0, 0.6, 1, 1.5, 3]`.
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
    /// Fit a natural cubic smile through paired strike/mid-IV observations.
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
    ///     std_dev_points (list[float] | None): Evaluation points for the
    ///         sampled curve. Defaults to `[-3, -1.5, -1, -0.6, 0, 0.6, 1,
    ///         1.5, 3]`; these are sample points, not spline knots.
    ///     extrapolate (bool): Extend the end cubic segments for out-of-range
    ///         queries and samples. Defaults to False.
    ///
    /// Raises:
    ///     ItofinError: If inputs are invalid or source points are duplicated.
    ///         Direct out-of-domain queries also fail unless extrapolation is enabled.
    #[new]
    #[pyo3(signature = (strikes, mid_ivs, forward, exercise_time, atm_vol, std_dev_points=None, extrapolate=false))]
    fn new(
        strikes: Vec<f64>,
        mid_ivs: Vec<f64>,
        forward: f64,
        exercise_time: f64,
        atm_vol: f64,
        std_dev_points: Option<Vec<f64>>,
        extrapolate: bool,
    ) -> PyResult<Self> {
        let mut inner = match std_dev_points {
            Some(points) => CubicSmileSection::with_std_dev_points(
                strikes,
                mid_ivs,
                forward,
                exercise_time,
                atm_vol,
                points,
            ),
            None => CubicSmileSection::new(strikes, mid_ivs, forward, exercise_time, atm_vol),
        }
        .map_err(PyQlError::from)?;
        inner = inner.with_extrapolation(extrapolate);
        Ok(Self { inner })
    }

    /// Return the fitted volatility at a strike.
    fn volatility(&self, strike: f64) -> PyResult<f64> {
        Ok(self.inner.volatility(strike).map_err(PyQlError::from)?)
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

    /// Volatilities sampled on `std_dev_points`; unavailable wings are None.
    #[getter]
    fn sampled_mid_ivs(&self) -> PyResult<Vec<Option<f64>>> {
        Ok(self.inner.sampled_mid_ivs().map_err(PyQlError::from)?)
    }

    /// Evaluation grid used by `sampled_mid_ivs`, in caller-specified order.
    #[getter]
    fn std_dev_points(&self) -> Vec<f64> {
        self.inner.std_dev_points().to_vec()
    }

    /// Standard-deviation coordinates of the sorted source observations.
    #[getter]
    fn node_std_dev_points(&self) -> Vec<f64> {
        self.inner.node_std_dev_points().to_vec()
    }

    /// Mid-IV observations paired with `node_std_dev_points`.
    #[getter]
    fn node_mid_ivs(&self) -> Vec<f64> {
        self.inner.node_mid_ivs().to_vec()
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

    /// Expiry time in years.
    #[getter]
    fn exercise_time(&self) -> f64 {
        self.inner.exercise_time()
    }

    /// Lowest observed strike.
    #[getter]
    fn min_strike(&self) -> f64 {
        self.inner.min_strike()
    }

    /// Highest observed strike.
    #[getter]
    fn max_strike(&self) -> f64 {
        self.inner.max_strike()
    }
}
