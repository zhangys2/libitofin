//! Python bindings for the Black implied-standard-deviation solver.
//!
//! [`black_formula_implied_std_dev`] returns total standard deviation.
//! [`black_formula_implied_volatility`] divides that result by `sqrt(expiry)`.

use crate::PyQlError;
use crate::option::PyOptionType;
use libitofin::pricingengines::black_formula_implied_std_dev as core_implied_std_dev;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// Total Black standard deviation implied by an option price.
///
/// Binds the core `black_formula_implied_std_dev` solver. The result is
/// `volatility * sqrt(expiry)`, not an annualized volatility. `guess=None`
/// selects the core's Corrado-Miller approximation as the starting point.
///
/// Args:
///     option_type (OptionType): Call or put.
///     strike (float): Strike in the same currency as `forward`.
///     forward (float): Forward price of the underlying.
///     black_price (float): Discounted Black premium in that same currency.
///         The binding does not convert exchange quote units.
///     discount (float): Positive discount factor applied to the undiscounted
///         Black value. Defaults to 1.
///     displacement (float): Non-negative lognormal shift applied to both
///         strike and forward. Defaults to 0.
///     guess (float | None): Initial total standard deviation. None uses the
///         core approximation.
///     accuracy (float): Solver tolerance on the total standard deviation.
///         Defaults to 1e-8.
///     max_iterations (int): Maximum solver iterations. Defaults to 100.
///
/// Returns:
///     float: The implied total standard deviation.
///
/// Raises:
///     ItofinError: If an input is rejected by the core, or the solver does
///         not converge. Failures are not reported as NaN or a clamped value.
#[allow(clippy::too_many_arguments)]
#[gen_stub_pyfunction(module = "itofin.pricingengines")]
#[pyfunction]
#[pyo3(signature = (
    option_type,
    strike,
    forward,
    black_price,
    discount = 1.0,
    displacement = 0.0,
    guess = None,
    accuracy = 1e-8,
    max_iterations = 100
))]
pub(crate) fn black_formula_implied_std_dev(
    option_type: PyOptionType,
    strike: f64,
    forward: f64,
    black_price: f64,
    discount: f64,
    displacement: f64,
    guess: Option<f64>,
    accuracy: f64,
    max_iterations: u32,
) -> PyResult<f64> {
    Ok(core_implied_std_dev(
        option_type.inner(),
        strike,
        forward,
        black_price,
        discount,
        displacement,
        guess,
        accuracy,
        max_iterations,
    )
    .map_err(PyQlError::from)?)
}

/// Annualized Black volatility implied by an option price.
///
/// Calls `black_formula_implied_std_dev` with the core approximation seed
/// and returns `std_dev / sqrt(expiry)`. The result is a decimal volatility
/// (`0.65`, not `65`). `expiry` is a positive year fraction.
///
/// Args:
///     option_type (OptionType): Call or put.
///     strike (float): Strike in the same currency as `forward`.
///     forward (float): Forward price of the underlying.
///     expiry (float): Time to expiry as a positive year fraction.
///     black_price (float): Discounted Black premium in that same currency.
///         The binding does not convert exchange quote units.
///     discount (float): Positive discount factor. Defaults to 1.
///     displacement (float): Non-negative lognormal shift. Defaults to 0.
///     accuracy (float): Solver tolerance on the total standard deviation.
///         Defaults to 1e-8.
///     max_iterations (int): Maximum solver iterations. Must be positive.
///         Defaults to 100.
///
/// Returns:
///     float: Annualized implied volatility as a decimal.
///
/// Raises:
///     ItofinError: If `expiry` is not a positive finite year fraction, the
///         solver settings are invalid, a core input check fails, or the
///         solver does not converge.
#[allow(clippy::too_many_arguments)]
#[gen_stub_pyfunction(module = "itofin.pricingengines")]
#[pyfunction]
#[pyo3(signature = (
    option_type,
    strike,
    forward,
    expiry,
    black_price,
    discount = 1.0,
    displacement = 0.0,
    accuracy = 1e-8,
    max_iterations = 100
))]
pub(crate) fn black_formula_implied_volatility(
    option_type: PyOptionType,
    strike: f64,
    forward: f64,
    expiry: f64,
    black_price: f64,
    discount: f64,
    displacement: f64,
    accuracy: f64,
    max_iterations: i64,
) -> PyResult<f64> {
    if !expiry.is_finite() || expiry <= 0.0 {
        return Err(crate::ItofinError::new_err(format!(
            "expiry ({expiry}) must be a positive finite year fraction"
        )));
    }
    if !accuracy.is_finite() || accuracy <= 0.0 {
        return Err(crate::ItofinError::new_err(format!(
            "accuracy ({accuracy}) must be positive and finite"
        )));
    }
    if max_iterations <= 0 {
        return Err(crate::ItofinError::new_err(format!(
            "max_iterations ({max_iterations}) must be positive"
        )));
    }
    let max_iterations = u32::try_from(max_iterations).map_err(|_| {
        crate::ItofinError::new_err(format!(
            "max_iterations ({max_iterations}) must fit in a 32-bit unsigned integer"
        ))
    })?;
    let std_dev = black_formula_implied_std_dev(
        option_type,
        strike,
        forward,
        black_price,
        discount,
        displacement,
        None,
        accuracy,
        max_iterations,
    )?;
    Ok(std_dev / expiry.sqrt())
}
