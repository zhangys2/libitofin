//! Python bindings for implied forward calculation.

use crate::PyQlError;
use libitofin::termstructures::forward::{
    AmericanLevel, AmericanModelConfig, ForwardStatus, ImpliedForward, ImpliedForwardConfig,
    MarketConvention, OptionQuotePair,
};
use pyo3::prelude::*;
#[allow(unused_imports)]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

/// A two-sided option quote pair at a single strike.
#[gen_stub_pyclass]
#[pyclass(
    name = "OptionQuotePair",
    module = "itofin.termstructures",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyOptionQuotePair {
    pub inner: OptionQuotePair,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyOptionQuotePair {
    #[new]
    pub fn new(strike: f64, call_bid: f64, call_ask: f64, put_bid: f64, put_ask: f64) -> Self {
        Self {
            inner: OptionQuotePair::new(strike, call_bid, call_ask, put_bid, put_ask),
        }
    }

    #[getter]
    pub fn strike(&self) -> f64 {
        self.inner.strike
    }

    #[getter]
    pub fn call_bid(&self) -> f64 {
        self.inner.call_bid
    }

    #[getter]
    pub fn call_ask(&self) -> f64 {
        self.inner.call_ask
    }

    #[getter]
    pub fn put_bid(&self) -> f64 {
        self.inner.put_bid
    }

    #[getter]
    pub fn put_ask(&self) -> f64 {
        self.inner.put_ask
    }

    #[getter]
    pub fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    fn __repr__(&self) -> String {
        format!(
            "OptionQuotePair(strike={}, C=[{}, {}], P=[{}, {}])",
            self.inner.strike,
            self.inner.call_bid,
            self.inner.call_ask,
            self.inner.put_bid,
            self.inner.put_ask
        )
    }
}

/// Diagnostic health metrics for the calculated forward.
#[gen_stub_pyclass]
#[pyclass(
    name = "ForwardDiagnostics",
    module = "itofin.termstructures",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyForwardDiagnostics {
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub is_valid: bool,
    #[pyo3(get)]
    pub total_pairs_received: usize,
    #[pyo3(get)]
    pub pairs_used: usize,
    #[pyo3(get)]
    pub pairs_pruned: usize,
    #[pyo3(get)]
    pub is_crossed: bool,
    #[pyo3(get)]
    pub box_arbitrage_violations: usize,
    #[pyo3(get)]
    pub wls_rmse: f64,
    #[pyo3(get)]
    pub min_spread: f64,
    #[pyo3(get)]
    pub max_spread: f64,
}

#[pymethods]
impl PyForwardDiagnostics {
    fn __repr__(&self) -> String {
        format!(
            "ForwardDiagnostics(status={:?}, is_valid={}, pairs_used={})",
            self.status, self.is_valid, self.pairs_used
        )
    }
}

/// Result of the implied forward estimation.
#[gen_stub_pyclass]
#[pyclass(
    name = "ImpliedForwardResult",
    module = "itofin.termstructures",
    from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyImpliedForwardResult {
    #[pyo3(get)]
    pub forward: f64,
    #[pyo3(get)]
    pub forward_bid_strict: f64,
    #[pyo3(get)]
    pub forward_ask_strict: f64,
    #[pyo3(get)]
    pub forward_bid_robust: f64,
    #[pyo3(get)]
    pub forward_ask_robust: f64,
    #[pyo3(get)]
    pub discount_factor: f64,
    #[pyo3(get)]
    pub implied_carry_rate: Option<f64>,
    #[pyo3(get)]
    pub diagnostics: PyForwardDiagnostics,
    /// Estimator that produced an American forward: "corridor", "model_fast" or
    /// "model_accurate"; None for other conventions.
    #[pyo3(get)]
    pub tier: Option<String>,
    /// Independent signals about an American forward: "schedule_ignored", "latency_fallback",
    /// "non_converged", "pairs_dropped".
    #[pyo3(get)]
    pub flags: Vec<String>,
    /// Why the model path failed (corridor fallback) or its Accurate stage fell back to Fast,
    /// for example "carry_not_bracketed" or "no_implied_volatility"; None when the model ran
    /// cleanly or was not attempted. "non_converged" is set with it.
    #[pyo3(get)]
    pub model_failure: Option<String>,
    /// Solved continuous yield, or the residual carry beside a dividend schedule; None on the
    /// corridor tier and for other conventions.
    #[pyo3(get)]
    pub solved_carry: Option<f64>,
    /// Mean call minus put implied volatility at the solution (model tiers only).
    #[pyo3(get)]
    pub call_put_iv_gap: Option<f64>,
    /// Gap evaluations spent by the carry solve (model tiers only).
    #[pyo3(get)]
    pub iterations: Option<usize>,
    /// Accurate-pricer evaluations spent on correction (model tiers only).
    #[pyo3(get)]
    pub accurate_prices_used: Option<usize>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyImpliedForwardResult {
    #[getter]
    pub fn is_valid(&self) -> bool {
        self.diagnostics.is_valid
    }

    #[getter]
    pub fn pairs_used(&self) -> usize {
        self.diagnostics.pairs_used
    }

    #[getter]
    pub fn status(&self) -> String {
        self.diagnostics.status.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ImpliedForwardResult(forward={:.4}, df={:.6}, status={:?}, pairs_used={})",
            self.forward,
            self.discount_factor,
            self.diagnostics.status,
            self.diagnostics.pairs_used
        )
    }
}

/// Calculate implied forward price and discount factor from option quotes.
///
/// Args:
///     quotes: A list of OptionQuotePair instances or (strike, call_bid, call_ask, put_bid, put_ask) tuples.
///     expiry_years: Time to expiration in years.
///     convention: "european" (default), "crypto_inverse", or "american".
///     spot: Underlying spot price (optional; required for American options).
///     discount_factor: Pre-specified discount factor D = exp(-r*T) (optional; if None, fits jointly).
///     min_pairs: Minimum valid strike pairs required (default: 2).
///     max_pairs: Maximum number of ATM strike pairs used (default: 50).
///     spread_floor: Spread floor regularizer (default: 1e-4).
///     max_spot_deviation: Maximum relative deviation from spot (default: 0.10).
///     american_kappa: American ATM moneyness corridor multiplier of the corridor tier (default: 0.5).
///     atm_vol: Annualized ATM volatility sizing the American strike window; estimated from the
///         quotes when omitted and a discount factor is given, otherwise 0.30.
///     dividends: Discrete cash dividends as (time_years, amount) pairs on the time basis of
///         `expiry_years`, each time in (0, expiry_years] (American model path only).
///     level: "auto" (default), "fast" or "accurate". Auto uses the accurate level when
///         dividends are given. Fast ignores a dividend schedule and flags it.
///     carry_search_range: (low, high) interval searched for the solved carry per year.
///     carry_flag_range: (low, high) interval outside of which the status is
///         WarningImplausibleCarry; must lie inside the search range.
///     correction_passes: Premium-correction passes of the accurate level (default: one, or
///         two with dividends; 0 means fast).
///     corridor_kappa: Strike window of the model path in standard deviations (default: 1.0).
///     max_accurate_prices: Work budget of accurate-pricer evaluations (default: 100).
///
/// For the American convention with a `discount_factor` the quotes are de-Americanized before
/// the parity estimate (tier "model_fast" or "model_accurate"); otherwise, or if the model cannot
/// be solved, the model-free corridor estimate is returned (tier "corridor").
#[gen_stub_pyfunction(module = "itofin.termstructures")]
#[pyfunction]
#[pyo3(signature = (quotes, expiry_years, convention="european", spot=None, discount_factor=None, min_pairs=2, max_pairs=50, spread_floor=1e-4, max_spot_deviation=Some(0.10), american_kappa=0.5, atm_vol=None, dividends=None, level="auto", carry_search_range=(-0.10, 0.25), carry_flag_range=(-0.05, 0.15), correction_passes=None, corridor_kappa=1.0, max_accurate_prices=100))]
#[allow(clippy::too_many_arguments)]
pub fn implied_forward(
    quotes: &Bound<'_, PyAny>,
    expiry_years: f64,
    convention: &str,
    spot: Option<f64>,
    discount_factor: Option<f64>,
    min_pairs: usize,
    max_pairs: usize,
    spread_floor: f64,
    max_spot_deviation: Option<f64>,
    american_kappa: f64,
    atm_vol: Option<f64>,
    dividends: Option<Vec<(f64, f64)>>,
    level: &str,
    carry_search_range: (f64, f64),
    carry_flag_range: (f64, f64),
    correction_passes: Option<usize>,
    corridor_kappa: f64,
    max_accurate_prices: usize,
) -> PyResult<PyImpliedForwardResult> {
    let american_level = match level.to_lowercase().as_str() {
        "auto" => AmericanLevel::Auto,
        "fast" => AmericanLevel::Fast,
        "accurate" => AmericanLevel::Accurate,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown level: {level}. Expected 'auto', 'fast' or 'accurate'"
            )));
        }
    };

    let mut pairs = Vec::new();
    for item in quotes.try_iter()? {
        let item = item?;
        if let Ok(py_pair) = item.extract::<PyRef<PyOptionQuotePair>>() {
            pairs.push(py_pair.inner);
        } else if let Ok(tuple) = item.extract::<(f64, f64, f64, f64, f64)>() {
            pairs.push(OptionQuotePair::new(
                tuple.0, tuple.1, tuple.2, tuple.3, tuple.4,
            ));
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "each quote must be an OptionQuotePair or a (strike, call_bid, call_ask, put_bid, put_ask) tuple",
            ));
        }
    }

    let m_convention = match convention.to_lowercase().as_str() {
        "european" | "european_vanilla" | "europeanvanilla" => MarketConvention::EuropeanVanilla,
        "crypto" | "crypto_inverse" | "cryptoinverse" | "inverse" => {
            MarketConvention::CryptoInverse
        }
        "american" | "american_equity" | "americanequity" => {
            let s = spot.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "American equity convention requires spot price",
                )
            })?;
            MarketConvention::AmericanEquity { spot: s, atm_vol }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown market convention: {convention}. Expected 'european', 'crypto_inverse', or 'american'"
            )));
        }
    };

    let config = ImpliedForwardConfig {
        discount_factor,
        spot,
        min_pairs,
        max_pairs,
        spread_floor,
        max_spot_deviation,
        american_kappa,
        american: AmericanModelConfig {
            level: american_level,
            dividends: dividends.unwrap_or_default(),
            corridor_kappa,
            carry_search_range,
            carry_flag_range,
            correction_passes,
            max_accurate_prices,
            ..AmericanModelConfig::default()
        },
    };

    let res = ImpliedForward::calculate(&pairs, m_convention, &config, expiry_years)
        .map_err(PyQlError)?;

    let status_str = match res.diagnostics.status {
        ForwardStatus::Valid => "Valid",
        ForwardStatus::WarningCrossedSyntheticQuotes => "WarningCrossedSyntheticQuotes",
        ForwardStatus::WarningBoxSpreadArbitrage => "WarningBoxSpreadArbitrage",
        ForwardStatus::WarningImplausibleDiscountFactor => "WarningImplausibleDiscountFactor",
        ForwardStatus::WarningSpotDeviationExceeded => "WarningSpotDeviationExceeded",
        ForwardStatus::DegradedAmericanBounds => "DegradedAmericanBounds",
        ForwardStatus::WarningImplausibleCarry => "WarningImplausibleCarry",
        ForwardStatus::WarningModelFallback => "WarningModelFallback",
    };

    let is_valid = res.diagnostics.status == ForwardStatus::Valid;

    Ok(PyImpliedForwardResult {
        forward: res.forward,
        forward_bid_strict: res.forward_bid_strict,
        forward_ask_strict: res.forward_ask_strict,
        forward_bid_robust: res.forward_bid_robust,
        forward_ask_robust: res.forward_ask_robust,
        discount_factor: res.discount_factor,
        implied_carry_rate: res.implied_carry_rate,
        diagnostics: PyForwardDiagnostics {
            status: status_str.to_string(),
            is_valid,
            total_pairs_received: res.diagnostics.total_pairs_received,
            pairs_used: res.diagnostics.pairs_used,
            pairs_pruned: res.diagnostics.pairs_pruned,
            is_crossed: res.diagnostics.is_crossed,
            box_arbitrage_violations: res.diagnostics.box_arbitrage_violations,
            wls_rmse: res.diagnostics.wls_rmse,
            min_spread: res.diagnostics.min_spread,
            max_spread: res.diagnostics.max_spread,
        },
        tier: res.tier.map(|tier| tier.as_str().to_string()),
        flags: res.flags.names().into_iter().map(String::from).collect(),
        model_failure: res
            .model_failure
            .map(|failure| failure.as_str().to_string()),
        solved_carry: res.american.map(|a| a.solved_carry),
        call_put_iv_gap: res.american.map(|a| a.call_put_iv_gap),
        iterations: res.american.map(|a| a.iterations),
        accurate_prices_used: res.american.map(|a| a.accurate_prices_used),
    })
}
