//! Merton-76 jump-diffusion process.
//!
//! Port of `ql/processes/merton76process.{hpp,cpp}`: GBS plus jump quotes.
//! Drift/diffusion fail as in C++. `apply` stays additive `x0 + dx` (`Real`
//! cannot `fail!`); `evolve`/`expectation` error through `drift`. The analytic
//! [`crate::pricingengines::JumpDiffusionEngine`] never evolves the process.

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::patterns::observable::{AsObservable, Observable, Observer, ResetThenNotify};
use crate::processes::BlackScholesMertonProcess;
use crate::quotes::Quote;
use crate::shared::{Shared, SharedMut, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::types::{Real, Time};

/// Merton-76 jump-diffusion process (`merton76process.hpp`).
pub struct Merton76Process {
    black: BlackScholesMertonProcess,
    jump_intensity: Handle<dyn Quote>,
    log_mean_jump: Handle<dyn Quote>,
    log_jump_volatility: Handle<dyn Quote>,
    observable: Shared<Observable>,
    _listener: SharedMut<ResetThenNotify>,
}

impl Merton76Process {
    /// `Merton76Process(stateVariable, dividendTS, riskFreeTS, blackVolTS,
    /// jumpInt, logJMean, logJVol)` with the default Euler discretization
    /// (unused: drift/diffusion fail).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state_variable: Handle<dyn Quote>,
        dividend_yield: Handle<dyn YieldTermStructure>,
        risk_free_rate: Handle<dyn YieldTermStructure>,
        black_vol: Handle<dyn BlackVolTermStructure>,
        jump_intensity: Handle<dyn Quote>,
        log_mean_jump: Handle<dyn Quote>,
        log_jump_volatility: Handle<dyn Quote>,
    ) -> Self {
        let black = BlackScholesMertonProcess::new(
            state_variable,
            dividend_yield,
            risk_free_rate,
            black_vol,
        );
        let observable = shared(Observable::new());
        let listener = ResetThenNotify::forwarding(Shared::clone(&observable));
        let observer = listener.clone() as SharedMut<dyn Observer>;
        black.observable().register_observer(&observer);
        jump_intensity.register_observer(&observer);
        log_mean_jump.register_observer(&observer);
        log_jump_volatility.register_observer(&observer);
        Self {
            black,
            jump_intensity,
            log_mean_jump,
            log_jump_volatility,
            observable,
            _listener: listener,
        }
    }

    /// Spot quote handle.
    pub fn state_variable(&self) -> Handle<dyn Quote> {
        self.black.state_variable()
    }

    /// Dividend-yield curve handle.
    pub fn dividend_yield(&self) -> Handle<dyn YieldTermStructure> {
        self.black.dividend_yield()
    }

    /// Risk-free curve handle.
    pub fn risk_free_rate(&self) -> Handle<dyn YieldTermStructure> {
        self.black.risk_free_rate()
    }

    /// Black-volatility curve handle.
    pub fn black_volatility(&self) -> Handle<dyn BlackVolTermStructure> {
        self.black.black_volatility()
    }

    /// Jump intensity λ.
    pub fn jump_intensity(&self) -> Handle<dyn Quote> {
        self.jump_intensity.clone()
    }

    /// Mean of the log jump size.
    pub fn log_mean_jump(&self) -> Handle<dyn Quote> {
        self.log_mean_jump.clone()
    }

    /// Volatility of the log jump size.
    pub fn log_jump_volatility(&self) -> Handle<dyn Quote> {
        self.log_jump_volatility.clone()
    }
}

impl AsObservable for Merton76Process {
    fn observable(&self) -> &Observable {
        &self.observable
    }
}

impl StochasticProcess1D for Merton76Process {
    fn x0(&self) -> QlResult<Real> {
        self.black.x0()
    }

    fn drift(&self, _t: Time, _x: Real) -> QlResult<Real> {
        fail!("Merton76Process does not implement drift")
    }

    fn diffusion(&self, _t: Time, _x: Real) -> QlResult<Real> {
        fail!("Merton76Process does not implement diffusion")
    }

    fn time(&self, date: &Date) -> QlResult<Time> {
        self.black.time(date)
    }
}
