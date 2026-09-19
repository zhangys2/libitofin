//! Black–Karasinski short-rate model.
//!
//! Port of `ql/models/shortrate/onefactormodels/blackkarasinski.{hpp,cpp}`:
//! first rates Black–Karasinski gap slice covering construction and
//! [`BlackKarasinskiDynamics`] (`r_t = e^{φ(t)+x_t}` with Ornstein–Uhlenbeck
//! `x`). QuantLib has no dedicated `blackkarasinski.cpp` suite; the dynamics
//! identity is pinned here. Numerical `tree()` / Brent `Helper` fitting and
//! the fitting-driven `dynamics()` path remain deferred.

use std::rc::Rc;

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::math::optimization::constraint::PositiveConstraint;
use crate::models::model::{CalibratedModel, TermStructureConsistentModel};
use crate::models::parameter::{ConstantParameter, Parameter};
use crate::models::shortrate::onefactormodel::ShortRateDynamics;
use crate::processes::OrnsteinUhlenbeckProcess;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::{Rate, Real, Time};

/// Standard Black–Karasinski model (`blackkarasinski.hpp`).
///
/// \f[ d\ln r_t = (\theta(t) - \alpha \ln r_t)\,dt + \sigma\,dW_t \f]
pub struct BlackKarasinski {
    model: CalibratedModel,
    ts_model: TermStructureConsistentModel,
}

impl BlackKarasinski {
    /// `BlackKarasinski(termStructure, a, sigma)`.
    ///
    /// # Errors
    ///
    /// Fails when `a` or `sigma` is not strictly positive.
    pub fn new(
        term_structure: Handle<dyn YieldTermStructure>,
        a: Real,
        sigma: Real,
    ) -> QlResult<Self> {
        let mut model = CalibratedModel::new(2);
        model.arguments_mut()[0] = ConstantParameter::new(a, Rc::new(PositiveConstraint))?;
        model.arguments_mut()[1] = ConstantParameter::new(sigma, Rc::new(PositiveConstraint))?;
        Ok(Self {
            model,
            ts_model: TermStructureConsistentModel::new(term_structure),
        })
    }

    /// QuantLib defaults (`a = 0.1`, `sigma = 0.1`).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn with_defaults(term_structure: Handle<dyn YieldTermStructure>) -> QlResult<Self> {
        Self::new(term_structure, 0.1, 0.1)
    }

    /// Mean-reversion speed `α` (`a()`).
    pub fn a(&self) -> Real {
        self.model.arguments()[0].value(0.0)
    }

    /// Volatility `σ`.
    pub fn sigma(&self) -> Real {
        self.model.arguments()[1].value(0.0)
    }

    /// Fitted yield curve handle.
    pub fn term_structure(&self) -> &Handle<dyn YieldTermStructure> {
        self.ts_model.term_structure()
    }
}

/// Short-rate dynamics in the Black–Karasinski model (`blackkarasinski.hpp`).
///
/// \f[ r_t = e^{\varphi(t) + x_t} \f]
/// with `x_t` an Ornstein–Uhlenbeck process of speed `α` and volatility `σ`.
pub struct BlackKarasinskiDynamics {
    process: Shared<dyn StochasticProcess1D>,
    fitting: Parameter,
}

impl BlackKarasinskiDynamics {
    /// `Dynamics(Parameter fitting, Real alpha, Real sigma)`.
    ///
    /// # Errors
    ///
    /// Fails when `sigma` is negative (Ornstein–Uhlenbeck constraint).
    pub fn new(fitting: Parameter, alpha: Real, sigma: Real) -> QlResult<Self> {
        Ok(Self {
            process: shared(OrnsteinUhlenbeckProcess::new(alpha, sigma, 0.0, 0.0)?)
                as Shared<dyn StochasticProcess1D>,
            fitting,
        })
    }
}

impl ShortRateDynamics for BlackKarasinskiDynamics {
    fn variable(&self, t: Time, r: Rate) -> Real {
        r.ln() - self.fitting.value(t)
    }

    fn short_rate(&self, t: Time, x: Real) -> Rate {
        (x + self.fitting.value(t)).exp()
    }

    fn process(&self) -> Shared<dyn StochasticProcess1D> {
        Shared::clone(&self.process)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interestrate::Compounding;
    use crate::models::parameter::{NumericalImpl, ParameterValue, TermStructureFittingParameter};
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn flat(rate: Rate) -> Shared<dyn YieldTermStructure> {
        shared(FlatForward::with_rate(
            Date::new(19, Month::May, 2026),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>
    }

    #[test]
    fn defaults_and_constructors() {
        let curve = Handle::new(flat(0.05));
        let model = BlackKarasinski::with_defaults(curve.clone()).unwrap();
        assert!((model.a() - 0.1).abs() < 1e-15);
        assert!((model.sigma() - 0.1).abs() < 1e-15);
        assert!(BlackKarasinski::new(curve, 0.0, 0.1).is_err());
        assert!(BlackKarasinski::new(Handle::new(flat(0.05)), 0.1, -0.01).is_err());
    }

    /// `blackkarasinski.hpp` Dynamics: `shortRate` / `variable` are inverses.
    #[test]
    fn dynamics_log_transform_round_trips() {
        let curve = Handle::new(flat(0.05));
        let phi_impl = NumericalImpl::new(curve);
        phi_impl.set(1.0, -0.5);
        let phi = TermStructureFittingParameter::new(phi_impl as Rc<dyn ParameterValue>);
        let dynamics = BlackKarasinskiDynamics::new(phi, 0.1, 0.01).unwrap();

        let r = 0.05;
        let x = dynamics.variable(1.0, r);
        assert!((x - (r.ln() + 0.5)).abs() < 1e-15);
        assert!((dynamics.short_rate(1.0, x) - r).abs() < 1e-15);
        assert_eq!(dynamics.process().x0().unwrap(), 0.0);
        assert!((dynamics.process().diffusion(0.0, 0.0).unwrap() - 0.01).abs() < 1e-15);
    }
}
