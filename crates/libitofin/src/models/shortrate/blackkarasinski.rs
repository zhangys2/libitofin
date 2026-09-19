//! Black–Karasinski short-rate model.
//!
//! Port of `ql/models/shortrate/onefactormodels/blackkarasinski.{hpp,cpp}`:
//! first rates Black–Karasinski gap slice covering construction (with curve
//! registration and an empty numerical `φ`) and
//! [`BlackKarasinskiDynamics`] (`r_t = e^{φ(t)+x_t}` with Ornstein–Uhlenbeck
//! `x`). QuantLib has no dedicated suite case; the dynamics identity is pinned
//! here. Numerical `tree()` / Brent `Helper` fitting and the fitting-driven
//! `dynamics()` path remain deferred.

use std::rc::Rc;

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::math::optimization::constraint::PositiveConstraint;
use crate::models::model::{
    register_with_term_structure, CalibratedModel, CalibratedModelHolder,
    TermStructureConsistentModel,
};
use crate::models::parameter::{
    ConstantParameter, NumericalImpl, Parameter, ParameterValue, TermStructureFittingParameter,
};
use crate::models::shortrate::onefactormodel::ShortRateDynamics;
use crate::patterns::observable::Observer;
use crate::processes::OrnsteinUhlenbeckProcess;
use crate::shared::{shared, shared_mut, Shared, SharedMut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::{Rate, Real, Time};

/// Standard Black–Karasinski model (`blackkarasinski.hpp`).
///
/// \f[ d\ln r_t = (\theta(t) - \alpha \ln r_t)\,dt + \sigma\,dW_t \f]
///
/// This slice covers construction and inspectors only; lattice `tree()` fit and
/// fitted `dynamics()` remain deferred.
pub struct BlackKarasinski {
    model: CalibratedModel,
    ts_model: TermStructureConsistentModel,
    /// Empty numerical fitting law (QL `phi_`); filled by deferred `tree()`.
    #[allow(dead_code)]
    phi: Parameter,
    #[allow(dead_code)]
    ts_observer: Option<SharedMut<dyn Observer>>,
}

impl BlackKarasinski {
    /// `BlackKarasinski(termStructure, a, sigma)`.
    ///
    /// Returns a [`SharedMut`] so the term-structure observer can be stashed
    /// after the model is shared (QL `registerWith(termStructure)`).
    ///
    /// # Errors
    ///
    /// Fails when `a` or `sigma` is not strictly positive.
    pub fn new(
        term_structure: Handle<dyn YieldTermStructure>,
        a: Real,
        sigma: Real,
    ) -> QlResult<SharedMut<Self>> {
        let mut model = CalibratedModel::new(2);
        model.arguments_mut()[0] = ConstantParameter::new(a, Rc::new(PositiveConstraint))?;
        model.arguments_mut()[1] = ConstantParameter::new(sigma, Rc::new(PositiveConstraint))?;
        let phi_impl = NumericalImpl::new(term_structure.clone());
        let phi = TermStructureFittingParameter::new(phi_impl as Rc<dyn ParameterValue>);
        let bk = Self {
            model,
            ts_model: TermStructureConsistentModel::new(term_structure.clone()),
            phi,
            ts_observer: None,
        };
        let shared = shared_mut(bk);
        let observer = register_with_term_structure(&shared, &term_structure);
        shared.borrow_mut().ts_observer = Some(observer);
        Ok(shared)
    }

    /// QuantLib defaults (`a = 0.1`, `sigma = 0.1`).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn with_defaults(
        term_structure: Handle<dyn YieldTermStructure>,
    ) -> QlResult<SharedMut<Self>> {
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

impl CalibratedModelHolder for BlackKarasinski {
    fn calibrated_model(&self) -> &CalibratedModel {
        &self.model
    }

    fn calibrated_model_mut(&mut self) -> &mut CalibratedModel {
        &mut self.model
    }
}

/// Short-rate dynamics in the Black–Karasinski model (`blackkarasinski.hpp`).
///
/// \f[ r_t = e^{\varphi(t) + x_t} \f]
/// with `x_t` an Ornstein–Uhlenbeck process of speed `α` and volatility `σ`.
///
/// Standalone construction takes a fitted `φ`; QL `model->dynamics()` first
/// runs `tree()` to fill `φ` and is deferred with that path.
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
    use crate::math::array::Array;
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
        assert!((model.borrow().a() - 0.1).abs() < 1e-15);
        assert!((model.borrow().sigma() - 0.1).abs() < 1e-15);
        assert!(BlackKarasinski::new(curve, 0.0, 0.1).is_err());
        assert!(BlackKarasinski::new(Handle::new(flat(0.05)), 0.1, -0.01).is_err());
    }

    #[test]
    fn set_params_updates_a_and_sigma() {
        let model = BlackKarasinski::with_defaults(Handle::new(flat(0.05))).unwrap();
        let params = Array::from(vec![0.2, 0.05]);
        model.borrow_mut().set_params(&params).unwrap();
        assert!((model.borrow().a() - 0.2).abs() < 1e-15);
        assert!((model.borrow().sigma() - 0.05).abs() < 1e-15);
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
        assert!((dynamics.process().drift(0.0, 0.05).unwrap() + 0.1 * 0.05).abs() < 1e-15);
    }
}
