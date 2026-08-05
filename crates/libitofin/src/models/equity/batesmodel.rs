//! The Bates stochastic-volatility jump-diffusion calibrated model.
//!
//! Port of `ql/models/equity/batesmodel.{hpp,cpp}`: an 8-parameter
//! [`CalibratedModel`] wrapping a [`BatesProcess`]. Arguments are
//! `(theta, kappa, sigma, rho, v0, nu, delta, lambda)` (`batesmodel.cpp:25-37`):
//! the five Heston parameters plus mean jump size, jump-size volatility, and
//! jump intensity.

use std::rc::Rc;

use crate::errors::QlResult;
use crate::math::optimization::constraint::{BoundaryConstraint, NoConstraint, PositiveConstraint};
use crate::models::model::{CalibratedModel, CalibratedModelHolder, register_with_term_structure};
use crate::models::parameter::ConstantParameter;
use crate::patterns::observable::Observer;
use crate::processes::BatesProcess;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::types::Real;

/// Bates model (`batesmodel.hpp:43`): Heston plus log-normal jumps.
pub struct BatesModel {
    model: CalibratedModel,
    process: Shared<BatesProcess>,
    #[allow(dead_code)]
    observer: Option<SharedMut<dyn Observer>>,
}

impl BatesModel {
    /// `BatesModel(const shared_ptr<BatesProcess>&)` (`batesmodel.cpp:25-37`).
    ///
    /// # Errors
    ///
    /// Fails if a constrained parameter is violated: Heston positives / `rho` in
    /// `[-1, 1]`, or `delta` / `lambda` not strictly positive. `nu` is
    /// unconstrained (`NoConstraint`).
    pub fn new(process: Shared<BatesProcess>) -> QlResult<SharedMut<BatesModel>> {
        let mut model = CalibratedModel::new(8);
        model.arguments_mut()[0] =
            ConstantParameter::new(process.theta(), Rc::new(PositiveConstraint))?;
        model.arguments_mut()[1] =
            ConstantParameter::new(process.kappa(), Rc::new(PositiveConstraint))?;
        model.arguments_mut()[2] =
            ConstantParameter::new(process.sigma(), Rc::new(PositiveConstraint))?;
        model.arguments_mut()[3] =
            ConstantParameter::new(process.rho(), Rc::new(BoundaryConstraint::new(-1.0, 1.0)))?;
        model.arguments_mut()[4] =
            ConstantParameter::new(process.v0(), Rc::new(PositiveConstraint))?;
        model.arguments_mut()[5] = ConstantParameter::new(process.nu(), Rc::new(NoConstraint))?;
        model.arguments_mut()[6] =
            ConstantParameter::new(process.delta(), Rc::new(PositiveConstraint))?;
        model.arguments_mut()[7] =
            ConstantParameter::new(process.lambda(), Rc::new(PositiveConstraint))?;

        let risk_free_rate = process.risk_free_rate();
        let dividend_yield = process.dividend_yield();
        let s0 = process.s0();

        let mut bates = BatesModel {
            model,
            process,
            observer: None,
        };
        bates.generate_arguments();

        let shared = shared_mut(bates);
        let observer = register_with_term_structure(&shared, &risk_free_rate);
        dividend_yield.register_observer(&observer);
        s0.register_observer(&observer);
        shared.borrow_mut().observer = Some(observer);
        Ok(shared)
    }

    /// Variance mean-reversion level `θ`.
    pub fn theta(&self) -> Real {
        self.model.arguments()[0].value(0.0)
    }

    /// Variance mean-reversion speed `κ`.
    pub fn kappa(&self) -> Real {
        self.model.arguments()[1].value(0.0)
    }

    /// Vol-of-vol `σ`.
    pub fn sigma(&self) -> Real {
        self.model.arguments()[2].value(0.0)
    }

    /// Spot/variance correlation `ρ`.
    pub fn rho(&self) -> Real {
        self.model.arguments()[3].value(0.0)
    }

    /// Spot variance `v0`.
    pub fn v0(&self) -> Real {
        self.model.arguments()[4].value(0.0)
    }

    /// Mean jump size `ν` (`arguments_[5]`).
    pub fn nu(&self) -> Real {
        self.model.arguments()[5].value(0.0)
    }

    /// Jump-size volatility `δ` (`arguments_[6]`).
    pub fn delta(&self) -> Real {
        self.model.arguments()[6].value(0.0)
    }

    /// Jump intensity `λ` (`arguments_[7]`).
    pub fn lambda(&self) -> Real {
        self.model.arguments()[7].value(0.0)
    }

    /// Underlying Bates process, rebuilt by every `generate_arguments`.
    pub fn process(&self) -> Shared<BatesProcess> {
        Shared::clone(&self.process)
    }
}

impl CalibratedModelHolder for BatesModel {
    fn calibrated_model(&self) -> &CalibratedModel {
        &self.model
    }

    fn calibrated_model_mut(&mut self) -> &mut CalibratedModel {
        &mut self.model
    }

    /// `generateArguments` (`batesmodel.cpp:39-45`).
    fn generate_arguments(&mut self) {
        self.process = shared(BatesProcess::new(
            self.process.risk_free_rate(),
            self.process.dividend_yield(),
            self.process.s0(),
            self.v0(),
            self.kappa(),
            self.theta(),
            self.sigma(),
            self.rho(),
            self.lambda(),
            self.nu(),
            self.delta(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::quotes::make_quote_handle;
    use crate::shared::Shared;
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::types::Rate;

    const S0: Real = 100.0;
    const V0: Real = 0.04;
    const KAPPA: Real = 1.2;
    const THETA: Real = 0.06;
    const SIGMA: Real = 0.3;
    const RHO: Real = -0.5;
    const LAMBDA: Real = 0.5;
    const NU: Real = -0.1;
    const DELTA: Real = 0.15;
    const R: Rate = 0.05;
    const Q: Rate = 0.02;

    fn reference() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn flat_yield(rate: Rate) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            reference(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn make_model() -> SharedMut<BatesModel> {
        let process = shared(BatesProcess::new(
            flat_yield(R),
            flat_yield(Q),
            make_quote_handle(S0).handle(),
            V0,
            KAPPA,
            THETA,
            SIGMA,
            RHO,
            LAMBDA,
            NU,
            DELTA,
        ));
        BatesModel::new(process).unwrap()
    }

    #[test]
    fn ctor_round_trips_all_eight_params() {
        let model = make_model();
        let m = model.borrow();
        assert_eq!(m.theta(), THETA);
        assert_eq!(m.kappa(), KAPPA);
        assert_eq!(m.sigma(), SIGMA);
        assert_eq!(m.rho(), RHO);
        assert_eq!(m.v0(), V0);
        assert_eq!(m.nu(), NU);
        assert_eq!(m.delta(), DELTA);
        assert_eq!(m.lambda(), LAMBDA);

        let p = m.process();
        assert_eq!(p.theta(), THETA);
        assert_eq!(p.lambda(), LAMBDA);
        assert_eq!(p.nu(), NU);
        assert_eq!(p.delta(), DELTA);
    }

    #[test]
    fn set_params_rebuilds_process_including_jumps() {
        use crate::math::array::Array;
        let model = make_model();
        let new_params = Array::from([0.05, 2.0, 0.4, -0.3, 0.03, 0.05, 0.2, 1.0]);
        model.borrow_mut().set_params(&new_params).unwrap();
        let m = model.borrow();
        assert!((m.lambda() - 1.0).abs() < 1e-15);
        assert!((m.nu() - 0.05).abs() < 1e-15);
        assert!((m.delta() - 0.2).abs() < 1e-15);
        let p = m.process();
        assert!((p.lambda() - 1.0).abs() < 1e-15);
        assert!((p.nu() - 0.05).abs() < 1e-15);
        assert!((p.v0() - 0.03).abs() < 1e-15);
    }
}
