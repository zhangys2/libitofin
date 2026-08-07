//! Finite-difference solver specialised to the G2++ short-rate model.
//!
//! Port of `ql/methods/finitedifferences/solvers/fdmg2solver.{hpp,cpp}`: a thin
//! lazy wrapper that builds [`FdmG2Op`] over directions `(0, 1)` and delegates
//! point queries to [`Fdm2DimSolver`].

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::methods::finitedifferences::operators::{FdmG2Op, FdmLinearOpComposite};
use crate::models::model::CalibratedModelHolder;
use crate::models::shortrate::G2;
use crate::patterns::lazyobject::LazyObject;
use crate::patterns::observable::{AsObservable, Observer};
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::types::Real;

use super::fdm2dimsolver::Fdm2DimSolver;
use super::fdmschemedesc::FdmSchemeDesc;
use super::fdmsolverdesc::FdmSolverDesc;

/// Clears the lazy cache (and the nested 2-D solver) when the G2 model notifies.
struct Updater {
    lazy: SharedMut<LazyObject>,
    solver: Shared<RefCell<Option<Shared<Fdm2DimSolver>>>>,
}

impl Observer for Updater {
    fn update(&mut self) {
        *self.solver.borrow_mut() = None;
        if let Some(update) = LazyObject::deferred_update(&self.lazy) {
            update.notify_observers();
        }
    }
}

/// Lazy G2++ 2-D FD solver (`fdmg2solver.hpp:37`).
pub struct FdmG2Solver {
    model: SharedMut<G2>,
    solver_desc: FdmSolverDesc,
    scheme_desc: FdmSchemeDesc,
    solver: Shared<RefCell<Option<Shared<Fdm2DimSolver>>>>,
    lazy: SharedMut<LazyObject>,
    _updater: SharedMut<Updater>,
}

impl FdmG2Solver {
    /// `FdmG2Solver(model, solverDesc, schemeDesc)` (`fdmg2solver.cpp:33-38`).
    ///
    /// Registers with the model so parameter / curve changes invalidate the
    /// cached rollback. Prefer [`FdmSchemeDesc::douglas`] until Hundsdorfer is
    /// ported (QL's default).
    pub fn new(
        model: SharedMut<G2>,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
    ) -> FdmG2Solver {
        let lazy = shared_mut(LazyObject::new(true));
        let solver = shared(RefCell::new(None));
        let updater = shared_mut(Updater {
            lazy: SharedMut::clone(&lazy),
            solver: Shared::clone(&solver),
        });
        let observer = updater.clone() as SharedMut<dyn Observer>;
        model
            .borrow()
            .calibrated_model()
            .observable()
            .register_observer(&observer);

        FdmG2Solver {
            model,
            solver_desc,
            scheme_desc,
            solver,
            lazy,
            _updater: updater,
        }
    }

    /// Solution at factor state `(x, y)` (`fdmg2solver.cpp:48-51`).
    pub fn value_at(&self, x: Real, y: Real) -> QlResult<Real> {
        self.calculate()?;
        self.solver
            .borrow()
            .as_ref()
            .expect("solver is filled by calculate")
            .interpolate_at(x, y)
    }

    fn calculate(&self) -> QlResult<()> {
        if !self.lazy.borrow_mut().start_calculation() {
            return Ok(());
        }
        let result = self.perform_calculations();
        self.lazy.borrow_mut().finish_calculation(&result);
        result
    }

    fn perform_calculations(&self) -> QlResult<()> {
        let op = shared_mut(FdmG2Op::new(
            Shared::clone(&self.solver_desc.mesher),
            SharedMut::clone(&self.model),
            0,
            1,
        )?) as SharedMut<dyn FdmLinearOpComposite>;
        let solver = shared(Fdm2DimSolver::new(
            self.solver_desc.clone(),
            self.scheme_desc,
            op,
        ));
        *self.solver.borrow_mut() = Some(solver);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::methods::finitedifferences::meshers::{FdmMesher, UniformGridMesher};
    use crate::methods::finitedifferences::operators::{FdmLinearOpIterator, FdmLinearOpLayout};
    use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
    use crate::methods::finitedifferences::utilities::FdmInnerValueCalculator;
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::types::Time;

    struct ConstantPayoff(Real);

    impl FdmInnerValueCalculator for ConstantPayoff {
        fn inner_value(&self, _iter: &FdmLinearOpIterator, _t: Time) -> Real {
            self.0
        }

        fn avg_inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
            self.inner_value(iter, t)
        }
    }

    fn model() -> SharedMut<G2> {
        let curve = Handle::new(shared(FlatForward::with_rate(
            Date::new(15, Month::January, 2026),
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        G2::new(curve, 0.1, 0.01, 0.2, 0.008, -0.75).unwrap()
    }

    fn solver(payoff: Real) -> FdmG2Solver {
        let layout = shared(FdmLinearOpLayout::new(vec![11, 11]));
        let mesher: Shared<dyn FdmMesher> = shared(
            UniformGridMesher::new(Shared::clone(&layout), &[(-0.05, 0.05), (-0.05, 0.05)])
                .unwrap(),
        );
        let desc = FdmSolverDesc {
            mesher,
            bc_set: Vec::new(),
            condition: shared(FdmStepConditionComposite::new(&[], Vec::new())),
            calculator: shared(ConstantPayoff(payoff)) as Shared<dyn FdmInnerValueCalculator>,
            maturity: 1.0,
            time_steps: 20,
            damping_steps: 0,
        };
        FdmG2Solver::new(model(), desc, FdmSchemeDesc::douglas())
    }

    #[test]
    fn constant_payoff_stays_near_discounted_constant() {
        // With a constant terminal value the FD roll should stay finite and
        // roughly scale with the discount — exact oracle deferred to the engine.
        let s = solver(1.0);
        let v = s.value_at(0.0, 0.0).unwrap();
        assert!(v.is_finite(), "value={v}");
        assert!(
            v > 0.0 && v < 1.0,
            "expected a pure discount factor-ish value, got {v}"
        );
    }

    #[test]
    fn zero_payoff_is_zero_at_origin() {
        let s = solver(0.0);
        let v = s.value_at(0.0, 0.0).unwrap();
        assert!(v.abs() < 1e-10, "got {v}");
    }
}
