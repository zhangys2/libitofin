//! Finite-difference solver specialised to the Hull–White short-rate model.
//!
//! Port of `ql/methods/finitedifferences/solvers/fdmhullwhitesolver.{hpp,cpp}`:
//! a thin lazy wrapper that builds [`FdmHullWhiteOp`] over direction `0` and
//! delegates point queries to [`Fdm1DimSolver`].

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::methods::finitedifferences::operators::{FdmHullWhiteOp, FdmLinearOpComposite};
use crate::models::model::CalibratedModelHolder;
use crate::models::shortrate::HullWhite;
use crate::patterns::lazyobject::LazyObject;
use crate::patterns::observable::{AsObservable, Observer};
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::types::Real;

use super::fdm1dimsolver::Fdm1DimSolver;
use super::fdmschemedesc::FdmSchemeDesc;
use super::fdmsolverdesc::FdmSolverDesc;

/// Clears the lazy cache (and the nested 1-D solver) when the HW model notifies.
struct Updater {
    lazy: SharedMut<LazyObject>,
    solver: Shared<RefCell<Option<Shared<Fdm1DimSolver>>>>,
}

impl Observer for Updater {
    fn update(&mut self) {
        *self.solver.borrow_mut() = None;
        if let Some(update) = LazyObject::deferred_update(&self.lazy) {
            update.notify_observers();
        }
    }
}

/// Lazy Hull–White 1-D FD solver (`fdmhullwhitesolver.hpp`).
pub struct FdmHullWhiteSolver {
    model: SharedMut<HullWhite>,
    solver_desc: FdmSolverDesc,
    scheme_desc: FdmSchemeDesc,
    solver: Shared<RefCell<Option<Shared<Fdm1DimSolver>>>>,
    lazy: SharedMut<LazyObject>,
    _updater: SharedMut<Updater>,
}

impl FdmHullWhiteSolver {
    /// `FdmHullWhiteSolver(model, solverDesc, schemeDesc)`
    /// (`fdmhullwhitesolver.cpp:33-38`).
    ///
    /// Registers with the model so parameter / curve changes invalidate the
    /// cached rollback. QL's `FdHullWhiteSwaptionEngine` defaults to
    /// [`FdmSchemeDesc::douglas`].
    pub fn new(
        model: SharedMut<HullWhite>,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
    ) -> FdmHullWhiteSolver {
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

        FdmHullWhiteSolver {
            model,
            solver_desc,
            scheme_desc,
            solver,
            lazy,
            _updater: updater,
        }
    }

    /// Solution at factor state `x` (`fdmhullwhitesolver.cpp:48-51`).
    ///
    /// C++ names the argument `r`; the mesher is the OU factor `x` (origin `0`),
    /// and the engine evaluates at `0.0`.
    pub fn value_at(&self, x: Real) -> QlResult<Real> {
        self.calculate()?;
        self.solver
            .borrow()
            .as_ref()
            .expect("solver is filled by calculate")
            .interpolate_at(x)
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
        let op = shared_mut(FdmHullWhiteOp::new(
            Shared::clone(&self.solver_desc.mesher),
            SharedMut::clone(&self.model),
            0,
        )) as SharedMut<dyn FdmLinearOpComposite>;
        let solver = shared(Fdm1DimSolver::new(
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

    fn model() -> SharedMut<HullWhite> {
        let curve = Handle::new(shared(FlatForward::with_rate(
            Date::new(15, Month::January, 2026),
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        HullWhite::new(curve, 0.1, 0.01).unwrap()
    }

    fn solver(payoff: Real) -> FdmHullWhiteSolver {
        let layout = shared(FdmLinearOpLayout::new(vec![11]));
        let mesher: Shared<dyn FdmMesher> =
            shared(UniformGridMesher::new(Shared::clone(&layout), &[(-0.05, 0.05)]).unwrap());
        let desc = FdmSolverDesc {
            mesher,
            bc_set: Vec::new(),
            condition: shared(FdmStepConditionComposite::new(&[], Vec::new())),
            calculator: shared(ConstantPayoff(payoff)) as Shared<dyn FdmInnerValueCalculator>,
            maturity: 1.0,
            time_steps: 20,
            damping_steps: 0,
        };
        FdmHullWhiteSolver::new(model(), desc, FdmSchemeDesc::douglas())
    }

    #[test]
    fn constant_payoff_stays_near_discounted_constant() {
        let s = solver(1.0);
        let v = s.value_at(0.0).unwrap();
        assert!(v.is_finite(), "value={v}");
        assert!(
            v > 0.0 && v < 1.0,
            "expected a pure discount factor-ish value, got {v}"
        );
    }

    #[test]
    fn zero_payoff_is_zero_at_origin() {
        let s = solver(0.0);
        let v = s.value_at(0.0).unwrap();
        assert!(v.abs() < 1e-10, "got {v}");
    }
}
