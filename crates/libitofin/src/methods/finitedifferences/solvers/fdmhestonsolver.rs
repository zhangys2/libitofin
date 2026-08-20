//! Finite-difference solver specialised to the Heston model.
//!
//! Port of `ql/methods/finitedifferences/solvers/fdmhestonsolver.{hpp,cpp}`: a
//! thin lazy wrapper that builds [`FdmHestonOp`] and delegates point queries
//! to [`Fdm2DimSolver`]. `valueAt(s, v)` interpolates at `(ln(s), v)`.
//!
//! Quanto and leverage-function arguments are omitted with the operator.

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::methods::finitedifferences::operators::{FdmHestonOp, FdmLinearOpComposite};
use crate::patterns::lazyobject::LazyObject;
use crate::patterns::observable::{AsObservable, Observer};
use crate::processes::HestonProcess;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::types::Real;

use super::fdm2dimsolver::Fdm2DimSolver;
use super::fdmschemedesc::FdmSchemeDesc;
use super::fdmsolverdesc::FdmSolverDesc;

/// Clears the lazy cache when the Heston process notifies.
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

/// Lazy Heston 2-D FD solver (`fdmhestonsolver.hpp:37`).
pub struct FdmHestonSolver {
    process: Shared<HestonProcess>,
    solver_desc: FdmSolverDesc,
    scheme_desc: FdmSchemeDesc,
    mixing_factor: Real,
    solver: Shared<RefCell<Option<Shared<Fdm2DimSolver>>>>,
    lazy: SharedMut<LazyObject>,
    _updater: SharedMut<Updater>,
}

impl FdmHestonSolver {
    /// `FdmHestonSolver(process, solverDesc, schemeDesc)` (`cpp:33-44`)
    /// without quanto / leverage. `mixing_factor` defaults to 1 in C++.
    pub fn new(
        process: Shared<HestonProcess>,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
        mixing_factor: Real,
    ) -> FdmHestonSolver {
        let lazy = shared_mut(LazyObject::new(true));
        let solver = shared(RefCell::new(None));
        let updater = shared_mut(Updater {
            lazy: SharedMut::clone(&lazy),
            solver: Shared::clone(&solver),
        });
        let observer = updater.clone() as SharedMut<dyn Observer>;
        process.observable().register_observer(&observer);

        FdmHestonSolver {
            process,
            solver_desc,
            scheme_desc,
            mixing_factor,
            solver,
            lazy,
            _updater: updater,
        }
    }

    /// Solution at spot `s` and variance `v` (`cpp:57-60`).
    pub fn value_at(&self, s: Real, v: Real) -> QlResult<Real> {
        self.calculate()?;
        self.solver
            .borrow()
            .as_ref()
            .expect("solver is filled by calculate")
            .interpolate_at(s.ln(), v)
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
        let op = shared_mut(FdmHestonOp::new(
            Shared::clone(&self.solver_desc.mesher),
            &self.process,
            self.mixing_factor,
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
    use crate::methods::finitedifferences::meshers::{
        FdmMesher, FdmMesherComposite, uniform_1d_mesher,
    };
    use crate::methods::finitedifferences::operators::FdmLinearOpIterator;
    use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
    use crate::methods::finitedifferences::utilities::FdmInnerValueCalculator;
    use crate::quotes::{Quote, SimpleQuote};
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

    fn process() -> Shared<HestonProcess> {
        let curve = Handle::new(shared(FlatForward::with_rate(
            Date::new(15, Month::January, 2026),
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        shared(HestonProcess::new(
            curve.clone(),
            curve,
            Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
            0.04,
            1.0,
            0.04,
            0.2,
            -0.5,
        ))
    }

    fn solver(payoff: Real) -> FdmHestonSolver {
        let mesher: Shared<dyn FdmMesher> = shared(FdmMesherComposite::new(vec![
            uniform_1d_mesher((80.0_f64).ln(), (120.0_f64).ln(), 11).unwrap(),
            uniform_1d_mesher(0.01, 0.09, 9).unwrap(),
        ])) as Shared<dyn FdmMesher>;
        let desc = FdmSolverDesc {
            mesher,
            bc_set: Vec::new(),
            condition: shared(FdmStepConditionComposite::new(&[], Vec::new())),
            calculator: shared(ConstantPayoff(payoff)) as Shared<dyn FdmInnerValueCalculator>,
            maturity: 0.5,
            time_steps: 20,
            damping_steps: 0,
        };
        FdmHestonSolver::new(process(), desc, FdmSchemeDesc::hundsdorfer(), 1.0)
    }

    #[test]
    fn constant_payoff_stays_near_discounted_constant() {
        let v = solver(1.0).value_at(100.0, 0.04).unwrap();
        assert!(v.is_finite(), "value={v}");
        assert!(
            v > 0.0 && v < 1.0,
            "expected a discount-factor-ish value, got {v}"
        );
    }

    #[test]
    fn zero_payoff_is_zero() {
        let v = solver(0.0).value_at(100.0, 0.04).unwrap();
        assert!(v.abs() < 1e-10, "got {v}");
    }
}
