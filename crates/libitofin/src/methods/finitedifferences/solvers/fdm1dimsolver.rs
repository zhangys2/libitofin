//! One-dimensional finite-difference solver with cubic interpolation.
//!
//! Port of `ql/methods/finitedifferences/solvers/fdm1dimsolver.{hpp,cpp}`:
//! rolls a 1-D grid from maturity to 0 via [`FdmBackwardSolver`], then answers
//! point queries through a [`MonotonicCubicNaturalSpline`] on the result.

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::interpolations::Interpolation;
use crate::math::interpolations::cubic::MonotonicCubicNaturalSpline;
use crate::methods::finitedifferences::operators::FdmLinearOpComposite;
use crate::methods::finitedifferences::stepconditions::{
    FdmSnapshotCondition, FdmStepConditionComposite,
};
use crate::patterns::lazyobject::LazyObject;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::types::Real;
use crate::utilities::null::Null;

use super::fdmbackwardsolver::FdmBackwardSolver;
use super::fdmschemedesc::FdmSchemeDesc;
use super::fdmsolverdesc::FdmSolverDesc;

/// Lazy 1-D FD solver (`fdm1dimsolver.hpp`).
pub struct Fdm1DimSolver {
    solver_desc: FdmSolverDesc,
    scheme_desc: FdmSchemeDesc,
    op: SharedMut<dyn FdmLinearOpComposite>,
    theta_condition: Shared<FdmSnapshotCondition>,
    conditions: Shared<FdmStepConditionComposite>,
    x: Vec<Real>,
    initial_values: Vec<Real>,
    interpolation: RefCell<Option<crate::math::interpolations::cubic::CubicInterpolation>>,
    result_values: RefCell<Vec<Real>>,
    lazy: SharedMut<LazyObject>,
}

impl Fdm1DimSolver {
    /// `Fdm1DimSolver(solverDesc, schemeDesc, op)` (`fdm1dimsolver.cpp:33-55`).
    pub fn new(
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
        op: SharedMut<dyn FdmLinearOpComposite>,
    ) -> Fdm1DimSolver {
        let layout = solver_desc.mesher.layout();
        let first_stop = solver_desc
            .condition
            .stopping_times()
            .first()
            .copied()
            .unwrap_or(solver_desc.maturity);
        let theta_t = 0.99 * Real::min(1.0 / 365.0, first_stop);
        let theta_condition = shared(FdmSnapshotCondition::new(theta_t));
        let conditions =
            FdmStepConditionComposite::join_conditions(&theta_condition, &solver_desc.condition);

        let mut x = Vec::with_capacity(layout.size());
        let mut initial_values = vec![0.0; layout.size()];
        let mut position = layout.begin();
        while position.index() < layout.size() {
            initial_values[position.index()] = solver_desc
                .calculator
                .avg_inner_value(&position, solver_desc.maturity);
            x.push(solver_desc.mesher.location(&position, 0));
            position.advance();
        }

        Fdm1DimSolver {
            solver_desc,
            scheme_desc,
            op,
            theta_condition,
            conditions,
            x,
            initial_values,
            interpolation: RefCell::new(None),
            result_values: RefCell::new(Vec::new()),
            lazy: shared_mut(LazyObject::new(true)),
        }
    }

    /// Interpolated solution at `x` (`fdm1dimsolver.cpp:70-73`).
    pub fn interpolate_at(&self, x: Real) -> QlResult<Real> {
        self.calculate()?;
        self.interpolation
            .borrow()
            .as_ref()
            .expect("interpolation is filled by calculate")
            .value(x)
    }

    /// Finite-difference theta at `x` (`fdm1dimsolver.cpp:75-88`).
    pub fn theta_at(&self, x: Real) -> QlResult<Real> {
        if self
            .conditions
            .stopping_times()
            .first()
            .is_some_and(|&t| t == 0.0)
        {
            return Ok(Real::null());
        }
        self.calculate()?;
        let theta_values = self.theta_condition.values().to_vec();
        let theta_interp = MonotonicCubicNaturalSpline::new(self.x.clone(), theta_values)?;
        Ok((theta_interp.value(x)? - self.interpolate_at(x)?) / self.theta_condition.time())
    }

    /// `∂/∂x` of the solution (`fdm1dimsolver.cpp:90-93`).
    pub fn derivative_x(&self, x: Real) -> QlResult<Real> {
        self.calculate()?;
        self.interpolation
            .borrow()
            .as_ref()
            .expect("interpolation is filled by calculate")
            .derivative(x)
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
        let mut rhs = Array::from(self.initial_values.clone());
        FdmBackwardSolver::new(
            SharedMut::clone(&self.op),
            self.solver_desc.bc_set.clone(),
            Some(Shared::clone(&self.conditions)),
            self.scheme_desc,
        )
        .rollback(
            &mut rhs,
            self.solver_desc.maturity,
            0.0,
            self.solver_desc.time_steps,
            self.solver_desc.damping_steps,
        )?;

        let result_values = rhs.to_vec();
        let interpolation =
            MonotonicCubicNaturalSpline::new(self.x.clone(), result_values.clone())?;
        *self.result_values.borrow_mut() = result_values;
        *self.interpolation.borrow_mut() = Some(interpolation);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::finitedifferences::meshers::{FdmMesher, UniformGridMesher};
    use crate::methods::finitedifferences::operators::{
        FdmLinearOp, FdmLinearOpIterator, FdmLinearOpLayout,
    };
    use crate::methods::finitedifferences::utilities::FdmInnerValueCalculator;
    use crate::shared::shared;
    use crate::types::{Size, Time};

    struct ZeroOp;

    impl FdmLinearOp for ZeroOp {
        fn apply(&self, r: &Array) -> Array {
            Array::with_size(r.size())
        }
    }

    impl FdmLinearOpComposite for ZeroOp {
        fn size(&self) -> Size {
            1
        }

        fn set_time(&mut self, _t1: Time, _t2: Time) -> QlResult<()> {
            Ok(())
        }

        fn apply_mixed(&self, r: &Array) -> Array {
            Array::with_size(r.size())
        }

        fn apply_direction(&self, _direction: Size, r: &Array) -> Array {
            Array::with_size(r.size())
        }

        fn solve_splitting(&self, _direction: Size, r: &Array, _s: Real) -> QlResult<Array> {
            Ok(r.clone())
        }

        fn preconditioner(&self, r: &Array, s: Real) -> QlResult<Array> {
            self.solve_splitting(0, r, s)
        }
    }

    struct MesherPayoff {
        mesher: Shared<dyn FdmMesher>,
        f: fn(Real) -> Real,
    }

    impl FdmInnerValueCalculator for MesherPayoff {
        fn inner_value(&self, iter: &FdmLinearOpIterator, _t: Time) -> Real {
            (self.f)(self.mesher.location(iter, 0))
        }

        fn avg_inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
            self.inner_value(iter, t)
        }
    }

    fn fixture(f: fn(Real) -> Real) -> Fdm1DimSolver {
        let layout = shared(FdmLinearOpLayout::new(vec![5]));
        let mesher: Shared<dyn FdmMesher> =
            shared(UniformGridMesher::new(Shared::clone(&layout), &[(0.0, 1.0)]).unwrap());
        let desc = FdmSolverDesc {
            mesher: Shared::clone(&mesher),
            bc_set: Vec::new(),
            condition: shared(FdmStepConditionComposite::new(&[], Vec::new())),
            calculator: shared(MesherPayoff { mesher, f }) as Shared<dyn FdmInnerValueCalculator>,
            maturity: 1.0,
            time_steps: 10,
            damping_steps: 0,
        };
        let op = shared_mut(ZeroOp) as SharedMut<dyn FdmLinearOpComposite>;
        Fdm1DimSolver::new(desc, FdmSchemeDesc::douglas(), op)
    }

    #[test]
    fn interpolates_terminal_payoff_when_generator_is_zero() {
        let solver = fixture(|x| 2.0 * x + 1.0);
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let got = solver.interpolate_at(x).unwrap();
            let expected = 2.0 * x + 1.0;
            assert!(
                (got - expected).abs() < 1e-10,
                "at {x}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn rejects_out_of_range_query() {
        let solver = fixture(|x| x);
        let err = solver.interpolate_at(2.0).unwrap_err();
        assert!(err.message().contains("extrapolation"));
    }
}
