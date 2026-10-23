//! Two-dimensional finite-difference solver with bicubic interpolation.
//!
//! Port of `ql/methods/finitedifferences/solvers/fdm2dimsolver.{hpp,cpp}`:
//! rolls a 2-D grid from maturity to 0 via [`FdmBackwardSolver`], then answers
//! point queries through a [`BicubicSpline`] on the result.

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::interpolations::Interpolation2D;
use crate::math::interpolations::bicubic::BicubicSpline;
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

/// Lazy 2-D FD solver (`fdm2dimsolver.hpp:39`).
pub struct Fdm2DimSolver {
    solver_desc: FdmSolverDesc,
    scheme_desc: FdmSchemeDesc,
    op: SharedMut<dyn FdmLinearOpComposite>,
    theta_condition: Shared<FdmSnapshotCondition>,
    conditions: Shared<FdmStepConditionComposite>,
    x: Vec<Real>,
    y: Vec<Real>,
    initial_values: Vec<Real>,
    interpolation: RefCell<Option<BicubicSpline>>,
    result_values: RefCell<Vec<Vec<Real>>>,
    lazy: SharedMut<LazyObject>,
}

impl Fdm2DimSolver {
    /// `Fdm2DimSolver(solverDesc, schemeDesc, op)` (`fdm2dimsolver.cpp:33-62`).
    pub fn new(
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
        op: SharedMut<dyn FdmLinearOpComposite>,
    ) -> Fdm2DimSolver {
        let layout = solver_desc.mesher.layout();
        let dim0 = layout.dim()[0];
        let dim1 = layout.dim()[1];

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

        let mut x = Vec::with_capacity(dim0);
        let mut y = Vec::with_capacity(dim1);
        let mut initial_values = vec![0.0; layout.size()];

        let mut position = layout.begin();
        while position.index() < layout.size() {
            initial_values[position.index()] = solver_desc
                .calculator
                .avg_inner_value(&position, solver_desc.maturity);
            if position.coordinates()[1] == 0 {
                x.push(solver_desc.mesher.location(&position, 0));
            }
            if position.coordinates()[0] == 0 {
                y.push(solver_desc.mesher.location(&position, 1));
            }
            position.advance();
        }

        Fdm2DimSolver {
            solver_desc,
            scheme_desc,
            op,
            theta_condition,
            conditions,
            x,
            y,
            initial_values,
            interpolation: RefCell::new(None),
            result_values: RefCell::new(Vec::new()),
            lazy: shared_mut(LazyObject::new(true)),
        }
    }

    /// Interpolated solution at `(x, y)` (`fdm2dimsolver.cpp:78-81`).
    pub fn interpolate_at(&self, x: Real, y: Real) -> QlResult<Real> {
        self.calculate()?;
        self.interpolation
            .borrow()
            .as_ref()
            .expect("interpolation is filled by calculate")
            .value(x, y)
    }

    /// Finite-difference theta at `(x, y)` (`fdm2dimsolver.cpp:83-96`).
    pub fn theta_at(&self, x: Real, y: Real) -> QlResult<Real> {
        if self
            .conditions
            .stopping_times()
            .first()
            .is_some_and(|&t| t == 0.0)
        {
            return Ok(Real::null());
        }
        self.calculate()?;
        let theta_values = flat_to_rows(&self.theta_condition.values(), self.y.len(), self.x.len());
        let theta_interp = BicubicSpline::new(self.x.clone(), self.y.clone(), theta_values)?;
        Ok((theta_interp.value(x, y)? - self.interpolate_at(x, y)?) / self.theta_condition.time())
    }

    /// `∂/∂x` of the solution (`fdm2dimsolver.cpp:98-101`).
    pub fn derivative_x(&self, x: Real, y: Real) -> QlResult<Real> {
        self.calculate()?;
        self.interpolation
            .borrow()
            .as_ref()
            .expect("interpolation is filled by calculate")
            .derivative_x(x, y)
    }

    /// `∂/∂y` of the solution (`fdm2dimsolver.cpp:103-106`).
    pub fn derivative_y(&self, x: Real, y: Real) -> QlResult<Real> {
        self.calculate()?;
        self.interpolation
            .borrow()
            .as_ref()
            .expect("interpolation is filled by calculate")
            .derivative_y(x, y)
    }

    /// `∂²/∂x²` of the solution (`fdm2dimsolver.cpp:108-111`).
    pub fn derivative_xx(&self, x: Real, y: Real) -> QlResult<Real> {
        self.calculate()?;
        self.interpolation
            .borrow()
            .as_ref()
            .expect("interpolation is filled by calculate")
            .second_derivative_x(x, y)
    }

    /// `∂²/∂y²` of the solution (`fdm2dimsolver.cpp:113-116`).
    pub fn derivative_yy(&self, x: Real, y: Real) -> QlResult<Real> {
        self.calculate()?;
        self.interpolation
            .borrow()
            .as_ref()
            .expect("interpolation is filled by calculate")
            .second_derivative_y(x, y)
    }

    /// `∂²/∂x∂y` of the solution (`fdm2dimsolver.cpp:118-121`).
    pub fn derivative_xy(&self, x: Real, y: Real) -> QlResult<Real> {
        self.calculate()?;
        self.interpolation
            .borrow()
            .as_ref()
            .expect("interpolation is filled by calculate")
            .derivative_xy(x, y)
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

        let rows = flat_to_rows(&rhs, self.y.len(), self.x.len());
        let interpolation = BicubicSpline::new(self.x.clone(), self.y.clone(), rows.clone())?;
        *self.result_values.borrow_mut() = rows;
        *self.interpolation.borrow_mut() = Some(interpolation);
        Ok(())
    }
}

/// Reshape a layout-flat array (x fastest) into `z[j][i]` for bicubic.
fn flat_to_rows(flat: &Array, n_y: usize, n_x: usize) -> Vec<Vec<Real>> {
    assert_eq!(flat.size(), n_x * n_y, "flat grid size mismatch");
    let mut rows = Vec::with_capacity(n_y);
    for j in 0..n_y {
        let mut row = Vec::with_capacity(n_x);
        for i in 0..n_x {
            row.push(flat[j * n_x + i]);
        }
        rows.push(row);
    }
    rows
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

    /// Zero generator: rollback leaves the terminal payoff unchanged.
    struct ZeroOp;

    impl FdmLinearOp for ZeroOp {
        fn apply(&self, r: &Array) -> Array {
            Array::with_size(r.size())
        }
    }

    impl FdmLinearOpComposite for ZeroOp {
        fn size(&self) -> Size {
            2
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
        f: fn(Real, Real) -> Real,
    }

    impl FdmInnerValueCalculator for MesherPayoff {
        fn inner_value(&self, iter: &FdmLinearOpIterator, _t: Time) -> Real {
            (self.f)(self.mesher.location(iter, 0), self.mesher.location(iter, 1))
        }

        fn avg_inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
            self.inner_value(iter, t)
        }
    }

    fn fixture(f: fn(Real, Real) -> Real) -> Fdm2DimSolver {
        let layout = shared(FdmLinearOpLayout::new(vec![5, 4]));
        let mesher: Shared<dyn FdmMesher> = shared(
            UniformGridMesher::new(Shared::clone(&layout), &[(0.0, 1.0), (0.0, 1.0)]).unwrap(),
        );
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
        Fdm2DimSolver::new(desc, FdmSchemeDesc::douglas(), op)
    }

    #[test]
    fn interpolates_terminal_payoff_when_generator_is_zero() {
        let solver = fixture(|x, y| x + 2.0 * y);
        // Grid nodes must reproduce exactly under bicubic of a bilinear field.
        for &(x, y) in &[(0.0, 0.0), (0.5, 0.0), (0.0, 1.0), (0.75, 0.5), (1.0, 1.0)] {
            let got = solver.interpolate_at(x, y).unwrap();
            let expected = x + 2.0 * y;
            assert!(
                (got - expected).abs() < 1e-10,
                "at ({x},{y}): {got} vs {expected}"
            );
        }
    }

    #[test]
    fn rejects_out_of_range_query() {
        let solver = fixture(|x, y| x * y);
        let err = solver.interpolate_at(2.0, 0.5).unwrap_err();
        assert!(err.message().contains("extrapolation"));
    }
}
