//! FDM-native boundary conditions.

use crate::math::array::Array;
use crate::methods::finitedifferences::operators::FdmLinearOp;
use crate::shared::{Shared, shared};
use crate::types::{Real, Time};
use std::cell::RefCell;

use super::{BoundaryCondition, BoundarySide};

/// A fixed value at one edge of a one-dimensional FDM array.
#[derive(Clone, Copy, Debug)]
pub struct DirichletBoundary {
    side: BoundarySide,
    value: Real,
}

impl DirichletBoundary {
    /// Creates a constant-value condition at `side`.
    pub fn new(side: BoundarySide, value: Real) -> Self {
        Self { side, value }
    }

    fn apply(&self, values: &mut Array) {
        if values.size() == 0 {
            return;
        }
        match self.side {
            BoundarySide::Lower => values[0] = self.value,
            BoundarySide::Upper => {
                let last = values.size() - 1;
                values[last] = self.value;
            }
            BoundarySide::None => {}
        }
    }
}

impl BoundaryCondition for DirichletBoundary {
    fn apply_before_applying(&self, _op: &mut dyn FdmLinearOp) {}

    fn apply_after_applying(&self, a: &mut Array) {
        self.apply(a);
    }

    fn apply_before_solving(&self, _op: &mut dyn FdmLinearOp, _rhs: &mut Array) {}

    fn apply_after_solving(&self, a: &mut Array) {
        self.apply(a);
    }

    fn set_time(&self, _t: Time) {}
}

/// A constant first derivative at one edge of a uniform one-dimensional grid.
#[derive(Clone, Copy, Debug)]
pub struct NeumannBoundary {
    side: BoundarySide,
    slope: Real,
    dx: Real,
}

impl NeumannBoundary {
    /// Creates a constant-slope condition using grid spacing `dx`.
    pub fn new(side: BoundarySide, slope: Real, dx: Real) -> Self {
        Self { side, slope, dx }
    }

    fn apply(&self, values: &mut Array) {
        if values.size() < 2 {
            return;
        }
        match self.side {
            BoundarySide::Lower => values[0] = values[1] - self.slope * self.dx,
            BoundarySide::Upper => {
                let last = values.size() - 1;
                values[last] = values[last - 1] + self.slope * self.dx;
            }
            BoundarySide::None => {}
        }
    }
}

impl BoundaryCondition for NeumannBoundary {
    fn apply_before_applying(&self, _op: &mut dyn FdmLinearOp) {}

    fn apply_after_applying(&self, a: &mut Array) {
        self.apply(a);
    }

    fn apply_before_solving(&self, _op: &mut dyn FdmLinearOp, _rhs: &mut Array) {}

    fn apply_after_solving(&self, a: &mut Array) {
        self.apply(a);
    }

    fn set_time(&self, _t: Time) {}
}

/// A Dirichlet condition whose value is recomputed at the scheme time.
pub struct TimeDependentDirichletBoundary {
    side: BoundarySide,
    value: Box<dyn Fn(Time) -> Real>,
    time: RefCell<Time>,
}

impl TimeDependentDirichletBoundary {
    /// Creates a condition whose value function receives the current time.
    pub fn new(side: BoundarySide, value: impl Fn(Time) -> Real + 'static) -> Shared<Self> {
        shared(Self {
            side,
            value: Box::new(value),
            time: RefCell::new(0.0),
        })
    }

    fn apply(&self, values: &mut Array) {
        if values.size() == 0 {
            return;
        }
        let value = (self.value)(*self.time.borrow());
        match self.side {
            BoundarySide::Lower => values[0] = value,
            BoundarySide::Upper => {
                let last = values.size() - 1;
                values[last] = value;
            }
            BoundarySide::None => {}
        }
    }
}

impl BoundaryCondition for TimeDependentDirichletBoundary {
    fn apply_before_applying(&self, _op: &mut dyn FdmLinearOp) {}

    fn apply_after_applying(&self, a: &mut Array) {
        self.apply(a);
    }

    fn apply_before_solving(&self, _op: &mut dyn FdmLinearOp, _rhs: &mut Array) {}

    fn apply_after_solving(&self, a: &mut Array) {
        self.apply(a);
    }

    fn set_time(&self, t: Time) {
        *self.time.borrow_mut() = t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirichlet_sets_only_the_selected_edge() {
        let condition = DirichletBoundary::new(BoundarySide::Lower, 9.0);
        let mut values = Array::from([1.0, 2.0, 3.0]);
        condition.apply_after_solving(&mut values);
        assert_eq!(values, Array::from([9.0, 2.0, 3.0]));
    }

    #[test]
    fn neumann_sets_the_selected_first_difference() {
        let condition = NeumannBoundary::new(BoundarySide::Upper, 2.0, 0.5);
        let mut values = Array::from([1.0, 2.0, 3.0]);
        condition.apply_after_solving(&mut values);
        assert_eq!(values, Array::from([1.0, 2.0, 3.0]));

        let condition = NeumannBoundary::new(BoundarySide::Lower, 2.0, 0.5);
        let mut values = Array::from([1.0, 2.0, 3.0]);
        condition.apply_after_solving(&mut values);
        assert_eq!(values, Array::from([1.0, 2.0, 3.0]));
    }
}
