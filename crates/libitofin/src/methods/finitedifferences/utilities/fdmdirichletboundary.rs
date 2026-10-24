//! Dirichlet condition on one face of an N-D finite-difference grid.
//!
//! Port of `ql/methods/finitedifferences/utilities/fdmdirichletboundary.{hpp,cpp}`:
//! unlike the 1-D [`DirichletBoundary`](crate::methods::finitedifferences::DirichletBoundary),
//! which writes the first or last entry of the flat array, this condition
//! writes **every** layout index whose coordinate along `direction` sits on
//! the requested face. That is what a 2-D Heston barrier needs: the rebate
//! along the whole `ln(S) = ln(H)` edge, for every variance node.

use crate::math::array::Array;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::methods::finitedifferences::operators::FdmLinearOp;
use crate::methods::finitedifferences::{BoundaryCondition, BoundarySide};
use crate::shared::Shared;
use crate::types::{Real, Size, Time};

/// Indices of the layout that lie on one face (`FdmIndicesOnBoundary`).
fn indices_on_boundary(mesher: &dyn FdmMesher, direction: Size, side: BoundarySide) -> Vec<Size> {
    let layout = mesher.layout();
    let last = layout.dim()[direction] - 1;
    layout
        .iter()
        .filter(|iter| {
            let c = iter.coordinates()[direction];
            match side {
                BoundarySide::Lower => c == 0,
                BoundarySide::Upper => c == last,
                BoundarySide::None => false,
            }
        })
        .map(|iter| iter.index())
        .collect()
}

/// Face-Dirichlet condition (`fdmdirichletboundary.hpp:35`).
pub struct FdmDirichletBoundary {
    side: BoundarySide,
    value_on_boundary: Real,
    indices: Vec<Size>,
    x_extreme: Real,
}

impl FdmDirichletBoundary {
    /// `FdmDirichletBoundary(mesher, valueOnBoundary, direction, side)`
    /// (`fdmdirichletboundary.cpp:33-52`).
    pub fn new(
        mesher: Shared<dyn FdmMesher>,
        value_on_boundary: Real,
        direction: Size,
        side: BoundarySide,
    ) -> Self {
        let locations = mesher.locations(direction);
        let x_extreme = match side {
            BoundarySide::Lower => locations[0],
            BoundarySide::Upper => locations[mesher.layout().dim()[direction] - 1],
            BoundarySide::None => locations[0],
        };
        FdmDirichletBoundary {
            side,
            value_on_boundary,
            indices: indices_on_boundary(&*mesher, direction, side),
            x_extreme,
        }
    }

    /// Scalar helper `applyAfterApplying(x, value)` (`cpp:69-72`): clamps an
    /// interpolated value that has walked past the face.
    pub fn apply_after_applying_at(&self, x: Real, value: Real) -> Real {
        if (self.side == BoundarySide::Lower && x < self.x_extreme)
            || (self.side == BoundarySide::Upper && x > self.x_extreme)
        {
            self.value_on_boundary
        } else {
            value
        }
    }
}

impl BoundaryCondition for FdmDirichletBoundary {
    fn apply_before_applying(&self, _op: &mut dyn FdmLinearOp) {}

    fn apply_after_applying(&self, a: &mut Array) {
        for &i in &self.indices {
            a[i] = self.value_on_boundary;
        }
    }

    fn apply_before_solving(&self, _op: &mut dyn FdmLinearOp, _rhs: &mut Array) {}

    fn apply_after_solving(&self, a: &mut Array) {
        self.apply_after_applying(a);
    }

    fn set_time(&self, _t: Time) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::finitedifferences::meshers::{FdmMesherComposite, uniform_1d_mesher};
    use crate::methods::finitedifferences::operators::FdmLinearOpLayout;
    use crate::shared::shared;

    #[test]
    fn lower_face_sets_every_index_with_first_coordinate_zero() {
        let composite = shared(FdmMesherComposite::new(vec![
            uniform_1d_mesher(0.0, 1.0, 3).unwrap(),
            uniform_1d_mesher(0.0, 2.0, 4).unwrap(),
        ]));
        let mesher: Shared<dyn FdmMesher> = composite.clone() as Shared<dyn FdmMesher>;
        let bc = FdmDirichletBoundary::new(Shared::clone(&mesher), 7.0, 0, BoundarySide::Lower);
        let mut values = Array::filled(mesher.layout().size(), 1.0);
        bc.apply_after_applying(&mut values);

        let layout: &FdmLinearOpLayout = mesher.layout();
        for iter in layout.iter() {
            if iter.coordinates()[0] == 0 {
                assert_eq!(values[iter.index()], 7.0, "index {}", iter.index());
            } else {
                assert_eq!(values[iter.index()], 1.0, "index {}", iter.index());
            }
        }
        assert_eq!(bc.apply_after_applying_at(-0.1, 3.0), 7.0);
        assert_eq!(bc.apply_after_applying_at(0.5, 3.0), 3.0);
    }

    #[test]
    fn upper_face_sets_the_last_coordinate_of_the_direction() {
        let composite = shared(FdmMesherComposite::new(vec![
            uniform_1d_mesher(0.0, 1.0, 3).unwrap(),
            uniform_1d_mesher(0.0, 2.0, 4).unwrap(),
        ]));
        let mesher: Shared<dyn FdmMesher> = composite as Shared<dyn FdmMesher>;
        let bc = FdmDirichletBoundary::new(Shared::clone(&mesher), 9.0, 1, BoundarySide::Upper);
        let mut values = Array::filled(mesher.layout().size(), 0.0);
        bc.apply_after_solving(&mut values);
        let last = mesher.layout().dim()[1] - 1;
        for iter in mesher.layout().iter() {
            if iter.coordinates()[1] == last {
                assert_eq!(values[iter.index()], 9.0);
            } else {
                assert_eq!(values[iter.index()], 0.0);
            }
        }
    }
}
