//! Fully explicit Euler stepping for finite-difference operators.

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::methods::finitedifferences::operators::FdmLinearOpComposite;
use crate::methods::finitedifferences::utilities::FdmBoundaryConditionSet;
use crate::shared::SharedMut;
use crate::types::Time;
use crate::{fail, require};

use super::boundaryconditionschemehelper::BoundaryConditionSchemeHelper;
use super::scheme::Scheme;

/// One explicit operator application per timestep.
pub struct ExplicitEulerScheme {
    dt: Option<Time>,
    map: SharedMut<dyn FdmLinearOpComposite>,
    bc_set: BoundaryConditionSchemeHelper,
}

impl ExplicitEulerScheme {
    pub fn new(map: SharedMut<dyn FdmLinearOpComposite>, bc_set: FdmBoundaryConditionSet) -> Self {
        Self {
            dt: None,
            map,
            bc_set: BoundaryConditionSchemeHelper::new(bc_set),
        }
    }
}

impl Scheme for ExplicitEulerScheme {
    fn set_step(&mut self, dt: Time) {
        self.dt = Some(dt);
    }

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn step(&mut self, a: &mut Array, t: Time) -> QlResult<()> {
        let Some(dt) = self.dt else {
            fail!("the timestep is not set: call set_step before stepping");
        };
        require!(t - dt > -1e-8, "a step towards negative time given");
        let start = (t - dt).max(0.0);

        let mut map = self.map.borrow_mut();
        map.set_time(start, t)?;
        self.bc_set.set_time(start);
        self.bc_set.apply_before_applying(&mut *map);
        let mut next = &*a + &(dt * &map.apply(a));
        self.bc_set.apply_after_applying(&mut next);
        self.bc_set.apply_after_solving(&mut next);
        *a = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::testops::{assert_close, probe, scaled_composite};
    use super::*;

    #[test]
    fn explicit_step_matches_the_closed_form() {
        let mut scheme = ExplicitEulerScheme::new(scaled_composite(&[0.3]), Vec::new());
        scheme.set_step(0.1);
        let input = probe(4);
        let mut actual = input.clone();
        scheme.step(&mut actual, 0.25).unwrap();
        assert_close(&actual, &(&input * 1.07));
    }
}
