//! Crank-Nicolson stepping.

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::methods::finitedifferences::operators::FdmLinearOpComposite;
use crate::methods::finitedifferences::utilities::FdmBoundaryConditionSet;
use crate::shared::SharedMut;
use crate::types::Time;

use super::douglasscheme::DouglasScheme;
use super::scheme::Scheme;

/// Crank-Nicolson is Douglas splitting with theta = 0.5 on the supported
/// one-dimensional operator path.
pub struct CrankNicolsonScheme {
    inner: DouglasScheme,
}

impl CrankNicolsonScheme {
    pub fn new(
        map: SharedMut<dyn FdmLinearOpComposite>,
        bc_set: FdmBoundaryConditionSet,
    ) -> Self {
        Self {
            inner: DouglasScheme::new(0.5, map, bc_set),
        }
    }
}

impl Scheme for CrankNicolsonScheme {
    fn set_step(&mut self, dt: Time) {
        self.inner.set_step(dt);
    }

    fn step(&mut self, a: &mut Array, t: Time) -> QlResult<()> {
        self.inner.step(a, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::testops::{assert_close, probe, scaled_composite};

    #[test]
    fn crank_nicolson_matches_douglas_half_theta() {
        let mut scheme = CrankNicolsonScheme::new(scaled_composite(&[0.3]), Vec::new());
        scheme.set_step(0.1);
        let input = probe(4);
        let mut actual = input.clone();
        scheme.step(&mut actual, 0.25).unwrap();
        let expected = Array::from_iter(
            input
                .iter()
                .map(|value| value * (1.07 - 0.015) / 0.985),
        );
        assert_close(&actual, &expected);
    }
}
