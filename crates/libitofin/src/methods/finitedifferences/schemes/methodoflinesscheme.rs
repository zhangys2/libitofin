//! Method of lines.
//!
//! Port of `ql/methods/finitedifferences/schemes/methodoflinesscheme.{hpp,cpp}`:
//! treat the spatial operator as an ODE in time and integrate it with
//! [`AdaptiveRungeKutta`](crate::math::ode::AdaptiveRungeKutta) from `t`
//! down to `max(0, t − dt)`.
//!
//! QuantLib's default [`FdmSchemeDesc::method_of_lines`](crate::methods::finitedifferences::solvers::FdmSchemeDesc::method_of_lines)
//! uses `eps = 0.001`, `relInitStepSize = 0.01`. `FdmBackwardSolver` passes
//! those as `theta` and `mu` (`fdmbackwardsolver.cpp:171-175`).

use std::cell::RefCell;

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::ode::AdaptiveRungeKutta;
use crate::methods::finitedifferences::operators::FdmLinearOpComposite;
use crate::methods::finitedifferences::utilities::FdmBoundaryConditionSet;
use crate::shared::SharedMut;
use crate::types::{Real, Time};
use crate::{fail, require};

use super::boundaryconditionschemehelper::BoundaryConditionSchemeHelper;
use super::scheme::Scheme;

/// The dummy end-time spread C++ adds when setting the operator
/// (`methodoflinesscheme.cpp:40`).
const SET_TIME_SPREAD: Time = 0.0001;

/// Method-of-lines step (`methodoflinesscheme.hpp:37`).
pub struct MethodOfLinesScheme {
    dt: Option<Time>,
    eps: Real,
    rel_init_step_size: Real,
    map: SharedMut<dyn FdmLinearOpComposite>,
    bc_set: BoundaryConditionSchemeHelper,
}

impl MethodOfLinesScheme {
    /// `MethodOfLinesScheme(eps, relInitStepSize, map, bcSet)`
    /// (`methodoflinesscheme.cpp:31-36`).
    pub fn new(
        eps: Real,
        rel_init_step_size: Real,
        map: SharedMut<dyn FdmLinearOpComposite>,
        bc_set: FdmBoundaryConditionSet,
    ) -> Self {
        MethodOfLinesScheme {
            dt: None,
            eps,
            rel_init_step_size,
            map,
            bc_set: BoundaryConditionSchemeHelper::new(bc_set),
        }
    }

    /// `cpp:39-45`: `dx/dt = −A(x)` at operator time `(t, t + 10⁻⁴)`.
    fn apply(&self, t: Time, u: &[Real]) -> QlResult<Vec<Real>> {
        let mut map = self.map.borrow_mut();
        map.set_time(t, t + SET_TIME_SPREAD)?;
        self.bc_set.apply_before_applying(&mut *map);
        let dxdt = -&map.apply(&Array::from(u.to_vec()));
        Ok(dxdt.to_vec())
    }
}

impl Scheme for MethodOfLinesScheme {
    /// `methodoflinesscheme.cpp:62-64`.
    fn set_step(&mut self, dt: Time) {
        self.dt = Some(dt);
    }

    /// `methodoflinesscheme.cpp:47-60`.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn step(&mut self, a: &mut Array, t: Time) -> QlResult<()> {
        let Some(dt) = self.dt else {
            fail!("the timestep is not set: call set_step before stepping");
        };
        require!(t - dt > -1e-8, "a step towards negative time given");

        let start = (t - dt).max(0.0);
        let rk = AdaptiveRungeKutta::new(self.eps, self.rel_init_step_size * dt, 0.0);
        let failure = RefCell::new(None);
        let this = &*self;
        let ode = |x: Real, u: &[Real]| match this.apply(x, u) {
            Ok(value) => value,
            Err(error) => {
                failure.borrow_mut().get_or_insert(error);
                vec![0.0; u.len()]
            }
        };
        let v = rk.solve(ode, a, t, start)?;
        if let Some(error) = failure.into_inner() {
            return Err(error);
        }

        let mut y = Array::from(v);
        self.bc_set.apply_after_solving(&mut y);
        *a = y;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::testops::{
        GRID, WHOLE, assert_close, black_scholes_op, call_log, mesher, probe, scaled_composite,
    };
    use crate::shared::shared_mut;

    const EPS: Real = 1e-12;
    const REL_INIT: Real = 0.01;
    const DT: Time = 0.1;
    const T: Time = 0.25;
    const C: Real = 0.3;

    fn method_of_lines(
        map: SharedMut<dyn FdmLinearOpComposite>,
        bc_set: FdmBoundaryConditionSet,
    ) -> MethodOfLinesScheme {
        let mut scheme = MethodOfLinesScheme::new(EPS, REL_INIT, map, bc_set);
        scheme.set_step(DT);
        scheme
    }

    /// Replay the C++ call sequence: integrate `dx/dt = −A(x)` with the same
    /// Runge–Kutta settings from `t` down to `t − dt`.
    #[test]
    fn a_step_replays_the_cpp_sequence_on_the_black_scholes_operator() {
        let mesher = mesher();
        let map: SharedMut<dyn FdmLinearOpComposite> = shared_mut(black_scholes_op(&mesher));
        let mut scheme = method_of_lines(map, Vec::new());

        let u = probe(GRID);
        let mut a = u.clone();
        scheme.step(&mut a, T).unwrap();

        let oracle: SharedMut<dyn FdmLinearOpComposite> = shared_mut(black_scholes_op(&mesher));
        let rk = AdaptiveRungeKutta::new(EPS, REL_INIT * DT, 0.0);
        let expected = rk
            .solve(
                |x, state| {
                    let mut map = oracle.borrow_mut();
                    map.set_time(x, x + SET_TIME_SPREAD).unwrap();
                    let dxdt = -&map.apply(&Array::from(state.to_vec()));
                    dxdt.to_vec()
                },
                &u,
                T,
                T - DT,
            )
            .unwrap();

        assert_close(&a, &Array::from(expected));
    }

    /// On a diagonal operator `A y = w y`, the ODE is `y' = −w y`, so a
    /// backward step of length `dt` multiplies by `e^{w dt}`.
    #[test]
    fn a_step_matches_the_closed_form_on_a_diagonal_operator() {
        let mut scheme = method_of_lines(scaled_composite(&[C]), Vec::new());

        let u = probe(4);
        let mut a = u.clone();
        scheme.step(&mut a, T).unwrap();

        let expected = &u * (WHOLE * DT).exp();
        for i in 0..a.size() {
            assert!(
                (a[i] - expected[i]).abs() <= 1e-9,
                "element {i}: {} != {}",
                a[i],
                expected[i]
            );
        }
    }

    /// C++ never calls `setTime` on the BC helper; each RK stage hits
    /// `applyBeforeApplying`, and the step finishes with `applyAfterSolving`.
    #[test]
    fn a_step_applies_bcs_only_around_the_ode_and_the_solve() {
        let map: SharedMut<dyn FdmLinearOpComposite> = scaled_composite(&[C]);
        let (log, bc_set) = call_log();
        let mut scheme = method_of_lines(map, bc_set);

        scheme.step(&mut probe(4), T).unwrap();

        let log = log.borrow();
        assert!(log.len() >= 2, "{log:?}");
        assert_eq!(log[0], "before_applying");
        assert_eq!(log[log.len() - 1], "after_solving");
        assert_eq!(log.iter().filter(|e| *e == "after_solving").count(), 1);
        assert!(
            log.iter()
                .all(|e| e == "before_applying" || e == "after_solving")
        );
    }

    #[test]
    fn stepping_before_the_timestep_is_set_fails() {
        let mut scheme =
            MethodOfLinesScheme::new(EPS, REL_INIT, scaled_composite(&[C]), Vec::new());
        assert!(scheme.step(&mut probe(4), T).is_err());
    }

    #[test]
    fn a_step_towards_negative_time_fails() {
        let mut scheme = method_of_lines(scaled_composite(&[C]), Vec::new());
        assert!(scheme.step(&mut probe(4), DT / 2.0).is_err());
    }
}
