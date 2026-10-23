//! Hundsdorfer–Verwer operator splitting.
//!
//! Port of `ql/methods/finitedifferences/schemes/hundsdorferscheme.{hpp,cpp}`:
//! an explicit update, a θ-weighted directional correction (as in Douglas),
//! then a μ-weighted predictor–corrector and a second directional pass.
//!
//! QuantLib's default [`FdmSchemeDesc::hundsdorfer`](crate::methods::finitedifferences::solvers::FdmSchemeDesc::hundsdorfer)
//! uses `θ = ½ + √3/6`, `μ = ½`.

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::methods::finitedifferences::operators::FdmLinearOpComposite;
use crate::methods::finitedifferences::utilities::FdmBoundaryConditionSet;
use crate::shared::SharedMut;
use crate::types::{Real, Time};
use crate::{fail, require};

use super::boundaryconditionschemehelper::BoundaryConditionSchemeHelper;
use super::scheme::Scheme;

/// Hundsdorfer–Verwer ADI step (`hundsdorferscheme.hpp:40`).
pub struct HundsdorferScheme {
    dt: Option<Time>,
    theta: Real,
    mu: Real,
    map: SharedMut<dyn FdmLinearOpComposite>,
    bc_set: BoundaryConditionSchemeHelper,
}

impl HundsdorferScheme {
    /// `HundsdorferScheme(theta, mu, map, bcSet)` (`hundsdorferscheme.cpp:32-35`).
    pub fn new(
        theta: Real,
        mu: Real,
        map: SharedMut<dyn FdmLinearOpComposite>,
        bc_set: FdmBoundaryConditionSet,
    ) -> Self {
        HundsdorferScheme {
            dt: None,
            theta,
            mu,
            map,
            bc_set: BoundaryConditionSchemeHelper::new(bc_set),
        }
    }
}

impl Scheme for HundsdorferScheme {
    /// `hundsdorferscheme.cpp:64-66`.
    fn set_step(&mut self, dt: Time) {
        self.dt = Some(dt);
    }

    /// `hundsdorferscheme.cpp:37-61`.
    ///
    /// Stage 1 mirrors Douglas: explicit `y = a + dt A(a)`, then per-direction
    /// corrections whose RHS reads the step input `a`. Stage 2 builds
    /// `yt = y0 + μ dt A(y − a)` from the post-stage-1 `y` and corrects again,
    /// now reading that fixed `y` on each directional RHS.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn step(&mut self, a: &mut Array, t: Time) -> QlResult<()> {
        let Some(dt) = self.dt else {
            fail!("the timestep is not set: call set_step before stepping");
        };
        require!(t - dt > -1e-8, "a step towards negative time given");
        let start = (t - dt).max(0.0);

        let mut yt = {
            let mut map = self.map.borrow_mut();
            map.set_time(start, t)?;
            self.bc_set.set_time(start);

            self.bc_set.apply_before_applying(&mut *map);
            let mut y = &*a + &(dt * &map.apply(a));
            self.bc_set.apply_after_applying(&mut y);

            let y0 = y.clone();

            for i in 0..map.size() {
                let rhs = &y - &((self.theta * dt) * &map.apply_direction(i, a));
                y = map.solve_splitting(i, &rhs, -self.theta * dt)?;
            }

            self.bc_set.apply_before_applying(&mut *map);
            let mut yt = &y0 + &((self.mu * dt) * &map.apply(&(&y - &*a)));
            self.bc_set.apply_after_applying(&mut yt);

            for i in 0..map.size() {
                let rhs = &yt - &((self.theta * dt) * &map.apply_direction(i, &y));
                yt = map.solve_splitting(i, &rhs, -self.theta * dt)?;
            }

            yt
        };
        self.bc_set.apply_after_solving(&mut yt);
        *a = yt;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::testops::{
        GRID, WHOLE, assert_close, black_scholes_op, call_log, mesher, probe, scaled_composite,
    };
    use crate::methods::finitedifferences::operators::FdmLinearOp;
    use crate::shared::shared_mut;

    fn theta() -> Real {
        0.5 + 3.0_f64.sqrt() / 6.0
    }
    const MU: Real = 0.5;
    const DT: Time = 0.1;
    const T: Time = 0.25;
    const COEFFICIENTS: [Real; 2] = [0.3, -0.45];

    fn hundsdorfer(
        map: SharedMut<dyn FdmLinearOpComposite>,
        bc_set: FdmBoundaryConditionSet,
    ) -> HundsdorferScheme {
        let mut scheme = HundsdorferScheme::new(theta(), MU, map, bc_set);
        scheme.set_step(DT);
        scheme
    }

    /// Replay the C++ call sequence against the Black-Scholes operator.
    #[test]
    fn a_step_replays_the_cpp_sequence_on_the_black_scholes_operator() {
        let mesher = mesher();
        let map: SharedMut<dyn FdmLinearOpComposite> = shared_mut(black_scholes_op(&mesher));
        let mut scheme = hundsdorfer(map, Vec::new());

        let u = probe(GRID);
        let mut a = u.clone();
        scheme.step(&mut a, T).unwrap();

        let mut oracle = black_scholes_op(&mesher);
        oracle.set_time(T - DT, T).unwrap();
        let mut y = &u + &(DT * &oracle.apply(&u));
        let y0 = y.clone();
        for i in 0..oracle.size() {
            let rhs = &y - &((theta() * DT) * &oracle.apply_direction(i, &u));
            y = oracle.solve_splitting(i, &rhs, -theta() * DT).unwrap();
        }
        let mut yt = &y0 + &((MU * DT) * &oracle.apply(&(&y - &u)));
        for i in 0..oracle.size() {
            let rhs = &yt - &((theta() * DT) * &oracle.apply_direction(i, &y));
            yt = oracle.solve_splitting(i, &rhs, -theta() * DT).unwrap();
        }

        assert_close(&a, &yt);
    }

    /// Closed form on a diagonal operator: stage-1 Douglas-like corrections,
    /// then stage-2 `y0 + μ dt w (y − a)` with a second directional pass that
    /// reads the fixed post-stage-1 `y`.
    #[test]
    fn a_step_matches_the_closed_form_on_a_diagonal_operator() {
        let mut scheme = hundsdorfer(scaled_composite(&COEFFICIENTS), Vec::new());

        let u = probe(4);
        let mut a = u.clone();
        scheme.step(&mut a, T).unwrap();

        let mut y = &u * (1.0 + DT * WHOLE);
        let y0 = y.clone();
        for c in COEFFICIENTS {
            y = &(&y - &((theta() * DT * c) * &u)) / (1.0 - theta() * DT * c);
        }
        let mut yt = &y0 + &((MU * DT * WHOLE) * &(&y - &u));
        for c in COEFFICIENTS {
            yt = &(&yt - &((theta() * DT * c) * &y)) / (1.0 - theta() * DT * c);
        }

        assert_close(&a, &yt);
    }

    /// BC helper sees two apply cycles plus a final after-solving.
    #[test]
    fn a_step_sets_operator_and_runs_both_bc_apply_cycles() {
        let raw = scaled_composite(&COEFFICIENTS[..1]);
        let map: SharedMut<dyn FdmLinearOpComposite> = raw.clone();
        let (log, bc_set) = call_log();
        let mut scheme = hundsdorfer(map, bc_set);

        let t = DT - 5e-9;
        scheme.step(&mut probe(4), t).unwrap();

        assert_eq!(raw.borrow().last_set_time, Some((0.0, t)));
        assert_eq!(
            *log.borrow(),
            vec![
                "set_time:0".to_string(),
                "before_applying".to_string(),
                "after_applying".to_string(),
                "before_applying".to_string(),
                "after_applying".to_string(),
                "after_solving".to_string(),
            ]
        );
    }

    #[test]
    fn stepping_before_the_timestep_is_set_fails() {
        let mut scheme =
            HundsdorferScheme::new(theta(), MU, scaled_composite(&COEFFICIENTS), Vec::new());
        assert!(scheme.step(&mut probe(4), T).is_err());
    }

    #[test]
    fn a_step_towards_negative_time_fails() {
        let mut scheme = hundsdorfer(scaled_composite(&COEFFICIENTS), Vec::new());
        assert!(scheme.step(&mut probe(4), DT / 2.0).is_err());
    }
}
