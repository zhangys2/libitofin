//! Trapezoidal BDF2 stepping.
//!
//! Port of `ql/methods/finitedifferences/schemes/trbdf2scheme.hpp`: a
//! trapezoidal predictor (QuantLib always uses Craig–Sneyd) over
//! `α Δt`, then a BDF2 corrector. The one-direction arm
//! (`hpp:123-125`) solves `(I − β A) fn = f` by splitting; the
//! multi-direction `BiCGstab` / `GMRES` branch (`hpp:126-150`) is deferred
//! the same way as [`ImplicitEulerScheme`](super::ImplicitEulerScheme), so
//! [`step`](Scheme::step) answers such an operator with an error.
//!
//! QuantLib's default [`FdmSchemeDesc::tr_bdf2`](crate::methods::finitedifferences::solvers::FdmSchemeDesc::tr_bdf2)
//! uses `α = 2 − √2` and `relTol = 1e-8`. `FdmBackwardSolver` always
//! constructs the predictor as Craig–Sneyd with `θ = μ = ½`
//! (`fdmbackwardsolver.cpp:179-186`).
//!
//! [`step`]: Scheme::step

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::methods::finitedifferences::operators::FdmLinearOpComposite;
use crate::methods::finitedifferences::utilities::FdmBoundaryConditionSet;
use crate::shared::SharedMut;
use crate::types::{Real, Time};
use crate::{fail, require};

use super::boundaryconditionschemehelper::BoundaryConditionSchemeHelper;
use super::craigsneydscheme::CraigSneydScheme;
use super::scheme::Scheme;

/// Trapezoidal BDF2 step (`trbdf2scheme.hpp:47`).
pub struct TrBDF2Scheme {
    dt: Option<Time>,
    beta: Option<Real>,
    alpha: Real,
    map: SharedMut<dyn FdmLinearOpComposite>,
    trapezoidal: CraigSneydScheme,
    bc_set: BoundaryConditionSchemeHelper,
}

impl TrBDF2Scheme {
    /// `TrBDF2Scheme(alpha, map, trapezoidalScheme, bcSet, relTol)`
    /// (`trbdf2scheme.hpp:85-93`).
    ///
    /// `relTol` and the C++ `solverType` only reach the deferred iterative
    /// branch, so they are omitted the same way
    /// [`ImplicitEulerScheme`](super::ImplicitEulerScheme) drops them.
    pub fn new(
        alpha: Real,
        map: SharedMut<dyn FdmLinearOpComposite>,
        trapezoidal: CraigSneydScheme,
        bc_set: FdmBoundaryConditionSet,
    ) -> Self {
        TrBDF2Scheme {
            dt: None,
            beta: None,
            alpha,
            map,
            trapezoidal,
            bc_set: BoundaryConditionSchemeHelper::new(bc_set),
        }
    }
}

impl Scheme for TrBDF2Scheme {
    /// `trbdf2scheme.hpp:96-99`: stores `dt` and
    /// `β = (1 − α) / (2 − α) · dt`.
    fn set_step(&mut self, dt: Time) {
        self.dt = Some(dt);
        self.beta = Some((1.0 - self.alpha) / (2.0 - self.alpha) * dt);
    }

    /// `trbdf2scheme.hpp:108-152`, the one-direction branch.
    ///
    /// The trapezoidal predictor sets the shared operator's time to
    /// `[t − α dt, t]`; the BDF2 solve uses that setting and does not call
    /// `set_time` again (`hpp:114-125`).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn step(&mut self, a: &mut Array, t: Time) -> QlResult<()> {
        let Some(dt) = self.dt else {
            fail!("the timestep is not set: call set_step before stepping");
        };
        let Some(beta) = self.beta else {
            fail!("the timestep is not set: call set_step before stepping");
        };
        require!(t - dt > -1e-8, "a step towards negative time given");

        let intermediate = dt * self.alpha;
        let mut f_star = a.clone();
        self.trapezoidal.set_step(intermediate);
        self.trapezoidal.step(&mut f_star, t)?;

        let start = (t - dt).max(0.0);
        {
            let mut map = self.map.borrow_mut();
            self.bc_set.set_time(start);
            self.bc_set.apply_before_solving(&mut *map, a);

            let size = map.size();
            if size != 1 {
                fail!(
                    "TrBDF2 over an operator splitting into {size} directions needs the \
                     iterative solvers deferred to #636"
                );
            }

            let inv_alpha = 1.0 / self.alpha;
            let one_m = 1.0 - self.alpha;
            let f = &(&(inv_alpha * &f_star) - &((one_m * one_m * inv_alpha) * &*a))
                / (2.0 - self.alpha);
            *a = map.solve_splitting(0, &f, -beta)?;
        }
        self.bc_set.apply_after_solving(a);
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

    const ALPHA: Real = 2.0 - std::f64::consts::SQRT_2;
    const CS_THETA: Real = 0.5;
    const CS_MU: Real = 0.5;
    const DT: Time = 0.1;
    const T: Time = 0.25;
    const C: Real = 0.3;

    fn tr_bdf2(
        map: SharedMut<dyn FdmLinearOpComposite>,
        bc_set: FdmBoundaryConditionSet,
    ) -> TrBDF2Scheme {
        let trapezoidal = CraigSneydScheme::new(CS_THETA, CS_MU, map.clone(), bc_set.clone());
        let mut scheme = TrBDF2Scheme::new(ALPHA, map, trapezoidal, bc_set);
        scheme.set_step(DT);
        scheme
    }

    fn craig_sneyd_step(u: &Array, dt: Time, c: Real) -> Array {
        // Mixed term is zero, so stage 1 is discarded and stage 2 starts from
        // the explicit update `y0` (`craigsneydscheme.cpp:53`).
        let y0 = u * (1.0 + dt * WHOLE);
        &(&y0 - &((CS_THETA * dt * c) * u)) / (1.0 - CS_THETA * dt * c)
    }

    /// Replay the C++ call sequence against the Black-Scholes operator.
    #[test]
    fn a_step_replays_the_cpp_sequence_on_the_black_scholes_operator() {
        let mesher = mesher();
        let map: SharedMut<dyn FdmLinearOpComposite> = shared_mut(black_scholes_op(&mesher));
        let mut scheme = tr_bdf2(map, Vec::new());

        let u = probe(GRID);
        let mut a = u.clone();
        scheme.step(&mut a, T).unwrap();

        let mut f_star = u.clone();
        let predictor_map: SharedMut<dyn FdmLinearOpComposite> =
            shared_mut(black_scholes_op(&mesher));
        let mut predictor = CraigSneydScheme::new(CS_THETA, CS_MU, predictor_map, Vec::new());
        predictor.set_step(ALPHA * DT);
        predictor.step(&mut f_star, T).unwrap();

        let inv_alpha = 1.0 / ALPHA;
        let one_m = 1.0 - ALPHA;
        let f = &(&(inv_alpha * &f_star) - &((one_m * one_m * inv_alpha) * &u)) / (2.0 - ALPHA);
        let beta = (1.0 - ALPHA) / (2.0 - ALPHA) * DT;
        let mut oracle = black_scholes_op(&mesher);
        oracle.set_time(T - ALPHA * DT, T).unwrap();
        let expected = oracle.solve_splitting(0, &f, -beta).unwrap();

        assert_close(&a, &expected);
    }

    /// Closed form on a one-direction diagonal operator: Craig–Sneyd over
    /// `α dt`, then the BDF2 combination and `f / (1 − β c)`.
    #[test]
    fn a_step_matches_the_closed_form_on_a_diagonal_operator() {
        let mut scheme = tr_bdf2(scaled_composite(&[C]), Vec::new());

        let u = probe(4);
        let mut a = u.clone();
        scheme.step(&mut a, T).unwrap();

        let f_star = craig_sneyd_step(&u, ALPHA * DT, C);
        let inv_alpha = 1.0 / ALPHA;
        let one_m = 1.0 - ALPHA;
        let f = &(&(inv_alpha * &f_star) - &((one_m * one_m * inv_alpha) * &u)) / (2.0 - ALPHA);
        let beta = (1.0 - ALPHA) / (2.0 - ALPHA) * DT;
        let expected = &f / (1.0 - beta * C);

        assert_close(&a, &expected);
    }

    /// The predictor's two apply cycles run first; the BDF2 corrector then
    /// uses `apply_before_solving` / `apply_after_solving`. The operator time
    /// stays the predictor's `[t − α dt, t]`.
    #[test]
    fn a_step_runs_the_predictor_then_the_bdf2_solving_calls() {
        let raw = scaled_composite(&[C]);
        let map: SharedMut<dyn FdmLinearOpComposite> = raw.clone();
        let (log, bc_set) = call_log();
        let mut scheme = tr_bdf2(map, bc_set);

        scheme.step(&mut probe(4), T).unwrap();

        let cs_start = T - DT * ALPHA;
        assert_eq!(raw.borrow().last_set_time, Some((cs_start, T)));
        let log = log.borrow();
        assert_eq!(log.len(), 9, "{log:?}");
        assert_eq!(log[0], format!("set_time:{cs_start}"));
        assert_eq!(
            &log[1..6],
            [
                "before_applying".to_string(),
                "after_applying".to_string(),
                "before_applying".to_string(),
                "after_applying".to_string(),
                "after_solving".to_string(),
            ]
        );
        assert_eq!(log[6], format!("set_time:{}", T - DT));
        assert_eq!(
            &log[7..],
            ["before_solving".to_string(), "after_solving".to_string()]
        );
    }

    /// The deferral of `hpp:126-150` is visible: more than one direction is
    /// refused rather than run through the one-direction arm.
    #[test]
    fn a_multi_direction_operator_reports_the_deferred_iterative_solvers() {
        let mut scheme = tr_bdf2(scaled_composite(&[C, -0.45]), Vec::new());
        let error = scheme.step(&mut probe(4), T).unwrap_err();
        assert!(error.message().contains("#636"), "{error}");
    }

    #[test]
    fn stepping_before_the_timestep_is_set_fails() {
        let map = scaled_composite(&[C]);
        let trapezoidal = CraigSneydScheme::new(CS_THETA, CS_MU, map.clone(), Vec::new());
        let mut scheme = TrBDF2Scheme::new(ALPHA, map, trapezoidal, Vec::new());
        assert!(scheme.step(&mut probe(4), T).is_err());
    }

    #[test]
    fn a_step_towards_negative_time_fails() {
        let mut scheme = tr_bdf2(scaled_composite(&[C]), Vec::new());
        assert!(scheme.step(&mut probe(4), DT / 2.0).is_err());
    }
}
