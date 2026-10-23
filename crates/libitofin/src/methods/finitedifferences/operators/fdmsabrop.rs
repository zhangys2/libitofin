//! Finite-difference generator for the SABR model.
//!
//! Port of `ql/methods/finitedifferences/operators/fdmsabrop.{hpp,cpp}`:
//! [`FdmSabrOp`] is the two-factor [`FdmLinearOpComposite`] for
//!
//! ```text
//! df = α f^β dW
//! dα = ν α dZ
//! ⟨dW, dZ⟩ = ρ dt
//! ```
//!
//! with absorbing boundary at `f = 0`. Direction 0 is the forward, direction 1
//! is `log α`. `f0` / `alpha` match the QL ctor but are unused by the operator
//! (the engine owns the mesher transform). The yield curve is snapshotted at
//! construction (QL `shared_ptr`, same as [`FdmBlackScholesOp`]).
//! `toMatrixDecomp` is deferred with the sparse-matrix work (#636).

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size, Time};

use super::fdmlinearop::FdmLinearOp;
use super::fdmlinearopcomposite::FdmLinearOpComposite;
use super::firstderivativeop::first_derivative_op;
use super::ninepointlinearop::NinePointLinearOp;
use super::secondderivativeop::second_derivative_op;
use super::secondordermixedderivativeop::second_order_mixed_derivative_op;
use super::triplebandlinearop::TripleBandLinearOp;

/// SABR finite-difference generator (`fdmsabrop.hpp`).
pub struct FdmSabrOp {
    /// Frozen curve link (QL stores `shared_ptr`, not a live handle).
    r_ts: Shared<dyn YieldTermStructure>,
    dff_map: TripleBandLinearOp,
    dx_map: TripleBandLinearOp,
    dxx_map: TripleBandLinearOp,
    correlation_map: NinePointLinearOp,
    map_f: TripleBandLinearOp,
    map_a: TripleBandLinearOp,
}

impl FdmSabrOp {
    /// `FdmSabrOp(mesher, rTS, f0, alpha, beta, nu, rho)`.
    ///
    /// # Errors
    ///
    /// Fails if the yield handle is empty or the mixed-derivative stencil
    /// cannot be built.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mesher: Shared<dyn FdmMesher>,
        r_ts: Handle<dyn YieldTermStructure>,
        _f0: Real,
        _alpha: Real,
        beta: Real,
        nu: Real,
        rho: Real,
    ) -> QlResult<Self> {
        let size = mesher.layout().size();
        let f = mesher.locations(0);
        let log_alpha = mesher.locations(1);

        let dff_map = second_derivative_op(0, Shared::clone(&mesher))
            .mult(&(0.5 * &(&log_alpha.exp().pow(2.0) * &f.pow(2.0 * beta))));
        let dx_map = first_derivative_op(1, Shared::clone(&mesher))
            .mult(&Array::filled(size, -0.5 * nu * nu));
        let dxx_map = second_derivative_op(1, Shared::clone(&mesher))
            .mult(&Array::filled(size, 0.5 * nu * nu));
        let correlation_map = second_order_mixed_derivative_op(0, 1, Shared::clone(&mesher))?
            .mult(&(rho * nu * &(&log_alpha.exp() * &f.pow(beta))));

        Ok(Self {
            r_ts: r_ts.current_link()?,
            dff_map,
            dx_map,
            dxx_map,
            correlation_map,
            map_f: TripleBandLinearOp::new(0, Shared::clone(&mesher)),
            map_a: TripleBandLinearOp::new(1, mesher),
        })
    }
}

impl FdmLinearOp for FdmSabrOp {
    fn apply(&self, r: &Array) -> Array {
        &(&self.map_f.apply(r) + &self.map_a.apply(r)) + &self.apply_mixed(r)
    }
}

impl FdmLinearOpComposite for FdmSabrOp {
    fn size(&self) -> Size {
        2
    }

    fn set_time(&mut self, t1: Time, t2: Time) -> QlResult<()> {
        let r = self
            .r_ts
            .forward_rate(
                t1,
                t2,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let half_r = Array::from([-0.5 * r]);
        self.map_f
            .axpyb(&Array::new(), &self.dff_map, &self.dff_map, &half_r);
        self.map_a
            .axpyb(&Array::from([1.0]), &self.dx_map, &self.dxx_map, &half_r);
        Ok(())
    }

    fn apply_mixed(&self, r: &Array) -> Array {
        self.correlation_map.apply(r)
    }

    fn apply_direction(&self, direction: Size, r: &Array) -> Array {
        match direction {
            0 => self.map_f.apply(r),
            1 => self.map_a.apply(r),
            _ => Array::with_size(r.size()),
        }
    }

    fn solve_splitting(&self, direction: Size, r: &Array, s: Real) -> QlResult<Array> {
        match direction {
            0 => self.map_f.solve_splitting(r, s, 1.0),
            1 => self.map_a.solve_splitting(r, s, 1.0),
            _ => fail!("direction too large"),
        }
    }

    fn preconditioner(&self, r: &Array, dt: Real) -> QlResult<Array> {
        self.solve_splitting(1, &self.solve_splitting(0, r, dt)?, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::finitedifferences::meshers::UniformGridMesher;
    use crate::methods::finitedifferences::operators::FdmLinearOpLayout;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;

    const BETA: Real = 0.5;
    const NU: Real = 0.8;
    const RHO: Real = -0.35;
    const R: Real = 0.04;
    const TOL: Real = 1e-10;

    fn flat(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            Date::new(22, Month::February, 2018),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn mesher() -> Shared<dyn FdmMesher> {
        let layout = shared(FdmLinearOpLayout::new(vec![11, 9]));
        shared(UniformGridMesher::new(Shared::clone(&layout), &[(0.2, 2.0), (-1.0, 1.0)]).unwrap())
    }

    #[test]
    fn closed_form_generator_on_polynomials() {
        let m = mesher();
        let layout = m.layout();
        let nf = layout.dim()[0];
        let nx = layout.dim()[1];
        let f_loc = m.locations(0);
        let x_loc = m.locations(1);

        let mut op = FdmSabrOp::new(Shared::clone(&m), flat(R), 1.0, 0.35, BETA, NU, RHO).unwrap();
        op.set_time(0.0, 1.0).unwrap();

        let mut u_f2 = Array::with_size(layout.size());
        let mut u_x2 = Array::with_size(layout.size());
        let mut u_fx = Array::with_size(layout.size());
        for iter in layout.iter() {
            let i = iter.index();
            let f = f_loc[i];
            let x = x_loc[i];
            u_f2[i] = f * f;
            u_x2[i] = x * x;
            u_fx[i] = f * x;
        }

        let lf2 = op.apply(&u_f2);
        let lx2 = op.apply(&u_x2);
        let lfx = op.apply(&u_fx);

        for iter in layout.iter() {
            let (i_f, i_x) = (iter.coordinates()[0], iter.coordinates()[1]);
            if i_f == 0 || i_f + 1 == nf || i_x == 0 || i_x + 1 == nx {
                continue;
            }
            let i = iter.index();
            let f = f_loc[i];
            let x = x_loc[i];
            let e2x = (2.0 * x).exp();
            let ex = x.exp();

            let e_f2 = e2x * f.powf(2.0 * BETA) - R * f * f;
            let e_x2 = -NU * NU * x + NU * NU - R * x * x;
            let e_fx = -0.5 * NU * NU * f + RHO * NU * ex * f.powf(BETA) - R * f * x;

            assert!(
                (lf2[i] - e_f2).abs() < TOL,
                "L[f²] at {i}: {} vs {e_f2}",
                lf2[i]
            );
            assert!(
                (lx2[i] - e_x2).abs() < TOL,
                "L[x²] at {i}: {} vs {e_x2}",
                lx2[i]
            );
            assert!(
                (lfx[i] - e_fx).abs() < TOL,
                "L[fx] at {i}: {} vs {e_fx}",
                lfx[i]
            );
        }

        // ρ = 0 kills the mixed contribution on u = fx (other terms unchanged).
        let mut z = FdmSabrOp::new(m, flat(R), 1.0, 0.35, BETA, NU, 0.0).unwrap();
        z.set_time(0.0, 1.0).unwrap();
        let mixed = z.apply_mixed(&u_fx);
        for i in 0..mixed.size() {
            assert!(mixed[i].abs() < 1e-14, "at {i}: {}", mixed[i]);
        }
    }
}
