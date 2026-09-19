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
//! (the engine owns the mesher transform). `toMatrixDecomp` is deferred with
//! the sparse-matrix work (#636).

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
    r_ts: Handle<dyn YieldTermStructure>,
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
    /// Fails if the mixed-derivative stencil cannot be built.
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
            r_ts,
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
            .current_link()?
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
        // Positive forward strip so f^β is well-defined.
        let layout = shared(FdmLinearOpLayout::new(vec![7, 9]));
        shared(UniformGridMesher::new(Shared::clone(&layout), &[(0.2, 2.0), (-1.0, 1.0)]).unwrap())
    }

    fn make_op(beta: Real, rho: Real, r: Real) -> (FdmSabrOp, Size) {
        let m = mesher();
        let n = m.layout().size();
        (
            FdmSabrOp::new(m, flat(r), 1.0, 0.35, beta, 1.0, rho).unwrap(),
            n,
        )
    }

    #[test]
    fn size_apply_split_and_zero_rho() {
        let (mut op, n) = make_op(0.5, -0.25, 0.0);
        assert_eq!(op.size(), 2);
        op.set_time(0.0, 1.0).unwrap();
        let u = Array::incremental(n, 1.0, 0.01);
        let full = op.apply(&u);
        let parts =
            &(&op.apply_direction(0, &u) + &op.apply_direction(1, &u)) + &op.apply_mixed(&u);
        for i in 0..full.size() {
            assert!((full[i] - parts[i]).abs() < 1e-12, "at {i}");
        }

        let (mut z, _) = make_op(0.5, 0.0, 0.0);
        z.set_time(0.0, 1.0).unwrap();
        let mixed = z.apply_mixed(&u);
        for i in 0..mixed.size() {
            assert!(mixed[i].abs() < 1e-14, "at {i}: {}", mixed[i]);
        }
    }

    #[test]
    fn flat_zero_rate_discount_on_constant_field() {
        let (mut op, n) = make_op(0.25, 0.25, 0.04);
        op.set_time(0.0, 1.0).unwrap();
        let ones = Array::filled(n, 1.0);
        let ax = op.apply_direction(0, &ones);
        let ay = op.apply_direction(1, &ones);
        // Constant field ⇒ pure diagonals −½ r on both maps (r flat ⇒ fwd = 0.04).
        for i in 0..ones.size() {
            assert!((ax[i] + 0.02).abs() < 1e-10, "mapF at {i}: {}", ax[i]);
            assert!((ay[i] + 0.02).abs() < 1e-10, "mapA at {i}: {}", ay[i]);
        }
    }

    #[test]
    fn solve_splitting_round_trips() {
        let (mut op, n) = make_op(0.6, 0.25, 0.0);
        op.set_time(0.0, 0.5).unwrap();
        let r = Array::incremental(n, 1.0, 0.05);
        let s = 0.01;
        for direction in [0, 1] {
            let x = op.solve_splitting(direction, &r, s).unwrap();
            let check = &x + &(s * &op.apply_direction(direction, &x));
            for i in 0..r.size() {
                assert!(
                    (check[i] - r[i]).abs() < 1e-10,
                    "dir {direction} at {i}: {} vs {}",
                    check[i],
                    r[i]
                );
            }
        }
    }
}
