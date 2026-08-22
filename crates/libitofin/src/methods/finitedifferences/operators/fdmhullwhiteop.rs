//! Finite-difference generator for the Hull–White one-factor short-rate model.
//!
//! Port of `ql/methods/finitedifferences/operators/fdmhullwhiteop.{hpp,cpp}`:
//! [`FdmHullWhiteOp`] is the one-direction [`FdmLinearOpComposite`]
//!
//! ```text
//! A = −a x ∂/∂x + ½ σ² ∂²/∂x² − (x + φ̄) I
//! ```
//!
//! with `φ̄ = ½ (φ(t₁) + φ(t₂))` rebuilt on every
//! [`set_time`](FdmLinearOpComposite::set_time) from the model's analytic
//! dynamics (`shortRate(t, 0)`). There is no mixed term (`size() = 1`).
//!
//! `toMatrixDecomp` is omitted with the sparse-matrix work (#636).

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::models::shortrate::HullWhite;
use crate::shared::{Shared, SharedMut};
use crate::types::{Real, Size, Time};

use super::fdmlinearop::FdmLinearOp;
use super::fdmlinearopcomposite::FdmLinearOpComposite;
use super::firstderivativeop::first_derivative_op;
use super::secondderivativeop::second_derivative_op;
use super::triplebandlinearop::TripleBandLinearOp;

/// Hull–White finite-difference generator (`fdmhullwhiteop.hpp:35`).
pub struct FdmHullWhiteOp {
    direction: Size,
    x: Array,
    dz_map: TripleBandLinearOp,
    map_t: TripleBandLinearOp,
    model: SharedMut<HullWhite>,
}

impl FdmHullWhiteOp {
    /// `FdmHullWhiteOp(mesher, model, direction)` (`fdmhullwhiteop.cpp:32-44`).
    ///
    /// The convection/diffusion map is built once from `(a, σ)`; the
    /// time-dependent discount diagonal is filled by
    /// [`set_time`](FdmLinearOpComposite::set_time).
    pub fn new(
        mesher: Shared<dyn FdmMesher>,
        model: SharedMut<HullWhite>,
        direction: Size,
    ) -> FdmHullWhiteOp {
        let (a, sigma) = {
            let hw = model.borrow();
            (hw.a(), hw.sigma())
        };
        let size = mesher.layout().size();
        let x = mesher.locations(direction);
        let dz_map = first_derivative_op(direction, Shared::clone(&mesher))
            .mult(&(-a * &x))
            .add_op(
                &second_derivative_op(direction, Shared::clone(&mesher))
                    .mult(&Array::filled(size, 0.5 * sigma * sigma)),
            );

        FdmHullWhiteOp {
            direction,
            x,
            dz_map,
            map_t: TripleBandLinearOp::new(direction, mesher),
            model,
        }
    }
}

impl FdmLinearOp for FdmHullWhiteOp {
    /// `apply` (`fdmhullwhiteop.cpp:61-63`).
    fn apply(&self, r: &Array) -> Array {
        self.map_t.apply(r)
    }
}

impl FdmLinearOpComposite for FdmHullWhiteOp {
    fn size(&self) -> Size {
        1
    }

    /// `setTime` (`fdmhullwhiteop.cpp:48-57`): average fitting short rate on
    /// the discount diagonal.
    fn set_time(&mut self, t1: Time, t2: Time) -> QlResult<()> {
        let phi = {
            let model = self.model.borrow();
            let dynamics = model.dynamics()?;
            0.5 * (dynamics.short_rate(t1, 0.0) + dynamics.short_rate(t2, 0.0))
        };
        let discount = -&(&self.x + phi);
        self.map_t
            .axpyb(&Array::new(), &self.dz_map, &self.dz_map, &discount);
        Ok(())
    }

    fn apply_mixed(&self, r: &Array) -> Array {
        Array::with_size(r.size())
    }

    fn apply_direction(&self, direction: Size, r: &Array) -> Array {
        if direction == self.direction {
            self.map_t.apply(r)
        } else {
            Array::with_size(r.size())
        }
    }

    fn solve_splitting(&self, direction: Size, r: &Array, s: Real) -> QlResult<Array> {
        if direction == self.direction {
            self.map_t.solve_splitting(r, s, 1.0)
        } else {
            Ok(Array::with_size(r.size()))
        }
    }

    fn preconditioner(&self, r: &Array, s: Real) -> QlResult<Array> {
        self.solve_splitting(self.direction, r, s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::interestrate::Compounding;
    use crate::methods::finitedifferences::meshers::UniformGridMesher;
    use crate::methods::finitedifferences::operators::FdmLinearOpLayout;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    const A: Real = 0.1;
    const SIGMA: Real = 0.01;

    fn flat_curve() -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            Date::new(15, Month::January, 2026),
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn mesher_1d() -> Shared<dyn FdmMesher> {
        let layout = shared(FdmLinearOpLayout::new(vec![11]));
        shared(UniformGridMesher::new(Shared::clone(&layout), &[(-0.05, 0.05)]).unwrap())
    }

    fn make_op() -> FdmHullWhiteOp {
        let model = HullWhite::new(flat_curve(), A, SIGMA).unwrap();
        FdmHullWhiteOp::new(mesher_1d(), model, 0)
    }

    #[test]
    fn size_is_one() {
        assert_eq!(make_op().size(), 1);
    }

    #[test]
    fn apply_equals_the_home_direction() {
        let mut op = make_op();
        op.set_time(0.0, 0.25).unwrap();
        let r = Array::incremental(op.x.size(), 1.0, 0.01);
        let full = op.apply(&r);
        let dir = op.apply_direction(0, &r);
        for i in 0..full.size() {
            assert!(
                (full[i] - dir[i]).abs() < 1e-12,
                "at {i}: {} vs {}",
                full[i],
                dir[i]
            );
        }
    }

    #[test]
    fn mixed_term_is_zero() {
        let mut op = make_op();
        op.set_time(0.0, 1.0).unwrap();
        let r = Array::incremental(op.x.size(), 0.5, 0.1);
        let mixed = op.apply_mixed(&r);
        for i in 0..mixed.size() {
            assert!(mixed[i].abs() < 1e-14, "at {i}: {}", mixed[i]);
        }
    }

    #[test]
    fn set_time_puts_average_phi_discount_on_the_map() {
        let model = HullWhite::new(flat_curve(), A, SIGMA).unwrap();
        let phi = {
            let hw = model.borrow();
            0.5 * (hw.phi(0.0) + hw.phi(1.0))
        };
        let mesher = mesher_1d();
        let mut op = FdmHullWhiteOp::new(Shared::clone(&mesher), SharedMut::clone(&model), 0);
        op.set_time(0.0, 1.0).unwrap();

        let ones = Array::filled(mesher.layout().size(), 1.0);
        let x = mesher.locations(0);
        let ax = op.apply_direction(0, &ones);
        for i in 1..ones.size() - 1 {
            let expected = -(x[i] + phi);
            assert!(
                (ax[i] - expected).abs() < 1e-12,
                "mapT at {i}: {} vs {expected}",
                ax[i]
            );
        }
    }

    #[test]
    fn solve_splitting_recovers_rhs_along_the_direction() {
        let mut op = make_op();
        op.set_time(0.0, 0.5).unwrap();
        let r = Array::incremental(op.x.size(), 1.0, 0.05);
        let s = 0.01;
        let x = op.solve_splitting(0, &r, s).unwrap();
        let check = &x + &(s * &op.apply_direction(0, &x));
        for i in 0..r.size() {
            assert!(
                (check[i] - r[i]).abs() < 1e-9,
                "at {i}: {} vs {}",
                check[i],
                r[i]
            );
        }
    }

    #[test]
    fn foreign_direction_solve_returns_zeros() {
        let mut op = make_op();
        op.set_time(0.0, 0.5).unwrap();
        let r = Array::filled(op.x.size(), 3.0);
        let out = op.solve_splitting(1, &r, 0.1).unwrap();
        for i in 0..out.size() {
            assert_eq!(out[i], 0.0);
        }
    }
}
