//! Finite-difference generator for the G2++ two-factor short-rate model.
//!
//! Port of `ql/methods/finitedifferences/operators/fdmg2op.{hpp,cpp}`:
//! [`FdmG2Op`] is the two-direction [`FdmLinearOpComposite`] whose directional
//! pieces are
//!
//! ```text
//! A_x = -a x ∂/∂x + ½ σ² ∂²/∂x² - ½ (x + y + φ̄) I
//! A_y = -b y ∂/∂y + ½ η² ∂²/∂y² - ½ (x + y + φ̄) I
//! A_xy = ρ σ η ∂²/∂x∂y
//! ```
//!
//! with `φ̄ = ½ (φ(t₁) + φ(t₂))` rebuilt on every
//! [`set_time`](FdmLinearOpComposite::set_time).
//!
//! `toMatrixDecomp` is omitted with the sparse-matrix work (#636).

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::models::shortrate::{G2, TwoFactorShortRateDynamics};
use crate::shared::{Shared, SharedMut};
use crate::types::{Real, Size, Time};

use super::fdmlinearop::FdmLinearOp;
use super::fdmlinearopcomposite::FdmLinearOpComposite;
use super::firstderivativeop::first_derivative_op;
use super::ninepointlinearop::NinePointLinearOp;
use super::secondderivativeop::second_derivative_op;
use super::secondordermixedderivativeop::second_order_mixed_derivative_op;
use super::triplebandlinearop::TripleBandLinearOp;

/// G2++ finite-difference generator (`fdmg2op.hpp:39`).
pub struct FdmG2Op {
    direction1: Size,
    direction2: Size,
    x: Array,
    y: Array,
    dx_map: TripleBandLinearOp,
    dy_map: TripleBandLinearOp,
    corr_map: NinePointLinearOp,
    map_x: TripleBandLinearOp,
    map_y: TripleBandLinearOp,
    model: SharedMut<G2>,
}

impl FdmG2Op {
    /// `FdmG2Op(mesher, model, direction1, direction2)` (`fdmg2op.cpp:33-55`).
    ///
    /// Directional diffusion/convection maps are built once from the model's
    /// `(a,σ,b,η,ρ)`; the time-dependent discount diagonal is filled by
    /// [`set_time`](FdmLinearOpComposite::set_time).
    ///
    /// # Errors
    ///
    /// Fails if the mixed-derivative stencil cannot be built (bad directions).
    pub fn new(
        mesher: Shared<dyn FdmMesher>,
        model: SharedMut<G2>,
        direction1: Size,
        direction2: Size,
    ) -> QlResult<FdmG2Op> {
        let g2 = model.borrow();
        let size = mesher.layout().size();
        let x = mesher.locations(direction1);
        let y = mesher.locations(direction2);

        let dx_map = first_derivative_op(direction1, Shared::clone(&mesher))
            .mult(&(-g2.a() * &x))
            .add_op(
                &second_derivative_op(direction1, Shared::clone(&mesher))
                    .mult(&Array::filled(size, 0.5 * g2.sigma() * g2.sigma())),
            );
        let dy_map = first_derivative_op(direction2, Shared::clone(&mesher))
            .mult(&(-g2.b() * &y))
            .add_op(
                &second_derivative_op(direction2, Shared::clone(&mesher))
                    .mult(&Array::filled(size, 0.5 * g2.eta() * g2.eta())),
            );
        let corr_map =
            second_order_mixed_derivative_op(direction1, direction2, Shared::clone(&mesher))?
                .mult(&Array::filled(size, g2.rho() * g2.sigma() * g2.eta()));
        drop(g2);

        Ok(FdmG2Op {
            direction1,
            direction2,
            x,
            y,
            dx_map,
            dy_map,
            corr_map,
            map_x: TripleBandLinearOp::new(direction1, Shared::clone(&mesher)),
            map_y: TripleBandLinearOp::new(direction2, mesher),
            model,
        })
    }
}

impl FdmLinearOp for FdmG2Op {
    /// `apply` (`fdmg2op.cpp:72-74`).
    fn apply(&self, r: &Array) -> Array {
        &(&self.map_x.apply(r) + &self.map_y.apply(r)) + &self.apply_mixed(r)
    }
}

impl FdmLinearOpComposite for FdmG2Op {
    fn size(&self) -> Size {
        2
    }

    /// `setTime` (`fdmg2op.cpp:59-70`): average fitting short rate on the
    /// diagonal of both directional maps.
    fn set_time(&mut self, t1: Time, t2: Time) -> QlResult<()> {
        let model = self.model.borrow();
        let dynamics = model.dynamics()?;
        let phi = 0.5 * (dynamics.short_rate(t1, 0.0, 0.0) + dynamics.short_rate(t2, 0.0, 0.0));
        drop(model);

        let hr = -0.5 * &(&(&self.x + &self.y) + phi);
        self.map_x
            .axpyb(&Array::new(), &self.dx_map, &self.dx_map, &hr);
        self.map_y
            .axpyb(&Array::new(), &self.dy_map, &self.dy_map, &hr);
        Ok(())
    }

    fn apply_mixed(&self, r: &Array) -> Array {
        self.corr_map.apply(r)
    }

    fn apply_direction(&self, direction: Size, r: &Array) -> Array {
        if direction == self.direction1 {
            self.map_x.apply(r)
        } else if direction == self.direction2 {
            self.map_y.apply(r)
        } else {
            Array::with_size(r.size())
        }
    }

    fn solve_splitting(&self, direction: Size, r: &Array, s: Real) -> QlResult<Array> {
        if direction == self.direction1 {
            self.map_x.solve_splitting(r, s, 1.0)
        } else if direction == self.direction2 {
            self.map_y.solve_splitting(r, s, 1.0)
        } else {
            // Matches C++ `fdmg2op.cpp:99` (zeros, not the identity).
            Ok(Array::with_size(r.size()))
        }
    }

    fn preconditioner(&self, r: &Array, s: Real) -> QlResult<Array> {
        self.solve_splitting(self.direction1, r, s)
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
    const B: Real = 0.2;
    const ETA: Real = 0.008;

    fn flat_curve() -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            Date::new(15, Month::January, 2026),
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn mesher_2d() -> Shared<dyn FdmMesher> {
        let layout = shared(FdmLinearOpLayout::new(vec![7, 9]));
        shared(
            UniformGridMesher::new(Shared::clone(&layout), &[(-0.05, 0.05), (-0.04, 0.04)])
                .unwrap(),
        )
    }

    fn make_op(rho: Real) -> FdmG2Op {
        let model = G2::new(flat_curve(), A, SIGMA, B, ETA, rho).unwrap();
        FdmG2Op::new(mesher_2d(), model, 0, 1).unwrap()
    }

    #[test]
    fn size_is_two() {
        assert_eq!(make_op(-0.75).size(), 2);
    }

    #[test]
    fn apply_splits_into_directions_plus_mixed() {
        let mut op = make_op(-0.75);
        op.set_time(0.0, 0.25).unwrap();
        let r = Array::incremental(op.x.size(), 1.0, 0.01);
        let full = op.apply(&r);
        let parts =
            &(&op.apply_direction(0, &r) + &op.apply_direction(1, &r)) + &op.apply_mixed(&r);
        for i in 0..full.size() {
            assert!(
                (full[i] - parts[i]).abs() < 1e-12,
                "at {i}: {} vs {}",
                full[i],
                parts[i]
            );
        }
    }

    #[test]
    fn zero_rho_kills_the_mixed_term() {
        let mut op = make_op(0.0);
        op.set_time(0.0, 1.0).unwrap();
        let r = Array::incremental(op.x.size(), 0.5, 0.1);
        let mixed = op.apply_mixed(&r);
        for i in 0..mixed.size() {
            assert!(mixed[i].abs() < 1e-14, "at {i}: {}", mixed[i]);
        }
    }

    #[test]
    fn set_time_puts_average_phi_discount_on_both_maps() {
        let model = G2::new(flat_curve(), A, SIGMA, B, ETA, -0.5).unwrap();
        let phi = {
            let g = model.borrow();
            0.5 * (g.phi(0.0) + g.phi(1.0))
        };
        let mesher = mesher_2d();
        let mut op = FdmG2Op::new(Shared::clone(&mesher), SharedMut::clone(&model), 0, 1).unwrap();
        op.set_time(0.0, 1.0).unwrap();

        // On a constant field the derivative bands vanish on a uniform grid's
        // interior for second derivatives of a constant and first derivatives
        // of a constant — both are zero — so apply_direction reduces to the
        // diagonal discount -½(x+y+φ).
        let ones = Array::filled(mesher.layout().size(), 1.0);
        let x = mesher.locations(0);
        let y = mesher.locations(1);
        let ax = op.apply_direction(0, &ones);
        let ay = op.apply_direction(1, &ones);
        for i in 0..ones.size() {
            let expected = -0.5 * (x[i] + y[i] + phi);
            assert!(
                (ax[i] - expected).abs() < 1e-12,
                "mapX at {i}: {} vs {expected}",
                ax[i]
            );
            assert!(
                (ay[i] - expected).abs() < 1e-12,
                "mapY at {i}: {} vs {expected}",
                ay[i]
            );
        }
    }

    #[test]
    fn solve_splitting_recovers_rhs_along_each_direction() {
        let mut op = make_op(-0.75);
        op.set_time(0.0, 0.5).unwrap();
        let r = Array::incremental(op.x.size(), 1.0, 0.05);
        let s = 0.01;
        for direction in [0, 1] {
            let x = op.solve_splitting(direction, &r, s).unwrap();
            let check = &x + &(s * &op.apply_direction(direction, &x));
            for i in 0..r.size() {
                assert!(
                    (check[i] - r[i]).abs() < 1e-9,
                    "dir {direction} at {i}: {} vs {}",
                    check[i],
                    r[i]
                );
            }
        }
    }

    #[test]
    fn foreign_direction_solve_returns_zeros() {
        let mut op = make_op(-0.75);
        op.set_time(0.0, 0.5).unwrap();
        let r = Array::filled(op.x.size(), 3.0);
        let out = op.solve_splitting(2, &r, 0.1).unwrap();
        for i in 0..out.size() {
            assert_eq!(out[i], 0.0);
        }
    }
}
