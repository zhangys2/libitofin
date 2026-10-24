//! Second-order mixed derivative operator ∂²/∂x∂y.
//!
//! Port of `ql/methods/finitedifferences/operators/secondordermixedderivativeop.{hpp,cpp}`:
//! a [`NinePointLinearOp`] whose stencil approximates the mixed second
//! derivative along directions `(d0, d1)`. Corner and edge one-sided stencils
//! match QuantLib so `f(x,y) = x y` is recovered exactly on a uniform grid
//! (including boundaries).

use crate::errors::QlResult;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::shared::Shared;
use crate::types::Size;

use super::ninepointlinearop::{NinePointCoeffs, NinePointLinearOp};

/// The ∂²/∂x_{d0}∂x_{d1} operator on `mesher`
/// (`secondordermixedderivativeop.cpp:28-114`).
///
/// # Errors
///
/// Propagates inconsistent-direction errors from [`NinePointLinearOp::new`].
pub fn second_order_mixed_derivative_op(
    d0: Size,
    d1: Size,
    mesher: Shared<dyn FdmMesher>,
) -> QlResult<NinePointLinearOp> {
    let dim0 = mesher.layout().dim()[d0];
    let dim1 = mesher.layout().dim()[d1];

    NinePointLinearOp::with_coeffs(d0, d1, mesher, move |mesher, position| {
        let hm_d0 = mesher.dminus(position, d0);
        let hp_d0 = mesher.dplus(position, d0);
        let hm_d1 = mesher.dminus(position, d1);
        let hp_d1 = mesher.dplus(position, d1);

        let zetam1 = hm_d0 * (hm_d0 + hp_d0);
        let zeta0 = hm_d0 * hp_d0;
        let zetap1 = hp_d0 * (hm_d0 + hp_d0);
        let phim1 = hm_d1 * (hm_d1 + hp_d1);
        let phi0 = hm_d1 * hp_d1;
        let phip1 = hp_d1 * (hm_d1 + hp_d1);

        let c0 = position.coordinates()[d0];
        let c1 = position.coordinates()[d1];
        let mut c = NinePointCoeffs::default();

        if c0 == 0 && c1 == 0 {
            // lower left corner
            let v = 1.0 / (hp_d0 * hp_d1);
            c.a11 = v;
            c.a22 = v;
            c.a21 = -v;
            c.a12 = -v;
        } else if c0 == dim0 - 1 && c1 == 0 {
            // upper left corner
            let v = 1.0 / (hm_d0 * hp_d1);
            c.a01 = v;
            c.a12 = v;
            c.a11 = -v;
            c.a02 = -v;
        } else if c0 == 0 && c1 == dim1 - 1 {
            // lower right corner
            let v = 1.0 / (hp_d0 * hm_d1);
            c.a10 = v;
            c.a21 = v;
            c.a20 = -v;
            c.a11 = -v;
        } else if c0 == dim0 - 1 && c1 == dim1 - 1 {
            // upper right corner
            let v = 1.0 / (hm_d0 * hm_d1);
            c.a00 = v;
            c.a11 = v;
            c.a10 = -v;
            c.a01 = -v;
        } else if c0 == 0 {
            // lower side
            c.a10 = hp_d1 / (hp_d0 * phim1);
            c.a20 = -c.a10;
            c.a21 = (hp_d1 - hm_d1) / (hp_d0 * phi0);
            c.a11 = -c.a21;
            c.a22 = hm_d1 / (hp_d0 * phip1);
            c.a12 = -c.a22;
        } else if c0 == dim0 - 1 {
            // upper side
            c.a00 = hp_d1 / (hm_d0 * phim1);
            c.a10 = -c.a00;
            c.a11 = (hp_d1 - hm_d1) / (hm_d0 * phi0);
            c.a01 = -c.a11;
            c.a12 = hm_d1 / (hm_d0 * phip1);
            c.a02 = -c.a12;
        } else if c1 == 0 {
            // left side
            c.a01 = hp_d0 / (zetam1 * hp_d1);
            c.a02 = -c.a01;
            c.a12 = (hp_d0 - hm_d0) / (zeta0 * hp_d1);
            c.a11 = -c.a12;
            c.a22 = hm_d0 / (zetap1 * hp_d1);
            c.a21 = -c.a22;
        } else if c1 == dim1 - 1 {
            // right side
            c.a00 = hp_d0 / (zetam1 * hm_d1);
            c.a01 = -c.a00;
            c.a11 = (hp_d0 - hm_d0) / (zeta0 * hm_d1);
            c.a10 = -c.a11;
            c.a21 = hm_d0 / (zetap1 * hm_d1);
            c.a20 = -c.a21;
        } else {
            c.a00 = hp_d0 * hp_d1 / (zetam1 * phim1);
            c.a10 = -(hp_d0 - hm_d0) * hp_d1 / (zeta0 * phim1);
            c.a20 = -hm_d0 * hp_d1 / (zetap1 * phim1);
            c.a01 = -hp_d0 * (hp_d1 - hm_d1) / (zetam1 * phi0);
            c.a11 = (hp_d0 - hm_d0) * (hp_d1 - hm_d1) / (zeta0 * phi0);
            c.a21 = hm_d0 * (hp_d1 - hm_d1) / (zetap1 * phi0);
            c.a02 = -hp_d0 * hm_d1 / (zetam1 * phip1);
            c.a12 = hm_d1 * (hp_d0 - hm_d0) / (zeta0 * phip1);
            c.a22 = hm_d0 * hm_d1 / (zetap1 * phip1);
        }
        c
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::array::Array;
    use crate::methods::finitedifferences::meshers::UniformGridMesher;
    use crate::methods::finitedifferences::operators::{FdmLinearOp, FdmLinearOpLayout};
    use crate::shared::shared;
    use crate::types::Real;

    fn mesher() -> Shared<dyn FdmMesher> {
        let layout = shared(FdmLinearOpLayout::new(vec![5, 6]));
        shared(UniformGridMesher::new(Shared::clone(&layout), &[(0.0, 1.0), (-0.5, 1.5)]).unwrap())
    }

    fn sample(mesher: &dyn FdmMesher, f: impl Fn(Real, Real) -> Real) -> Array {
        let layout = mesher.layout();
        let mut values = Array::with_size(layout.size());
        let mut position = layout.begin();
        while position.index() < layout.size() {
            let x = mesher.location(&position, 0);
            let y = mesher.location(&position, 1);
            values[position.index()] = f(x, y);
            position.advance();
        }
        values
    }

    #[test]
    fn recovers_one_on_product_xy() {
        let mesher = mesher();
        let op = second_order_mixed_derivative_op(0, 1, Shared::clone(&mesher)).unwrap();
        let t = op.apply(&sample(&*mesher, |x, y| x * y));
        for i in 0..t.size() {
            assert!((t[i] - 1.0).abs() < 1e-12, "at {i}: {}", t[i]);
        }
    }

    #[test]
    fn annihilates_constants_and_pure_functions_of_one_variable() {
        let mesher = mesher();
        let op = second_order_mixed_derivative_op(0, 1, Shared::clone(&mesher)).unwrap();
        for (name, f) in [
            ("const", (|_x, _y| 3.0) as fn(Real, Real) -> Real),
            ("f(x)", |x, _y| x * x + 2.0 * x),
            ("g(y)", |_x, y| y.sin() + y * y),
        ] {
            let t = op.apply(&sample(&*mesher, f));
            for i in 0..t.size() {
                assert!(t[i].abs() < 1e-12, "{name} at {i}: {}", t[i]);
            }
        }
    }
}
