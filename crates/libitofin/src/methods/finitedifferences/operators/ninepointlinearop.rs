//! Nine-point linear operator over two mesh directions.
//!
//! Port of `ql/methods/finitedifferences/operators/ninepointlinearop.{hpp,cpp}`:
//! stores a 3×3 stencil of coefficients around each grid point in directions
//! `(d0, d1)` and applies them against the eight neighbours plus the centre.
//!
//! Index naming follows QuantLib: digit `0/1/2` means step `-1/0/+1` along
//! `d0` (first digit) and `d1` (second digit). The centre uses coefficient
//! `a11` against the point itself (no separate `i11` index).
//!
//! `toMatrix` is omitted with the rest of the sparse-matrix work (#636).

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::require;
use crate::shared::Shared;
use crate::types::{Real, Size};

use super::fdmlinearop::FdmLinearOp;
use super::fdmlinearopiterator::FdmLinearOpIterator;

/// The nine stencil weights at one grid point
/// (`a00`…`a22` in `ninepointlinearop.hpp`).
#[derive(Clone, Copy, Debug, Default)]
pub struct NinePointCoeffs {
    pub a00: Real,
    pub a10: Real,
    pub a20: Real,
    pub a01: Real,
    pub a11: Real,
    pub a21: Real,
    pub a02: Real,
    pub a12: Real,
    pub a22: Real,
}

/// A nine-point operator over two directions of an [`FdmMesher`].
#[derive(Clone)]
pub struct NinePointLinearOp {
    d0: Size,
    d1: Size,
    i00: Vec<Size>,
    i10: Vec<Size>,
    i20: Vec<Size>,
    i01: Vec<Size>,
    i21: Vec<Size>,
    i02: Vec<Size>,
    i12: Vec<Size>,
    i22: Vec<Size>,
    a00: Vec<Real>,
    a10: Vec<Real>,
    a20: Vec<Real>,
    a01: Vec<Real>,
    a11: Vec<Real>,
    a21: Vec<Real>,
    a02: Vec<Real>,
    a12: Vec<Real>,
    a22: Vec<Real>,
    mesher: Shared<dyn FdmMesher>,
}

impl NinePointLinearOp {
    /// The operator over directions `(d0, d1)` of `mesher`, with all nine
    /// coefficients zero (`ninepointlinearop.cpp:29-68`).
    ///
    /// # Errors
    ///
    /// Fails when `d0 == d1` or either direction is out of range for the mesh.
    pub fn new(d0: Size, d1: Size, mesher: Shared<dyn FdmMesher>) -> QlResult<Self> {
        let layout = Shared::clone(mesher.layout());
        let dim = layout.dim().len();
        require!(
            d0 != d1 && d0 < dim && d1 < dim,
            "inconsistent derivative directions"
        );

        let size = layout.size();
        let mut i00 = vec![0; size];
        let mut i10 = vec![0; size];
        let mut i20 = vec![0; size];
        let mut i01 = vec![0; size];
        let mut i21 = vec![0; size];
        let mut i02 = vec![0; size];
        let mut i12 = vec![0; size];
        let mut i22 = vec![0; size];

        let mut position = layout.begin();
        while position.index() < size {
            let i = position.index();
            i10[i] = layout.neighbourhood(&position, d1, -1);
            i01[i] = layout.neighbourhood(&position, d0, -1);
            i21[i] = layout.neighbourhood(&position, d0, 1);
            i12[i] = layout.neighbourhood(&position, d1, 1);
            i00[i] = layout.neighbourhood2(&position, d0, -1, d1, -1);
            i20[i] = layout.neighbourhood2(&position, d0, 1, d1, -1);
            i02[i] = layout.neighbourhood2(&position, d0, -1, d1, 1);
            i22[i] = layout.neighbourhood2(&position, d0, 1, d1, 1);
            position.advance();
        }

        Ok(NinePointLinearOp {
            d0,
            d1,
            i00,
            i10,
            i20,
            i01,
            i21,
            i02,
            i12,
            i22,
            a00: vec![0.0; size],
            a10: vec![0.0; size],
            a20: vec![0.0; size],
            a01: vec![0.0; size],
            a11: vec![0.0; size],
            a21: vec![0.0; size],
            a02: vec![0.0; size],
            a12: vec![0.0; size],
            a22: vec![0.0; size],
            mesher,
        })
    }

    /// Builds the operator and fills coefficients via `coeffs`
    /// (Rust form of C++ derived constructors writing protected `a**_` bands).
    ///
    /// # Errors
    ///
    /// Propagates [`new`](Self::new) direction errors.
    pub fn with_coeffs(
        d0: Size,
        d1: Size,
        mesher: Shared<dyn FdmMesher>,
        mut coeffs: impl FnMut(&dyn FdmMesher, &FdmLinearOpIterator) -> NinePointCoeffs,
    ) -> QlResult<Self> {
        let mut operator = Self::new(d0, d1, Shared::clone(&mesher))?;
        let layout = Shared::clone(mesher.layout());
        let mut position = layout.begin();
        while position.index() < layout.size() {
            let i = position.index();
            let c = coeffs(&*mesher, &position);
            operator.a00[i] = c.a00;
            operator.a10[i] = c.a10;
            operator.a20[i] = c.a20;
            operator.a01[i] = c.a01;
            operator.a11[i] = c.a11;
            operator.a21[i] = c.a21;
            operator.a02[i] = c.a02;
            operator.a12[i] = c.a12;
            operator.a22[i] = c.a22;
            position.advance();
        }
        Ok(operator)
    }

    /// First derivative direction `d0`.
    pub fn d0(&self) -> Size {
        self.d0
    }

    /// Second derivative direction `d1`.
    pub fn d1(&self) -> Size {
        self.d1
    }

    /// Mesh this operator is defined over.
    pub fn mesher(&self) -> &Shared<dyn FdmMesher> {
        &self.mesher
    }

    fn size(&self) -> Size {
        self.mesher.layout().size()
    }

    /// A copy with row `i` scaled by `u[i]`, i.e. `diag(u) * self`
    /// (`ninepointlinearop.cpp:152-168`).
    pub fn mult(&self, u: &Array) -> NinePointLinearOp {
        assert_eq!(u.size(), self.size(), "inconsistent size of u");
        let mut result = self.clone();
        for i in 0..self.size() {
            let s = u[i];
            result.a00[i] *= s;
            result.a10[i] *= s;
            result.a20[i] *= s;
            result.a01[i] *= s;
            result.a11[i] *= s;
            result.a21[i] *= s;
            result.a02[i] *= s;
            result.a12[i] *= s;
            result.a22[i] *= s;
        }
        result
    }
}

impl FdmLinearOp for NinePointLinearOp {
    /// `apply` (`ninepointlinearop.cpp:108-133`).
    fn apply(&self, u: &Array) -> Array {
        assert_eq!(
            u.size(),
            self.size(),
            "inconsistent length of r {} vs {}",
            u.size(),
            self.size()
        );
        let mut ret = Array::with_size(u.size());
        for i in 0..ret.size() {
            ret[i] = self.a00[i] * u[self.i00[i]]
                + self.a01[i] * u[self.i01[i]]
                + self.a02[i] * u[self.i02[i]]
                + self.a10[i] * u[self.i10[i]]
                + self.a11[i] * u[i]
                + self.a12[i] * u[self.i12[i]]
                + self.a20[i] * u[self.i20[i]]
                + self.a21[i] * u[self.i21[i]]
                + self.a22[i] * u[self.i22[i]];
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::finitedifferences::meshers::UniformGridMesher;
    use crate::methods::finitedifferences::operators::FdmLinearOpLayout;
    use crate::shared::shared;

    fn mesher_2d(nx: Size, ny: Size) -> Shared<dyn FdmMesher> {
        let layout = shared(FdmLinearOpLayout::new(vec![nx, ny]));
        shared(UniformGridMesher::new(Shared::clone(&layout), &[(0.0, 1.0), (0.0, 1.0)]).unwrap())
    }

    #[test]
    fn rejects_identical_or_out_of_range_directions() {
        let m = mesher_2d(3, 4);
        assert!(NinePointLinearOp::new(0, 0, Shared::clone(&m)).is_err());
        assert!(NinePointLinearOp::new(0, 2, m).is_err());
    }

    #[test]
    fn apply_routes_through_hand_indexed_neighbours() {
        let mesher = mesher_2d(3, 4);
        let layout = Shared::clone(mesher.layout());
        let op = NinePointLinearOp::with_coeffs(0, 1, Shared::clone(&mesher), |_, pos| {
            // Distinct weights so each neighbour contributes uniquely.
            let i = pos.index() as Real + 1.0;
            NinePointCoeffs {
                a00: 1.0 * i,
                a10: 2.0 * i,
                a20: 3.0 * i,
                a01: 4.0 * i,
                a11: 5.0 * i,
                a21: 6.0 * i,
                a02: 7.0 * i,
                a12: 8.0 * i,
                a22: 9.0 * i,
            }
        })
        .unwrap();

        let u = Array::incremental(layout.size(), 1.0, 1.0);
        let got = op.apply(&u);

        let mut position = layout.begin();
        while position.index() < layout.size() {
            let i = position.index();
            let scale = i as Real + 1.0;
            let expected = scale
                * (1.0 * u[layout.neighbourhood2(&position, 0, -1, 1, -1)]
                    + 4.0 * u[layout.neighbourhood(&position, 0, -1)]
                    + 7.0 * u[layout.neighbourhood2(&position, 0, -1, 1, 1)]
                    + 2.0 * u[layout.neighbourhood(&position, 1, -1)]
                    + 5.0 * u[i]
                    + 8.0 * u[layout.neighbourhood(&position, 1, 1)]
                    + 3.0 * u[layout.neighbourhood2(&position, 0, 1, 1, -1)]
                    + 6.0 * u[layout.neighbourhood(&position, 0, 1)]
                    + 9.0 * u[layout.neighbourhood2(&position, 0, 1, 1, 1)]);
            assert!(
                (got[i] - expected).abs() < 1e-12,
                "at {i}: {} vs {expected}",
                got[i]
            );
            position.advance();
        }
    }

    #[test]
    fn mult_scales_each_row() {
        let mesher = mesher_2d(3, 3);
        let op =
            NinePointLinearOp::with_coeffs(0, 1, Shared::clone(&mesher), |_, _| NinePointCoeffs {
                a00: 1.0,
                a10: 1.0,
                a20: 1.0,
                a01: 1.0,
                a11: 1.0,
                a21: 1.0,
                a02: 1.0,
                a12: 1.0,
                a22: 1.0,
            })
            .unwrap();
        let w = Array::incremental(mesher.layout().size(), 2.0, 0.5);
        let u = Array::incremental(mesher.layout().size(), 0.0, 1.0);
        let scaled = op.mult(&w).apply(&u);
        let direct = op.apply(&u);
        for i in 0..u.size() {
            assert!((scaled[i] - w[i] * direct[i]).abs() < 1e-12);
        }
    }
}
