//! Householder reflection and transformation.
//!
//! Port of `ql/math/matrixutilities/householder.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::matrix::Matrix;
use crate::require;
use crate::types::Real;

/// Householder reflection built from a unit direction `e`.
pub struct HouseholderReflection {
    e: Array,
}

impl HouseholderReflection {
    /// Reflection with respect to the hyperplane orthogonal to `e`.
    pub fn new(e: Array) -> Self {
        Self { e }
    }

    /// The reflection vector for target `a`.
    pub fn reflection_vector(&self, a: &Array) -> QlResult<Array> {
        let na = a.norm2();
        require!(na > 0.0, "vector of length zero given");

        let a_dot_e = a.dot(&self.e);
        let a1 = &self.e * a_dot_e;
        let a2 = a - &a1;

        let eps = a2.dot(&a2) / (a_dot_e * a_dot_e);
        if eps < Real::EPSILON * Real::EPSILON {
            Ok(Array::with_size(a.size()))
        } else if eps < 1e-4 {
            let eps2 = eps * eps;
            let eps3 = eps * eps2;
            let eps4 = eps2 * eps2;
            let numerator =
                &a2 - &(&a1 * (eps / 2.0 - eps2 / 8.0 + eps3 / 16.0 - 5.0 / 128.0 * eps4));
            let denom = a_dot_e * (eps + eps2 / 4.0 - eps3 / 8.0 + 5.0 / 64.0 * eps4).sqrt();
            Ok(&numerator / denom)
        } else {
            let c = a - &(&self.e * na);
            Ok(&c / c.norm2())
        }
    }

    /// Applies the Householder reflection to `a`.
    pub fn apply(&self, a: &Array) -> QlResult<Array> {
        let v = self.reflection_vector(a)?;
        Ok(HouseholderTransformation::new(v).apply(a))
    }
}

/// Householder transformation `H = I - 2 v v^T / ||v||^2`.
pub struct HouseholderTransformation {
    v: Array,
}

impl HouseholderTransformation {
    /// Builds the transformation from reflection vector `v`.
    pub fn new(v: Array) -> Self {
        Self { v }
    }

    /// Applies the transformation to vector `x`.
    pub fn apply(&self, x: &Array) -> Array {
        x - &(&self.v * (2.0 * self.v.dot(x)))
    }

    /// The explicit Householder matrix.
    pub fn matrix(&self) -> Matrix {
        let y = &self.v / self.v.norm2();
        let n = y.size();
        let mut m = Matrix::with_size(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = if i == j { 1.0 } else { 0.0 } - 2.0 * y[i] * y[j];
            }
        }
        m
    }
}

/// Convenience: `HouseholderTransformation(HouseholderReflection(e).reflection_vector(q1))`.
pub fn householder_transformation(e: Array, target: &Array) -> QlResult<Matrix> {
    let reflection = HouseholderReflection::new(e);
    let v = reflection.reflection_vector(target)?;
    Ok(HouseholderTransformation::new(v).matrix())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_maps_target_to_scaled_e() {
        let e = Array::from([1.0, 0.0, 0.0]);
        let a = Array::from([3.0, 4.0, 0.0]);
        let reflected = HouseholderReflection::new(e).apply(&a).unwrap();
        let expected_norm = a.norm2();
        assert!((reflected[0] - expected_norm).abs() < 1e-12);
        assert!(reflected[1].abs() < 1e-12);
        assert!(reflected[2].abs() < 1e-12);
    }
}
