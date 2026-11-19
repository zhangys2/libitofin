//! Chebyshev-node barycentric interpolation.
//!
//! Port of `ql/math/interpolations/chebyshevinterpolation.{hpp,cpp}` on top of
//! the barycentric weights in `ql/math/interpolations/lagrangeinterpolation.hpp`
//! (Berrut and Trefethen, 2004). QuantLib evaluates the interpolant through
//! Lagrange on the Chebyshev nodes; only that value is needed by the QD+
//! American engine, so the derivative and primitive (unimplemented in
//! QuantLib's Lagrange) are not exposed.

use std::f64::consts::PI;

use crate::errors::QlResult;
use crate::fail;
use crate::require;
use crate::types::{Real, Size};

/// Which Chebyshev grid the nodes sit on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChebyshevKind {
    /// Nodes `-cos((i+1/2) π / n)`.
    First,
    /// Nodes `-cos(i π / (n-1))`, including the endpoints `±1`.
    Second,
}

/// Barycentric Lagrange interpolation on Chebyshev nodes.
#[derive(Clone, Debug)]
pub struct ChebyshevInterpolation {
    x: Vec<Real>,
    y: Vec<Real>,
    lambda: Vec<Real>,
}

impl ChebyshevInterpolation {
    /// Interpolates samples `f` taken on `n` Chebyshev nodes of `kind`.
    ///
    /// # Errors
    ///
    /// Returns an error if `n < 2`, or if `f` fails.
    pub fn from_function(
        n: Size,
        kind: ChebyshevKind,
        mut f: impl FnMut(Real) -> QlResult<Real>,
    ) -> QlResult<Self> {
        let x = Self::nodes(n, kind)?;
        let mut y = Vec::with_capacity(x.len());
        for &node in &x {
            y.push(f(node)?);
        }
        Self::new(x, y)
    }

    /// The `n` Chebyshev nodes of `kind`, in increasing order.
    ///
    /// # Errors
    ///
    /// Returns an error if `n < 2` (a second-kind grid divides by `n - 1`).
    pub fn nodes(n: Size, kind: ChebyshevKind) -> QlResult<Vec<Real>> {
        require!(n >= 2, "Chebyshev interpolation needs at least 2 points");
        let n_real = n as Real;
        let mut t = Vec::with_capacity(n);
        match kind {
            ChebyshevKind::First => {
                for i in 0..n {
                    t.push(-((i as Real + 0.5) * PI / n_real).cos());
                }
            }
            ChebyshevKind::Second => {
                for i in 0..n {
                    t.push(-(i as Real * PI / (n_real - 1.0)).cos());
                }
            }
        }
        Ok(t)
    }

    fn new(x: Vec<Real>, y: Vec<Real>) -> QlResult<Self> {
        if x.len() != y.len() || x.len() < 2 {
            fail!(
                "Chebyshev interpolation needs matching x/y of length at least 2, got {} and {}",
                x.len(),
                y.len()
            );
        }
        let lambda = barycentric_weights(&x);
        Ok(ChebyshevInterpolation { x, y, lambda })
    }

    /// The interpolant at `x`, extrapolating outside the nodes.
    ///
    /// A node within `10 ε |x|` is returned exactly, matching Lagrange's
    /// `lower_bound` snap (`lagrangeinterpolation.hpp`).
    pub fn value(&self, x: Real) -> Real {
        let eps = 10.0 * Real::EPSILON * x.abs();
        let idx = self.x.partition_point(|&node| node < x - eps);
        if idx < self.x.len() && self.x[idx] - x < eps {
            return self.y[idx];
        }
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for i in 0..self.x.len() {
            let alpha = self.lambda[i] / (x - self.x[i]);
            numerator += alpha * self.y[i];
            denominator += alpha;
        }
        numerator / denominator
    }
}

/// `λ_i = 1 / ∏_{j≠i} c (x_i - x_j)` with `c = 4 / (x_last - x_first)`.
fn barycentric_weights(x: &[Real]) -> Vec<Real> {
    let n = x.len();
    let scale = 4.0 / (x[n - 1] - x[0]);
    let mut lambda = vec![0.0; n];
    for i in 0..n {
        let mut weight = 1.0;
        let x_i = x[i];
        for (j, &x_j) in x.iter().enumerate() {
            if i != j {
                weight *= scale * (x_i - x_j);
            }
        }
        lambda[i] = 1.0 / weight;
    }
    lambda
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_kind_nodes_include_the_endpoints() {
        let nodes = ChebyshevInterpolation::nodes(5, ChebyshevKind::Second).unwrap();
        assert!((nodes[0] + 1.0).abs() <= 1e-15);
        assert!(nodes[2].abs() <= 1e-15);
        assert!((nodes[4] - 1.0).abs() <= 1e-15);
    }

    #[test]
    fn recovers_a_quadratic_on_second_kind_nodes() {
        let interpolation = ChebyshevInterpolation::from_function(6, ChebyshevKind::Second, |x| {
            Ok(2.0 + 3.0 * x - x * x)
        })
        .unwrap();
        for x in [-1.0, -0.4, 0.0, 0.3, 0.8, 1.0] {
            let expected = 2.0 + 3.0 * x - x * x;
            assert!(
                (interpolation.value(x) - expected).abs() <= 1e-12,
                "x={x}: {} vs {expected}",
                interpolation.value(x)
            );
        }
    }
}
