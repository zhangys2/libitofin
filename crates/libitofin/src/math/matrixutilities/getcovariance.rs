//! Covariance matrix from correlation and standard deviations.
//!
//! Port of `ql/math/matrixutilities/getcovariance.hpp`.

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::matrix::Matrix;
use crate::require;
use crate::types::Real;

/// Builds the covariance matrix from standard deviations and a correlation matrix.
///
/// Only the symmetric part of `corr` is used; diagonal entries must equal one
/// within `tolerance`.
pub fn get_covariance(std_dev: &Array, corr: &Matrix, tolerance: Real) -> QlResult<Matrix> {
    let std_dev: &[Real] = std_dev;
    let size = std_dev.len();
    require!(
        corr.rows() == size,
        "dimension mismatch between volatilities ({size}) and correlation rows ({})",
        corr.rows()
    );
    require!(
        corr.columns() == size,
        "correlation matrix is not square: {size} rows and {} columns",
        corr.columns()
    );

    let mut covariance = Matrix::with_size(size, size);
    for i in 0..size {
        for j in 0..i {
            require!(
                (corr[(i, j)] - corr[(j, i)]).abs() <= tolerance,
                "correlation matrix not symmetric: c[{i},{j}] = {} c[{j},{i}] = {}",
                corr[(i, j)],
                corr[(j, i)]
            );
            let entry = std_dev[i] * std_dev[j] * 0.5 * (corr[(i, j)] + corr[(j, i)]);
            covariance[(i, j)] = entry;
            covariance[(j, i)] = entry;
        }
        require!(
            (corr[(i, i)] - 1.0).abs() <= tolerance,
            "invalid correlation matrix, diagonal element of row {i} is {} instead of 1.0",
            corr[(i, i)]
        );
        covariance[(i, i)] = std_dev[i] * std_dev[i];
    }
    Ok(covariance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_covariance_from_unit_correlation() {
        let std_dev = [0.2, 0.3, 0.4];
        let corr = Matrix::from([[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]]);
        let cov = get_covariance(&Array::from(std_dev), &corr, 1e-12).unwrap();
        assert!((cov[(0, 0)] - 0.04).abs() < 1e-15);
        assert!((cov[(0, 1)] - 0.03).abs() < 1e-15);
        assert!((cov[(2, 2)] - 0.16).abs() < 1e-15);
    }
}
