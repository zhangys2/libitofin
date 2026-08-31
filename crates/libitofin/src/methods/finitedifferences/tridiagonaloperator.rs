//! Classic 1-D tridiagonal finite-difference operator.
//!
//! Port of `ql/methods/finitedifferences/tridiagonaloperator.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::fail;
use crate::math::array::Array;
use crate::math::comparison::close;
use crate::require;
use crate::types::{Real, Size};

/// Tridiagonal linear operator on a 1-D grid.
#[derive(Clone, Debug)]
pub struct TridiagonalOperator {
    n: Size,
    diagonal: Array,
    lower_diagonal: Array,
    upper_diagonal: Array,
}

impl TridiagonalOperator {
    /// Empty or size-`n` (`n == 0` or `n >= 2`) operator with zero diagonals.
    pub fn with_size(size: Size) -> QlResult<Self> {
        if size >= 2 {
            Ok(Self {
                n: size,
                diagonal: Array::with_size(size),
                lower_diagonal: Array::with_size(size - 1),
                upper_diagonal: Array::with_size(size - 1),
            })
        } else if size == 0 {
            Ok(Self {
                n: 0,
                diagonal: Array::new(),
                lower_diagonal: Array::new(),
                upper_diagonal: Array::new(),
            })
        } else {
            fail!(
                "invalid size ({size}) for tridiagonal operator (must be null or >= 2)"
            );
        }
    }

    /// Builds from lower, main, and upper diagonals.
    pub fn from_diagonals(low: Array, mid: Array, high: Array) -> QlResult<Self> {
        let n = mid.size();
        require!(
            low.size() == n.saturating_sub(1),
            "low diagonal vector of size {} instead of {}",
            low.size(),
            n.saturating_sub(1)
        );
        require!(
            high.size() == n.saturating_sub(1),
            "high diagonal vector of size {} instead of {}",
            high.size(),
            n.saturating_sub(1)
        );
        Ok(Self {
            n,
            diagonal: mid,
            lower_diagonal: low,
            upper_diagonal: high,
        })
    }

    /// Identity operator of the given size.
    pub fn identity(size: Size) -> QlResult<Self> {
        require!(size >= 2, "identity size must be >= 2");
        Self::from_diagonals(
            Array::filled(size - 1, 0.0),
            Array::filled(size, 1.0),
            Array::filled(size - 1, 0.0),
        )
    }

    pub fn size(&self) -> Size {
        self.n
    }

    pub fn lower_diagonal(&self) -> &Array {
        &self.lower_diagonal
    }

    pub fn diagonal(&self) -> &Array {
        &self.diagonal
    }

    pub fn upper_diagonal(&self) -> &Array {
        &self.upper_diagonal
    }

    pub fn set_first_row(&mut self, val_b: Real, val_c: Real) {
        self.diagonal[0] = val_b;
        self.upper_diagonal[0] = val_c;
    }

    pub fn set_mid_row(&mut self, i: Size, val_a: Real, val_b: Real, val_c: Real) -> QlResult<()> {
        require!(
            i >= 1 && i <= self.n - 2,
            "out of range in TridiagonalOperator::set_mid_row"
        );
        self.lower_diagonal[i - 1] = val_a;
        self.diagonal[i] = val_b;
        self.upper_diagonal[i] = val_c;
        Ok(())
    }

    pub fn set_mid_rows(&mut self, val_a: Real, val_b: Real, val_c: Real) {
        for i in 1..=self.n - 2 {
            self.lower_diagonal[i - 1] = val_a;
            self.diagonal[i] = val_b;
            self.upper_diagonal[i] = val_c;
        }
    }

    pub fn set_last_row(&mut self, val_a: Real, val_b: Real) {
        self.lower_diagonal[self.n - 2] = val_a;
        self.diagonal[self.n - 1] = val_b;
    }

    /// Applies the operator to `v`.
    pub fn apply_to(&self, v: &Array) -> QlResult<Array> {
        require!(self.n != 0, "uninitialized TridiagonalOperator");
        require!(
            v.size() == self.n,
            "vector of the wrong size {} instead of {}",
            v.size(),
            self.n
        );
        let mut result = Array::with_size(self.n);
        for i in 0..self.n {
            result[i] = self.diagonal[i] * v[i];
        }
        result[0] += self.upper_diagonal[0] * v[1];
        for j in 1..=self.n - 2 {
            result[j] += self.lower_diagonal[j - 1] * v[j - 1] + self.upper_diagonal[j] * v[j + 1];
        }
        result[self.n - 1] += self.lower_diagonal[self.n - 2] * v[self.n - 2];
        Ok(result)
    }

    /// Solves `L x = rhs` (Thomas algorithm).
    pub fn solve_for(&self, rhs: &Array) -> QlResult<Array> {
        let mut result = Array::with_size(rhs.size());
        self.solve_for_into(rhs, &mut result)?;
        Ok(result)
    }

    /// Solves into `result` without allocating the output array.
    pub fn solve_for_into(&self, rhs: &Array, result: &mut Array) -> QlResult<()> {
        require!(self.n != 0, "uninitialized TridiagonalOperator");
        require!(
            rhs.size() == self.n,
            "rhs vector of size {} instead of {}",
            rhs.size(),
            self.n
        );
        require!(result.size() == self.n, "result size mismatch");

        let mut temp = Array::with_size(self.n);
        let mut bet = self.diagonal[0];
        require!(
            !close(bet, 0.0),
            "diagonal's first element ({bet}) cannot be close to zero"
        );
        result[0] = rhs[0] / bet;
        for j in 1..self.n {
            temp[j] = self.upper_diagonal[j - 1] / bet;
            bet = self.diagonal[j] - self.lower_diagonal[j - 1] * temp[j];
            require!(!close(bet, 0.0), "division by zero");
            result[j] = (rhs[j] - self.lower_diagonal[j - 1] * result[j - 1]) / bet;
        }
        for j in (1..self.n - 1).rev() {
            result[j] -= temp[j + 1] * result[j + 1];
        }
        result[0] -= temp[1] * result[1];
        Ok(())
    }
}

impl std::ops::Add for &TridiagonalOperator {
    type Output = TridiagonalOperator;

    fn add(self, rhs: &TridiagonalOperator) -> TridiagonalOperator {
        TridiagonalOperator::from_diagonals(
            &self.lower_diagonal + &rhs.lower_diagonal,
            &self.diagonal + &rhs.diagonal,
            &self.upper_diagonal + &rhs.upper_diagonal,
        )
        .expect("compatible sizes")
    }
}

impl std::ops::Sub for &TridiagonalOperator {
    type Output = TridiagonalOperator;

    fn sub(self, rhs: &TridiagonalOperator) -> TridiagonalOperator {
        TridiagonalOperator::from_diagonals(
            &self.lower_diagonal - &rhs.lower_diagonal,
            &self.diagonal - &rhs.diagonal,
            &self.upper_diagonal - &rhs.upper_diagonal,
        )
        .expect("compatible sizes")
    }
}

impl std::ops::Mul<Real> for &TridiagonalOperator {
    type Output = TridiagonalOperator;

    fn mul(self, a: Real) -> TridiagonalOperator {
        TridiagonalOperator::from_diagonals(
            &self.lower_diagonal * a,
            &self.diagonal * a,
            &self.upper_diagonal * a,
        )
        .expect("valid diagonals")
    }
}

impl std::ops::Mul<&TridiagonalOperator> for Real {
    type Output = TridiagonalOperator;

    fn mul(self, rhs: &TridiagonalOperator) -> TridiagonalOperator {
        rhs * self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_to_and_solve_round_trip() {
        let mut op = TridiagonalOperator::with_size(3).unwrap();
        op.set_first_row(2.0, 1.0);
        op.set_mid_row(1, 1.0, 2.0, 1.0).unwrap();
        op.set_last_row(1.0, 2.0);
        let v = Array::from([1.0, 2.0, 3.0]);
        let av = op.apply_to(&v).unwrap();
        let x = op.solve_for(&av).unwrap();
        for i in 0..3 {
            assert!((x[i] - v[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn identity_apply_is_noop() {
        let id = TridiagonalOperator::identity(4).unwrap();
        let v = Array::from([1.0, -2.0, 3.5, 0.25]);
        let out = id.apply_to(&v).unwrap();
        for i in 0..4 {
            assert_eq!(out[i], v[i]);
        }
    }
}
