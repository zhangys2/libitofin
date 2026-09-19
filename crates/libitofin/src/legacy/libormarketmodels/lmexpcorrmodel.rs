//! Exponential Libor-forward correlation model.
//!
//! Port of `ql/legacy/libormarketmodels/lmexpcorrmodel.{hpp,cpp}`:
//! `ρ_{i,j} = exp(-β |i-j|)`.

use std::rc::Rc;

use crate::errors::QlResult;
use crate::math::matrix::Matrix;
use crate::math::matrixutilities::{SalvagingAlgorithm, pseudo_sqrt};
use crate::math::optimization::constraint::PositiveConstraint;
use crate::models::parameter::{ConstantParameter, Parameter};
use crate::require;
use crate::types::{Real, Size, Time};

/// Exponential correlation model (`lmexpcorrmodel.hpp`).
pub struct LmExponentialCorrelationModel {
    size: Size,
    corr_matrix: Matrix,
    pseudo_sqrt: Matrix,
    #[allow(dead_code)]
    arguments: Vec<Parameter>,
}

impl LmExponentialCorrelationModel {
    /// `LmExponentialCorrelationModel(size, rho)`.
    ///
    /// # Errors
    ///
    /// Fails when `rho` violates [`PositiveConstraint`].
    pub fn new(size: Size, rho: Real) -> QlResult<Self> {
        let arguments = vec![ConstantParameter::new(rho, Rc::new(PositiveConstraint))?];
        let mut model = Self {
            size,
            corr_matrix: Matrix::with_size(size, size),
            pseudo_sqrt: Matrix::with_size(size, size),
            arguments,
        };
        model.generate_arguments();
        Ok(model)
    }

    /// Number of forward rates.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Number of factors (full rank for this model).
    pub fn factors(&self) -> Size {
        self.size
    }

    /// Correlation matrix (time-independent).
    pub fn correlation(&self, _t: Time) -> Matrix {
        self.corr_matrix.clone()
    }

    /// Element `ρ_{i,j}`.
    ///
    /// # Errors
    ///
    /// Fails when `i` or `j` is out of range.
    pub fn correlation_ij(&self, i: Size, j: Size, _t: Time) -> QlResult<Real> {
        require!(
            i < self.size && j < self.size,
            "correlation index ({i},{j}) out of range [0..{})",
            self.size
        );
        Ok(self.corr_matrix[(i, j)])
    }

    /// Cached spectral pseudo square root.
    pub fn pseudo_sqrt(&self, _t: Time) -> Matrix {
        self.pseudo_sqrt.clone()
    }

    /// Whether the correlation is independent of calendar time.
    pub fn is_time_independent(&self) -> bool {
        true
    }

    fn generate_arguments(&mut self) {
        let rho = self.arguments[0].value(0.0);
        for i in 0..self.size {
            for j in i..self.size {
                let v = (-rho * (i as Real - j as Real).abs()).exp();
                self.corr_matrix[(i, j)] = v;
                self.corr_matrix[(j, i)] = v;
            }
        }
        self.pseudo_sqrt = pseudo_sqrt(&self.corr_matrix, SalvagingAlgorithm::Spectral);
    }
}
