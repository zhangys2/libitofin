//! Abcd mathematical function.
//!
//! Port of `ql/math/abcdmathfunction.{hpp,cpp}`:
//! `f(t) = [a + b t] e^{-c t} + d`.

use crate::errors::QlResult;
use crate::require;
use crate::types::{Real, Time};

/// Abcd functional form (`abcdmathfunction.hpp`).
#[derive(Clone, Debug)]
pub struct AbcdMathFunction {
    a: Real,
    b: Real,
    c: Real,
    d: Real,
}

impl AbcdMathFunction {
    /// `AbcdMathFunction(a, b, c, d)`.
    ///
    /// # Errors
    ///
    /// Fails when coefficients violate [`validate`].
    pub fn new(a: Real, b: Real, c: Real, d: Real) -> QlResult<Self> {
        validate(a, b, c, d)?;
        Ok(Self { a, b, c, d })
    }

    pub fn a(&self) -> Real {
        self.a
    }
    pub fn b(&self) -> Real {
        self.b
    }
    pub fn c(&self) -> Real {
        self.c
    }
    pub fn d(&self) -> Real {
        self.d
    }

    /// `f(t)`; zero for `t < 0`.
    pub fn value(&self, t: Time) -> Real {
        if t < 0.0 {
            0.0
        } else {
            (self.a + self.b * t) * (-self.c * t).exp() + self.d
        }
    }
}

/// Validates Abcd coefficients (`AbcdMathFunction::validate`).
pub fn validate(a: Real, b: Real, c: Real, d: Real) -> QlResult<()> {
    require!(c > 0.0, "c ({c}) must be positive");
    require!(d >= 0.0, "d ({d}) must be non negative");
    require!(a + d >= 0.0, "a+d ({a}+{d}) must be non negative");
    if b >= 0.0 {
        return Ok(());
    }
    let z = 1.0 / c - a / b;
    if z >= 0.0 {
        let bound = -(d * c) / (c * a / b - 1.0).exp();
        require!(
            b >= bound,
            "b ({b}) less than {bound}: negative function value at stationary point {z}"
        );
    }
    Ok(())
}
