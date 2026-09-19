//! Linear-exponential Libor-forward volatility model.
//!
//! Port of `ql/legacy/libormarketmodels/lmlinexpvolmodel.{hpp,cpp}`:
//! `σ_i(t) = (a(T_i-t)+d) e^{-b(T_i-t)} + c` for `T_i > t`, else 0.

use std::rc::Rc;

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::optimization::constraint::PositiveConstraint;
use crate::models::parameter::{ConstantParameter, Parameter};
use crate::types::{Real, Size, Time};

/// Caplet volatility model with linear-exponential time structure.
pub struct LmLinearExponentialVolatilityModel {
    size: Size,
    fixing_times: Vec<Time>,
    arguments: Vec<Parameter>,
}

impl LmLinearExponentialVolatilityModel {
    /// `LmLinearExponentialVolatilityModel(fixingTimes, a, b, c, d)`.
    ///
    /// # Errors
    ///
    /// Fails when any of `a`,`b`,`c`,`d` violates [`PositiveConstraint`].
    pub fn new(fixing_times: Vec<Time>, a: Real, b: Real, c: Real, d: Real) -> QlResult<Self> {
        let size = fixing_times.len();
        let pos: Rc<dyn crate::math::optimization::constraint::Constraint> =
            Rc::new(PositiveConstraint);
        Ok(Self {
            size,
            fixing_times,
            arguments: vec![
                ConstantParameter::new(a, Rc::clone(&pos))?,
                ConstantParameter::new(b, Rc::clone(&pos))?,
                ConstantParameter::new(c, Rc::clone(&pos))?,
                ConstantParameter::new(d, pos)?,
            ],
        })
    }

    /// Number of forward rates.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Instantaneous volatilities at `t`.
    pub fn volatility(&self, t: Time) -> Array {
        let (a, b, c, d) = self.abcd();
        let mut tmp = Array::with_size(self.size);
        for i in 0..self.size {
            let t_fix = self.fixing_times[i];
            if t_fix > t {
                tmp[i] = (a * (t_fix - t) + d) * (-b * (t_fix - t)).exp() + c;
            }
        }
        tmp
    }

    /// Instantaneous volatility of forward `i` at `t`.
    pub fn volatility_i(&self, i: Size, t: Time) -> Real {
        let (a, b, c, d) = self.abcd();
        let t_fix = self.fixing_times[i];
        if t_fix > t {
            (a * (t_fix - t) + d) * (-b * (t_fix - t)).exp() + c
        } else {
            0.0
        }
    }

    fn abcd(&self) -> (Real, Real, Real, Real) {
        (
            self.arguments[0].value(0.0),
            self.arguments[1].value(0.0),
            self.arguments[2].value(0.0),
            self.arguments[3].value(0.0),
        )
    }
}
