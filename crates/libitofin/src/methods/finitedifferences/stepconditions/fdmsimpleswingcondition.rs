//! Simple swing step condition. Port of `fdmsimpleswingcondition.{hpp,cpp}`.

use crate::math::array::Array;
use crate::methods::finitedifferences::StepCondition;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::methods::finitedifferences::utilities::FdmInnerValueCalculator;
use crate::shared::Shared;
use crate::types::{Integer, Size, Time};

pub struct FdmSimpleSwingCondition {
    exercise_times: Vec<Time>,
    mesher: Shared<dyn FdmMesher>,
    calculator: Shared<dyn FdmInnerValueCalculator>,
    min_exercises: Size,
    swing_direction: Size,
}

impl FdmSimpleSwingCondition {
    #[rustfmt::skip]
    pub fn new(exercise_times: Vec<Time>, mesher: Shared<dyn FdmMesher>, calculator: Shared<dyn FdmInnerValueCalculator>, swing_direction: Size, min_exercises: Size) -> Self {
        Self { exercise_times, mesher, calculator, min_exercises, swing_direction }
    }
}

impl StepCondition for FdmSimpleSwingCondition {
    #[rustfmt::skip]
    fn apply_to(&self, a: &mut Array, t: Time) {
        let Some(idx) = self.exercise_times.iter().position(|&x| x == t) else { return; };
        let layout = self.mesher.layout();
        let max_ex = layout.dim()[self.swing_direction] - 1;
        let remaining = self.exercise_times.len() - idx;
        let mut ret = a.clone();
        for iter in layout.iter() {
            let used = iter.coordinates()[self.swing_direction];
            if used < max_ex {
                let cash = self.calculator.inner_value(&iter, t);
                let cur = a[iter.index()];
                let plus = a[layout.neighbourhood(&iter, self.swing_direction, 1 as Integer)];
                if cur < plus + cash || used + remaining <= self.min_exercises {
                    ret[iter.index()] = plus + cash;
                }
            }
        }
        *a = ret;
    }
}
