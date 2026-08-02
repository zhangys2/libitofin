//! American early-exercise step condition.
//!
//! Port of `ql/methods/finitedifferences/stepconditions/fdmamericanstepcondition.{hpp,cpp}`.

use crate::math::array::Array;
use crate::methods::finitedifferences::StepCondition;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::methods::finitedifferences::utilities::FdmInnerValueCalculator;
use crate::shared::Shared;
use crate::types::Time;

/// Applies the American early-exercise condition `V = max(V, intrinsic)` at
/// every time on or after [`exercise_start`](Self::new).
pub struct FdmAmericanStepCondition {
    mesher: Shared<dyn FdmMesher>,
    calculator: Shared<dyn FdmInnerValueCalculator>,
    exercise_start: Time,
}

impl FdmAmericanStepCondition {
    /// Builds the condition over `mesher` with payoff `calculator`.
    ///
    /// `exercise_start` defaults to `0.0` in QuantLib (exercisable from t = 0).
    pub fn new(
        mesher: Shared<dyn FdmMesher>,
        calculator: Shared<dyn FdmInnerValueCalculator>,
        exercise_start: Time,
    ) -> Self {
        Self {
            mesher,
            calculator,
            exercise_start,
        }
    }
}

impl StepCondition for FdmAmericanStepCondition {
    fn apply_to(&self, a: &mut Array, t: Time) {
        if t < self.exercise_start {
            return;
        }
        assert_eq!(
            self.mesher.layout().size(),
            a.size(),
            "inconsistent array dimensions"
        );
        for iter in self.mesher.layout().iter() {
            let inner = self.calculator.inner_value(&iter, t);
            let index = iter.index();
            if inner > a[index] {
                a[index] = inner;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruments::PlainVanillaPayoff;
    use crate::methods::finitedifferences::meshers::UniformGridMesher;
    use crate::methods::finitedifferences::operators::FdmLinearOpLayout;
    use crate::methods::finitedifferences::utilities::fdm_log_inner_value;
    use crate::option::OptionType;
    use crate::payoff::Payoff;
    use crate::shared::shared;

    #[test]
    fn american_condition_lifts_values_to_intrinsic() {
        let layout = shared(FdmLinearOpLayout::new(vec![5]));
        let mesher: Shared<dyn FdmMesher> =
            shared(UniformGridMesher::new(layout, &[(4.0_f64.ln(), 5.0_f64.ln())]).unwrap());
        let payoff: Shared<dyn Payoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, (4.5_f64).exp()));
        let calculator = shared(fdm_log_inner_value(
            Shared::clone(&payoff),
            Shared::clone(&mesher),
            0,
        ));
        let condition = FdmAmericanStepCondition::new(mesher, calculator, 0.0);
        let mut values = Array::from([0.0, 0.0, 0.0, 0.0, 0.0]);
        condition.apply_to(&mut values, 0.5);
        assert!(values.iter().any(|&v| v > 0.0));
    }
}
