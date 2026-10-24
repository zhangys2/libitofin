//! Bermudan early-exercise step condition.
//!
//! Port of `ql/methods/finitedifferences/stepconditions/fdmbermudanstepcondition.{hpp,cpp}`.

use crate::math::array::Array;
use crate::methods::finitedifferences::StepCondition;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::methods::finitedifferences::utilities::FdmInnerValueCalculator;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::Time;

/// Applies the Bermudan early-exercise condition only on discrete exercise
/// times (exact equality, matching QuantLib's `std::find` on the time vector).
pub struct FdmBermudanStepCondition {
    mesher: Shared<dyn FdmMesher>,
    calculator: Shared<dyn FdmInnerValueCalculator>,
    exercise_times: Vec<Time>,
}

impl FdmBermudanStepCondition {
    /// Builds the condition from calendar exercise dates.
    pub fn new(
        exercise_dates: &[Date],
        reference_date: Date,
        day_counter: &DayCounter,
        mesher: Shared<dyn FdmMesher>,
        calculator: Shared<dyn FdmInnerValueCalculator>,
    ) -> Self {
        let exercise_times = exercise_dates
            .iter()
            .map(|d| day_counter.year_fraction(reference_date, *d))
            .collect();
        Self {
            mesher,
            calculator,
            exercise_times,
        }
    }

    /// The exercise times in year-fraction units.
    pub fn exercise_times(&self) -> &[Time] {
        &self.exercise_times
    }
}

impl StepCondition for FdmBermudanStepCondition {
    fn apply_to(&self, a: &mut Array, t: Time) {
        if !self.exercise_times.contains(&t) {
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
