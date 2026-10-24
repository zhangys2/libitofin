//! A step condition that fans out over a list of step conditions.
//!
//! Port of `ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp:43`
//! and its `.cpp`.

use super::{FdmAmericanStepCondition, FdmBermudanStepCondition, FdmSnapshotCondition};
use crate::cashflows::Dividend;
use crate::errors::QlResult;
use crate::exercise::{Exercise, ExerciseType};
use crate::math::array::Array;
use crate::methods::finitedifferences::StepCondition;
use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::methods::finitedifferences::utilities::{FdmDividendHandler, FdmInnerValueCalculator};
use crate::shared::{Shared, shared};
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{Real, Time};

/// The step conditions of one solver, applied in order at every step.
pub struct FdmStepConditionComposite {
    stopping_times: Vec<Time>,
    conditions: Vec<Shared<dyn StepCondition>>,
}

impl FdmStepConditionComposite {
    /// Merges the per-condition stopping times into one sorted, unique grid
    /// (`cpp:36-46`).
    ///
    /// C++ funnels the lists through a `std::set<Real>`, which orders and
    /// deduplicates on exact equality; the Rust sort-and-dedup is exact for the
    /// same reason, unlike the `close_enough` dedup a
    /// [`TimeGrid`](crate::math::timegrid::TimeGrid) applies.
    pub fn new(stopping_times: &[Vec<Time>], conditions: Vec<Shared<dyn StepCondition>>) -> Self {
        let mut all_stopping_times: Vec<Time> = stopping_times.iter().flatten().copied().collect();
        all_stopping_times.sort_by(Real::total_cmp);
        all_stopping_times.dedup();

        Self {
            stopping_times: all_stopping_times,
            conditions,
        }
    }

    /// The merged stopping times (`cpp:53-55`).
    pub fn stopping_times(&self) -> &[Time] {
        &self.stopping_times
    }

    /// The conditions, in application order (`cpp:48-51`).
    pub fn conditions(&self) -> &[Shared<dyn StepCondition>] {
        &self.conditions
    }

    /// Wraps a snapshot around an existing composite (`cpp:63-78`).
    ///
    /// The stopping times are `c2`'s plus `c1`'s capture time, and `c2` enters
    /// the condition list whole and first, `c1` second: every step applies the
    /// wrapped composite before the snapshot records the grid.
    pub fn join_conditions(
        c1: &Shared<FdmSnapshotCondition>,
        c2: &Shared<FdmStepConditionComposite>,
    ) -> Shared<FdmStepConditionComposite> {
        let stopping_times = [c2.stopping_times().to_vec(), vec![c1.time()]];

        let composite: Shared<dyn StepCondition> = c2.clone();
        let snapshot: Shared<dyn StepCondition> = c1.clone();

        shared(FdmStepConditionComposite::new(
            &stopping_times,
            vec![composite, snapshot],
        ))
    }

    /// `FdmStepConditionComposite::vanillaComposite` (`cpp:80-145`).
    ///
    /// Wires discrete cash dividends (filtered to `[ref_date, maturity]`) and
    /// the American / Bermudan early-exercise condition. European exercise
    /// adds no extra condition.
    pub fn vanilla_composite(
        cash_flow: &[Shared<dyn Dividend>],
        exercise: &dyn Exercise,
        mesher: Shared<dyn FdmMesher>,
        calculator: Shared<dyn FdmInnerValueCalculator>,
        ref_date: Date,
        day_counter: DayCounter,
    ) -> QlResult<Shared<FdmStepConditionComposite>> {
        let mut stopping_times = Vec::new();
        let mut step_conditions: Vec<Shared<dyn StepCondition>> = Vec::new();

        if !cash_flow.is_empty() {
            let maturity_date = exercise.last_date();
            let dividends: Vec<Shared<dyn Dividend>> = cash_flow
                .iter()
                .filter(|div| div.date() >= ref_date && div.date() <= maturity_date)
                .cloned()
                .collect();

            let dividend_condition = shared(FdmDividendHandler::new(
                &dividends,
                Shared::clone(&mesher),
                ref_date,
                day_counter.clone(),
                0,
            )?);
            step_conditions.push(Shared::clone(&dividend_condition) as Shared<dyn StepCondition>);

            let maturity_time = day_counter.year_fraction(ref_date, exercise.last_date());
            let mut dividend_times = dividend_condition.dividend_times().to_vec();
            for t in &mut dividend_times {
                *t = (*t).min(maturity_time);
            }
            stopping_times.push(dividend_times.clone());
            for t in &mut dividend_times {
                *t = (*t + 1e-5).min(maturity_time);
            }
            stopping_times.push(dividend_times);
        }

        match exercise.exercise_type() {
            ExerciseType::American => {
                let exercise_start = day_counter.year_fraction(ref_date, exercise.dates()[0]);
                step_conditions.push(shared(FdmAmericanStepCondition::new(
                    mesher,
                    calculator,
                    exercise_start,
                )) as Shared<dyn StepCondition>);
            }
            ExerciseType::Bermudan => {
                let bermudan = shared(FdmBermudanStepCondition::new(
                    exercise.dates(),
                    ref_date,
                    &day_counter,
                    mesher,
                    calculator,
                ));
                stopping_times.push(bermudan.exercise_times().to_vec());
                step_conditions.push(bermudan as Shared<dyn StepCondition>);
            }
            ExerciseType::European => {}
        }

        Ok(shared(FdmStepConditionComposite::new(
            &stopping_times,
            step_conditions,
        )))
    }
}

impl StepCondition for FdmStepConditionComposite {
    /// `cpp:57-61`: applies every condition in turn.
    fn apply_to(&self, a: &mut Array, t: Time) {
        for condition in &self.conditions {
            condition.apply_to(a, t);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    struct Recorder {
        tag: &'static str,
        scale: Real,
        offset: Real,
        log: Shared<RefCell<Vec<String>>>,
    }

    impl StepCondition for Recorder {
        fn apply_to(&self, a: &mut Array, t: Time) {
            self.log.borrow_mut().push(format!("{}:{t}", self.tag));
            a[0] = a[0] * self.scale + self.offset;
        }
    }

    fn recorder(
        tag: &'static str,
        scale: Real,
        offset: Real,
        log: &Shared<RefCell<Vec<String>>>,
    ) -> Shared<dyn StepCondition> {
        shared(Recorder {
            tag,
            scale,
            offset,
            log: Shared::clone(log),
        })
    }

    #[test]
    fn conditions_are_applied_in_order() {
        let log = shared(RefCell::new(Vec::new()));
        let composite = FdmStepConditionComposite::new(
            &[],
            vec![
                recorder("first", 2.0, 1.0, &log),
                recorder("second", 3.0, 0.0, &log),
            ],
        );
        let mut values = Array::from([1.0]);

        composite.apply_to(&mut values, 0.25);

        assert_eq!(values[0], 9.0);
        assert_eq!(
            *log.borrow(),
            vec!["first:0.25".to_string(), "second:0.25".to_string()]
        );
    }

    #[test]
    fn stopping_times_are_sorted_and_deduplicated_across_the_lists() {
        let composite = FdmStepConditionComposite::new(
            &[vec![2.0, 0.5, 1.0], vec![1.0, 0.25], vec![0.5]],
            Vec::new(),
        );

        assert_eq!(composite.stopping_times(), &[0.25, 0.5, 1.0, 2.0]);
    }

    #[test]
    fn joined_conditions_merge_the_snapshot_time_into_the_stopping_times() {
        let inner = shared(FdmStepConditionComposite::new(
            &[vec![0.25, 1.0, 2.0]],
            Vec::new(),
        ));
        let snapshot = shared(FdmSnapshotCondition::new(0.75));

        let joined = FdmStepConditionComposite::join_conditions(&snapshot, &inner);

        assert_eq!(joined.stopping_times(), &[0.25, 0.75, 1.0, 2.0]);
    }

    #[test]
    fn joined_conditions_hold_the_composite_first_and_the_snapshot_second() {
        let log = shared(RefCell::new(Vec::new()));
        let inner = shared(FdmStepConditionComposite::new(
            &[vec![1.0]],
            vec![recorder("inner", 2.0, 1.0, &log)],
        ));
        let snapshot = shared(FdmSnapshotCondition::new(1.0));

        let joined = FdmStepConditionComposite::join_conditions(&snapshot, &inner);

        assert_eq!(joined.conditions().len(), 2);
        let inner_dyn: Shared<dyn StepCondition> = inner.clone();
        let snapshot_dyn: Shared<dyn StepCondition> = snapshot.clone();
        assert!(Shared::ptr_eq(&joined.conditions()[0], &inner_dyn));
        assert!(Shared::ptr_eq(&joined.conditions()[1], &snapshot_dyn));

        let mut values = Array::from([1.0]);
        joined.apply_to(&mut values, 1.0);

        assert_eq!(*log.borrow(), vec!["inner:1".to_string()]);
        assert_eq!(snapshot.values(), Array::from([3.0]));
    }

    #[test]
    fn a_composite_is_itself_a_step_condition() {
        let log = shared(RefCell::new(Vec::new()));
        let composite: Shared<dyn StepCondition> = shared(FdmStepConditionComposite::new(
            &[vec![0.5]],
            vec![recorder("only", 2.0, 1.0, &log)],
        ));
        let mut values = Array::from([1.0]);

        composite.apply_to(&mut values, 0.5);

        assert_eq!(values[0], 3.0);
        assert_eq!(*log.borrow(), vec!["only:0.5".to_string()]);
    }

    #[test]
    fn vanilla_composite_clamps_dividend_times_and_adds_the_smoother() {
        use crate::cashflows::FixedDividend;
        use crate::exercise::EuropeanExercise;
        use crate::methods::finitedifferences::meshers::{FdmMesher, UniformGridMesher};
        use crate::methods::finitedifferences::operators::{
            FdmLinearOpIterator, FdmLinearOpLayout,
        };
        use crate::methods::finitedifferences::utilities::FdmInnerValueCalculator;
        use crate::time::date::Month;
        use crate::time::daycounters::actual365fixed::Actual365Fixed;

        struct ZeroCalc;
        impl FdmInnerValueCalculator for ZeroCalc {
            fn inner_value(&self, _iter: &FdmLinearOpIterator, _t: Time) -> Real {
                0.0
            }
            fn avg_inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
                self.inner_value(iter, t)
            }
        }

        let today = Date::new(11, Month::February, 2018);
        let dc = Actual365Fixed::new();
        let maturity = Date::new(11, Month::February, 2019);
        let div_date = Date::new(11, Month::August, 2018);
        let layout = shared(FdmLinearOpLayout::new(vec![5]));
        let mesher: Shared<dyn FdmMesher> =
            shared(UniformGridMesher::new(layout, &[(80.0_f64.ln(), 120.0_f64.ln())]).unwrap());
        let exercise = EuropeanExercise::new(maturity);
        let cash: Vec<Shared<dyn Dividend>> = vec![shared(FixedDividend::new(30.0, div_date))];
        let composite = FdmStepConditionComposite::vanilla_composite(
            &cash,
            &exercise,
            mesher,
            shared(ZeroCalc),
            today,
            dc.clone(),
        )
        .unwrap();

        let t = dc.year_fraction(today, div_date);
        let times = composite.stopping_times();
        assert!(times.iter().any(|&x| (x - t).abs() < 1e-14), "{times:?}");
        assert!(
            times.iter().any(|&x| (x - (t + 1e-5)).abs() < 1e-14),
            "{times:?}"
        );
        assert_eq!(composite.conditions().len(), 1);
    }
}
