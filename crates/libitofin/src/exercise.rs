//! Option exercise classes.
//!
//! Port of `ql/exercise.{hpp,cpp}`: the [`Exercise`] trait is the base exercise
//! contract, with [`EuropeanExercise`], [`EarlyExercise`], [`AmericanExercise`]
//! and [`BermudanExercise`] as the concrete styles.

use crate::errors::QlResult;
use crate::require;
use crate::time::date::Date;

/// Exercise style of an option (QuantLib's `Exercise::Type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExerciseType {
    /// Exercisable at any time between two predefined dates.
    American,
    /// Exercisable only at a set of fixed dates.
    Bermudan,
    /// Exercisable only at one (expiry) date.
    European,
}

/// Base exercise contract.
///
/// Implementors guarantee at least one exercise date (their constructors
/// enforce it), so [`last_date`](Exercise::last_date) is infallible where
/// QuantLib's `lastDate()` throws on an empty date vector.
pub trait Exercise {
    /// The exercise style.
    fn exercise_type(&self) -> ExerciseType;

    /// All exercise dates, in ascending order.
    fn dates(&self) -> &[Date];

    /// The last exercise date.
    fn last_date(&self) -> Date {
        *self
            .dates()
            .last()
            .expect("no exercise date given (implementors guarantee at least one)")
    }
}

/// Early-exercise base: American or Bermudan, with an optional
/// payoff-at-expiry flag (QuantLib's `EarlyExercise`).
pub trait EarlyExercise: Exercise {
    /// Whether the payoff is settled at expiry rather than at exercise.
    fn payoff_at_expiry(&self) -> bool;
}

/// European exercise: the option can only be exercised at one (expiry) date.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EuropeanExercise {
    dates: [Date; 1],
}

impl EuropeanExercise {
    /// Builds a European exercise at the given expiry date.
    pub fn new(date: Date) -> EuropeanExercise {
        EuropeanExercise { dates: [date] }
    }
}

impl Exercise for EuropeanExercise {
    fn exercise_type(&self) -> ExerciseType {
        ExerciseType::European
    }

    fn dates(&self) -> &[Date] {
        &self.dates
    }
}

/// American exercise: any time between an earliest and a latest date.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AmericanExercise {
    dates: [Date; 2],
    payoff_at_expiry: bool,
}

impl AmericanExercise {
    /// Builds an American exercise window `[earliest, latest]`.
    ///
    /// # Errors
    ///
    /// Fails when `earliest > latest`.
    pub fn new(earliest: Date, latest: Date, payoff_at_expiry: bool) -> QlResult<Self> {
        require!(earliest <= latest, "earliest > latest exercise date");
        Ok(Self {
            dates: [earliest, latest],
            payoff_at_expiry,
        })
    }

    /// Builds an American exercise from the minimum date through `latest`.
    pub fn from_latest(latest: Date, payoff_at_expiry: bool) -> Self {
        Self {
            dates: [Date::min_date(), latest],
            payoff_at_expiry,
        }
    }
}

impl Exercise for AmericanExercise {
    fn exercise_type(&self) -> ExerciseType {
        ExerciseType::American
    }

    fn dates(&self) -> &[Date] {
        &self.dates
    }
}

impl EarlyExercise for AmericanExercise {
    fn payoff_at_expiry(&self) -> bool {
        self.payoff_at_expiry
    }
}

/// Bermudan exercise: exercisable only on a discrete set of dates.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BermudanExercise {
    dates: Vec<Date>,
    payoff_at_expiry: bool,
}

impl BermudanExercise {
    /// Builds a Bermudan exercise on the given dates (sorted ascending).
    ///
    /// # Errors
    ///
    /// Fails when `dates` is empty.
    pub fn new(mut dates: Vec<Date>, payoff_at_expiry: bool) -> QlResult<Self> {
        require!(!dates.is_empty(), "no exercise date given");
        dates.sort();
        Ok(Self {
            dates,
            payoff_at_expiry,
        })
    }
}

impl Exercise for BermudanExercise {
    fn exercise_type(&self) -> ExerciseType {
        ExerciseType::Bermudan
    }

    fn dates(&self) -> &[Date] {
        &self.dates
    }
}

impl EarlyExercise for BermudanExercise {
    fn payoff_at_expiry(&self) -> bool {
        self.payoff_at_expiry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::date::Month;

    #[test]
    fn european_exercise_holds_the_single_expiry() {
        let expiry = Date::new(17, Month::May, 2027);
        let exercise = EuropeanExercise::new(expiry);
        assert_eq!(exercise.exercise_type(), ExerciseType::European);
        assert_eq!(exercise.dates(), &[expiry]);
        assert_eq!(exercise.last_date(), expiry);
    }

    #[test]
    fn american_exercise_holds_the_window() {
        let earliest = Date::new(15, Month::June, 2026);
        let latest = Date::new(15, Month::June, 2027);
        let exercise = AmericanExercise::new(earliest, latest, false).unwrap();
        assert_eq!(exercise.exercise_type(), ExerciseType::American);
        assert_eq!(exercise.dates(), &[earliest, latest]);
        assert_eq!(exercise.last_date(), latest);
        assert!(!exercise.payoff_at_expiry());
    }

    #[test]
    fn american_from_latest_starts_at_min_date() {
        let latest = Date::new(31, Month::December, 2030);
        let exercise = AmericanExercise::from_latest(latest, true);
        assert_eq!(exercise.dates()[0], Date::min_date());
        assert_eq!(exercise.last_date(), latest);
        assert!(exercise.payoff_at_expiry());
    }

    #[test]
    fn bermudan_exercise_sorts_dates() {
        let d1 = Date::new(15, Month::January, 2027);
        let d2 = Date::new(15, Month::July, 2027);
        let d3 = Date::new(15, Month::April, 2027);
        let exercise = BermudanExercise::new(vec![d2, d1, d3], false).unwrap();
        assert_eq!(exercise.dates(), &[d1, d3, d2]);
        assert_eq!(exercise.exercise_type(), ExerciseType::Bermudan);
    }

    #[test]
    fn usable_as_trait_object() {
        let expiry = Date::new(31, Month::December, 2030);
        let exercise: &dyn Exercise = &EuropeanExercise::new(expiry);
        assert_eq!(exercise.last_date(), expiry);
        assert_eq!(exercise.dates().len(), 1);
    }
}
