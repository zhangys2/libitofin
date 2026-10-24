//! Vanilla swing option. Port of `ql/instruments/vanillaswingoption.{hpp,cpp}`.

use std::any::Any;

use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::exercise::{BermudanExercise, Exercise, ExerciseType};
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::StrikedTypePayoff;
use crate::option::OptionType;
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::{Real, Size};

/// Bermudan swing exercise (`SwingExercise`); seconds default to 0.
pub struct SwingExercise {
    inner: BermudanExercise,
    seconds: Vec<Size>,
}

impl SwingExercise {
    #[rustfmt::skip]
    pub fn new(dates: Vec<Date>) -> QlResult<Self> {
        let inner = BermudanExercise::new(dates, false)?;
        let seconds = vec![0; inner.dates().len()];
        Ok(Self { inner, seconds })
    }
    #[rustfmt::skip]
    pub fn seconds(&self) -> &[Size] { &self.seconds }
}

#[rustfmt::skip]
impl Exercise for SwingExercise {
    fn exercise_type(&self) -> ExerciseType { self.inner.exercise_type() }
    fn dates(&self) -> &[Date] { self.inner.dates() }
}

/// Unfloored forward payoff (`VanillaForwardPayoff`); call `S−K`, put `K−S`.
#[derive(Clone, Copy, Debug)]
pub struct VanillaForwardPayoff {
    option_type: OptionType,
    strike: Real,
}

impl VanillaForwardPayoff {
    #[rustfmt::skip]
    pub fn new(option_type: OptionType, strike: Real) -> Self { Self { option_type, strike } }
}

#[rustfmt::skip]
impl Payoff for VanillaForwardPayoff {
    fn name(&self) -> String { "ForwardTypePayoff".into() }
    fn description(&self) -> String { format!("ForwardTypePayoff {:?} {}", self.option_type, self.strike) }
    fn value(&self, price: Real) -> Real {
        match self.option_type { OptionType::Call => price - self.strike, OptionType::Put => self.strike - price }
    }
}

#[rustfmt::skip]
impl crate::instruments::TypePayoff for VanillaForwardPayoff {
    fn option_type(&self) -> OptionType { self.option_type }
}

#[rustfmt::skip]
impl StrikedTypePayoff for VanillaForwardPayoff {
    fn strike(&self) -> Real { self.strike }
}

#[derive(Default)]
pub struct VanillaSwingArguments {
    pub min_exercise_rights: Option<Size>,
    pub max_exercise_rights: Option<Size>,
    pub payoff: Option<Shared<dyn StrikedTypePayoff>>,
    pub exercise: Option<Shared<SwingExercise>>,
}

impl Arguments for VanillaSwingArguments {
    #[rustfmt::skip]
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        let Some(min) = self.min_exercise_rights else { fail!("no minExerciseRights"); };
        let Some(max) = self.max_exercise_rights else { fail!("no maxExerciseRights"); };
        require!(min <= max, "minExerciseRights <= maxExerciseRights");
        let n = self.exercise.as_ref().unwrap().dates().len();
        require!(n >= max, "number of exercise rights exceeds number of exercise dates");
        Ok(())
    }
}

#[derive(Default)]
pub struct VanillaSwingResults {
    pub instrument: InstrumentResults,
}

#[rustfmt::skip]
impl Results for VanillaSwingResults {
    fn reset(&mut self) { self.instrument.reset(); }
    fn as_instrument_results(&self) -> Option<&InstrumentResults> { Some(&self.instrument) }
}

pub struct VanillaSwingOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    payoff: Shared<dyn StrikedTypePayoff>,
    exercise: Shared<SwingExercise>,
    min_exercise_rights: Size,
    max_exercise_rights: Size,
}

impl VanillaSwingOption {
    #[rustfmt::skip]
    pub fn new(payoff: Shared<dyn StrikedTypePayoff>, exercise: Shared<SwingExercise>, min_exercise_rights: Size, max_exercise_rights: Size, settings: Shared<Settings<Date>>) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self { base, settings, payoff, exercise, min_exercise_rights, max_exercise_rights }
    }
}

#[rustfmt::skip]
impl Instrument for VanillaSwingOption {
    fn base(&self) -> &InstrumentBase { &self.base }
    fn base_mut(&mut self) -> &mut InstrumentBase { &mut self.base }
    fn is_expired(&self) -> QlResult<bool> {
        event_has_occurred(self.exercise.last_date(), &self.settings, None, None)
    }
    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<VanillaSwingArguments>() else { fail!("wrong argument type"); };
        arguments.payoff = Some(Shared::clone(&self.payoff));
        arguments.exercise = Some(Shared::clone(&self.exercise));
        arguments.min_exercise_rights = Some(self.min_exercise_rights);
        arguments.max_exercise_rights = Some(self.max_exercise_rights);
        Ok(())
    }
    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let Some(results) = (results as &dyn Any).downcast_ref::<VanillaSwingResults>() else { fail!("wrong result type"); };
        self.base_mut().store_results(&results.instrument);
        Ok(())
    }
}
