//! Basket option on multiple underlyings.
//!
//! Port of `ql/instruments/basketoption.hpp` (AverageBasketPayoff and
//! `BasketOption` instrument slice used by Choi engines).

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::instruments::{PlainVanillaPayoff, StrikedTypePayoff, TypePayoff};
use crate::math::array::Array;
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::{Real, Size};

/// Weighted average basket payoff.
#[derive(Clone, Debug)]
pub struct AverageBasketPayoff {
    base_payoff: PlainVanillaPayoff,
    weights: Array,
}

impl AverageBasketPayoff {
    /// Payoff on the weighted sum of underlyings.
    pub fn new(base_payoff: PlainVanillaPayoff, weights: Array) -> Self {
        Self {
            base_payoff,
            weights,
        }
    }

    /// Equal weights for `n` underlyings.
    pub fn with_equal_weights(base_payoff: PlainVanillaPayoff, n: Size) -> Self {
        Self::new(base_payoff, Array::filled(n, 1.0 / n as Real))
    }

    pub fn base_payoff(&self) -> &PlainVanillaPayoff {
        &self.base_payoff
    }

    pub fn weights(&self) -> &Array {
        &self.weights
    }

    pub fn accumulate(&self, a: &Array) -> Real {
        assert_eq!(a.size(), self.weights.size(), "basket size mismatch");
        a.iter().zip(self.weights.iter()).map(|(x, w)| x * w).sum()
    }

    pub fn value_on_forwards(&self, forwards: &Array) -> Real {
        self.base_payoff.value(self.accumulate(forwards))
    }
}

impl Payoff for AverageBasketPayoff {
    fn name(&self) -> String {
        self.base_payoff.name()
    }

    fn description(&self) -> String {
        format!("Average basket {}", self.base_payoff.description())
    }

    fn value(&self, price: Real) -> Real {
        self.base_payoff.value(price)
    }
}

impl TypePayoff for AverageBasketPayoff {
    fn option_type(&self) -> crate::option::OptionType {
        self.base_payoff.option_type()
    }
}

impl StrikedTypePayoff for AverageBasketPayoff {
    fn strike(&self) -> Real {
        self.base_payoff.strike()
    }
}

/// Spread basket payoff (two underlyings only).
#[derive(Clone, Debug)]
pub struct SpreadBasketPayoff {
    base_payoff: PlainVanillaPayoff,
}

impl SpreadBasketPayoff {
    pub fn new(base_payoff: PlainVanillaPayoff) -> Self {
        Self { base_payoff }
    }

    pub fn base_payoff(&self) -> &PlainVanillaPayoff {
        &self.base_payoff
    }

    pub fn accumulate(&self, a: &Array) -> Real {
        assert_eq!(a.size(), 2, "payoff is only defined for two underlyings");
        a[0] - a[1]
    }
}

impl Payoff for SpreadBasketPayoff {
    fn name(&self) -> String {
        self.base_payoff.name()
    }

    fn description(&self) -> String {
        format!("Spread basket {}", self.base_payoff.description())
    }

    fn value(&self, price: Real) -> Real {
        self.base_payoff.value(price)
    }
}

impl TypePayoff for SpreadBasketPayoff {
    fn option_type(&self) -> crate::option::OptionType {
        self.base_payoff.option_type()
    }
}

impl StrikedTypePayoff for SpreadBasketPayoff {
    fn strike(&self) -> Real {
        self.base_payoff.strike()
    }
}

/// Minimum basket payoff.
#[derive(Clone, Debug)]
pub struct MinBasketPayoff {
    base_payoff: PlainVanillaPayoff,
}

impl MinBasketPayoff {
    pub fn new(base_payoff: PlainVanillaPayoff) -> Self {
        Self { base_payoff }
    }

    pub fn base_payoff(&self) -> &PlainVanillaPayoff {
        &self.base_payoff
    }

    pub fn accumulate(&self, a: &Array) -> Real {
        assert!(!a.is_empty(), "empty array");
        *a.iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .expect("non-empty")
    }
}

impl Payoff for MinBasketPayoff {
    fn name(&self) -> String {
        self.base_payoff.name()
    }

    fn description(&self) -> String {
        format!("Min basket {}", self.base_payoff.description())
    }

    fn value(&self, price: Real) -> Real {
        self.base_payoff.value(price)
    }
}

impl TypePayoff for MinBasketPayoff {
    fn option_type(&self) -> crate::option::OptionType {
        self.base_payoff.option_type()
    }
}

impl StrikedTypePayoff for MinBasketPayoff {
    fn strike(&self) -> Real {
        self.base_payoff.strike()
    }
}

/// Maximum basket payoff.
#[derive(Clone, Debug)]
pub struct MaxBasketPayoff {
    base_payoff: PlainVanillaPayoff,
}

impl MaxBasketPayoff {
    pub fn new(base_payoff: PlainVanillaPayoff) -> Self {
        Self { base_payoff }
    }

    pub fn base_payoff(&self) -> &PlainVanillaPayoff {
        &self.base_payoff
    }

    pub fn accumulate(&self, a: &Array) -> Real {
        assert!(!a.is_empty(), "empty array");
        *a.iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .expect("non-empty")
    }
}

impl Payoff for MaxBasketPayoff {
    fn name(&self) -> String {
        self.base_payoff.name()
    }

    fn description(&self) -> String {
        format!("Max basket {}", self.base_payoff.description())
    }

    fn value(&self, price: Real) -> Real {
        self.base_payoff.value(price)
    }
}

impl TypePayoff for MaxBasketPayoff {
    fn option_type(&self) -> crate::option::OptionType {
        self.base_payoff.option_type()
    }
}

impl StrikedTypePayoff for MaxBasketPayoff {
    fn strike(&self) -> Real {
        self.base_payoff.strike()
    }
}

/// Payoff on a basket of assets.
#[derive(Clone, Debug)]
pub enum BasketPayoff {
    Average(AverageBasketPayoff),
    Spread(SpreadBasketPayoff),
    Min(MinBasketPayoff),
    Max(MaxBasketPayoff),
}

impl BasketPayoff {
    pub fn base_payoff(&self) -> &PlainVanillaPayoff {
        match self {
            Self::Average(p) => p.base_payoff(),
            Self::Spread(p) => p.base_payoff(),
            Self::Min(p) => p.base_payoff(),
            Self::Max(p) => p.base_payoff(),
        }
    }

    pub fn accumulate(&self, a: &Array) -> Real {
        match self {
            Self::Average(p) => p.accumulate(a),
            Self::Spread(p) => p.accumulate(a),
            Self::Min(p) => p.accumulate(a),
            Self::Max(p) => p.accumulate(a),
        }
    }
}

impl Payoff for BasketPayoff {
    fn name(&self) -> String {
        self.base_payoff().name()
    }

    fn description(&self) -> String {
        match self {
            Self::Average(p) => p.description(),
            Self::Spread(p) => p.description(),
            Self::Min(p) => p.description(),
            Self::Max(p) => p.description(),
        }
    }

    fn value(&self, price: Real) -> Real {
        self.base_payoff().value(price)
    }
}

impl TypePayoff for BasketPayoff {
    fn option_type(&self) -> crate::option::OptionType {
        self.base_payoff().option_type()
    }
}

impl StrikedTypePayoff for BasketPayoff {
    fn strike(&self) -> Real {
        self.base_payoff().strike()
    }
}

impl From<AverageBasketPayoff> for BasketPayoff {
    fn from(p: AverageBasketPayoff) -> Self {
        Self::Average(p)
    }
}

impl From<SpreadBasketPayoff> for BasketPayoff {
    fn from(p: SpreadBasketPayoff) -> Self {
        Self::Spread(p)
    }
}

impl From<MinBasketPayoff> for BasketPayoff {
    fn from(p: MinBasketPayoff) -> Self {
        Self::Min(p)
    }
}

impl From<MaxBasketPayoff> for BasketPayoff {
    fn from(p: MaxBasketPayoff) -> Self {
        Self::Max(p)
    }
}

/// Arguments for basket-option engines.
#[derive(Default)]
pub struct BasketArguments {
    pub payoff: Option<BasketPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for BasketArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        Ok(())
    }
}

/// Results for basket-option engines.
#[derive(Default)]
pub struct BasketResults {
    pub instrument: InstrumentResults,
}

impl Results for BasketResults {
    fn reset(&mut self) {
        self.instrument.reset();
    }

    fn as_instrument_results(&self) -> Option<&InstrumentResults> {
        Some(&self.instrument)
    }
}

/// Engine base for basket options.
pub type BasketEngine = GenericEngine<BasketArguments, BasketResults>;

/// Basket option on multiple assets.
pub struct BasketOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    payoff: BasketPayoff,
    exercise: Shared<dyn Exercise>,
}

impl BasketOption {
    pub fn new<P: Into<BasketPayoff>>(
        payoff: P,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Self {
            base,
            settings,
            payoff: payoff.into(),
            exercise,
        }
    }

    pub fn payoff(&self) -> &BasketPayoff {
        &self.payoff
    }

    pub fn exercise(&self) -> &Shared<dyn Exercise> {
        &self.exercise
    }

    pub fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }
}

impl Instrument for BasketOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        crate::event::event_has_occurred(self.exercise.last_date(), &self.settings, None, None)
    }

    fn setup_arguments(&self, args: &mut dyn Arguments) -> QlResult<()> {
        let args = (args as &mut dyn Any)
            .downcast_mut::<BasketArguments>()
            .expect("BasketOption expects BasketArguments");
        args.payoff = Some(self.payoff.clone());
        args.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        let results = (results as &dyn Any)
            .downcast_ref::<BasketResults>()
            .expect("BasketOption expects BasketResults");
        self.base.store_results(&results.instrument);
        Ok(())
    }
}
