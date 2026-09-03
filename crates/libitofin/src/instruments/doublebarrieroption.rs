//! Double-barrier option instrument.
//!
//! Port of `ql/instruments/doublebarrieroption.{hpp,cpp}`.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::fail;
use crate::instrument::{Instrument, InstrumentBase};
use crate::instruments::PlainVanillaPayoff;
use crate::pricingengine::Arguments;
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::Real;

/// Double-barrier type (QuantLib `DoubleBarrier::Type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DoubleBarrierType {
    KnockIn,
    KnockOut,
}

/// Arguments for double-barrier engines.
#[derive(Default)]
pub struct DoubleBarrierArguments {
    pub barrier_type: Option<DoubleBarrierType>,
    pub barrier_lo: Option<Real>,
    pub barrier_hi: Option<Real>,
    pub rebate: Option<Real>,
    pub payoff: Option<PlainVanillaPayoff>,
    pub exercise: Option<Shared<dyn Exercise>>,
}

impl Arguments for DoubleBarrierArguments {
    fn validate(&self) -> QlResult<()> {
        require!(self.barrier_type.is_some(), "no double-barrier type");
        require!(self.barrier_lo.is_some(), "no low barrier given");
        require!(self.barrier_hi.is_some(), "no high barrier given");
        require!(self.rebate.is_some(), "no rebate given");
        require!(self.payoff.is_some(), "no payoff given");
        require!(self.exercise.is_some(), "no exercise given");
        Ok(())
    }
}

/// Double-barrier option on a single asset.
pub struct DoubleBarrierOption {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    barrier_type: DoubleBarrierType,
    barrier_lo: Real,
    barrier_hi: Real,
    rebate: Real,
    payoff: PlainVanillaPayoff,
    exercise: Shared<dyn Exercise>,
}

impl DoubleBarrierOption {
    /// Builds a double-barrier option.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        barrier_type: DoubleBarrierType,
        barrier_lo: Real,
        barrier_hi: Real,
        rebate: Real,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(
            barrier_lo > 0.0 && barrier_hi > 0.0,
            "barriers must be positive"
        );
        require!(
            barrier_lo < barrier_hi,
            "low barrier must be below high barrier"
        );
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            barrier_type,
            barrier_lo,
            barrier_hi,
            rebate,
            payoff,
            exercise,
        })
    }

    pub fn barrier_type(&self) -> DoubleBarrierType {
        self.barrier_type
    }
    pub fn barrier_lo(&self) -> Real {
        self.barrier_lo
    }
    pub fn barrier_hi(&self) -> Real {
        self.barrier_hi
    }
    pub fn rebate(&self) -> Real {
        self.rebate
    }
}

impl Instrument for DoubleBarrierOption {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        crate::event::event_has_occurred(self.exercise.last_date(), &self.settings, None, None)
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        let Some(arguments) = (arguments as &mut dyn Any).downcast_mut::<DoubleBarrierArguments>()
        else {
            fail!("wrong argument type");
        };
        arguments.barrier_type = Some(self.barrier_type);
        arguments.barrier_lo = Some(self.barrier_lo);
        arguments.barrier_hi = Some(self.barrier_hi);
        arguments.rebate = Some(self.rebate);
        arguments.payoff = Some(self.payoff);
        arguments.exercise = Some(Shared::clone(&self.exercise));
        Ok(())
    }
}

/// Returns true when the spot has touched either barrier.
pub fn double_barrier_triggered(spot: Real, barrier_lo: Real, barrier_hi: Real) -> bool {
    spot <= barrier_lo || spot >= barrier_hi
}
