//! Forward rate agreement.
//!
//! Port of `ql/instruments/forwardrateagreement.{hpp,cpp}`: an FRA settles on
//! the value date for the NPV of a forward loan/deposit over
//! `[value_date, maturity_date]`.

use crate::errors::QlResult;
use crate::event::event_has_occurred;
use crate::fail;
use crate::handle::Handle;
use crate::indexes::IborIndex;
use crate::indexes::index::Index;
use crate::indexes::interestrateindex::InterestRateIndex;
use crate::instrument::{Instrument, InstrumentBase, InstrumentResults};
use crate::interestrate::{Compounding, InterestRate};
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::time::frequency::Frequency;
use crate::types::{Integer, Rate, Real};

/// Long = purchase (borrower); Short = sale (lender). QuantLib `Position::Type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Position {
    Long,
    Short,
}

/// A forward-rate agreement on an [`IborIndex`].
pub struct ForwardRateAgreement {
    base: InstrumentBase,
    settings: Shared<Settings<Date>>,
    index: Shared<IborIndex>,
    value_date: Date,
    maturity_date: Date,
    position: Position,
    strike: InterestRate,
    notional: Real,
    use_indexed_coupon: bool,
    discount_curve: Handle<dyn YieldTermStructure>,
    amount: Option<Real>,
    forward_rate: Option<InterestRate>,
}

impl ForwardRateAgreement {
    /// Indexed FRA: maturity from `index.maturity_date(value_date)`, forward
    /// from the index fixing.
    pub fn new(
        index: Shared<IborIndex>,
        value_date: Date,
        position: Position,
        strike_forward_rate: Rate,
        notional: Real,
        discount_curve: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let maturity = index.maturity_date(value_date)?;
        let mut fra = Self::with_maturity(
            Shared::clone(&index),
            value_date,
            maturity,
            position,
            strike_forward_rate,
            notional,
            discount_curve,
            settings,
        )?;
        fra.use_indexed_coupon = true;
        Ok(fra)
    }

    /// Par-coupon FRA over an explicit maturity date.
    #[allow(clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]
    pub fn with_maturity(
        index: Shared<IborIndex>,
        value_date: Date,
        maturity_date: Date,
        position: Position,
        strike_forward_rate: Rate,
        notional: Real,
        discount_curve: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(notional > 0.0, "notionalAmount must be positive");
        let maturity_date = index
            .fixing_calendar()
            .adjust(maturity_date, index.business_day_convention());
        require!(
            value_date < maturity_date,
            "valueDate must be earlier than maturityDate"
        );
        let strike = InterestRate::new(
            strike_forward_rate,
            index.day_counter().clone(),
            Compounding::Simple,
            Frequency::Once,
        )?;
        let base = InstrumentBase::new();
        settings.register_eval_date_observer(&base.observer());
        base.register_with(index.observable());
        discount_curve.register_observer(&base.observer());
        Ok(Self {
            base,
            settings,
            index,
            value_date,
            maturity_date,
            position,
            strike,
            notional,
            use_indexed_coupon: false,
            discount_curve,
            amount: None,
            forward_rate: None,
        })
    }

    /// The FRA fixing date.
    pub fn fixing_date(&self) -> Date {
        self.index.fixing_date(self.value_date)
    }

    /// Settlement amount before discounting (after calculation).
    pub fn amount(&mut self) -> QlResult<Real> {
        self.calculate()?;
        match self.amount {
            Some(a) => Ok(a),
            None => fail!("FRA amount not available"),
        }
    }

    /// The forecast forward rate (after calculation).
    pub fn forward_rate(&mut self) -> QlResult<InterestRate> {
        self.calculate()?;
        match self.forward_rate.clone() {
            Some(r) => Ok(r),
            None => fail!("FRA forward rate not available"),
        }
    }

    fn calculate_forward_rate(&mut self) -> QlResult<()> {
        let rate = if self.use_indexed_coupon {
            self.index.fixing(self.fixing_date(), true)?
        } else {
            let curve = self.index.forwarding_term_structure().current_link()?;
            let d1 = curve.discount_date(self.value_date, false)?;
            let d2 = curve.discount_date(self.maturity_date, false)?;
            let t = self
                .index
                .day_counter()
                .year_fraction(self.value_date, self.maturity_date);
            (d1 / d2 - 1.0) / t
        };
        self.forward_rate = Some(InterestRate::new(
            rate,
            self.index.day_counter().clone(),
            Compounding::Simple,
            Frequency::Once,
        )?);
        Ok(())
    }

    fn calculate_amount(&mut self) -> QlResult<()> {
        self.calculate_forward_rate()?;
        let sign: Integer = match self.position {
            Position::Long => 1,
            Position::Short => -1,
        };
        let forward = self.forward_rate.as_ref().expect("forward set");
        let f = forward.rate();
        let k = self.strike.rate();
        let t = forward
            .day_counter()
            .year_fraction(self.value_date, self.maturity_date);
        self.amount = Some(self.notional * sign as Real * (f - k) * t / (1.0 + f * t));
        Ok(())
    }
}

impl Instrument for ForwardRateAgreement {
    fn base(&self) -> &InstrumentBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        &mut self.base
    }

    fn is_expired(&self) -> QlResult<bool> {
        event_has_occurred(self.value_date, &self.settings, None, None)
    }

    fn setup_expired(&mut self) {
        let expired = InstrumentResults {
            value: Some(0.0),
            error_estimate: Some(0.0),
            ..InstrumentResults::default()
        };
        self.base_mut().store_results(&expired);
        let _ = self.calculate_forward_rate();
    }

    fn perform_calculations(&mut self) -> QlResult<()> {
        self.calculate_amount()?;
        let discount = if self.discount_curve.is_empty() {
            self.index.forwarding_term_structure().current_link()?
        } else {
            self.discount_curve.current_link()?
        };
        let npv =
            self.amount.expect("amount set") * discount.discount_date(self.value_date, false)?;
        self.base_mut().store_results(&InstrumentResults {
            value: Some(npv),
            error_estimate: Some(0.0),
            ..InstrumentResults::default()
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexes::ibor::Euribor;
    use crate::instrument::Instrument;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    #[test]
    fn fra_long_has_positive_npv_when_forward_exceeds_strike() {
        let today = Date::new(15, Month::June, 2026);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let curve = Handle::new(shared(FlatForward::with_rate(
            today,
            0.03,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let index = shared(
            Euribor::new(
                Period::new(6, TimeUnit::Months),
                curve.clone(),
                Shared::clone(&settings),
            )
            .unwrap(),
        );
        let value = today + 60;
        let mut fra = ForwardRateAgreement::new(
            index,
            value,
            Position::Long,
            0.01,
            1_000_000.0,
            curve,
            Shared::clone(&settings),
        )
        .unwrap();
        let npv = fra.npv().unwrap();
        assert!(npv > 0.0, "expected positive NPV, got {npv}");
        assert!(fra.amount().unwrap() > 0.0);
    }
}
