//! Equity total return swap.
//!
//! Port of `ql/instruments/equitytotalreturnswap.{hpp,cpp}`: an equity total
//! return swap exchanges the total return of an [`EquityIndex`] for a set of
//! floating cash flows linked to either an [`IborIndex`] or an [`OvernightIndex`].
//!
//! The equity leg future value is:
//!
//! $$FV^{\text{equity}} = N \left[ \frac{I(t, T_M)}{I(T_0)} - 1 \right]$$
//!
//! where $N$ is the swap notional, $I(T_0)$ is the value of the equity index on
//! the start date, and $I(t, T_M)$ is the value at maturity.
//!
//! The floating leg payments are linked to either an Ibor index (via an
//! [`IborLeg`]) or an overnight index (via an [`OvernightLeg`]). For an overnight
//! index, the interest rate fixings are compounded over each accrual period.
//!
//! Swap type ([`SwapType::Payer`] or [`SwapType::Receiver`]) refers to the
//! equity leg: a payer pays equity return and receives floating rate.

use crate::cashflow::{CashFlow, Leg};
use crate::cashflows::{EquityCashFlow, IborLeg, OvernightLeg};
use crate::errors::QlResult;
use crate::indexes::EquityIndex;
use crate::indexes::iborindex::{IborIndex, OvernightIndex};
use crate::indexes::index::Index;
use crate::instrument::{Instrument, InstrumentBase};
use crate::instruments::swap::{Swap, SwapType};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared};
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendar::Calendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::schedule::Schedule;
use crate::time::timeunit::TimeUnit;
use crate::types::{Integer, Natural, Rate, Real};

/// Basis point factor used for fair margin calculation ($10^{-4}$).
const BASIS_POINT: Real = 1.0e-4;

/// Reference to either an Ibor index or an Overnight index.
#[derive(Clone)]
pub enum InterestRateIndexKind {
    /// An Ibor index (e.g. Libor, Euribor).
    Ibor(Shared<IborIndex>),
    /// An overnight index (e.g. SOFR, ESTR).
    Overnight(Shared<OvernightIndex>),
}

impl InterestRateIndexKind {
    /// The name of the underlying index.
    pub fn name(&self) -> String {
        match self {
            Self::Ibor(idx) => idx.name(),
            Self::Overnight(idx) => idx.name(),
        }
    }
}

impl From<Shared<IborIndex>> for InterestRateIndexKind {
    fn from(idx: Shared<IborIndex>) -> Self {
        InterestRateIndexKind::Ibor(idx)
    }
}

impl From<Shared<OvernightIndex>> for InterestRateIndexKind {
    fn from(idx: Shared<OvernightIndex>) -> Self {
        InterestRateIndexKind::Overnight(idx)
    }
}

/// Equity total return swap (`EquityTotalReturnSwap`).
pub struct EquityTotalReturnSwap {
    swap: Swap,
    swap_type: SwapType,
    nominal: Real,
    schedule: Schedule,
    equity_index: Shared<EquityIndex>,
    interest_rate_index: InterestRateIndexKind,
    day_counter: DayCounter,
    margin: Rate,
    gearing: Real,
    payment_calendar: Calendar,
    payment_convention: BusinessDayConvention,
    payment_delay: Natural,
}

impl EquityTotalReturnSwap {
    /// Builds an equity total return swap.
    ///
    /// # Errors
    ///
    /// - Returns an error if `nominal` is negative.
    /// - Returns an error if both `payment_calendar` (if provided) and `schedule.calendar()` are empty.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        swap_type: SwapType,
        nominal: Real,
        schedule: Schedule,
        equity_index: Shared<EquityIndex>,
        interest_rate_index: impl Into<InterestRateIndexKind>,
        day_counter: DayCounter,
        margin: Rate,
        gearing: Real,
        payment_calendar: Option<Calendar>,
        payment_convention: BusinessDayConvention,
        payment_delay: Natural,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(nominal >= 0.0, "Nominal cannot be negative");

        let payment_cal = payment_calendar.unwrap_or_default();
        let cal = if !payment_cal.is_empty() {
            payment_cal.clone()
        } else {
            require!(
                !schedule.calendar().is_empty(),
                "Calendar in schedule cannot be empty"
            );
            schedule.calendar().clone()
        };

        let start_date = schedule.start_date();
        let end_date = schedule.end_date();
        let payment_date = cal.advance(
            end_date,
            payment_delay as Integer,
            TimeUnit::Days,
            payment_convention,
            schedule.end_of_month(),
        );

        let equity_cash_flow = shared(EquityCashFlow::new(
            nominal,
            Shared::clone(&equity_index),
            start_date,
            end_date,
            payment_date,
            true, // growthOnly = true
        ));
        let equity_leg: Leg = vec![equity_cash_flow as Shared<dyn CashFlow>];

        let interest_rate_index = interest_rate_index.into();
        let interest_rate_leg: Leg = match &interest_rate_index {
            InterestRateIndexKind::Ibor(ibor) => {
                IborLeg::new(schedule.clone(), Shared::clone(ibor))
                    .with_notional(nominal)
                    .with_payment_day_counter(day_counter.clone())
                    .with_spread(margin)
                    .with_gearing(gearing)
                    .with_payment_calendar(cal.clone())
                    .with_payment_adjustment(payment_convention)
                    .with_payment_lag(payment_delay as Integer)
                    .build()?
            }
            InterestRateIndexKind::Overnight(overnight) => {
                OvernightLeg::new(schedule.clone(), Shared::clone(overnight))
                    .with_notional(nominal)
                    .with_payment_day_counter(day_counter.clone())
                    .with_spread(margin)
                    .with_gearing(gearing)
                    .with_payment_calendar(cal.clone())
                    .with_payment_adjustment(payment_convention)
                    .with_payment_lag(payment_delay as Integer)
                    .build()?
            }
        };

        // Payer refers to equity leg:
        // Payer: equity leg is paid (-1), floating leg received (+1) -> payer flags: [true, false]
        // Receiver: equity leg received (+1), floating leg paid (-1) -> payer flags: [false, true]
        let payer_flags = match swap_type {
            SwapType::Payer => vec![true, false],
            SwapType::Receiver => vec![false, true],
        };

        let swap = Swap::new(
            vec![equity_leg, interest_rate_leg],
            payer_flags,
            Shared::clone(&settings),
        )?;

        Ok(EquityTotalReturnSwap {
            swap,
            swap_type,
            nominal,
            schedule,
            equity_index,
            interest_rate_index,
            day_counter,
            margin,
            gearing,
            payment_calendar: payment_cal,
            payment_convention,
            payment_delay,
        })
    }

    /// The swap type (referring to the equity leg).
    pub fn swap_type(&self) -> SwapType {
        self.swap_type
    }

    /// The swap nominal.
    pub fn nominal(&self) -> Real {
        self.nominal
    }

    /// The equity index.
    pub fn equity_index(&self) -> &Shared<EquityIndex> {
        &self.equity_index
    }

    /// The floating interest rate index.
    pub fn interest_rate_index(&self) -> &InterestRateIndexKind {
        &self.interest_rate_index
    }

    /// The schedule.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// The floating leg day counter.
    pub fn day_counter(&self) -> &DayCounter {
        &self.day_counter
    }

    /// The margin on the floating leg.
    pub fn margin(&self) -> Rate {
        self.margin
    }

    /// The gearing on the floating leg.
    pub fn gearing(&self) -> Real {
        self.gearing
    }

    /// The payment calendar.
    pub fn payment_calendar(&self) -> &Calendar {
        &self.payment_calendar
    }

    /// The payment convention.
    pub fn payment_convention(&self) -> BusinessDayConvention {
        self.payment_convention
    }

    /// The payment delay in business days.
    pub fn payment_delay(&self) -> Natural {
        self.payment_delay
    }

    /// The equity leg.
    pub fn equity_leg(&self) -> &Leg {
        &self.swap.legs()[0]
    }

    /// The interest rate floating leg.
    pub fn interest_rate_leg(&self) -> &Leg {
        &self.swap.legs()[1]
    }

    /// The embedded base swap.
    pub fn swap(&self) -> &Swap {
        &self.swap
    }

    /// The embedded base swap, mutably.
    pub fn swap_mut(&mut self) -> &mut Swap {
        &mut self.swap
    }

    /// Sets the pricing engine on the swap.
    pub fn set_pricing_engine(&mut self, engine: SharedMut<dyn PricingEngine>) {
        self.swap.base_mut().set_pricing_engine(engine);
    }

    /// The NPV of the swap.
    pub fn npv(&mut self) -> QlResult<Real> {
        self.swap.npv()
    }

    /// The NPV of the equity leg.
    pub fn equity_leg_npv(&mut self) -> QlResult<Real> {
        self.swap.leg_npv(0)
    }

    /// The NPV of the floating interest rate leg.
    pub fn interest_rate_leg_npv(&mut self) -> QlResult<Real> {
        self.swap.leg_npv(1)
    }

    /// Implies the fair margin that makes the swap NPV zero.
    ///
    /// $$NPV = NPV^{\text{equity}} + NPV^{\text{floating}} = 0$$
    ///
    /// where $NPV^{\text{floating}} = NPV^{\text{floating, ex-margin}} + \text{margin} \cdot \frac{BPS}{10^{-4}}$.
    pub fn fair_margin(&mut self) -> QlResult<Rate> {
        let interest_leg_bps = self.swap.leg_bps(1)? / BASIS_POINT;
        let ex_margin_interest_leg_npv =
            self.interest_rate_leg_npv()? - self.margin * interest_leg_bps;
        Ok(-(self.equity_leg_npv()? + ex_margin_interest_leg_npv) / interest_leg_bps)
    }
}

impl Instrument for EquityTotalReturnSwap {
    fn base(&self) -> &InstrumentBase {
        self.swap.base()
    }

    fn base_mut(&mut self) -> &mut InstrumentBase {
        self.swap.base_mut()
    }

    fn is_expired(&self) -> QlResult<bool> {
        self.swap.is_expired()
    }

    fn setup_expired(&mut self) {
        self.swap.setup_expired();
    }

    fn setup_arguments(&self, arguments: &mut dyn Arguments) -> QlResult<()> {
        self.swap.setup_arguments(arguments)
    }

    fn fetch_results(&mut self, results: &dyn Results) -> QlResult<()> {
        self.swap.fetch_results(results)
    }
}
