//! Black-formula engine for European callable bonds.
//!
//! Port of QuantLib's
//! `ql/experimental/callablebonds/blackcallablebondengine.{hpp,cpp}`: the
//! embedded European call/put is priced as a Hull Ch.20 bond option. Quoted
//! forward yield volatility is converted to forward price volatility via
//! modified duration, then fed to [`black_formula`].
//!
//! Only the constant yield-vol (`Handle<Quote>`) constructor is ported; the
//! full `CallableBondVolatilityStructure` surface is deferred.

use crate::cashflows::{CashFlows, Duration};
use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instruments::{BondResults, CallabilityType, CallableBondArguments};
use crate::interestrate::{Compounding, InterestRate};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::blackformula::black_formula;
use crate::quotes::Quote;
use crate::require;
use crate::settings::Settings;
use crate::shared::Shared;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::daycounters::actual365fixed::Actual365Fixed;
use crate::time::frequency::Frequency;
use crate::types::{Real, Time, Volatility};

type CallableBondEngineBase = GenericEngine<CallableBondArguments, BondResults>;

/// Black-formula engine for a European callable / puttable fixed-rate bond.
pub struct BlackCallableFixedRateBondEngine {
    base: CallableBondEngineBase,
    yield_vol: Handle<dyn Quote>,
    vol_day_counter: DayCounter,
    discount_curve: Handle<dyn YieldTermStructure>,
    settings: Shared<Settings<Date>>,
}

impl BlackCallableFixedRateBondEngine {
    /// Builds the engine from a constant forward yield volatility quote.
    ///
    /// Matches QuantLib's `Handle<Quote>` constructor, which wraps the quote in
    /// a `CallableBondConstantVolatility(0, NullCalendar(), …, Actual365Fixed)`.
    pub fn new(
        fwd_yield_vol: Handle<dyn Quote>,
        discount_curve: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
    ) -> BlackCallableFixedRateBondEngine {
        let base =
            CallableBondEngineBase::new(CallableBondArguments::default(), BondResults::default());
        fwd_yield_vol.register_observer(&base.observer());
        discount_curve.register_observer(&base.observer());
        BlackCallableFixedRateBondEngine {
            base,
            yield_vol: fwd_yield_vol,
            vol_day_counter: Actual365Fixed::new(),
            discount_curve,
            settings,
        }
    }

    /// Present value of coupons paid between settlement and the exercise date,
    /// rebased to settlement (`BlackCallableFixedRateBondEngine::spotIncome`).
    fn spot_income(
        args: &CallableBondArguments,
        curve: &dyn YieldTermStructure,
        settings: &Settings<Date>,
    ) -> QlResult<Real> {
        let settlement = args.settlement_date.expect("validated settlement date");
        let option_maturity = args.callability_dates[0];
        let mut income = 0.0;
        let count = args.cashflows.len();
        require!(count > 0, "empty cash-flow leg");
        for flow in &args.cashflows[..count - 1] {
            if !flow.has_occurred(settings, Some(settlement), Some(false))? {
                if flow.has_occurred(settings, Some(option_maturity), Some(false))? {
                    income += flow.amount()? * curve.discount_date(flow.date(), true)?;
                } else {
                    break;
                }
            }
        }
        Ok(income / curve.discount_date(settlement, true)?)
    }

    /// Converts quoted forward yield vol into forward cash-price vol
    /// (`BlackCallableFixedRateBondEngine::forwardPriceVolatility`).
    fn forward_price_volatility(
        &self,
        args: &CallableBondArguments,
        curve: &dyn YieldTermStructure,
    ) -> QlResult<Volatility> {
        let exercise_date = args.callability_dates[0];
        let fwd_npv = CashFlows::npv(
            &args.cashflows,
            curve,
            &self.settings,
            Some(false),
            Some(exercise_date),
            Some(exercise_date),
        )?;

        let Some(day_counter) = args.payment_day_counter.clone() else {
            fail!("null payment day counter");
        };
        let mut frequency = args.frequency;
        if matches!(frequency, Frequency::NoFrequency | Frequency::Once) {
            frequency = Frequency::Annual;
        }

        let fwd_ytm = CashFlows::solve_yield(
            &args.cashflows,
            fwd_npv,
            day_counter.clone(),
            Compounding::Compounded,
            frequency,
            &self.settings,
            Some(false),
            Some(exercise_date),
            Some(exercise_date),
            None,
            None,
            None,
        )?;
        let fwd_rate = InterestRate::new(fwd_ytm, day_counter, Compounding::Compounded, frequency)?;
        let fwd_dur = CashFlows::duration(
            &args.cashflows,
            &fwd_rate,
            Duration::Modified,
            &self.settings,
            Some(false),
            Some(exercise_date),
            Some(exercise_date),
        )?;

        // Constant yield vol: `CallableBondConstantVolatility` ignores option
        // time / bond length / strike, so only the quote value is needed.
        let yield_vol = self.yield_vol.current_link()?.value()?;
        Ok(yield_vol * fwd_dur * fwd_ytm)
    }
}

impl AsObservable for BlackCallableFixedRateBondEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BlackCallableFixedRateBondEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn calculate(&mut self) -> QlResult<()> {
        require!(
            !self.discount_curve.is_empty(),
            "discounting term structure handle is empty"
        );
        require!(
            self.base.arguments().callability_dates.len() == 1,
            "Must have exactly one call/put date to use Black Engine"
        );

        let settle = self
            .base
            .arguments()
            .settlement_date
            .expect("validated settlement date");
        let exercise_date = self.base.arguments().callability_dates[0];
        require!(
            exercise_date >= settle,
            "must have exercise Date >= settlement Date"
        );

        let curve = self.discount_curve.current_link()?;
        let reference_date = curve.reference_date()?;

        let value = CashFlows::npv(
            &self.base.arguments().cashflows,
            &*curve,
            &self.settings,
            Some(false),
            Some(settle),
            Some(settle),
        )?;
        let npv = CashFlows::npv(
            &self.base.arguments().cashflows,
            &*curve,
            &self.settings,
            Some(false),
            Some(reference_date),
            Some(reference_date),
        )?;

        let spot_income = Self::spot_income(self.base.arguments(), &*curve, &self.settings)?;
        let fwd_cash_price = (value - spot_income) / curve.discount_date(exercise_date, true)?;
        let cash_strike =
            self.base.arguments().callability_prices[0] * self.base.arguments().face_amount / 100.0;

        let option_type = match self.base.arguments().callability_types[0] {
            CallabilityType::Call => OptionType::Call,
            CallabilityType::Put => OptionType::Put,
        };

        let price_vol = self.forward_price_volatility(self.base.arguments(), &*curve)?;
        let Some(vol_ref) = self.settings.evaluation_date() else {
            fail!("null evaluation date for yield volatility");
        };
        let exercise_time: Time = self.vol_day_counter.year_fraction(vol_ref, exercise_date);

        let discount = curve.discount_date(exercise_date, true)?;
        let discount_to_settlement = discount / curve.discount_date(settle, true)?;

        let embedded_option_value = black_formula(
            option_type,
            cash_strike,
            fwd_cash_price,
            price_vol * exercise_time.sqrt(),
            1.0,
            0.0,
        )?;

        let (value_out, settlement_out) = match option_type {
            OptionType::Call => (
                npv - embedded_option_value * discount,
                value - embedded_option_value * discount_to_settlement,
            ),
            OptionType::Put => (
                npv + embedded_option_value * discount,
                value + embedded_option_value * discount_to_settlement,
            ),
        };

        let results = self.base.results_mut();
        results.instrument.value = Some(value_out);
        results.settlement_value = Some(settlement_out);
        Ok(())
    }
}

/// Black-formula engine for a European callable zero-coupon bond.
///
/// QuantLib's `BlackCallableZeroCouponBondEngine` is a thin alias of
/// [`BlackCallableFixedRateBondEngine`].
pub type BlackCallableZeroCouponBondEngine = BlackCallableFixedRateBondEngine;
