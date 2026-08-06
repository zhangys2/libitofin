//! G2++ European swaption engine.
//!
//! Port of `ql/pricingengines/swaption/g2swaptionengine.hpp`: prices a
//! physically settled European swaption under [`G2`] by correcting the fixed
//! rate for any floating-leg spread (via [`DiscountingSwapEngine`]) and
//! calling [`G2::swaption`](crate::models::shortrate::G2::swaption).
//!
//! ## Warning (from QuantLib)
//!
//! The engine assumes the exercise date equals the start date of the passed
//! swap.

use crate::errors::QlResult;
use crate::fail;
use crate::instrument::Instrument;
use crate::instrument::InstrumentResults;
use crate::instruments::{SettlementType, SwaptionArguments, SwaptionEngine};
use crate::models::model::CalibratedModelHolder;
use crate::models::shortrate::G2;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::swap::DiscountingSwapEngine;
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::time::date::Date;
use crate::types::{Real, Size};

/// European swaption engine for the two-factor Gaussian model G2++
/// (`g2swaptionengine.hpp:39`).
pub struct G2SwaptionEngine {
    base: SwaptionEngine,
    model: SharedMut<G2>,
    range: Real,
    intervals: Size,
    /// Needed by [`DiscountingSwapEngine`] (C++ reads global `Settings`; Rust
    /// engines carry the handle explicitly).
    settings: Shared<Settings<Date>>,
}

impl G2SwaptionEngine {
    /// `G2SwaptionEngine(model, range, intervals)` (`g2swaptionengine.hpp:45`):
    /// `range` is the number of standard deviations for the integral limits;
    /// `intervals` is the [`SegmentIntegral`](crate::math::integrals::segment::SegmentIntegral)
    /// partition count. Typical QL example values: `range = 6.0`,
    /// `intervals = 16`.
    pub fn new(
        model: SharedMut<G2>,
        range: Real,
        intervals: Size,
        settings: Shared<Settings<Date>>,
    ) -> G2SwaptionEngine {
        let base = SwaptionEngine::new(SwaptionArguments::default(), InstrumentResults::default());
        base.register_with(model.borrow().calibrated_model().observable());
        G2SwaptionEngine {
            base,
            model,
            range,
            intervals,
            settings,
        }
    }
}

impl AsObservable for G2SwaptionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for G2SwaptionEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    /// `calculate` (`g2swaptionengine.hpp:50-66`).
    fn calculate(&mut self) -> QlResult<()> {
        let (nominal, swap_type, floating_reset_dates, fixed_pay_dates, fixed_rate) = {
            let arguments = self.base.arguments();
            require!(
                arguments.settlement_type == SettlementType::Physical,
                "cash-settled swaptions not priced with G2 engine"
            );

            let Some(swap) = arguments.swap.as_ref() else {
                fail!("swap not set");
            };
            let swap_args = &arguments.swap_arguments;

            let Some(nominal) = swap_args.nominal else {
                fail!("non-constant nominals are not supported yet");
            };
            let Some(swap_type) = swap_args.swap_type else {
                fail!("swap type not set");
            };
            require!(
                !swap_args.floating_reset_dates.is_empty(),
                "swap has no floating resets"
            );
            require!(
                !swap_args.fixed_pay_dates.is_empty(),
                "swap has no fixed payment dates"
            );

            // Adjust the fixed rate for the floating-leg spread (not taken into
            // account by the model) — same correction as BlackSwaptionEngine.
            let discounting = shared_mut(DiscountingSwapEngine::new(
                self.model.borrow().term_structure(),
                Some(false),
                None,
                None,
                Shared::clone(&self.settings),
            )) as SharedMut<dyn PricingEngine>;

            let mut swap_ref = swap.borrow_mut();
            swap_ref
                .base_mut()
                .set_pricing_engine_silent(SharedMut::clone(&discounting));
            let spread = swap_ref.spread();
            let fixed_rate = swap_ref.fixed_rate();
            let correction = if spread != 0.0 {
                spread * (swap_ref.floating_leg_bps()? / swap_ref.fixed_leg_bps()?).abs()
            } else {
                0.0
            };
            drop(swap_ref);

            (
                nominal,
                swap_type,
                swap_args.floating_reset_dates.clone(),
                swap_args.fixed_pay_dates.clone(),
                fixed_rate - correction,
            )
        };

        let value = self.model.borrow().swaption(
            nominal,
            swap_type,
            &floating_reset_dates,
            &fixed_pay_dates,
            fixed_rate,
            self.range,
            self.intervals,
        )?;

        self.base.results_mut().value = Some(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::indexes::IborIndex;
    use crate::indexes::ibor::Euribor;
    use crate::instruments::{SettlementMethod, SettlementType, SwapType, Swaption, VanillaSwap};
    use crate::interestrate::Compounding;
    use crate::shared::{Shared, shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::target::Target;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;
    use crate::time::schedule::{MakeSchedule, Schedule};

    const NOMINAL: Real = 100.0;
    const FIXED_RATE: Real = 0.05;

    fn today() -> Date {
        Date::new(15, Month::January, 2026)
    }

    fn settings() -> Shared<Settings<Date>> {
        let s = shared(Settings::new());
        s.set_evaluation_date(today());
        s
    }

    fn flat_curve() -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            0.05,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn schedule(from: Date, to: Date, frequency: Frequency) -> Schedule {
        MakeSchedule::new()
            .from(from)
            .to(to)
            .with_frequency(frequency)
            .with_calendar(Target::new())
            .with_convention(BusinessDayConvention::Unadjusted)
            .with_termination_date_convention(BusinessDayConvention::Unadjusted)
            .forwards()
            .end_of_month(false)
            .build()
    }

    /// European swaption: exercise equals swap start (G2 engine assumption).
    fn fixture_swaption(swap_type: SwapType, spread: Real) -> (Swaption, SharedMut<G2>) {
        let settings = settings();
        let curve = flat_curve();
        let model = G2::new(curve.clone(), 0.1, 0.01, 0.2, 0.008, -0.75).unwrap();
        let index: Shared<IborIndex> = shared(Euribor::six_months(curve, Shared::clone(&settings)));
        let start = Date::new(15, Month::January, 2027);
        let end = Date::new(15, Month::January, 2032);
        let swap = shared_mut(
            VanillaSwap::new(
                swap_type,
                NOMINAL,
                schedule(start, end, Frequency::Annual),
                FIXED_RATE,
                Thirty360::with_convention(Convention::BondBasis),
                schedule(start, end, Frequency::Semiannual),
                index,
                spread,
                Actual360::new(),
                None,
                Shared::clone(&settings),
            )
            .unwrap()
            .into_fixed_vs_floating(),
        );
        let mut swaption = Swaption::new(
            swap,
            shared(EuropeanExercise::new(start)) as Shared<dyn Exercise>,
            SettlementType::Physical,
            SettlementMethod::PhysicalOTC,
            Shared::clone(&settings),
        );
        let engine = shared_mut(G2SwaptionEngine::new(
            SharedMut::clone(&model),
            6.0,
            16,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;
        swaption.base_mut().set_pricing_engine(engine);
        (swaption, model)
    }

    #[test]
    fn rejects_cash_settlement() {
        let mut engine = G2SwaptionEngine::new(
            G2::new(flat_curve(), 0.1, 0.01, 0.2, 0.008, -0.75).unwrap(),
            6.0,
            16,
            settings(),
        );
        let args = (engine.arguments_mut() as &mut dyn std::any::Any)
            .downcast_mut::<SwaptionArguments>()
            .unwrap();
        args.settlement_type = SettlementType::Cash;
        assert_eq!(
            engine.calculate().unwrap_err().message(),
            "cash-settled swaptions not priced with G2 engine"
        );
    }

    #[test]
    fn both_flavours_are_positive_options() {
        let (mut payer, _) = fixture_swaption(SwapType::Payer, 0.0);
        let (mut receiver, _) = fixture_swaption(SwapType::Receiver, 0.0);
        let p = payer.npv().unwrap();
        let r = receiver.npv().unwrap();
        assert!(p > 0.0 && r > 0.0, "payer={p} receiver={r}");
    }

    /// European identity: payer − receiver = underlying forward-starting swap NPV
    /// (exercise date = swap start).
    ///
    /// G2 accrues fixed coupons with the discount-curve day counter
    /// (`g2.cpp:162-168`), so the fixture uses that same day counter on the
    /// fixed leg; otherwise DiscountingSwapEngine and the G2 integral price
    /// slightly different cashflows.
    #[test]
    fn payer_minus_receiver_matches_swap_npv() {
        let settings = settings();
        let curve = flat_curve();
        let model = G2::new(curve.clone(), 0.1, 0.01, 0.2, 0.008, -0.75).unwrap();
        let start = Date::new(15, Month::January, 2027);
        let end = Date::new(15, Month::January, 2032);
        // Off-market strike so the swap NPV is non-trivial.
        let strike = 0.04;
        let mk_swap = |swap_type: SwapType| {
            let index: Shared<IborIndex> =
                shared(Euribor::six_months(curve.clone(), Shared::clone(&settings)));
            shared_mut(
                VanillaSwap::new(
                    swap_type,
                    NOMINAL,
                    schedule(start, end, Frequency::Annual),
                    strike,
                    Actual365Fixed::new(),
                    schedule(start, end, Frequency::Semiannual),
                    index,
                    0.0,
                    Actual360::new(),
                    None,
                    Shared::clone(&settings),
                )
                .unwrap()
                .into_fixed_vs_floating(),
            )
        };
        let mk_swaption = |swap_type: SwapType| {
            let swap = mk_swap(swap_type);
            let mut swaption = Swaption::new(
                swap,
                shared(EuropeanExercise::new(start)) as Shared<dyn Exercise>,
                SettlementType::Physical,
                SettlementMethod::PhysicalOTC,
                Shared::clone(&settings),
            );
            // Finer integral than the QL example defaults so the European
            // put-call identity holds tightly once day counts agree.
            let engine = shared_mut(G2SwaptionEngine::new(
                SharedMut::clone(&model),
                8.0,
                64,
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>;
            swaption.base_mut().set_pricing_engine(engine);
            swaption
        };

        let mut payer = mk_swaption(SwapType::Payer);
        let mut receiver = mk_swaption(SwapType::Receiver);
        let p = payer.npv().unwrap();
        let r = receiver.npv().unwrap();

        let swap = mk_swap(SwapType::Payer);
        let discounting = shared_mut(DiscountingSwapEngine::new(
            curve,
            Some(false),
            None,
            None,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;
        swap.borrow_mut()
            .base_mut()
            .set_pricing_engine_silent(discounting);
        let swap_npv = swap.borrow_mut().npv().unwrap();

        assert!(
            ((p - r) - swap_npv).abs() < 1e-3,
            "payer {p} - receiver {r} = {} vs swap NPV {swap_npv}",
            p - r
        );
    }

    #[test]
    fn higher_vol_raises_atm_payer_value() {
        let settings = settings();
        let curve = flat_curve();
        let start = Date::new(15, Month::January, 2027);
        let end = Date::new(15, Month::January, 2032);
        let mk = |sigma: Real, eta: Real| {
            let model = G2::new(curve.clone(), 0.1, sigma, 0.2, eta, -0.75).unwrap();
            let index: Shared<IborIndex> =
                shared(Euribor::six_months(curve.clone(), Shared::clone(&settings)));
            let swap = shared_mut(
                VanillaSwap::new(
                    SwapType::Payer,
                    NOMINAL,
                    schedule(start, end, Frequency::Annual),
                    FIXED_RATE,
                    Thirty360::with_convention(Convention::BondBasis),
                    schedule(start, end, Frequency::Semiannual),
                    index,
                    0.0,
                    Actual360::new(),
                    None,
                    Shared::clone(&settings),
                )
                .unwrap()
                .into_fixed_vs_floating(),
            );
            let mut swaption = Swaption::new(
                swap,
                shared(EuropeanExercise::new(start)) as Shared<dyn Exercise>,
                SettlementType::Physical,
                SettlementMethod::PhysicalOTC,
                Shared::clone(&settings),
            );
            let engine = shared_mut(G2SwaptionEngine::new(
                model,
                6.0,
                16,
                Shared::clone(&settings),
            )) as SharedMut<dyn PricingEngine>;
            swaption.base_mut().set_pricing_engine(engine);
            swaption.npv().unwrap()
        };

        let low = mk(0.005, 0.004);
        let high = mk(0.02, 0.015);
        assert!(
            high > low,
            "higher factor vols must raise the ATM payer value: low={low} high={high}"
        );
    }

    #[test]
    fn spread_correction_moves_value() {
        let (mut flat_spread, _) = fixture_swaption(SwapType::Payer, 0.0);
        let (mut with_spread, _) = fixture_swaption(SwapType::Payer, 0.001);
        let v0 = flat_spread.npv().unwrap();
        let v1 = with_spread.npv().unwrap();
        assert!(
            (v0 - v1).abs() > 1e-4,
            "floating spread correction must move the NPV: {v0} vs {v1}"
        );
    }
}
