//! A swap exchanging a constant-maturity-swap (CMS) leg for a fixed leg.
//!
//! `CmsSwap` pays or receives a fixed leg against a leg of [`CmsCoupon`]s that
//! track a [`SwapIndex`]. It reuses the generic [`Swap`] base and prices through
//! a swap engine (e.g. `DiscountingSwapEngine`), mirroring
//! [`VanillaSwap`](crate::instruments::VanillaSwap).
//!
//! This first slice uses the raw (unadjusted) CMS rate - `gearing · swapRate +
//! spread` - since the convexity-adjusting CMS pricers are still a follow-up, so
//! it is exact only where a convexity adjustment is not required. There is no
//! single cached `test-suite` oracle, so the behaviour is pinned by the
//! fair-rate identities in the tests.

use crate::cashflow::{CashFlow, Leg};
use crate::cashflows::{CmsCoupon, FixedRateLeg};
use crate::errors::QlResult;
use crate::indexes::SwapIndex;
use crate::indexes::index::Index;
use crate::indexes::interestrateindex::InterestRateIndex;
use crate::instrument::{Instrument, InstrumentBase};
use crate::instruments::{Swap, SwapType};
use crate::interestrate::Compounding;
use crate::pricingengine::{Arguments, Results};
use crate::settings::Settings;
use crate::shared::{Shared, shared};
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::calendar::Calendar;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::time::frequency::Frequency;
use crate::time::schedule::Schedule;
use crate::types::{Rate, Real, Spread};

const BASIS_POINT: Real = 1.0e-4;

/// A fixed-vs-CMS swap. Leg 0 is the fixed leg, leg 1 the CMS leg; the
/// [`SwapType`] chooses which is paid (`Payer` pays fixed, receives CMS).
pub struct CmsSwap {
    swap: Swap,
    fixed_rate: Rate,
}

impl CmsSwap {
    /// Builds a fixed-vs-CMS swap on a single `nominal`.
    ///
    /// `Payer` pays the fixed leg and receives the CMS leg. The CMS leg tracks
    /// `swap_index` with `cms_gearing`/`cms_spread`; its payment calendar and
    /// convention are taken from the index's fixing calendar and
    /// `payment_convention` (defaulting to `Following`).
    ///
    /// # Errors
    ///
    /// Propagates any leg-building error and any [`Swap`] construction error.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        swap_type: SwapType,
        nominal: Real,
        fixed_schedule: Schedule,
        fixed_rate: Rate,
        fixed_day_count: DayCounter,
        cms_schedule: Schedule,
        swap_index: Shared<SwapIndex>,
        cms_gearing: Real,
        cms_spread: Spread,
        cms_day_count: DayCounter,
        payment_convention: Option<BusinessDayConvention>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<CmsSwap> {
        let convention = payment_convention.unwrap_or(BusinessDayConvention::Following);

        let fixed_leg = FixedRateLeg::new(fixed_schedule)
            .with_notional(nominal)
            .with_coupon_rate(
                fixed_rate,
                fixed_day_count,
                Compounding::Simple,
                Frequency::Annual,
            )?
            .with_payment_adjustment(convention)
            .build()?;

        let cms_leg = build_cms_leg(
            &cms_schedule,
            &swap_index,
            nominal,
            cms_gearing,
            cms_spread,
            cms_day_count,
            convention,
        )?;

        let payer = match swap_type {
            SwapType::Payer => vec![true, false],
            SwapType::Receiver => vec![false, true],
        };
        let swap = Swap::new(vec![fixed_leg, cms_leg], payer, settings)?;
        Ok(CmsSwap { swap, fixed_rate })
    }

    /// The generic swap this wraps.
    pub fn swap(&self) -> &Swap {
        &self.swap
    }

    /// The fixed leg's NPV.
    pub fn fixed_leg_npv(&mut self) -> QlResult<Real> {
        self.swap.leg_npv(0)
    }

    /// The CMS leg's NPV.
    pub fn cms_leg_npv(&mut self) -> QlResult<Real> {
        self.swap.leg_npv(1)
    }

    /// The fixed rate that zeroes the swap NPV.
    ///
    /// Uses the same first-order relation as the discounting engine's
    /// `FixedVsFloatingSwap` fallback: `fixedRate − NPV / (fixedLegBPS / 1bp)`.
    ///
    /// # Errors
    ///
    /// The engine must have priced the swap and provided the fixed leg's BPS.
    pub fn fair_rate(&mut self) -> QlResult<Rate> {
        self.calculate()?;
        let npv = self.npv()?;
        let bps = self.swap.leg_bps(0)?;
        Ok(self.fixed_rate - npv / (bps / BASIS_POINT))
    }
}

fn build_cms_leg(
    schedule: &Schedule,
    swap_index: &Shared<SwapIndex>,
    nominal: Real,
    gearing: Real,
    spread: Spread,
    day_counter: DayCounter,
    convention: BusinessDayConvention,
) -> QlResult<Leg> {
    let calendar: Calendar = Index::fixing_calendar(&**swap_index);
    let fixing_days = InterestRateIndex::fixing_days(&**swap_index);
    let periods = schedule.len() - 1;
    let mut leg: Leg = Vec::with_capacity(periods);
    for i in 0..periods {
        let start = schedule.date(i);
        let end = schedule.date(i + 1);
        let payment_date = calendar.adjust(end, convention);
        let coupon = shared(CmsCoupon::new(
            payment_date,
            nominal,
            start,
            end,
            fixing_days,
            Shared::clone(swap_index),
            gearing,
            spread,
            start,
            end,
            day_counter.clone(),
            false,
        )?);
        leg.push(coupon as Shared<dyn CashFlow>);
    }
    Ok(leg)
}

impl Instrument for CmsSwap {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::currency::Currency;
    use crate::handle::Handle;
    use crate::indexes::ibor::Euribor;
    use crate::pricingengine::PricingEngine;
    use crate::pricingengines::swap::DiscountingSwapEngine;
    use crate::shared::{SharedMut, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::calendars::target::Target;
    use crate::time::date::Month;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    const NOMINAL: Real = 1_000_000.0;
    const RATE: Real = 0.03;

    fn today() -> Date {
        Date::new(15, Month::January, 2020)
    }

    fn settings() -> Shared<Settings<Date>> {
        let s = shared(Settings::new());
        s.set_evaluation_date(today());
        s
    }

    fn curve() -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            RATE,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn swap_index(s: &Shared<Settings<Date>>) -> Shared<SwapIndex> {
        let ibor = shared(Euribor::six_months(curve(), Shared::clone(s)));
        shared(SwapIndex::new(
            "EuriborSwap".into(),
            Period::new(5, TimeUnit::Years),
            2,
            Currency::eur(),
            Target::new(),
            Period::new(1, TimeUnit::Years),
            BusinessDayConvention::ModifiedFollowing,
            Thirty360::with_convention(Convention::BondBasis),
            ibor,
            Shared::clone(s),
        ))
    }

    fn schedule() -> Schedule {
        let calendar = Target::new();
        let settlement = calendar.advance(
            today(),
            2,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let maturity = calendar.advance(
            settlement,
            5,
            TimeUnit::Years,
            BusinessDayConvention::Following,
            false,
        );
        crate::time::schedule::MakeSchedule::new()
            .from(settlement)
            .to(maturity)
            .with_frequency(Frequency::Annual)
            .with_calendar(calendar)
            .with_convention(BusinessDayConvention::ModifiedFollowing)
            .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
            .forwards()
            .end_of_month(false)
            .build()
    }

    fn cms_swap(swap_type: SwapType, fixed_rate: Rate, s: &Shared<Settings<Date>>) -> CmsSwap {
        let mut swap = CmsSwap::new(
            swap_type,
            NOMINAL,
            schedule(),
            fixed_rate,
            Thirty360::with_convention(Convention::BondBasis),
            schedule(),
            swap_index(s),
            1.0,
            0.0,
            Actual360::new(),
            Some(BusinessDayConvention::ModifiedFollowing),
            Shared::clone(s),
        )
        .unwrap();
        let engine = shared_mut(DiscountingSwapEngine::new(
            curve(),
            None,
            None,
            None,
            Shared::clone(s),
        ));
        swap.base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        swap
    }

    #[test]
    fn the_fair_rate_zeroes_the_npv() {
        let s = settings();
        let mut off_market = cms_swap(SwapType::Payer, 0.01, &s);
        let fair = off_market.fair_rate().unwrap();
        let mut repriced = cms_swap(SwapType::Payer, fair, &s);
        assert!(
            repriced.npv().unwrap().abs() < 1e-6,
            "npv at the fair rate should vanish, got {}",
            repriced.npv().unwrap()
        );
    }

    #[test]
    fn paying_a_higher_fixed_rate_lowers_the_payer_value() {
        let s = settings();
        let low = cms_swap(SwapType::Payer, 0.02, &s).npv().unwrap();
        let high = cms_swap(SwapType::Payer, 0.04, &s).npv().unwrap();
        assert!(
            high < low,
            "payer value should fall with the fixed rate: {high} !< {low}"
        );
    }

    #[test]
    fn the_fair_rate_tracks_the_forward_cms_level() {
        // On a flat curve the fair fixed rate should sit near the CMS (swap)
        // rate the coupons forecast.
        let s = settings();
        let mut swap = cms_swap(SwapType::Payer, 0.01, &s);
        let fair = swap.fair_rate().unwrap();

        let index = swap_index(&s);
        let fixing_date = Target::new().advance(
            today(),
            2,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let cms_rate = Index::fixing(&*index, fixing_date, false).unwrap();
        assert!(
            (fair - cms_rate).abs() < 0.01,
            "fair rate {fair} should be near the forecast CMS rate {cms_rate}"
        );
    }
}
