//! Binomial Tsiveriotis–Fernandes engine for convertible bonds.
//!
//! Port of QuantLib's `ql/pricingengines/bond/binomialconvertibleengine.hpp`
//! specialised on the Cox–Ross–Rubinstein tree. Flat `r`, `q`, `σ` are taken
//! from the process at the conversion exercise date; a CRR lattice rolls a
//! [`DiscretizedConvertible`] back with blended equity/debt discounting.

use crate::discretizedasset::DiscretizedAsset;
use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::instruments::{BondResults, ConvertibleBondArguments};
use crate::interestrate::Compounding;
use crate::math::timegrid::TimeGrid;
use crate::methods::lattices::binomialtree::CoxRossRubinstein;
use crate::methods::lattices::lattice::Lattice;
use crate::methods::lattices::tree::Tree;
use crate::methods::lattices::treelattice::{TreeLattice1D, TreeLatticeImpl};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::bond::discretizedconvertible::{
    DiscretizedConvertible, DividendSchedule,
};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size};

/// Constant-coefficient Black–Scholes lattice used only for `grid` /
/// `initialize` (rollback is overridden on [`DiscretizedConvertible`]).
struct BlackScholesLattice {
    tree: CoxRossRubinstein,
    grid: TimeGrid,
    risk_free_rate: Real,
}

impl TreeLatticeImpl for BlackScholesLattice {
    type Tree = CoxRossRubinstein;

    fn tree(&self) -> &CoxRossRubinstein {
        &self.tree
    }

    fn discount(&self, i: Size, _index: Size) -> Real {
        (-self.risk_free_rate * self.grid.dt(i)).exp()
    }
}

/// Binomial TF engine for convertible bonds.
pub struct BinomialConvertibleEngine {
    base: GenericEngine<ConvertibleBondArguments, BondResults>,
    process: Shared<GeneralizedBlackScholesProcess>,
    time_steps: Size,
    credit_spread: Handle<dyn Quote>,
    dividends: DividendSchedule,
}

impl BinomialConvertibleEngine {
    /// Builds the engine over `process` with `time_steps` binomial steps and a
    /// credit-spread quote. Optional stock dividends are subtracted from the
    /// spot before the tree is built (as in QuantLib).
    ///
    /// # Errors
    ///
    /// Fails when `time_steps` is zero.
    pub fn new(
        process: Shared<GeneralizedBlackScholesProcess>,
        time_steps: Size,
        credit_spread: Handle<dyn Quote>,
        dividends: DividendSchedule,
    ) -> QlResult<Self> {
        require!(
            time_steps > 0,
            "timeSteps must be positive, {time_steps} not allowed"
        );
        let base = GenericEngine::new(ConvertibleBondArguments::default(), BondResults::default());
        base.register_with(process.observable());
        credit_spread.register_observer(&base.observer());
        Ok(Self {
            base,
            process,
            time_steps,
            credit_spread,
            dividends,
        })
    }

    /// The credit-spread handle.
    pub fn credit_spread(&self) -> &Handle<dyn Quote> {
        &self.credit_spread
    }

    /// The dividend schedule.
    pub fn dividends(&self) -> &DividendSchedule {
        &self.dividends
    }
}

impl AsObservable for BinomialConvertibleEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for BinomialConvertibleEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn calculate(&mut self) -> QlResult<()> {
        let arguments = {
            let args = self.base.arguments();
            ConvertibleBondArguments {
                exercise: args.exercise.clone(),
                conversion_ratio: args.conversion_ratio,
                callability_dates: args.callability_dates.clone(),
                callability_types: args.callability_types.clone(),
                callability_prices: args.callability_prices.clone(),
                callability_triggers: args.callability_triggers.clone(),
                cashflows: args.cashflows.clone(),
                issue_date: args.issue_date,
                settlement_date: args.settlement_date,
                settlement_days: args.settlement_days,
                redemption: args.redemption,
            }
        };

        let Some(exercise) = &arguments.exercise else {
            fail!("no exercise given");
        };
        let maturity_date = exercise.last_date();
        let settlement = arguments
            .settlement_date
            .expect("validated settlement date");

        let risk_free = self.process.risk_free_rate().current_link()?;
        let dividend_ts = self.process.dividend_yield().current_link()?;
        let vol_ts = self.process.black_volatility().current_link()?;
        let rfdc = risk_free.require_day_counter()?;
        let divdc = dividend_ts.require_day_counter()?;

        let mut s0 = self.process.x0()?;
        require!(s0 > 0.0, "negative or null underlying");
        let v = vol_ts.black_vol_date(maturity_date, s0, true)?;
        let risk_free_rate = risk_free
            .zero_rate_date(
                maturity_date,
                rfdc.clone(),
                Compounding::Continuous,
                Frequency::NoFrequency,
                true,
            )?
            .rate();
        let q = dividend_ts
            .zero_rate_date(
                maturity_date,
                divdc,
                Compounding::Continuous,
                Frequency::NoFrequency,
                true,
            )?
            .rate();

        let reference_date = risk_free.reference_date()?;
        for dividend in &self.dividends {
            if dividend.date() >= reference_date {
                s0 -= dividend.amount()? * risk_free.discount_date(dividend.date(), true)?;
            }
        }
        require!(s0 > 0.0, "negative value after subtracting dividends");

        let maturity = rfdc.year_fraction(settlement, maturity_date);
        require!(
            maturity > 0.0,
            "the convertible engine needs a positive maturity"
        );

        let tree = CoxRossRubinstein::new(s0, risk_free_rate, q, v, maturity, self.time_steps)?;
        let pu = tree.probability(0, 0, 1);
        let pd = tree.probability(0, 0, 0);
        let dt = maturity / self.time_steps as Real;
        let grid = TimeGrid::new(maturity, self.time_steps)?;
        let bsl = BlackScholesLattice {
            tree,
            grid: grid.clone(),
            risk_free_rate,
        };
        let lattice: Shared<dyn Lattice> = shared(TreeLattice1D::new(bsl, grid.clone())?);

        let credit_spread = self.credit_spread.current_link()?.value()?;
        let mut convertible = DiscretizedConvertible::new(
            arguments,
            Shared::clone(&self.process),
            self.dividends.clone(),
            credit_spread,
            risk_free_rate,
            pu,
            pd,
            dt,
            &grid,
        )?;
        convertible.initialize(Shared::clone(&lattice), maturity)?;
        convertible.rollback(0.0)?;
        let value = convertible.present_value()?;
        require!(value.is_finite(), "floating-point overflow on tree grid");

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.settlement_value = Some(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflow::CashFlow;
    use crate::cashflows::{Dividend, FixedDividend, SimpleCashFlow};
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::indexes::Euribor;
    use crate::instrument::Instrument;
    use crate::instruments::{
        ConvertibleBondArguments, ConvertibleFixedCouponBond, ConvertibleFloatingRateBond,
        ConvertibleZeroCouponBond, FixedRateBond, FloatingRateBond, PlainVanillaPayoff,
        StrikedTypePayoff, VanillaOption, ZeroCouponBond,
    };
    use crate::math::interpolations::flat::BackwardFlat;
    use crate::option::OptionType;
    use crate::pricingengines::bond::DiscountingBondEngine;
    use crate::pricingengines::vanilla::BinomialVanillaEngine;
    use crate::processes::{BlackProcess, BlackScholesMertonProcess};
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::volatility::BlackVolTermStructure;
    use crate::termstructures::yields::{FlatForward, ForwardCurve, ZeroSpreadedTermStructure};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::NullCalendar;
    use crate::time::calendars::target::Target;
    use crate::time::calendars::unitedstates::{Market, UnitedStates};
    use crate::time::date::{Date, Month};
    use crate::time::dategenerationrule::DateGeneration;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::schedule::{MakeSchedule, Schedule};
    use crate::time::timeunit::TimeUnit;

    struct Vars {
        settings: Shared<Settings<Date>>,
        today: Date,
        issue_date: Date,
        maturity_date: Date,
        process: Shared<BlackScholesMertonProcess>,
        credit_spread: Handle<dyn Quote>,
        risk_free: Handle<dyn YieldTermStructure>,
        conversion_ratio: Real,
        redemption: Real,
        settlement_days: Size,
    }

    fn vars() -> Vars {
        let today = Date::new(15, Month::January, 2020);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let calendar = Target::new();
        let day_counter = Actual360::new();
        let settlement_days = 3usize;
        let fixing_days = 2i32;
        let mut issue_date = calendar.advance(
            today,
            fixing_days,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let maturity_date = calendar.advance(
            issue_date,
            10,
            TimeUnit::Years,
            BusinessDayConvention::Following,
            false,
        );
        issue_date = calendar.advance(
            maturity_date,
            -10,
            TimeUnit::Years,
            BusinessDayConvention::Following,
            false,
        );

        let underlying = 50.0;
        let spot = shared(SimpleQuote::new(underlying));
        let q_rate = shared(SimpleQuote::new(0.02));
        let r_rate = shared(SimpleQuote::new(0.05));
        let vol = shared(SimpleQuote::new(0.15));
        let quote_handle =
            |q: &Shared<SimpleQuote>| Handle::new(Shared::clone(q) as Shared<dyn Quote>);
        let flat = |q: &Shared<SimpleQuote>| {
            Handle::new(shared(FlatForward::moving(
                0,
                NullCalendar::new(),
                quote_handle(q),
                day_counter.clone(),
                Compounding::Continuous,
                Frequency::Annual,
                Shared::clone(&settings),
            )) as Shared<dyn YieldTermStructure>)
        };
        let risk_free = flat(&r_rate);
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat(&q_rate),
            risk_free.clone(),
            Handle::new(shared(BlackConstantVol::moving_with_quote(
                0,
                NullCalendar::new(),
                quote_handle(&vol),
                day_counter.clone(),
                Shared::clone(&settings),
            )) as Shared<dyn BlackVolTermStructure>),
        ));
        let credit = shared(SimpleQuote::new(0.005));
        Vars {
            settings,
            today,
            issue_date,
            maturity_date,
            process,
            credit_spread: Handle::new(credit as Shared<dyn Quote>),
            risk_free,
            conversion_ratio: 100.0 / underlying,
            redemption: 100.0,
            settlement_days,
        }
    }

    fn discount_curve(v: &Vars) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(ZeroSpreadedTermStructure::new(
            v.risk_free.clone(),
            v.credit_spread.clone(),
        )) as Shared<dyn YieldTermStructure>)
    }

    #[test]
    fn out_of_the_money_convertible_matches_the_vanilla_bond() {
        // QuantLib convertiblebonds.cpp `testBond`: with a vanishing conversion
        // ratio the convertible collapses to the credit-spread vanilla bond.
        let v = vars();
        let conversion_ratio = 1.0e-16;
        // QuantLib's convertiblebonds.cpp uses 1001 steps for this oracle.
        let time_steps = 1001usize;
        let engine = shared_mut(
            BinomialConvertibleEngine::new(
                Shared::clone(&v.process),
                time_steps,
                v.credit_spread.clone(),
                Vec::new(),
            )
            .unwrap(),
        );

        let eu_exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(v.maturity_date));
        let am_exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::new(v.issue_date, v.maturity_date, false).unwrap());

        // Zero-coupon
        let zero_schedule = MakeSchedule::new()
            .from(v.issue_date)
            .to(v.maturity_date)
            .with_frequency(Frequency::Once)
            .with_calendar(Target::new())
            .backwards()
            .build();
        let mut eu_zero = ConvertibleZeroCouponBond::new(
            Shared::clone(&eu_exercise),
            conversion_ratio,
            Vec::new(),
            v.issue_date,
            v.settlement_days as u32,
            Actual360::new(),
            zero_schedule.clone(),
            v.redemption,
            Shared::clone(&v.settings),
        )
        .unwrap();
        eu_zero
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);
        let mut am_zero = ConvertibleZeroCouponBond::new(
            Shared::clone(&am_exercise),
            conversion_ratio,
            Vec::new(),
            v.issue_date,
            v.settlement_days as u32,
            Actual360::new(),
            zero_schedule,
            v.redemption,
            Shared::clone(&v.settings),
        )
        .unwrap();
        am_zero
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);

        let mut zero = ZeroCouponBond::new(
            v.settlement_days as u32,
            Target::new(),
            100.0,
            v.maturity_date,
            BusinessDayConvention::Following,
            v.redemption,
            Some(v.issue_date),
            Shared::clone(&v.settings),
        )
        .unwrap();
        let bond_engine = shared_mut(DiscountingBondEngine::new(
            discount_curve(&v),
            None,
            Shared::clone(&v.settings),
        ));
        zero.bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&bond_engine) as SharedMut<dyn PricingEngine>);
        let expected_zero = zero.bond_mut().settlement_value().unwrap();
        let tol_zero = 1.0e-2;
        let eu_zero_npv = eu_zero.npv().unwrap();
        let am_zero_npv = am_zero.npv().unwrap();
        assert!(
            (eu_zero_npv - expected_zero).abs() < tol_zero,
            "eu zero convertible {eu_zero_npv} vs vanilla {expected_zero}"
        );
        assert!(
            (am_zero_npv - expected_zero).abs() < tol_zero,
            "am zero convertible {am_zero_npv} vs vanilla {expected_zero}"
        );

        // Fixed coupon
        let fixed_schedule = MakeSchedule::new()
            .from(v.issue_date)
            .to(v.maturity_date)
            .with_frequency(Frequency::Annual)
            .with_calendar(Target::new())
            .backwards()
            .build();
        let coupons = vec![0.05];
        let mut eu_fixed = ConvertibleFixedCouponBond::new(
            Shared::clone(&eu_exercise),
            conversion_ratio,
            Vec::new(),
            v.issue_date,
            v.settlement_days as u32,
            coupons.clone(),
            Actual360::new(),
            fixed_schedule.clone(),
            v.redemption,
            Shared::clone(&v.settings),
        )
        .unwrap();
        eu_fixed
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);
        let mut am_fixed = ConvertibleFixedCouponBond::new(
            Shared::clone(&am_exercise),
            conversion_ratio,
            Vec::new(),
            v.issue_date,
            v.settlement_days as u32,
            coupons.clone(),
            Actual360::new(),
            fixed_schedule.clone(),
            v.redemption,
            Shared::clone(&v.settings),
        )
        .unwrap();
        am_fixed
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);

        let mut fixed = FixedRateBond::new(
            v.settlement_days as u32,
            100.0,
            fixed_schedule,
            coupons,
            Actual360::new(),
            BusinessDayConvention::Following,
            v.redemption,
            Some(v.issue_date),
            None,
            None,
            Target::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(&v.settings),
        )
        .unwrap();
        fixed
            .bond_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&bond_engine) as SharedMut<dyn PricingEngine>);
        let expected_fixed = fixed.bond_mut().settlement_value().unwrap();
        let tol_fixed = 2.0e-2;
        let eu_fixed_npv = eu_fixed.npv().unwrap();
        let am_fixed_npv = am_fixed.npv().unwrap();
        assert!(
            (eu_fixed_npv - expected_fixed).abs() < tol_fixed,
            "eu fixed convertible {eu_fixed_npv} vs vanilla {expected_fixed}"
        );
        assert!(
            (am_fixed_npv - expected_fixed).abs() < tol_fixed,
            "am fixed convertible {am_fixed_npv} vs vanilla {expected_fixed}"
        );

        // Floating-rate
        let calendar = Target::new();
        let fixing_days = calendar.business_days_between(v.today, v.issue_date, true, false) as u32;
        let index = shared(Euribor::one_year(
            discount_curve(&v),
            Shared::clone(&v.settings),
        ));
        let spreads = Vec::new();
        let mut eu_floating = ConvertibleFloatingRateBond::new(
            Shared::clone(&eu_exercise),
            conversion_ratio,
            Vec::new(),
            v.issue_date,
            v.settlement_days as u32,
            Shared::clone(&index),
            fixing_days,
            spreads.clone(),
            Actual360::new(),
            fixed_schedule.clone(),
            v.redemption,
            Shared::clone(&v.settings),
        )
        .unwrap();
        eu_floating
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);
        let mut am_floating = ConvertibleFloatingRateBond::new(
            Shared::clone(&am_exercise),
            conversion_ratio,
            Vec::new(),
            v.issue_date,
            v.settlement_days as u32,
            Shared::clone(&index),
            fixing_days,
            spreads.clone(),
            Actual360::new(),
            fixed_schedule,
            v.redemption,
            Shared::clone(&v.settings),
        )
        .unwrap();
        am_floating
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);

        let float_schedule = Schedule::new(
            v.issue_date,
            v.maturity_date,
            Period::new(1, TimeUnit::Years),
            calendar,
            BusinessDayConvention::Following,
            BusinessDayConvention::Following,
            DateGeneration::Backward,
            false,
            Date::null(),
            Date::null(),
        );
        let mut floating = FloatingRateBond::new(
            v.settlement_days as u32,
            100.0,
            float_schedule,
            index,
            Actual360::new(),
            BusinessDayConvention::Following,
            Some(fixing_days),
            vec![1.0],
            spreads,
            Vec::new(),
            Vec::new(),
            v.redemption,
            Some(v.issue_date),
            None,
            NullCalendar::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(&v.settings),
        )
        .unwrap();
        floating
            .bond_mut()
            .base_mut()
            .set_pricing_engine(bond_engine as SharedMut<dyn PricingEngine>);
        let expected_float = floating.bond_mut().settlement_value().unwrap();
        let tol_float = 2.0e-2;
        let eu_float_npv = eu_floating.npv().unwrap();
        let am_float_npv = am_floating.npv().unwrap();
        assert!(
            (eu_float_npv - expected_float).abs() < tol_float,
            "eu floating convertible {eu_float_npv} vs vanilla {expected_float}"
        );
        assert!(
            (am_float_npv - expected_float).abs() < tol_float,
            "am floating convertible {am_float_npv} vs vanilla {expected_float}"
        );
    }

    #[test]
    fn at_the_money_convertible_exceeds_the_straight_bond() {
        let v = vars();
        let time_steps = 401usize;
        let engine = shared_mut(
            BinomialConvertibleEngine::new(
                Shared::clone(&v.process),
                time_steps,
                v.credit_spread.clone(),
                Vec::new(),
            )
            .unwrap(),
        );
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(v.maturity_date));
        let schedule = MakeSchedule::new()
            .from(v.issue_date)
            .to(v.maturity_date)
            .with_frequency(Frequency::Annual)
            .with_calendar(Target::new())
            .backwards()
            .build();
        let mut convertible = ConvertibleFixedCouponBond::new(
            exercise,
            v.conversion_ratio,
            Vec::new(),
            v.issue_date,
            v.settlement_days as u32,
            vec![0.05],
            Actual360::new(),
            schedule.clone(),
            v.redemption,
            Shared::clone(&v.settings),
        )
        .unwrap();
        convertible
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

        let mut straight = FixedRateBond::new(
            v.settlement_days as u32,
            100.0,
            schedule,
            vec![0.05],
            Actual360::new(),
            BusinessDayConvention::Following,
            v.redemption,
            Some(v.issue_date),
            None,
            None,
            Target::new(),
            BusinessDayConvention::Unadjusted,
            false,
            None,
            Shared::clone(&v.settings),
        )
        .unwrap();
        let bond_engine = shared_mut(DiscountingBondEngine::new(
            discount_curve(&v),
            None,
            Shared::clone(&v.settings),
        ));
        straight
            .bond_mut()
            .base_mut()
            .set_pricing_engine(bond_engine as SharedMut<dyn PricingEngine>);

        let conv = convertible.npv().unwrap();
        let bond = straight.bond_mut().settlement_value().unwrap();
        assert!(
            conv > bond + 0.5,
            "ATM convertible {conv} should exceed the straight bond {bond}"
        );
        let _ = v.today;
    }

    #[test]
    fn zero_coupon_convertible_matches_vanilla_call() {
        // QuantLib convertiblebonds.cpp `testOption`: a European zero-coupon
        // convertible with a vanishing credit spread is discounted redemption
        // plus conversion-ratio times a vanilla call (strike = redemption /
        // conversion ratio). Settlement is T+0 so the bond and option share
        // the same discounting origin.
        let v = vars();
        let settlement_days = 0u32;
        let time_steps = 2001usize;
        let credit_spread = Handle::new(shared(SimpleQuote::new(0.0)) as Shared<dyn Quote>);
        let engine = shared_mut(
            BinomialConvertibleEngine::new(
                Shared::clone(&v.process),
                time_steps,
                credit_spread,
                Vec::new(),
            )
            .unwrap(),
        );
        let vanilla_engine =
            shared_mut(BinomialVanillaEngine::new(Shared::clone(&v.process), time_steps).unwrap());

        let conversion_strike = v.redemption / v.conversion_ratio;
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, conversion_strike));
        let eu_exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(v.maturity_date));

        let schedule = MakeSchedule::new()
            .from(v.issue_date)
            .to(v.maturity_date)
            .with_frequency(Frequency::Once)
            .with_calendar(Target::new())
            .backwards()
            .build();
        let mut eu_zero = ConvertibleZeroCouponBond::new(
            Shared::clone(&eu_exercise),
            v.conversion_ratio,
            Vec::new(),
            v.issue_date,
            settlement_days,
            Actual360::new(),
            schedule,
            v.redemption,
            Shared::clone(&v.settings),
        )
        .unwrap();
        eu_zero
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);

        let mut eu_option = VanillaOption::new(
            payoff,
            Shared::clone(&eu_exercise),
            Shared::clone(&v.settings),
        );
        eu_option
            .base_mut()
            .set_pricing_engine(vanilla_engine as SharedMut<dyn PricingEngine>);

        let discount = v
            .risk_free
            .current_link()
            .unwrap()
            .discount_date(v.maturity_date, false)
            .unwrap();
        let expected = v.redemption * discount + v.conversion_ratio * eu_option.npv().unwrap();
        let calculated = eu_zero.npv().unwrap();
        let tolerance = 5.0e-2;
        assert!(
            (calculated - expected).abs() < tolerance,
            "zero convertible {calculated} vs discounted redemption + call {expected}"
        );
    }

    #[test]
    fn dividends_spanning_settlement_keep_only_future_amounts() {
        // QuantLib convertiblebonds.cpp `testDividendsSpanningSettlementDate`:
        // a dividend on or before bond settlement is dropped; a later fixed
        // dividend is stored as amount × risk-free discount.
        let v = vars();
        let calendar = Target::new();
        let settlement = calendar.advance(
            v.today,
            v.settlement_days as i32,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let before_settlement = calendar.advance(
            v.today,
            1,
            TimeUnit::Days,
            BusinessDayConvention::Following,
            false,
        );
        let after_settlement = calendar.advance(
            settlement,
            1,
            TimeUnit::Years,
            BusinessDayConvention::Following,
            false,
        );

        let args = ConvertibleBondArguments {
            exercise: Some(shared(EuropeanExercise::new(v.maturity_date)) as Shared<dyn Exercise>),
            conversion_ratio: v.conversion_ratio,
            cashflows: vec![
                shared(SimpleCashFlow::new(v.redemption, v.maturity_date).unwrap())
                    as Shared<dyn CashFlow>,
            ],
            issue_date: Some(v.issue_date),
            settlement_date: Some(settlement),
            settlement_days: v.settlement_days as u32,
            redemption: v.redemption,
            ..ConvertibleBondArguments::default()
        };

        let future_amount = 10.0;
        let dividends: DividendSchedule = vec![
            shared(FixedDividend::new(1.0, before_settlement)) as Shared<dyn Dividend>,
            shared(FixedDividend::new(future_amount, after_settlement)) as Shared<dyn Dividend>,
        ];

        let convertible = DiscretizedConvertible::new(
            args,
            Shared::clone(&v.process),
            dividends,
            v.credit_spread.current_link().unwrap().value().unwrap(),
            0.05,
            0.5,
            0.5,
            0.01,
            &TimeGrid::default(),
        )
        .unwrap();

        let expected = future_amount
            * v.risk_free
                .current_link()
                .unwrap()
                .discount_date(after_settlement, false)
                .unwrap();
        assert_eq!(convertible.dividend_values().size(), 1);
        let calculated = convertible.dividend_values()[0];
        assert!(
            (calculated - expected).abs() <= 1.0e-12 * expected.abs(),
            "dividend PV {calculated} vs {expected}"
        );
    }

    #[test]
    fn known_regression_detects_tree_overflow() {
        // QuantLib convertiblebonds.cpp `testRegression`: a historically
        // overflowing CRR tree must fail with an Error rather than returning
        // Inf. Evaluation is 24 Dec 2008; the process is Black (q = r) with
        // a ~2168% constant vol.
        let today = Date::new(23, Month::December, 2008);
        let tomorrow = today + 1;
        let settings = shared(Settings::new());
        settings.set_evaluation_date(tomorrow);

        let dates = vec![
            Date::new(29, Month::December, 2008),
            Date::new(5, Month::January, 2009),
            Date::new(29, Month::January, 2009),
            Date::new(27, Month::February, 2009),
            Date::new(30, Month::March, 2009),
            Date::new(29, Month::June, 2009),
            Date::new(29, Month::December, 2009),
            Date::new(29, Month::December, 2010),
            Date::new(29, Month::December, 2011),
            Date::new(31, Month::December, 2012),
            Date::new(30, Month::December, 2013),
            Date::new(29, Month::December, 2014),
            Date::new(29, Month::December, 2015),
            Date::new(29, Month::December, 2016),
            Date::new(29, Month::December, 2017),
            Date::new(31, Month::December, 2018),
            Date::new(30, Month::December, 2019),
            Date::new(29, Month::December, 2020),
            Date::new(29, Month::December, 2021),
            Date::new(29, Month::December, 2022),
            Date::new(29, Month::December, 2023),
            Date::new(29, Month::December, 2028),
            Date::new(29, Month::December, 2033),
            Date::new(29, Month::December, 2038),
            Date::new(31, Month::December, 2199),
        ];
        let forwards = vec![
            0.002_599_934_280_0,
            0.002_599_934_280_0,
            0.005_312_327_550_0,
            0.019_704_959_872_1,
            0.022_052_484_529_6,
            0.021_707_639_564_3,
            0.023_034_962_747_8,
            0.008_763_164_747_6,
            0.021_908_429_949_9,
            0.024_479_876_621_9,
            0.026_788_549_845_6,
            0.026_692_286_756_2,
            0.027_105_212_638_6,
            0.026_882_989_164_8,
            0.026_459_474_449_8,
            0.027_345_036_742_4,
            0.029_485_261_474_9,
            0.028_555_611_971_9,
            0.030_555_776_465_9,
            0.029_224_473_842_2,
            0.026_391_700_419_4,
            0.023_962_697_024_3,
            0.021_641_710_809_0,
            0.022_834_383_842_2,
            0.022_834_383_842_2,
        ];
        let risk_free: Handle<dyn YieldTermStructure> = Handle::new(shared(
            ForwardCurve::new(dates, forwards, Actual360::new(), BackwardFlat).unwrap(),
        )
            as Shared<dyn YieldTermStructure>);
        // QuantLib `BlackProcess(u, r, sigma)` sets q = r.
        let process = shared(BlackProcess::new(
            Handle::new(shared(SimpleQuote::new(2.908_438_281_879_744_3)) as Shared<dyn Quote>),
            risk_free.clone(),
            risk_free,
            Handle::new(shared(BlackConstantVol::new(
                tomorrow,
                Some(NullCalendar::new()),
                21.685_235_548_092_248,
                Thirty360::with_convention(Convention::BondBasis),
            )) as Shared<dyn BlackVolTermStructure>),
        ));
        let spread =
            Handle::new(shared(SimpleQuote::new(0.114_987_006_780_128_74)) as Shared<dyn Quote>);

        let issue_date = Date::new(23, Month::July, 2008);
        let maturity_date = Date::new(1, Month::August, 2013);
        let calendar = UnitedStates::new(Market::GovernmentBond);
        let schedule = MakeSchedule::new()
            .from(issue_date)
            .to(maturity_date)
            .with_tenor(Period::new(6, TimeUnit::Months))
            .with_calendar(calendar)
            .with_convention(BusinessDayConvention::Unadjusted)
            .build();
        let coupons = vec![0.05; schedule.len().saturating_sub(1)];
        let mut bond = ConvertibleFixedCouponBond::new(
            shared(EuropeanExercise::new(maturity_date)) as Shared<dyn Exercise>,
            100.0 / 20.3175,
            Vec::new(),
            issue_date,
            3,
            coupons,
            Thirty360::with_convention(Convention::BondBasis),
            schedule,
            100.0,
            Shared::clone(&settings),
        )
        .unwrap();
        bond.base_mut().set_pricing_engine(shared_mut(
            BinomialConvertibleEngine::new(process, 600, spread, Vec::new()).unwrap(),
        ) as SharedMut<dyn PricingEngine>);

        let err = bond.npv().expect_err("INF result was not detected");
        assert!(
            err.message().contains("overflow") || err.message().contains("finite"),
            "expected overflow Error, got: {}",
            err.message()
        );
    }
}
