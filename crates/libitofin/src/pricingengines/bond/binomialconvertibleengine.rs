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
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{
        ConvertibleFixedCouponBond, ConvertibleZeroCouponBond, FixedRateBond, ZeroCouponBond,
    };
    use crate::pricingengines::bond::DiscountingBondEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::volatility::BlackVolTermStructure;
    use crate::termstructures::yields::{FlatForward, ZeroSpreadedTermStructure};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::NullCalendar;
    use crate::time::calendars::target::Target;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::time::schedule::MakeSchedule;
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
            .set_pricing_engine(bond_engine as SharedMut<dyn PricingEngine>);
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
}
