//! Finite-difference G2++ swaption engine.
//!
//! Port of `ql/pricingengines/swaption/fdg2swaptionengine.{hpp,cpp}`: builds a
//! two-factor process mesher, an
//! [`FdmAffineModelSwapInnerValue`](crate::methods::finitedifferences::utilities::FdmAffineModelSwapInnerValue),
//! Bermudan (or empty European) step conditions, and rolls back with
//! [`FdmG2Solver`](crate::methods::finitedifferences::solvers::FdmG2Solver).
//!
//! ## Divergences from QuantLib
//!
//! - `vanillaComposite` is hand-rolled (empty dividends + Bermudan / European),
//!   matching [`FdmBermudanEngine`](crate::pricingengines::vanilla).

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instrument::InstrumentResults;
use crate::instruments::{SwaptionArguments, SwaptionEngine};
use crate::methods::finitedifferences::StepCondition;
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_simple_process_1d_mesher,
};
use crate::methods::finitedifferences::solvers::{FdmG2Solver, FdmSchemeDesc, FdmSolverDesc};
use crate::methods::finitedifferences::stepconditions::{
    FdmBermudanStepCondition, FdmStepConditionComposite,
};
use crate::methods::finitedifferences::utilities::{
    FdmAffineModelSwapInnerValue, FdmInnerValueCalculator,
};
use crate::models::model::CalibratedModelHolder;
use crate::models::shortrate::G2;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::processes::OrnsteinUhlenbeckProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared};
use crate::types::{Real, Size, Time};

/// Finite-difference swaption engine under G2++ (`fdg2swaptionengine.hpp:36`).
pub struct FdG2SwaptionEngine {
    base: SwaptionEngine,
    model: SharedMut<G2>,
    t_grid: Size,
    x_grid: Size,
    y_grid: Size,
    damping_steps: Size,
    inv_eps: Real,
    scheme_desc: FdmSchemeDesc,
}

impl FdG2SwaptionEngine {
    /// QL defaults (`fdg2swaptionengine.hpp:40-45`):
    /// `tGrid=100`, `xGrid=50`, `yGrid=50`, `dampingSteps=0`, `invEps=1e-5`,
    /// Hundsdorfer.
    pub fn new(model: SharedMut<G2>) -> FdG2SwaptionEngine {
        Self::with_params(model, 100, 50, 50, 0, 1e-5, FdmSchemeDesc::hundsdorfer())
    }

    /// Full constructor (`fdg2swaptionengine.hpp:40-45`).
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        model: SharedMut<G2>,
        t_grid: Size,
        x_grid: Size,
        y_grid: Size,
        damping_steps: Size,
        inv_eps: Real,
        scheme_desc: FdmSchemeDesc,
    ) -> FdG2SwaptionEngine {
        let base = SwaptionEngine::new(SwaptionArguments::default(), InstrumentResults::default());
        base.register_with(model.borrow().calibrated_model().observable());
        FdG2SwaptionEngine {
            base,
            model,
            t_grid,
            x_grid,
            y_grid,
            damping_steps,
            inv_eps,
            scheme_desc,
        }
    }
}

impl AsObservable for FdG2SwaptionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdG2SwaptionEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.base.arguments_mut()
    }

    fn results(&self) -> &dyn Results {
        self.base.results()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    /// `calculate` (`fdg2swaptionengine.cpp:54-128`).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn calculate(&mut self) -> QlResult<()> {
        let arguments = self.base.arguments();
        let Some(exercise) = arguments.exercise.as_ref() else {
            fail!("exercise not set");
        };
        let Some(swap) = arguments.swap.as_ref() else {
            fail!("swap not set");
        };

        let (a, sigma, b, eta, rho, maturity, reference_date, day_counter, t2d) = {
            let model = self.model.borrow();
            let ts = model.term_structure().current_link()?;
            let day_counter = ts.require_day_counter()?;
            let reference_date = ts.reference_date()?;
            let maturity = day_counter.year_fraction(reference_date, exercise.last_date());

            let mut t2d = Vec::with_capacity(exercise.dates().len());
            for &date in exercise.dates() {
                let t = day_counter.year_fraction(reference_date, date);
                require!(t >= 0.0, "exercise dates must not contain past date");
                t2d.push((t, date));
            }

            (
                model.a(),
                model.sigma(),
                model.b(),
                model.eta(),
                model.rho(),
                maturity,
                reference_date,
                day_counter,
                t2d,
            )
        };

        let process1 = OrnsteinUhlenbeckProcess::new(a, sigma, 0.0, 0.0)?;
        let process2 = OrnsteinUhlenbeckProcess::new(b, eta, 0.0, 0.0)?;
        let x_mesher =
            fdm_simple_process_1d_mesher(self.x_grid, &process1, maturity, 1, self.inv_eps, None)?;
        let y_mesher =
            fdm_simple_process_1d_mesher(self.y_grid, &process2, maturity, 1, self.inv_eps, None)?;
        let mesher =
            shared(FdmMesherComposite::new(vec![x_mesher, y_mesher])) as Shared<dyn FdmMesher>;

        let fwd_ts = {
            let swap = swap.borrow();
            let fwd_ts = swap.ibor_index().forwarding_term_structure().clone();
            let fwd = fwd_ts.current_link()?;
            require!(
                fwd.require_day_counter()? == day_counter,
                "day counter of forward and discount curve must match"
            );
            require!(
                fwd.reference_date()? == reference_date,
                "reference date of forward and discount curve must match"
            );
            fwd_ts
        };
        let fwd_model = G2::new(fwd_ts, a, sigma, b, eta, rho)?;

        let calculator = shared(FdmAffineModelSwapInnerValue::new(
            SharedMut::clone(&self.model),
            fwd_model,
            &swap.borrow(),
            t2d,
            Shared::clone(&mesher),
            0,
        )?) as Shared<dyn FdmInnerValueCalculator>;

        let condition = match exercise.exercise_type() {
            ExerciseType::European => shared(FdmStepConditionComposite::new(&[], Vec::new())),
            ExerciseType::Bermudan => {
                let bermudan = FdmBermudanStepCondition::new(
                    exercise.dates(),
                    reference_date,
                    &day_counter,
                    Shared::clone(&mesher),
                    Shared::clone(&calculator),
                );
                let times: Vec<Time> = bermudan.exercise_times().to_vec();
                shared(FdmStepConditionComposite::new(
                    &[times],
                    vec![shared(bermudan) as Shared<dyn StepCondition>],
                ))
            }
            ExerciseType::American => {
                fail!("American exercise is not supported by FdG2SwaptionEngine");
            }
        };

        let solver_desc = FdmSolverDesc {
            mesher,
            bc_set: Vec::new(),
            condition,
            calculator,
            maturity,
            time_steps: self.t_grid,
            damping_steps: self.damping_steps,
        };
        let solver = FdmG2Solver::new(SharedMut::clone(&self.model), solver_desc, self.scheme_desc);
        self.base.results_mut().value = Some(solver.value_at(0.0, 0.0)?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{BermudanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::indexes::IborIndex;
    use crate::indexes::InterestRateIndex;
    use crate::indexes::ibor::Euribor;
    use crate::instrument::Instrument;
    use crate::instruments::{
        FixedVsFloatingSwap, SettlementMethod, SettlementType, SwapType, Swaption, VanillaSwap,
    };
    use crate::interestrate::Compounding;
    use crate::pricingengines::swap::DiscountingSwapEngine;
    use crate::pricingengines::swaption::G2SwaptionEngine;
    use crate::settings::Settings;
    use crate::shared::{shared, shared_mut};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::businessdayconvention::BusinessDayConvention;
    use crate::time::calendars::target::Target;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::daycounters::thirty360::{Convention, Thirty360};
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::schedule::{MakeSchedule, Schedule};
    use crate::time::timeunit::TimeUnit;

    const NOMINAL: Real = 100.0;

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

    fn model(curve: Handle<dyn YieldTermStructure>) -> SharedMut<G2> {
        G2::new(curve, 0.1, 0.01, 0.2, 0.008, -0.75).unwrap()
    }

    fn make_swap(
        curve: Handle<dyn YieldTermStructure>,
        settings: &Shared<Settings<Date>>,
        swap_type: SwapType,
        fixed_rate: Real,
        start: Date,
        end: Date,
    ) -> SharedMut<crate::instruments::FixedVsFloatingSwap> {
        let index: Shared<IborIndex> = shared(Euribor::six_months(curve, Shared::clone(settings)));
        shared_mut(
            VanillaSwap::new(
                swap_type,
                NOMINAL,
                schedule(start, end, Frequency::Annual),
                fixed_rate,
                Actual365Fixed::new(),
                schedule(start, end, Frequency::Semiannual),
                index,
                0.0,
                Actual360::new(),
                None,
                Shared::clone(settings),
            )
            .unwrap()
            .into_fixed_vs_floating(),
        )
    }

    #[test]
    fn european_itm_payer_is_finite_and_positive() {
        let settings = settings();
        let curve = flat_curve();
        let g2 = model(curve.clone());
        let start = Date::new(15, Month::January, 2027);
        let end = Date::new(15, Month::January, 2032);
        let swap = make_swap(curve, &settings, SwapType::Payer, 0.03, start, end);
        let mut swaption = Swaption::new(
            swap,
            shared(EuropeanExercise::new(start)) as Shared<dyn Exercise>,
            SettlementType::Physical,
            SettlementMethod::PhysicalOTC,
            Shared::clone(&settings),
        );
        let engine = shared_mut(FdG2SwaptionEngine::with_params(
            g2,
            20,
            21,
            21,
            0,
            1e-4,
            FdmSchemeDesc::hundsdorfer(),
        )) as SharedMut<dyn PricingEngine>;
        swaption.base_mut().set_pricing_engine(engine);
        let npv = swaption.npv().unwrap();
        assert!(npv.is_finite() && npv > 0.0, "npv={npv}");
    }

    #[test]
    fn european_tracks_analytic_g2_engine() {
        let settings = settings();
        let curve = flat_curve();
        let g2 = model(curve.clone());
        let start = Date::new(15, Month::January, 2027);
        let end = Date::new(15, Month::January, 2032);
        let strike = 0.04;

        let mk = |engine: SharedMut<dyn PricingEngine>| {
            let swap = make_swap(
                curve.clone(),
                &settings,
                SwapType::Payer,
                strike,
                start,
                end,
            );
            let mut swaption = Swaption::new(
                swap,
                shared(EuropeanExercise::new(start)) as Shared<dyn Exercise>,
                SettlementType::Physical,
                SettlementMethod::PhysicalOTC,
                Shared::clone(&settings),
            );
            swaption.base_mut().set_pricing_engine(engine);
            swaption.npv().unwrap()
        };

        let analytic = mk(shared_mut(G2SwaptionEngine::new(
            SharedMut::clone(&g2),
            8.0,
            64,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>);
        let fd = mk(shared_mut(FdG2SwaptionEngine::with_params(
            g2,
            40,
            31,
            31,
            0,
            1e-4,
            FdmSchemeDesc::hundsdorfer(),
        )) as SharedMut<dyn PricingEngine>);

        assert!(analytic > 0.0 && fd > 0.0, "analytic={analytic} fd={fd}");
        assert!(
            (fd - analytic).abs() < 1.0,
            "fd={fd} analytic={analytic} diff={}",
            (fd - analytic).abs()
        );
    }

    #[test]
    fn bermudan_is_at_least_european() {
        let settings = settings();
        let curve = flat_curve();
        let g2 = model(curve.clone());
        let start = Date::new(15, Month::January, 2027);
        let mid = Date::new(15, Month::January, 2028);
        let end = Date::new(15, Month::January, 2032);
        let strike = 0.04;

        let european_swap = make_swap(
            curve.clone(),
            &settings,
            SwapType::Payer,
            strike,
            start,
            end,
        );
        let bermudan_swap = make_swap(curve, &settings, SwapType::Payer, strike, start, end);

        let fd = |model: SharedMut<G2>| {
            shared_mut(FdG2SwaptionEngine::with_params(
                model,
                24,
                21,
                21,
                0,
                1e-4,
                FdmSchemeDesc::hundsdorfer(),
            )) as SharedMut<dyn PricingEngine>
        };

        let mut european = Swaption::new(
            european_swap,
            shared(EuropeanExercise::new(start)) as Shared<dyn Exercise>,
            SettlementType::Physical,
            SettlementMethod::PhysicalOTC,
            Shared::clone(&settings),
        );
        european
            .base_mut()
            .set_pricing_engine(fd(SharedMut::clone(&g2)));

        let mut bermudan = Swaption::new(
            bermudan_swap,
            shared(BermudanExercise::new(vec![start, mid], false).unwrap()) as Shared<dyn Exercise>,
            SettlementType::Physical,
            SettlementMethod::PhysicalOTC,
            Shared::clone(&settings),
        );
        bermudan.base_mut().set_pricing_engine(fd(g2));

        let e = european.npv().unwrap();
        let b = bermudan.npv().unwrap();
        assert!(b + 1e-8 >= e, "bermudan={b} european={e}");
    }

    #[test]
    fn rejects_american_exercise() {
        let settings = settings();
        let curve = flat_curve();
        let g2 = model(curve.clone());
        let start = Date::new(15, Month::January, 2027);
        let end = Date::new(15, Month::January, 2032);
        let swap = make_swap(curve, &settings, SwapType::Payer, 0.05, start, end);
        let mut swaption = Swaption::new(
            swap,
            shared(crate::exercise::AmericanExercise::new(start, end, false).unwrap())
                as Shared<dyn Exercise>,
            SettlementType::Physical,
            SettlementMethod::PhysicalOTC,
            Shared::clone(&settings),
        );
        let engine = shared_mut(FdG2SwaptionEngine::with_params(
            g2,
            4,
            5,
            5,
            0,
            1e-4,
            FdmSchemeDesc::hundsdorfer(),
        )) as SharedMut<dyn PricingEngine>;
        swaption.base_mut().set_pricing_engine(engine);
        let err = swaption.npv().unwrap_err();
        assert!(err.message().contains("American"));
    }

    /// Port of `bermudanswaption.cpp` `testCachedG2Values` — FDM half only.
    ///
    /// Tree half deferred until `TreeSwaptionEngine` accepts G2 (`G2::tree` landed).
    /// Expecteds are the at-par coupon branch (`Settings::using_at_par_coupons`
    /// defaults to `true`).
    #[test]
    fn cached_g2_fdm_bermudan_values() {
        let today = Date::new(15, Month::September, 2016);
        let settlement = Date::new(19, Month::September, 2016);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        assert!(
            settings.using_at_par_coupons(),
            "oracle expects the at-par coupon branch"
        );

        let calendar = Target::new();
        let curve = Handle::new(shared(FlatForward::with_rate(
            settlement,
            0.04875825,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>);
        let index: Shared<IborIndex> =
            shared(Euribor::six_months(curve.clone(), Shared::clone(&settings)));

        let make_swap = |fixed_rate: Real| -> SharedMut<FixedVsFloatingSwap> {
            let start = calendar.advance_by_period(
                settlement,
                Period::new(1, TimeUnit::Years),
                BusinessDayConvention::Following,
                false,
            );
            let maturity = calendar.advance_by_period(
                start,
                Period::new(5, TimeUnit::Years),
                BusinessDayConvention::Following,
                false,
            );
            let fixed_schedule = MakeSchedule::new()
                .from(start)
                .to(maturity)
                .with_frequency(Frequency::Annual)
                .with_calendar(calendar.clone())
                .with_convention(BusinessDayConvention::Unadjusted)
                .with_termination_date_convention(BusinessDayConvention::Unadjusted)
                .forwards()
                .end_of_month(false)
                .build();
            let float_schedule = MakeSchedule::new()
                .from(start)
                .to(maturity)
                .with_frequency(Frequency::Semiannual)
                .with_calendar(calendar.clone())
                .with_convention(BusinessDayConvention::ModifiedFollowing)
                .with_termination_date_convention(BusinessDayConvention::ModifiedFollowing)
                .forwards()
                .end_of_month(false)
                .build();
            let floating_dc = index.day_counter().clone();
            shared_mut(
                VanillaSwap::new(
                    SwapType::Payer,
                    1000.0,
                    fixed_schedule,
                    fixed_rate,
                    Thirty360::with_convention(Convention::BondBasis),
                    float_schedule,
                    Shared::clone(&index),
                    0.0,
                    floating_dc,
                    None,
                    Shared::clone(&settings),
                )
                .unwrap()
                .into_fixed_vs_floating(),
            )
        };

        let discounting = shared_mut(DiscountingSwapEngine::new(
            curve.clone(),
            None,
            None,
            None,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;
        let atm_swap = make_swap(0.0);
        atm_swap
            .borrow_mut()
            .base_mut()
            .set_pricing_engine(SharedMut::clone(&discounting));
        let atm_rate = atm_swap.borrow_mut().fair_rate().unwrap();

        let moneyness = [0.5, 0.75, 1.0, 1.25, 1.5];
        let expected = [103.227, 54.6502, 20.0469, 5.26924, 1.07093];
        let tol = 0.005;

        let g2 = G2::new(curve, 0.1, 0.01, 0.2, 0.013, -0.5).unwrap();
        let engine = shared_mut(FdG2SwaptionEngine::with_params(
            g2,
            50,
            75,
            75,
            0,
            1e-3,
            FdmSchemeDesc::hundsdorfer(),
        )) as SharedMut<dyn PricingEngine>;

        for (i, &s) in moneyness.iter().enumerate() {
            let swap = make_swap(s * atm_rate);
            let exercise_dates: Vec<Date> = swap
                .borrow()
                .fixed_leg()
                .iter()
                .map(|flow| {
                    flow.as_coupon()
                        .expect("fixed leg carries coupons")
                        .accrual_start_date()
                })
                .collect();
            let mut swaption = Swaption::new(
                swap,
                shared(BermudanExercise::new(exercise_dates, false).unwrap())
                    as Shared<dyn Exercise>,
                SettlementType::Physical,
                SettlementMethod::PhysicalOTC,
                Shared::clone(&settings),
            );
            swaption
                .base_mut()
                .set_pricing_engine(SharedMut::clone(&engine));
            let got = swaption.npv().unwrap();
            assert!(
                (got - expected[i]).abs() <= tol,
                "moneyness {s}: got {got}, expected {}, |diff|={}",
                expected[i],
                (got - expected[i]).abs()
            );
        }
    }
}
