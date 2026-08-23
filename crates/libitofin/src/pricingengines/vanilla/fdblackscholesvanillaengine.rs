//! Finite-difference Black–Scholes engine for vanillas with cash dividends.
//!
//! Port of `ql/pricingengines/vanilla/fdblackscholesvanillaengine.{hpp,cpp}`
//! on the Spot cash-dividend model (the default) and the Escrowed model
//! ([`CashDividendModel`]). Quanto is [`with_quanto_helper`](FdBlackScholesVanillaEngine::with_quanto_helper)
//! (incompatible with the escrowed cash-dividend model). Local vol is the Dupire
//! branch of [`FdmBlackScholesSolver`]. American and Bermudan exercise use the
//! Spot model via [`FdmStepConditionComposite::vanilla_composite`]; escrowed
//! early exercise needs `FdmEscrowedLogInnerValueCalculator` and is rejected.

use crate::errors::QlResult;
use crate::exercise::{Exercise, ExerciseType};
use crate::fail;
use crate::instruments::{
    Greeks, MoreGreeks, OneAssetOptionEngine, OneAssetOptionResults, OptionArguments,
    StrikedTypePayoff,
};
use crate::methods::finitedifferences::meshers::{
    FdmMesher, FdmMesherComposite, fdm_black_scholes_mesher_with_quanto,
};
use crate::methods::finitedifferences::solvers::{
    FdmBlackScholesSolver, FdmSchemeDesc, FdmSolverDesc,
};
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{
    EscrowedDividendAdjustment, FdmInnerValueCalculator, FdmQuantoHelper, fdm_log_inner_value,
};
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::pricingengines::DividendSchedule;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size};
use crate::utilities::null::Null;

/// QuantLib `FdBlackScholesVanillaEngine::CashDividendModel`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CashDividendModel {
    /// Discrete cash drops as FD step conditions (C++ default).
    #[default]
    Spot,
    /// Remaining dividends escrowed out of the spot; no cash-drop steps.
    Escrowed,
}

/// Finite-difference Black–Scholes vanilla engine (European / American /
/// Bermudan + cash dividends on the Spot model).
pub struct FdBlackScholesVanillaEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    dividends: DividendSchedule,
    t_grid: Size,
    x_grid: Size,
    damping_steps: Size,
    scheme_desc: FdmSchemeDesc,
    local_vol: bool,
    illegal_local_vol_overwrite: Real,
    cash_dividend_model: CashDividendModel,
    quanto: Option<Shared<FdmQuantoHelper>>,
}

impl FdBlackScholesVanillaEngine {
    /// `FdBlackScholesVanillaEngine(process)` with QuantLib defaults:
    /// `tGrid = xGrid = 100`, no damping, Douglas, no dividends.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_params(process, Vec::new(), 100, 100, 0, FdmSchemeDesc::douglas())
    }

    /// `FdBlackScholesVanillaEngine(process, dividends)` with the same grid
    /// defaults.
    pub fn with_dividends(
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
    ) -> Self {
        Self::with_params(process, dividends, 100, 100, 0, FdmSchemeDesc::douglas())
    }

    /// Full constructor matching the C++ six-argument form (local-vol off,
    /// Spot cash-dividend model).
    pub fn with_params(
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
        t_grid: Size,
        x_grid: Size,
        damping_steps: Size,
        scheme_desc: FdmSchemeDesc,
    ) -> Self {
        Self::with_local_vol(
            process,
            dividends,
            t_grid,
            x_grid,
            damping_steps,
            scheme_desc,
            false,
            -Real::null(),
        )
    }

    /// As [`with_params`](Self::with_params), with the C++ `localVol` /
    /// `illegalLocalVolOverwrite` arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn with_local_vol(
        process: Shared<GeneralizedBlackScholesProcess>,
        dividends: DividendSchedule,
        t_grid: Size,
        x_grid: Size,
        damping_steps: Size,
        scheme_desc: FdmSchemeDesc,
        local_vol: bool,
        illegal_local_vol_overwrite: Real,
    ) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            dividends,
            t_grid,
            x_grid,
            damping_steps,
            scheme_desc,
            local_vol,
            illegal_local_vol_overwrite,
            cash_dividend_model: CashDividendModel::Spot,
            quanto: None,
        }
    }

    /// C++ `withQuantoHelper` / the quanto-helper constructors.
    pub fn with_quanto_helper(mut self, helper: Shared<FdmQuantoHelper>) -> Self {
        self.base.register_with(helper.observable());
        self.quanto = Some(helper);
        self
    }

    /// C++ `withCashDividendModel`.
    pub fn with_cash_dividend_model(mut self, model: CashDividendModel) -> Self {
        self.cash_dividend_model = model;
        self
    }

    /// Fills the arguments and returns the NPV.
    pub fn price(
        &mut self,
        payoff: Shared<dyn StrikedTypePayoff>,
        exercise: Shared<dyn Exercise>,
    ) -> QlResult<Real> {
        {
            let args = self.base.arguments_mut();
            args.payoff = Some(payoff);
            args.exercise = Some(exercise);
        }
        self.calculate()?;
        match self.base.results().instrument.value {
            Some(value) => Ok(value),
            None => fail!("no results returned"),
        }
    }
}

impl AsObservable for FdBlackScholesVanillaEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for FdBlackScholesVanillaEngine {
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
        let arguments = self.base.arguments();
        let Some(exercise) = arguments.exercise.as_ref() else {
            fail!("no exercise given");
        };
        let Some(payoff) = arguments.payoff.as_ref() else {
            fail!("no payoff given");
        };
        let strike = payoff.strike();
        require!(strike >= 0.0, "strike must be non-negative");

        let maturity = self.process.time(&exercise.last_date())?;
        let spot = self.process.x0()?;
        require!(spot > 0.0, "negative or null underlying given");

        let (dividend_schedule, spot_adjustment) = match self.cash_dividend_model {
            CashDividendModel::Spot => (self.dividends.clone(), 0.0),
            CashDividendModel::Escrowed => {
                require!(
                    exercise.exercise_type() == ExerciseType::European,
                    "Escrowed dividend model is not supported for American/Bermudan options"
                );
                require!(
                    self.quanto.is_none(),
                    "Escrowed dividend model is not supported for Quanto-Options"
                );
                let process = Shared::clone(&self.process);
                let escrowed = EscrowedDividendAdjustment::new(
                    self.dividends.clone(),
                    process.risk_free_rate(),
                    process.dividend_yield(),
                    move |d| process.time(&d),
                    maturity,
                );
                let settlement = self
                    .process
                    .risk_free_rate()
                    .current_link()?
                    .reference_date()?;
                let t_settlement = self.process.time(&settlement)?;
                let spot_adjustment = escrowed.dividend_adjustment(t_settlement)?;
                require!(
                    spot + spot_adjustment > 0.0,
                    "spot minus dividends becomes negative"
                );
                (Vec::new(), spot_adjustment)
            }
        };

        let equity = fdm_black_scholes_mesher_with_quanto(
            self.x_grid,
            &self.process,
            maturity,
            strike,
            None,
            None,
            0.0001,
            1.5,
            Some((strike, 0.1)),
            &dividend_schedule,
            self.quanto.as_deref(),
            spot_adjustment,
        )?;
        let mesher = shared(FdmMesherComposite::new(vec![equity]));
        let mesher_dyn: Shared<dyn FdmMesher> = mesher.clone() as Shared<dyn FdmMesher>;

        let payoff_dyn: Shared<dyn Payoff> = Shared::clone(payoff) as Shared<dyn Payoff>;
        let calculator: Shared<dyn FdmInnerValueCalculator> = shared(fdm_log_inner_value(
            payoff_dyn,
            Shared::clone(&mesher_dyn),
            0,
        ));

        let r_ts = self.process.risk_free_rate().current_link()?;
        let conditions = FdmStepConditionComposite::vanilla_composite(
            &dividend_schedule,
            &**exercise,
            Shared::clone(&mesher_dyn),
            Shared::clone(&calculator),
            r_ts.reference_date()?,
            r_ts.require_day_counter()?,
        )?;

        let solver_desc = FdmSolverDesc {
            mesher: mesher_dyn,
            bc_set: Vec::new(),
            condition: conditions,
            calculator,
            maturity,
            time_steps: self.t_grid,
            damping_steps: self.damping_steps,
        };
        let solver = FdmBlackScholesSolver::with_quanto(
            &self.process,
            strike,
            solver_desc,
            self.scheme_desc,
            self.local_vol,
            self.illegal_local_vol_overwrite,
            self.quanto.clone(),
        )?;
        let s = spot + spot_adjustment;
        let results = self.base.results_mut();
        results.instrument.value = Some(solver.value_at(s)?);
        results.greeks = Greeks {
            delta: Some(solver.delta_at(s)?),
            gamma: None,
            theta: Some(solver.theta_at(s)?),
            ..Greeks::default()
        };
        results.more_greeks = MoreGreeks::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::PlainVanillaPayoff;
    use crate::instruments::VanillaOption;
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared_mut;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;
    use crate::utilities::null::Null;

    fn process(
        today: Date,
        spot: Real,
        q: Real,
        r: Real,
        vol: Real,
    ) -> Shared<GeneralizedBlackScholesProcess> {
        let dc = Actual365Fixed::new();
        shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
            Handle::new(shared(FlatForward::with_rate(
                today,
                q,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(FlatForward::with_rate(
                today,
                r,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>),
            Handle::new(shared(BlackConstantVol::new(today, None, vol, dc))
                as Shared<dyn BlackVolTermStructure>),
        ))
    }

    #[test]
    fn no_dividend_european_is_close_to_analytic() {
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let process = process(today, 100.0, 0.0, 0.05, 0.20);
        let expiry = today + Period::new(1, TimeUnit::Years);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, 105.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));

        let mut analytic =
            VanillaOption::new(Shared::clone(&payoff), Shared::clone(&exercise), settings);
        analytic
            .base_mut()
            .set_pricing_engine(
                shared_mut(AnalyticEuropeanEngine::new(Shared::clone(&process)))
                    as crate::shared::SharedMut<dyn PricingEngine>,
            );
        let expected = analytic.npv().unwrap();

        let mut fd = FdBlackScholesVanillaEngine::with_params(
            process,
            Vec::new(),
            100,
            100,
            0,
            FdmSchemeDesc::douglas(),
        );
        let calculated = fd.price(payoff, exercise).unwrap();
        assert!(
            (calculated - expected).abs() < 0.05,
            "{calculated} vs {expected}"
        );
    }

    /// Dupire of a constant Black vol is that vol, so `localVol = true` must
    /// match the same analytic European within the existing FD band.
    #[test]
    fn local_vol_european_matches_constant_vol_analytic() {
        let today = Date::new(11, Month::February, 2018);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let process = process(today, 100.0, 0.0, 0.05, 0.20);
        let expiry = today + Period::new(1, TimeUnit::Years);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Put, 105.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));

        let mut analytic =
            VanillaOption::new(Shared::clone(&payoff), Shared::clone(&exercise), settings);
        analytic
            .base_mut()
            .set_pricing_engine(
                shared_mut(AnalyticEuropeanEngine::new(Shared::clone(&process)))
                    as crate::shared::SharedMut<dyn PricingEngine>,
            );
        let expected = analytic.npv().unwrap();

        let mut fd = FdBlackScholesVanillaEngine::with_local_vol(
            process,
            Vec::new(),
            100,
            100,
            0,
            FdmSchemeDesc::douglas(),
            true,
            -Real::null(),
        );
        let calculated = fd.price(payoff, exercise).unwrap();
        assert!(
            (calculated - expected).abs() < 0.05,
            "local-vol vanilla {calculated} vs analytic {expected}"
        );
    }

    fn analytic_npv(
        process: Shared<GeneralizedBlackScholesProcess>,
        payoff: Shared<dyn StrikedTypePayoff>,
        exercise: Shared<dyn Exercise>,
        settings: Shared<Settings<Date>>,
    ) -> Real {
        let mut option = VanillaOption::new(payoff, exercise, settings);
        option
            .base_mut()
            .set_pricing_engine(shared_mut(AnalyticEuropeanEngine::new(process))
                as crate::shared::SharedMut<dyn PricingEngine>);
        option.npv().unwrap()
    }

    /// `dividendoption.cpp` `testFdEuropeanDegenerate` (Escrowed): empty and
    /// zero-amount dividend schedules must not move the NPV.
    #[test]
    fn escrowed_degenerate_dividends_leave_the_npv_unchanged() {
        let today = Date::new(27, Month::February, 2005);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let expiry = Date::new(13, Month::April, 2005);
        let process = {
            let dc = crate::time::daycounters::actual360::Actual360::new();
            shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(54.625)) as Shared<dyn Quote>),
                Handle::new(shared(FlatForward::with_rate(
                    today,
                    0.0,
                    dc.clone(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>),
                Handle::new(shared(FlatForward::with_rate(
                    today,
                    0.052706,
                    dc.clone(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>),
                Handle::new(shared(BlackConstantVol::new(today, None, 0.282922, dc))
                    as Shared<dyn BlackVolTermStructure>),
            ))
        };
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, 55.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));

        let ref_npv = FdBlackScholesVanillaEngine::with_params(
            Shared::clone(&process),
            Vec::new(),
            100,
            300,
            0,
            FdmSchemeDesc::douglas(),
        )
        .with_cash_dividend_model(CashDividendModel::Escrowed)
        .price(Shared::clone(&payoff), Shared::clone(&exercise))
        .unwrap();

        let empty = FdBlackScholesVanillaEngine::with_params(
            Shared::clone(&process),
            Vec::new(),
            100,
            300,
            0,
            FdmSchemeDesc::douglas(),
        )
        .with_cash_dividend_model(CashDividendModel::Escrowed)
        .price(Shared::clone(&payoff), Shared::clone(&exercise))
        .unwrap();
        assert!(
            (empty - ref_npv).abs() < 1e-6,
            "empty dividends {empty} vs {ref_npv}"
        );

        let zeros: DividendSchedule = (1..=6)
            .map(|i| {
                shared(crate::cashflows::FixedDividend::new(0.0, today + i))
                    as Shared<dyn crate::cashflows::Dividend>
            })
            .collect();
        let zero_npv = FdBlackScholesVanillaEngine::with_params(
            process,
            zeros,
            100,
            300,
            0,
            FdmSchemeDesc::douglas(),
        )
        .with_cash_dividend_model(CashDividendModel::Escrowed)
        .price(payoff, exercise)
        .unwrap();
        assert!(
            (zero_npv - ref_npv).abs() < 1e-6,
            "zero dividends {zero_npv} vs {ref_npv}"
        );
    }

    /// `dividendoption.cpp` `testEscrowedDividendModel`: FD escrowed vs Black
    /// on `S + dividendAdjustment(0)` at the C++ 0.001 tolerance (200×400).
    #[test]
    fn escrowed_european_matches_black_on_the_prepaid_spot() {
        let today = Date::new(11, Month::November, 2025);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let expiry = today + Period::new(18, TimeUnit::Months);
        let (spot, q, r, vol) = (100.0, 0.05, 0.025, 0.30);
        let bsm = process(today, spot, q, r, vol);
        let amount = 5.0;

        let dates = [
            today - Period::new(1, TimeUnit::Days),
            today + Period::new(6, TimeUnit::Months),
            expiry,
            expiry + Period::new(1, TimeUnit::Days),
        ];
        for div_date in dates {
            let dividends = vec![
                shared(crate::cashflows::FixedDividend::new(amount, div_date))
                    as Shared<dyn crate::cashflows::Dividend>,
            ];
            let process_for_time = Shared::clone(&bsm);
            let maturity = bsm.time(&expiry).unwrap();
            let adj = EscrowedDividendAdjustment::new(
                dividends.clone(),
                bsm.risk_free_rate(),
                bsm.dividend_yield(),
                move |d| process_for_time.time(&d),
                maturity,
            );
            let s_star = spot + adj.dividend_adjustment(0.0).unwrap();
            assert!(s_star > 0.0, "prepaid spot must stay positive");
            let prepaid = process(today, s_star, q, r, vol);

            for option_type in [OptionType::Call, OptionType::Put] {
                let payoff: Shared<dyn StrikedTypePayoff> =
                    shared(PlainVanillaPayoff::new(option_type, 95.0));
                let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));
                let expected = analytic_npv(
                    Shared::clone(&prepaid),
                    Shared::clone(&payoff),
                    Shared::clone(&exercise),
                    Shared::clone(&settings),
                );
                let calculated = FdBlackScholesVanillaEngine::with_params(
                    Shared::clone(&bsm),
                    dividends.clone(),
                    200,
                    400,
                    0,
                    FdmSchemeDesc::douglas(),
                )
                .with_cash_dividend_model(CashDividendModel::Escrowed)
                .price(payoff, exercise)
                .unwrap();
                assert!(
                    (calculated - expected).abs() < 0.001,
                    "{option_type:?} div={div_date}: fd={calculated} black={expected}"
                );
            }
        }
    }

    #[test]
    fn escrowed_rejects_dividends_that_wipe_out_the_spot() {
        let today = Date::new(11, Month::November, 2025);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let expiry = today + Period::new(1, TimeUnit::Years);
        let process = process(today, 100.0, 0.0, 0.0, 0.20);
        let dividends = vec![shared(crate::cashflows::FixedDividend::new(
            150.0,
            today + Period::new(1, TimeUnit::Months),
        )) as Shared<dyn crate::cashflows::Dividend>];
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, 100.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));
        let err = FdBlackScholesVanillaEngine::with_params(
            process,
            dividends,
            20,
            20,
            0,
            FdmSchemeDesc::douglas(),
        )
        .with_cash_dividend_model(CashDividendModel::Escrowed)
        .price(payoff, exercise)
        .unwrap_err();
        assert_eq!(err.message(), "spot minus dividends becomes negative");
    }

    fn quanto_helper(
        today: Date,
        r_d: Real,
        r_f: Real,
        fx_vol: Real,
        rho: Real,
        dc: crate::time::daycounter::DayCounter,
    ) -> Shared<FdmQuantoHelper> {
        shared(FdmQuantoHelper::new(
            shared(FlatForward::with_rate(
                today,
                r_d,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
            shared(FlatForward::with_rate(
                today,
                r_f,
                dc.clone(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
            shared(BlackConstantVol::new(today, None, fx_vol, dc))
                as Shared<dyn BlackVolTermStructure>,
            rho,
            1.0,
        ))
    }

    /// `quantooption.cpp` `testPDEOptionValues`: FD quanto vs Black with
    /// `q + (r_d − r_f + ρ σ σ_fx)`.
    #[test]
    fn pde_quanto_values_track_adjusted_black() {
        use crate::time::daycounters::actual360::Actual360;

        let dc = Actual360::new();
        let today = Date::new(21, Month::April, 2019);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);

        // type, strike, spot, q, r, t, vol, fxr, fxv, corr
        let cases = [
            (
                OptionType::Call,
                105.0,
                100.0,
                0.04,
                0.08,
                0.5,
                0.2,
                0.05,
                0.10,
                0.3,
            ),
            (
                OptionType::Call,
                100.0,
                100.0,
                0.16,
                0.08,
                0.25,
                0.15,
                0.05,
                0.20,
                -0.3,
            ),
            (
                OptionType::Put,
                105.0,
                100.0,
                0.04,
                0.08,
                0.5,
                0.2,
                0.05,
                0.10,
                0.3,
            ),
            (
                OptionType::Call,
                0.0,
                100.0,
                0.04,
                0.08,
                0.3,
                0.3,
                0.05,
                0.10,
                0.75,
            ),
        ];

        for (ty, strike, spot, q, r, t, vol, fxr, fxv, corr) in cases {
            let process = shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
                Handle::new(shared(FlatForward::with_rate(
                    today,
                    q,
                    dc.clone(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>),
                Handle::new(shared(FlatForward::with_rate(
                    today,
                    r,
                    dc.clone(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>),
                Handle::new(shared(BlackConstantVol::new(today, None, vol, dc.clone()))
                    as Shared<dyn BlackVolTermStructure>),
            ));
            let helper = quanto_helper(today, r, fxr, fxv, corr, dc.clone());
            let days = (t * 360.0 + 0.5) as i32;
            let expiry = today + days;
            let payoff: Shared<dyn StrikedTypePayoff> = shared(PlainVanillaPayoff::new(ty, strike));
            let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));

            let mut fd_opt = VanillaOption::new(
                Shared::clone(&payoff),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            );
            let t_grid = (t * 200.0) as Size;
            fd_opt.base_mut().set_pricing_engine(shared_mut(
                FdBlackScholesVanillaEngine::with_params(
                    Shared::clone(&process),
                    Vec::new(),
                    t_grid,
                    500,
                    1,
                    FdmSchemeDesc::douglas(),
                )
                .with_quanto_helper(helper),
            )
                as crate::shared::SharedMut<dyn PricingEngine>);

            let q_adj = q + (r - fxr + corr * vol * fxv);
            let adj_process = shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>),
                Handle::new(shared(FlatForward::with_rate(
                    today,
                    q_adj,
                    dc.clone(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>),
                Handle::new(shared(FlatForward::with_rate(
                    today,
                    r,
                    dc.clone(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>),
                Handle::new(shared(BlackConstantVol::new(today, None, vol, dc.clone()))
                    as Shared<dyn BlackVolTermStructure>),
            ));
            let mut analytic = VanillaOption::new(payoff, exercise, Shared::clone(&settings));
            analytic
                .base_mut()
                .set_pricing_engine(shared_mut(AnalyticEuropeanEngine::new(adj_process))
                    as crate::shared::SharedMut<dyn PricingEngine>);

            let fd_npv = fd_opt.npv().unwrap();
            let an_npv = analytic.npv().unwrap();
            assert!(
                (fd_npv - an_npv).abs() < 2e-4,
                "{ty:?} K={strike} T={t}: fd={fd_npv} analytic={an_npv}"
            );
            let fd_delta = fd_opt.delta().unwrap();
            let an_delta = analytic.delta().unwrap();
            assert!(
                (fd_delta - an_delta).abs() < 1e-4,
                "{ty:?} K={strike} T={t}: fd delta={fd_delta} analytic={an_delta}"
            );
        }
    }

    #[test]
    fn escrowed_quanto_is_rejected() {
        let today = Date::new(21, Month::April, 2019);
        let process = process(today, 100.0, 0.04, 0.08, 0.20);
        let helper = quanto_helper(today, 0.08, 0.05, 0.10, 0.3, Actual365Fixed::new());
        let expiry = today + Period::new(6, TimeUnit::Months);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, 105.0));
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));
        let err = FdBlackScholesVanillaEngine::with_params(
            process,
            Vec::new(),
            10,
            10,
            0,
            FdmSchemeDesc::douglas(),
        )
        .with_quanto_helper(helper)
        .with_cash_dividend_model(CashDividendModel::Escrowed)
        .price(payoff, exercise)
        .unwrap_err();
        assert!(err.message().contains("Escrowed dividend model"));
    }

    /// `quantooption.cpp` `testAmericanQuantoOption`: cached American quanto
    /// with one cash dividend, Douglas 100×400 + 1 damping step.
    #[test]
    fn american_quanto_matches_cached_npv() {
        let dc = Actual365Fixed::new();
        let today = Date::new(21, Month::April, 2019);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let maturity = today + Period::new(9, TimeUnit::Months);
        let (spot, q, r, vol) = (100.0, 0.03, 0.025, 0.30);
        let process = process(today, spot, q, r, vol);
        let helper = quanto_helper(today, r, 0.075, 0.15, -0.75, dc);
        let dividends = vec![shared(crate::cashflows::FixedDividend::new(
            8.0,
            today + Period::new(6, TimeUnit::Months),
        )) as Shared<dyn crate::cashflows::Dividend>];
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, 105.0));
        let exercise: Shared<dyn Exercise> = shared(AmericanExercise::from_latest(maturity, false));

        let expected = 8.90611734;
        let price = |local_vol: bool| {
            let mut option = VanillaOption::new(
                Shared::clone(&payoff),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            );
            option.base_mut().set_pricing_engine(shared_mut(
                FdBlackScholesVanillaEngine::with_local_vol(
                    Shared::clone(&process),
                    dividends.clone(),
                    100,
                    400,
                    1,
                    FdmSchemeDesc::douglas(),
                    local_vol,
                    -Real::null(),
                )
                .with_quanto_helper(Shared::clone(&helper)),
            )
                as crate::shared::SharedMut<dyn PricingEngine>);
            option.npv().unwrap()
        };

        let bs = price(false);
        assert!(
            (bs - expected).abs() < 1e-4,
            "Black-Scholes American quanto {bs} vs {expected}"
        );
        let local = price(true);
        assert!(
            (local - expected).abs() < 1e-4,
            "local-vol American quanto {local} vs {expected}"
        );
        assert!(
            (bs - local).abs() < 1e-6,
            "BS vs local-vol American quanto {bs} vs {local}"
        );
    }

    #[test]
    fn escrowed_american_is_rejected() {
        let today = Date::new(21, Month::April, 2019);
        let process = process(today, 100.0, 0.03, 0.025, 0.30);
        let expiry = today + Period::new(9, TimeUnit::Months);
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, 105.0));
        let exercise: Shared<dyn Exercise> = shared(AmericanExercise::from_latest(expiry, false));
        let err = FdBlackScholesVanillaEngine::with_params(
            process,
            Vec::new(),
            10,
            10,
            0,
            FdmSchemeDesc::douglas(),
        )
        .with_cash_dividend_model(CashDividendModel::Escrowed)
        .price(payoff, exercise)
        .unwrap_err();
        assert!(err.message().contains("American/Bermudan"));
    }
}
