//! Finite-difference Black–Scholes engine for vanillas with cash dividends.
//!
//! Port of `ql/pricingengines/vanilla/fdblackscholesvanillaengine.{hpp,cpp}`
//! on the Spot cash-dividend model (the default) and the Escrowed model
//! ([`CashDividendModel`]). Quanto is [`with_quanto_helper`](FdBlackScholesVanillaEngine::with_quanto_helper)
//! (incompatible with the escrowed cash-dividend model). Local vol is the Dupire
//! branch of [`FdmBlackScholesSolver`]. American and Bermudan exercise use
//! [`FdmStepConditionComposite::vanilla_composite`]; the escrowed model prices
//! early exercise through [`FdmEscrowedLogInnerValueCalculator`] and plants
//! zero-amount dividend dates as stopping times (`cpp:126-130`).

use crate::cashflows::{Dividend, FixedDividend};
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
    EscrowedDividendAdjustment, FdmEscrowedLogInnerValueCalculator, FdmInnerValueCalculator,
    FdmQuantoHelper, fdm_log_inner_value,
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
/// Bermudan + cash dividends on the Spot and Escrowed models).
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

        let (dividend_schedule, spot_adjustment, escrowed) = match self.cash_dividend_model {
            CashDividendModel::Spot => (self.dividends.clone(), 0.0, None),
            CashDividendModel::Escrowed => {
                require!(
                    self.quanto.is_none(),
                    "Escrowed dividend model is not supported for Quanto-Options"
                );
                let process = Shared::clone(&self.process);
                let escrowed = shared(EscrowedDividendAdjustment::new(
                    self.dividends.clone(),
                    process.risk_free_rate(),
                    process.dividend_yield(),
                    move |d| process.time(&d),
                    maturity,
                ));
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
                // American / Bermudan: zero-amount dates so the time grid stops
                // when remaining escrowed cash drops (`cpp:126-130`).
                let dividend_schedule = if exercise.exercise_type() == ExerciseType::European {
                    Vec::new()
                } else {
                    self.dividends
                        .iter()
                        .map(|cf| {
                            shared(FixedDividend::new(0.0, cf.date())) as Shared<dyn Dividend>
                        })
                        .collect()
                };
                (dividend_schedule, spot_adjustment, Some(escrowed))
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
        let calculator: Shared<dyn FdmInnerValueCalculator> = match escrowed {
            Some(escrowed) => shared(FdmEscrowedLogInnerValueCalculator::new(
                escrowed,
                payoff_dyn,
                Shared::clone(&mesher_dyn),
                0,
            )),
            None => shared(fdm_log_inner_value(
                payoff_dyn,
                Shared::clone(&mesher_dyn),
                0,
            )),
        };

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
            gamma: Some(solver.gamma_at(s)?),
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
    use crate::exercise::{AmericanExercise, BermudanExercise, EuropeanExercise};
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

        // A dividend exactly on expiry stays in `dividendAdjustment(T)`, so
        // `FdmEscrowedLogInnerValueCalculator` pays off `S* + D` and is not
        // the prepaid-Black identity (`cpp:41-44`). C++'s
        // `testEscrowedDividendModel` uses 3M/9M on a 1Y option.
        let dates = [
            today - Period::new(1, TimeUnit::Days),
            today + Period::new(6, TimeUnit::Months),
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

    /// `americanoption.cpp` `testEscrowedVsSpotAmericanOption`: escrowed
    /// American call vs spot model with vol scaled by `S / (S − D)` @ 1e-2.
    #[test]
    fn escrowed_american_matches_spot_model() {
        use crate::time::daycounters::actual360::Actual360;

        let today = Date::new(27, Month::February, 2021);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let maturity = today + Period::new(12, TimeUnit::Months);
        let dc = Actual360::new();
        let make_process = |vol: Real| {
            shared(BlackScholesMertonProcess::new(
                Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
                Handle::new(shared(FlatForward::with_rate(
                    today,
                    0.08,
                    dc.clone(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>),
                Handle::new(shared(FlatForward::with_rate(
                    today,
                    0.04,
                    dc.clone(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn YieldTermStructure>),
                Handle::new(shared(BlackConstantVol::new(today, None, vol, dc.clone()))
                    as Shared<dyn BlackVolTermStructure>),
            ))
        };
        let dividends = vec![shared(crate::cashflows::FixedDividend::new(
            10.0,
            today + Period::new(10, TimeUnit::Months),
        )) as Shared<dyn crate::cashflows::Dividend>];
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, 100.0));
        let exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, maturity, false).unwrap());

        let price = |vol: Real, model: CashDividendModel| {
            let mut option = VanillaOption::new(
                Shared::clone(&payoff),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            );
            option.base_mut().set_pricing_engine(shared_mut(
                FdBlackScholesVanillaEngine::with_params(
                    make_process(vol),
                    dividends.clone(),
                    100,
                    400,
                    0,
                    FdmSchemeDesc::douglas(),
                )
                .with_cash_dividend_model(model),
            )
                as crate::shared::SharedMut<dyn PricingEngine>);
            (option.npv().unwrap(), option.delta().unwrap())
        };

        let (spot_npv, spot_delta) = price(0.3, CashDividendModel::Spot);
        let (esc_npv, esc_delta) = price(100.0 / 90.0 * 0.3, CashDividendModel::Escrowed);
        assert!(
            (esc_npv - spot_npv).abs() < 1e-2,
            "American escrowed NPV {esc_npv} vs spot {spot_npv}"
        );
        assert!(
            (esc_delta - spot_delta).abs() < 1e-2,
            "American escrowed delta {esc_delta} vs spot {spot_delta}"
        );
    }

    /// `dividendoption.cpp` `testFdAmericanDegenerate` (Escrowed): empty and
    /// zero-amount dividend schedules must not move the American NPV.
    #[test]
    fn escrowed_american_degenerate_dividends_leave_the_npv_unchanged() {
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
        let exercise: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, expiry, false).unwrap());

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

    /// Escrowed Bermudan is the same wiring as American (zero-amount stopping
    /// times + escrowed inner value); it must price and sit above European.
    #[test]
    fn escrowed_bermudan_is_at_least_european() {
        let today = Date::new(27, Month::February, 2021);
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today);
        let expiry = today + Period::new(12, TimeUnit::Months);
        let process = process(today, 100.0, 0.08, 0.04, 0.30);
        let dividends = vec![shared(crate::cashflows::FixedDividend::new(
            10.0,
            today + Period::new(10, TimeUnit::Months),
        )) as Shared<dyn crate::cashflows::Dividend>];
        let payoff: Shared<dyn StrikedTypePayoff> =
            shared(PlainVanillaPayoff::new(OptionType::Call, 100.0));
        let european: Shared<dyn Exercise> = shared(EuropeanExercise::new(expiry));
        let bermudan: Shared<dyn Exercise> = shared(
            BermudanExercise::new(
                vec![
                    today + Period::new(3, TimeUnit::Months),
                    today + Period::new(6, TimeUnit::Months),
                    today + Period::new(9, TimeUnit::Months),
                    expiry,
                ],
                false,
            )
            .unwrap(),
        );

        let euro = FdBlackScholesVanillaEngine::with_params(
            Shared::clone(&process),
            dividends.clone(),
            50,
            100,
            0,
            FdmSchemeDesc::douglas(),
        )
        .with_cash_dividend_model(CashDividendModel::Escrowed)
        .price(Shared::clone(&payoff), european)
        .unwrap();
        let berm = FdBlackScholesVanillaEngine::with_params(
            process,
            dividends,
            50,
            100,
            0,
            FdmSchemeDesc::douglas(),
        )
        .with_cash_dividend_model(CashDividendModel::Escrowed)
        .price(payoff, bermudan)
        .unwrap();
        assert!(
            berm + 1e-12 >= euro,
            "escrowed Bermudan {berm} should be ≥ European {euro}"
        );
    }
}

#[cfg(test)]
mod test_fd_engines {
    //! The `testFdEngines` oracle of `test-suite/europeanoption.cpp:1241-1256`,
    //! which runs the shared `testEngineConsistency` harness (`:192-278`) with
    //! the finite-difference engine on a 500 by 500 grid and checks it against
    //! the analytic engine over the full market sweep.

    use std::time::Instant;

    use super::super::test_market::{market, today};
    use super::FdBlackScholesVanillaEngine;
    use crate::instrument::Instrument;
    use crate::instruments::{EuropeanOption, PlainVanillaPayoff};
    use crate::methods::finitedifferences::solvers::FdmSchemeDesc;
    use crate::option::OptionType::{self, Call, Put};
    use crate::pricingengine::PricingEngine;
    use crate::shared::{Shared, SharedMut, shared, shared_mut};
    use crate::time::date::Date;
    use crate::types::{Rate, Real, Size, Volatility};

    /// `timeSteps` and `gridPoints` of `europeanoption.cpp:1246-1247`.
    const T_GRID: Size = 500;
    const X_GRID: Size = 500;

    const UNDERLYING: Real = 100.0;

    /// `relativeTol` of `europeanoption.cpp:1249-1252`.
    const VALUE_TOLERANCE: Real = 1.0e-4;
    const DELTA_TOLERANCE: Real = 1.0e-6;
    const GAMMA_TOLERANCE: Real = 1.0e-6;
    const THETA_TOLERANCE: Real = 1.0e-3;

    fn relative_error(x1: Real, x2: Real, reference: Real) -> Real {
        if reference != 0.0 {
            (x1 - x2).abs() / reference
        } else {
            (x1 - x2).abs()
        }
    }

    fn fd_option(
        market: &super::super::test_market::Market,
        option_type: OptionType,
        strike: Real,
        expiry: Date,
    ) -> EuropeanOption {
        let payoff = shared(PlainVanillaPayoff::new(option_type, strike));
        let exercise = shared(crate::exercise::EuropeanExercise::new(expiry));
        let mut option = EuropeanOption::new(payoff, exercise, Shared::clone(&market.settings));
        let engine = shared_mut(FdBlackScholesVanillaEngine::with_params(
            Shared::clone(&market.process),
            Vec::new(),
            T_GRID,
            X_GRID,
            0,
            FdmSchemeDesc::douglas(),
        ));
        option
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        option
    }

    /// Builds both options once per (type, strike) and mutates shared quotes,
    /// so repricings run through the observer chain.
    #[test]
    fn fd_engine_matches_the_analytic_engine_over_the_market_sweep() {
        let started = Instant::now();
        let market = market();
        let expiry = today() + 360;

        let q_rates: [Rate; 2] = [0.00, 0.05];
        let r_rates: [Rate; 3] = [0.01, 0.05, 0.15];
        let vols: [Volatility; 3] = [0.11, 0.50, 1.20];

        let mut worst = [
            ("value", 0.0),
            ("delta", 0.0),
            ("gamma", 0.0),
            ("theta", 0.0),
        ];

        for option_type in [Call, Put] {
            for strike in [75.0, 100.0, 125.0] {
                let mut reference = market.option(option_type, strike, expiry);
                let mut option = fd_option(&market, option_type, strike, expiry);

                for q in q_rates {
                    for r in r_rates {
                        for vol in vols {
                            market.set(UNDERLYING, q, r, vol);

                            let value = option.npv().unwrap();
                            let mut checks =
                                vec![("value", reference.npv().unwrap(), value, VALUE_TOLERANCE)];
                            if value > UNDERLYING * 1.0e-5 {
                                checks.push((
                                    "delta",
                                    reference.delta().unwrap(),
                                    option.delta().unwrap(),
                                    DELTA_TOLERANCE,
                                ));
                                checks.push((
                                    "gamma",
                                    reference.gamma().unwrap(),
                                    option.gamma().unwrap(),
                                    GAMMA_TOLERANCE,
                                ));
                                checks.push((
                                    "theta",
                                    reference.theta().unwrap(),
                                    option.theta().unwrap(),
                                    THETA_TOLERANCE,
                                ));
                            }

                            for (name, expected, calculated, tolerance) in checks {
                                let error = relative_error(expected, calculated, UNDERLYING);
                                assert!(
                                    error <= tolerance,
                                    "{name} of {option_type:?} K={strike} q={q} r={r} v={vol}: \
                                     analytic {expected} vs finite difference {calculated} \
                                     (relative error {error} over {tolerance})"
                                );
                                for slot in &mut worst {
                                    if slot.0 == name && error > slot.1 {
                                        slot.1 = error;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        println!(
            "testFdEngines: {:?} for 108 combinations; worst relative errors {worst:?}",
            started.elapsed()
        );
    }
}

#[cfg(test)]
mod test_fd_values {
    //! American options on a 100 by 400 grid against the Ju-1998 table within
    //! 8e-2 (`test-suite/americanoption.cpp:375-427`).

    use super::super::test_market::{Market, market, time_to_days, today};
    use super::FdBlackScholesVanillaEngine;
    use crate::exercise::{AmericanExercise, Exercise};
    use crate::instrument::Instrument;
    use crate::instruments::{OneAssetOption, PlainVanillaPayoff};
    use crate::methods::finitedifferences::solvers::FdmSchemeDesc;
    use crate::option::OptionType::{self, Call, Put};
    use crate::pricingengine::PricingEngine;
    use crate::shared::{Shared, SharedMut, shared, shared_mut};
    use crate::types::{Rate, Real, Size, Time, Volatility};

    const T_GRID: Size = 100;
    const X_GRID: Size = 400;
    const TOLERANCE: Real = 8.0e-2;

    struct JuValue {
        option_type: OptionType,
        strike: Real,
        spot: Real,
        q: Rate,
        r: Rate,
        t: Time,
        vol: Volatility,
        expected: Real,
    }

    fn price(market: &Market, ju: &JuValue) -> Real {
        market.set(ju.spot, ju.q, ju.r, ju.vol);

        let exercise = AmericanExercise::over(today(), today() + time_to_days(ju.t)).unwrap();
        let mut option = OneAssetOption::new(
            shared(PlainVanillaPayoff::new(ju.option_type, ju.strike)),
            shared(exercise) as Shared<dyn Exercise>,
            Shared::clone(&market.settings),
        );
        let engine = shared_mut(FdBlackScholesVanillaEngine::with_params(
            Shared::clone(&market.process),
            Vec::new(),
            T_GRID,
            X_GRID,
            0,
            FdmSchemeDesc::douglas(),
        ));
        option
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        option.npv().unwrap()
    }

    #[test]
    fn american_options_reproduce_the_ju_values() {
        let market = market();
        let rows = [
            JuValue {
                option_type: Put,
                strike: 40.0,
                spot: 40.0,
                q: 0.0,
                r: 0.0488,
                t: 0.3333,
                vol: 0.2,
                expected: 1.576,
            },
            JuValue {
                option_type: Put,
                strike: 45.0,
                spot: 40.0,
                q: 0.0,
                r: 0.0488,
                t: 0.5833,
                vol: 0.2,
                expected: 5.260,
            },
            JuValue {
                option_type: Call,
                strike: 100.0,
                spot: 100.0,
                q: 0.07,
                r: 0.03,
                t: 3.0,
                vol: 0.2,
                expected: 9.065,
            },
            JuValue {
                option_type: Call,
                strike: 100.0,
                spot: 120.0,
                q: 0.07,
                r: 0.03,
                t: 3.0,
                vol: 0.2,
                expected: 21.398,
            },
        ];

        for ju in &rows {
            let calculated = price(&market, ju);
            let error = (calculated - ju.expected).abs();
            println!(
                "testFdValues: {:?} K={} S={} t={}: Ju {} finite difference {calculated}",
                ju.option_type, ju.strike, ju.spot, ju.t, ju.expected
            );
            assert!(
                error <= TOLERANCE,
                "{:?} K={} S={} q={} r={} t={} v={}: Ju {} vs finite difference {calculated} \
                 (absolute error {error} over {TOLERANCE})",
                ju.option_type,
                ju.strike,
                ju.spot,
                ju.q,
                ju.r,
                ju.t,
                ju.vol,
                ju.expected
            );
        }
    }
}

#[cfg(test)]
mod test_fd_earliest_exercise_date {
    //! The `testFdEarliestExerciseDate` oracle of
    //! `test-suite/americanoption.cpp:2173-2255`: a deep in-the-money put whose
    //! exercise window is narrowed from the front.
    //!
    //! This is the only oracle that drives a non-zero `exercise_start`.
    //! `testFdValues` opens every window at the reference date, so the
    //! early-return of `FdmAmericanStepCondition::apply_to`
    //! (`fdmamericanstepcondition.cpp:37-38`) never fires there and an engine
    //! ignoring the earliest exercise date would pass it.

    use super::super::AnalyticEuropeanEngine;
    use super::FdBlackScholesVanillaEngine;
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{OneAssetOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::methods::finitedifferences::solvers::FdmSchemeDesc;
    use crate::option::OptionType::Put;
    use crate::pricingengine::PricingEngine;
    use crate::processes::{BlackScholesMertonProcess, GeneralizedBlackScholesProcess};
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{Shared, SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;
    use crate::types::{Rate, Real, Size, Volatility};

    /// The market of `:2186-2195`.
    pub(super) const S0: Real = 80.0;
    pub(super) const STRIKE: Real = 100.0;
    const SIGMA: Volatility = 0.25;
    const R: Rate = 0.05;
    const Q: Rate = 0.0;

    /// The grid of `:2206`.
    pub(super) const T_GRID: Size = 200;
    pub(super) const X_GRID: Size = 200;

    pub(super) fn today() -> Date {
        Date::new(15, Month::January, 2025)
    }

    fn flat_rate(rate: Rate) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    pub(super) fn process() -> Shared<GeneralizedBlackScholesProcess> {
        let spot = Handle::new(shared(SimpleQuote::new(S0)) as Shared<dyn Quote>);
        let vol = Handle::new(shared(BlackConstantVol::new(
            today(),
            None,
            SIGMA,
            Actual365Fixed::new(),
        )) as Shared<dyn BlackVolTermStructure>);
        shared(BlackScholesMertonProcess::new(
            spot,
            flat_rate(Q),
            flat_rate(R),
            vol,
        ))
    }

    /// The American put over `[earliest, maturity]`, priced by the
    /// finite-difference engine (`:2203-2208`).
    fn american_price(
        settings: &Shared<Settings<Date>>,
        process: &Shared<GeneralizedBlackScholesProcess>,
        earliest: Date,
        maturity: Date,
    ) -> Real {
        let exercise = AmericanExercise::over(earliest, maturity).unwrap();
        let mut option = OneAssetOption::new(
            shared(PlainVanillaPayoff::new(Put, STRIKE)),
            shared(exercise) as Shared<dyn Exercise>,
            Shared::clone(settings),
        );
        let engine = shared_mut(FdBlackScholesVanillaEngine::with_params(
            Shared::clone(process),
            Vec::new(),
            T_GRID,
            X_GRID,
            0,
            FdmSchemeDesc::douglas(),
        ));
        option
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        option.npv().unwrap()
    }

    #[test]
    fn narrowing_the_exercise_window_lowers_the_price_toward_the_european_one() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let process = process();
        let maturity = today() + Period::new(1, TimeUnit::Years);

        let full = american_price(&settings, &process, today(), maturity);
        let mid = american_price(
            &settings,
            &process,
            maturity - Period::new(6, TimeUnit::Months),
            maturity,
        );
        let late = american_price(
            &settings,
            &process,
            maturity - Period::new(3, TimeUnit::Months),
            maturity,
        );

        let mut european = OneAssetOption::new(
            shared(PlainVanillaPayoff::new(Put, STRIKE)),
            shared(EuropeanExercise::new(maturity)) as Shared<dyn Exercise>,
            Shared::clone(&settings),
        );
        let analytic = shared_mut(AnalyticEuropeanEngine::new(Shared::clone(&process)));
        european
            .base_mut()
            .set_pricing_engine(analytic as SharedMut<dyn PricingEngine>);
        let euro = european.npv().unwrap();

        println!("testFdEarliestExerciseDate: full {full} 6M {mid} 3M {late} european {euro}");

        assert!(
            full - euro > 1.0,
            "the early-exercise premium should be significant: full {full} european {euro}"
        );
        assert!(
            full - late > 0.01,
            "restricting the exercise window should reduce the price: full {full} late {late}"
        );
        assert!(
            late > euro + 0.01,
            "the restricted American should exceed the European: late {late} european {euro}"
        );
        assert!(
            mid >= late - 1e-8,
            "a wider window should give a higher price: 6M {mid} 3M {late}"
        );
        assert!(
            full >= mid - 1e-8,
            "the full window should give the highest price: full {full} 6M {mid}"
        );
    }
}

#[cfg(test)]
mod test_fd_bermudan {
    //! A degeneracy oracle for the Bermudan branch, not a port: the C++ test
    //! suite prices no Bermudan option through this engine, so there is no
    //! upstream number to reproduce and inventing one would pin nothing.
    //!
    //! It runs on the deep in-the-money put of
    //! `test-suite/americanoption.cpp:2186-2195`, whose early-exercise premium
    //! is around two, and prices every arm through this same engine on the same
    //! grid so the arms differ only in the exercise.

    use super::FdBlackScholesVanillaEngine;
    use super::test_fd_earliest_exercise_date::{S0, STRIKE, T_GRID, X_GRID, process, today};
    use crate::exercise::{AmericanExercise, BermudanExercise, EuropeanExercise, Exercise};
    use crate::instrument::Instrument;
    use crate::instruments::{OneAssetOption, PlainVanillaPayoff};
    use crate::methods::finitedifferences::solvers::FdmSchemeDesc;
    use crate::option::OptionType::Put;
    use crate::pricingengine::PricingEngine;
    use crate::processes::GeneralizedBlackScholesProcess;
    use crate::settings::Settings;
    use crate::shared::{Shared, SharedMut, shared, shared_mut};
    use crate::time::date::Date;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;
    use crate::types::Real;

    fn fd_price(
        settings: &Shared<Settings<Date>>,
        process: &Shared<GeneralizedBlackScholesProcess>,
        exercise: Shared<dyn Exercise>,
    ) -> Real {
        let mut option = OneAssetOption::new(
            shared(PlainVanillaPayoff::new(Put, STRIKE)),
            exercise,
            Shared::clone(settings),
        );
        let engine = shared_mut(FdBlackScholesVanillaEngine::with_params(
            Shared::clone(process),
            Vec::new(),
            T_GRID,
            X_GRID,
            0,
            FdmSchemeDesc::douglas(),
        ));
        option
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        option.npv().unwrap()
    }

    /// The exercise dates `months` after the reference date, the last of which
    /// is the maturity when it is `12`.
    fn every(months: &[i32]) -> Shared<dyn Exercise> {
        let dates = months
            .iter()
            .map(|m| today() + Period::new(*m, TimeUnit::Months))
            .collect();
        shared(BermudanExercise::new(dates, false).unwrap()) as Shared<dyn Exercise>
    }

    fn maturity() -> Date {
        today() + Period::new(1, TimeUnit::Years)
    }

    fn fixture() -> (
        Shared<Settings<Date>>,
        Shared<GeneralizedBlackScholesProcess>,
    ) {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        (settings, process())
    }

    #[test]
    fn binding_market_regressions_cover_european_american_and_quarterly_bermudan() {
        let (settings, process) = fixture();
        let (european_theta, bermudan_theta) = if cfg!(target_os = "linux") {
            (0.377_587_111_592_534_55, 0.385_210_817_371_187)
        } else {
            (0.377_587_111_591_224_7, 0.385_210_817_369_876_67)
        };
        let rows: [(&str, Shared<dyn Exercise>, [Real; 4]); 3] = [
            (
                "European",
                shared(EuropeanExercise::new(maturity())),
                [
                    18.266147644485358,
                    -0.714_918_249_077_874_9,
                    0.016_981_361_087_847_3,
                    european_theta,
                ],
            ),
            (
                "American",
                shared(AmericanExercise::over(today(), maturity()).unwrap()),
                [
                    20.357667204554883,
                    -0.859_029_794_934_689_8,
                    0.026600330114210077,
                    -0.863_258_093_716_587,
                ],
            ),
            (
                "Bermudan",
                every(&[3, 6, 9, 12]),
                [
                    19.954434523211695,
                    -0.817_505_453_287_635_2,
                    0.019414167279763642,
                    bermudan_theta,
                ],
            ),
        ];

        for (name, exercise, expected) in rows {
            let mut option = OneAssetOption::new(
                shared(PlainVanillaPayoff::new(Put, STRIKE)),
                exercise,
                Shared::clone(&settings),
            );
            let engine = shared_mut(FdBlackScholesVanillaEngine::with_params(
                Shared::clone(&process),
                Vec::new(),
                T_GRID,
                X_GRID,
                0,
                FdmSchemeDesc::douglas(),
            ));
            option
                .base_mut()
                .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
            let actual = [
                option.npv().unwrap(),
                option.delta().unwrap(),
                option.gamma().unwrap(),
                option.theta().unwrap(),
            ];
            for (field, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 1.0e-12,
                    "{name} field {field}: {actual} != {expected}"
                );
            }
        }
    }

    /// The single-date form, whose one exercise opportunity is the expiry, is
    /// the European option. It agrees to `3.8e-7`, not to the last bit, and the
    /// residual is mechanism rather than noise: that lone exercise time reaches
    /// the solver as a stopping time equal to the rollback's own starting time,
    /// so the model applies the condition once before it steps
    /// (`finitedifferencemodel.rs:96-100`), taking `max` against a terminal grid
    /// that the solver seeded with `avg_inner_value` while the condition reads
    /// `inner_value`. Where the node value exceeds the cell average the grid is
    /// lifted, and the measured gap is the sum of those lifts. C++ does the
    /// identical thing.
    ///
    /// This arm pins that the Bermudan branch does not *corrupt* a price it
    /// should reproduce. It cannot pin that the condition fires, because a
    /// condition that never fired would pass it just as well - that is
    /// [`dense_exercise_dates_price_above_the_european_and_under_the_american`].
    #[test]
    fn a_single_exercise_date_at_expiry_degenerates_to_the_european_price() {
        let (settings, process) = fixture();

        let euro = fd_price(
            &settings,
            &process,
            shared(EuropeanExercise::new(maturity())) as Shared<dyn Exercise>,
        );
        let bermudan = fd_price(&settings, &process, every(&[12]));

        let error = (bermudan - euro).abs();
        assert!(
            error <= 1.0e-5,
            "a Bermudan exercisable only at expiry should be the European option: \
             Bermudan {bermudan} european {euro} (absolute error {error})"
        );
    }

    /// The load-bearing arm. Twelve monthly exercise opportunities on a put
    /// this deep in the money capture almost all of an early-exercise premium
    /// worth about two, so the price sits far above the European one and just
    /// under the continuously exercisable American one.
    ///
    /// A condition that never fired - a wrong exercise-time clock, a dropped
    /// stopping-time push leaving the solver stepping past every exercise date,
    /// an inverted comparison - prices the European value and misses the floor
    /// by two hundred times its slack.
    #[test]
    fn dense_exercise_dates_price_above_the_european_and_under_the_american() {
        let (settings, process) = fixture();

        let euro = fd_price(
            &settings,
            &process,
            shared(EuropeanExercise::new(maturity())) as Shared<dyn Exercise>,
        );
        let monthly = fd_price(
            &settings,
            &process,
            every(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        );
        let american = fd_price(
            &settings,
            &process,
            shared(AmericanExercise::over(today(), maturity()).unwrap()) as Shared<dyn Exercise>,
        );

        println!("testFdBermudan: S0={S0} european {euro} monthly {monthly} american {american}");

        assert!(
            monthly > euro + 1.0,
            "monthly exercise should capture most of the early-exercise premium: \
             monthly {monthly} european {euro}"
        );
        assert!(
            monthly <= american,
            "a Bermudan cannot beat the American it is a subset of: \
             monthly {monthly} american {american}"
        );
    }

    /// Adding exercise opportunities cannot make the option worth less. The
    /// quarterly dates are a subset of the monthly ones, so this compares two
    /// exercise sets rather than two grids of different fineness.
    #[test]
    fn price_is_monotone_in_the_exercise_date_set() {
        let (settings, process) = fixture();

        let quarterly = fd_price(&settings, &process, every(&[3, 6, 9, 12]));
        let monthly = fd_price(
            &settings,
            &process,
            every(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        );

        assert!(
            monthly >= quarterly - 1.0e-8,
            "more exercise dates should not lower the price: \
             quarterly {quarterly} monthly {monthly}"
        );
    }
}

#[cfg(test)]
mod test_fd_engine_with_non_constant_parameters {
    //! The `testFdEngineWithNonConstantParameters` oracle of
    //! `test-suite/europeanoption.cpp:1578-1631`: the only European arm whose
    //! risk-free curve is not flat, and so the only one that pins the
    //! per-step forward reads of `set_time` to intervals that telescope to
    //! the right integrated rate. The flat sweep above cannot: every interval
    //! of a flat curve returns the same forward. What survives here is a read
    //! over a fixed short interval, which integrates too little rate over the
    //! year; a read over the whole life still integrates correctly and this
    //! oracle does not separate it from the correct one.

    use super::super::AnalyticEuropeanEngine;
    use super::super::test_market::today;
    use super::FdBlackScholesVanillaEngine;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{EuropeanOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::math::interpolations::flat::BackwardFlat;
    use crate::methods::finitedifferences::solvers::FdmSchemeDesc;
    use crate::option::OptionType::Call;
    use crate::pricingengine::PricingEngine;
    use crate::processes::GeneralizedBlackScholesProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{Shared, SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::{FlatForward, ForwardCurve};
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::types::{Real, Size, Volatility};

    /// `u` and `v` of `cpp:1582-1583`; the strike of `cpp:1610` is the spot.
    const UNDERLYING: Real = 190.0;
    const STRIKE: Real = 190.0;
    const VOLATILITY: Volatility = 0.20;

    /// `timeSteps` and `gridPoints` of `cpp:1617-1618`.
    const T_GRID: Size = 200;
    const X_GRID: Size = 201;

    /// `tolerance` of `cpp:1623`, absolute on the price rather than the
    /// relative measure the sweep uses.
    const TOLERANCE: Real = 0.01;

    /// The process of `cpp:1602-1605`. C++ names `BlackScholesProcess`, whose
    /// constructor supplies the dividend yield the generalized process needs
    /// as a flat zero `FlatForward` on Actual/365 Fixed
    /// (`ql/processes/blackscholesprocess.cpp:238-239`); with a zero rate the
    /// day counter is numerically inert.
    fn process() -> GeneralizedBlackScholesProcess {
        let day_counter = Actual360::new();
        let spot = shared(SimpleQuote::new(UNDERLYING)) as Shared<dyn Quote>;
        let vol = shared(BlackConstantVol::new(
            today(),
            None,
            VOLATILITY,
            day_counter.clone(),
        )) as Shared<dyn BlackVolTermStructure>;

        let dates = vec![
            today(),
            today() + 90,
            today() + 180,
            today() + 270,
            today() + 360,
        ];
        let forwards = vec![0.0, 0.001, 0.002, 0.005, 0.01];
        let risk_free =
            shared(ForwardCurve::new(dates, forwards, day_counter.clone(), BackwardFlat).unwrap())
                as Shared<dyn YieldTermStructure>;

        let dividend = shared(FlatForward::with_rate(
            today(),
            0.0,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>;

        GeneralizedBlackScholesProcess::new(
            Handle::new(spot),
            Handle::new(dividend),
            Handle::new(risk_free),
            Handle::new(vol),
        )
    }

    /// The forward the curve carries rises from 0.001 to 0.01 across the
    /// year, so a `set_time` pinned to a fixed short interval integrates too
    /// little rate and the price misses by around thirty times the tolerance.
    #[test]
    fn fd_engine_matches_the_analytic_engine_under_a_time_varying_rate() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let process = shared(process());

        let payoff = shared(PlainVanillaPayoff::new(Call, STRIKE));
        let exercise = shared(EuropeanExercise::new(today() + 360));
        let mut option = EuropeanOption::new(payoff, exercise, Shared::clone(&settings));

        let analytic = shared_mut(AnalyticEuropeanEngine::new(Shared::clone(&process)));
        option
            .base_mut()
            .set_pricing_engine(analytic as SharedMut<dyn PricingEngine>);
        let expected = option.npv().unwrap();

        let finite_difference = shared_mut(FdBlackScholesVanillaEngine::with_params(
            Shared::clone(&process),
            Vec::new(),
            T_GRID,
            X_GRID,
            0,
            FdmSchemeDesc::douglas(),
        ));
        option
            .base_mut()
            .set_pricing_engine(finite_difference as SharedMut<dyn PricingEngine>);
        let calculated = option.npv().unwrap();

        let error = (expected - calculated).abs();
        assert!(
            error <= TOLERANCE,
            "analytic {expected} vs finite difference {calculated} \
             (absolute error {error} over {TOLERANCE})"
        );
    }
}
