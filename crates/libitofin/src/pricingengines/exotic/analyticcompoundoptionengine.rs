//! Analytic compound-option engine (Wystup 2002).
//!
//! Port of `ql/pricingengines/exotic/analyticcompoundoptionengine.{hpp,cpp}`:
//! NPV, delta, gamma, vega, and theta. Critical spot via Brent + `black_formula`.

use crate::errors::QlResult;
use crate::instrument::Instrument;
use crate::instruments::{CompoundArguments, CompoundResults, StrikedTypePayoff, TypePayoff};
use crate::interestrate::Compounding;
use crate::math::distributions::bivariatenormal::BivariateCumulativeNormalDistributionDr78;
use crate::math::distributions::normal::{CumulativeNormalDistribution, NormalDistribution};
use crate::math::solver1d::Solver1D;
use crate::math::solvers1d::brent::Brent;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::black_formula;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::Real;

type EngineBase = GenericEngine<CompoundArguments, CompoundResults>;

/// Pricing engine for European compound options.
pub struct AnalyticCompoundOptionEngine {
    base: EngineBase,
    process: Shared<GeneralizedBlackScholesProcess>,
    n: CumulativeNormalDistribution,
}

impl AnalyticCompoundOptionEngine {
    /// `AnalyticCompoundOptionEngine(process)`.
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        let base = EngineBase::new(CompoundArguments::default(), CompoundResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            n: CumulativeNormalDistribution::standard(),
        }
    }
}

impl AsObservable for AnalyticCompoundOptionEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticCompoundOptionEngine {
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
        let a = self.base.arguments();
        let m_pay = a.mother_payoff.expect("validated");
        let d_pay = a.daughter_payoff.expect("validated");
        let m_ex = a.mother_exercise.as_ref().expect("validated");
        let d_ex = a.daughter_exercise.as_ref().expect("validated");
        let str_m = m_pay.strike();
        let str_d = d_pay.strike();
        require!(str_d > 0.0, "Daughter strike must be positive");
        require!(str_m > 0.0, "Mother strike must be positive");
        let s = self.process.x0()?;
        require!(s > 0.0, "negative or null underlying given");
        let phi = Real::from(d_pay.option_type() as i32);
        let w = Real::from(m_pay.option_type() as i32);

        let rf = self.process.risk_free_rate().current_link()?;
        let qy = self.process.dividend_yield().current_link()?;
        let vol = self.process.black_volatility().current_link()?;
        let mat_m = m_ex.last_date();
        let mat_d = d_ex.last_date();
        let t_m = self.process.time(&mat_m)?;
        let t_d = self.process.time(&mat_d)?;
        let help_mat = rf.reference_date()? + (mat_d - mat_m);
        let t_help = self.process.time(&help_mat)?;
        let std_help = vol.black_vol_date(help_mat, str_d, true)? * t_help.sqrt();
        let q_help = qy.discount_date(help_mat, false)?;
        let r_help = rf.discount_date(help_mat, false)?;
        let d_ty = d_pay.option_type();
        let solved = Brent::new().with_max_evaluations(1000).solve_bracketed(
            |spot| {
                let fwd = spot * q_help / r_help;
                black_formula(d_ty, str_d, fwd, std_help, r_help, 0.0).unwrap_or(Real::NAN) - str_m
            },
            1.0e-6,
            str_d,
            1.0e-6,
            str_d * 1000.0,
        )?;

        let dd_d = qy.discount(t_d, false)?;
        let rd_d = rf.discount(t_d, false)?;
        let dd_m = qy.discount(t_m, false)?;
        let rd_m = rf.discount(t_m, false)?;
        let sd_m = vol.black_vol_date(mat_m, str_m, true)? * t_m.sqrt();
        let v_d = vol.black_vol_date(mat_d, str_d, true)?;
        let sd_d = v_d * t_d.sqrt();
        let x = ((rd_m * solved / (s * dd_m)) * (0.5 * sd_m * sd_m).exp()).ln() / sd_m;
        let fwd_d = s * dd_d / rd_d;
        let d_plus = (fwd_d / str_d).ln() / sd_d + 0.5 * sd_d;
        let d_minus = d_plus - sd_d;
        let n2 = BivariateCumulativeNormalDistributionDr78::new(w * (t_m / t_d).sqrt())?;

        let tau_12 = t_d - t_m;
        let dd_12 = qy.discount(tau_12, false)?;
        let rd_12 = rf.discount(tau_12, false)?;
        let fwd_12 = solved * dd_12 / rd_12;
        let sd_12 = v_d * tau_12.sqrt();
        let d_p_t12 = (fwd_12 / str_d).ln() / sd_12 + 0.5 * sd_12;

        let e_x = (x * t_d.sqrt() + t_m.sqrt() * d_minus) / tau_12.sqrt();
        let r_d = rf
            .zero_rate(t_d, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate();
        let d_d = qy
            .zero_rate(t_d, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate();

        let norm = NormalDistribution::standard();
        let x_m_sm = x - sd_m;
        let n2_xmsm = n2.value(-phi * w * x_m_sm, phi * d_plus);
        let n2_x = n2.value(-phi * w * x, phi * d_minus);
        let n_ex = self.n.value(-phi * w * e_x);
        let nx = self.n.value(-phi * w * x);
        let n_t12 = self.n.value(phi * d_p_t12);
        let n_dp = norm.value(d_plus);
        let n_xm = norm.value(x_m_sm);
        let inv_m_time = 1.0 / t_m.sqrt();
        let inv_d_time = 1.0 / t_d.sqrt();

        let value =
            phi * w * s * dd_d * n2_xmsm - phi * w * str_d * rd_d * n2_x - w * str_m * rd_m * nx;
        let delta = phi * w * dd_d * n2_xmsm;
        let gamma = (dd_d / (v_d * s)) * (inv_m_time * n_xm * n_t12 + w * inv_d_time * n_dp * n_ex);
        let vega = dd_d * s * (t_m.sqrt() * n_xm * n_t12 + w * t_d.sqrt() * n_dp * n_ex);
        let mut theta = phi * w * d_d * s * dd_d * n2_xmsm
            - phi * w * r_d * str_d * rd_d * n2_x
            - w * r_d * str_m * rd_m * nx;
        theta -= 0.5 * v_d * s * dd_d * (inv_m_time * n_xm * n_t12 + w * inv_d_time * n_dp * n_ex);

        let res = self.base.results_mut();
        res.instrument.value = Some(value);
        res.greeks.delta = Some(delta);
        res.greeks.gamma = Some(gamma);
        res.greeks.vega = Some(vega);
        res.greeks.theta = Some(theta);
        Ok(())
    }
}

/// Attaches [`AnalyticCompoundOptionEngine`] to `option`.
pub fn set_analytic_compound_option_engine(
    option: &mut crate::instruments::CompoundOption,
    process: Shared<GeneralizedBlackScholesProcess>,
) {
    let engine =
        shared_mut(AnalyticCompoundOptionEngine::new(process)) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{CompoundOption, EuropeanOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType::{self, Call, Put};
    use crate::pricingengines::vanilla::AnalyticEuropeanEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn flat_rate(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            reference,
            quote_handle(quote),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::with_quote(
            reference,
            None,
            quote_handle(quote),
            Actual360::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_option(
        tm: OptionType,
        td: OptionType,
        km: Real,
        kd: Real,
        s: Real,
        q: Real,
        r: Real,
        t_m: Real,
        t_d: Real,
        v: Real,
    ) -> (
        CompoundOption,
        Shared<BlackScholesMertonProcess>,
        Date,
        Shared<Settings<Date>>,
    ) {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);
        let process = shared(BlackScholesMertonProcess::new(
            quote_handle(&shared(SimpleQuote::new(s))),
            flat_rate(today, &shared(SimpleQuote::new(q))),
            flat_rate(today, &shared(SimpleQuote::new(r))),
            flat_vol(today, &shared(SimpleQuote::new(v))),
        ));
        let mother: Shared<dyn Exercise> =
            shared(EuropeanExercise::new(today + (t_m * 360.0).round() as i32));
        let daughter: Shared<dyn Exercise> =
            shared(EuropeanExercise::new(today + (t_d * 360.0).round() as i32));
        let mut option = CompoundOption::new(
            PlainVanillaPayoff::new(tm, km),
            mother,
            PlainVanillaPayoff::new(td, kd),
            daughter,
            Shared::clone(&settings),
        );
        set_analytic_compound_option_engine(&mut option, Shared::clone(&process));
        (option, process, today, settings)
    }

    /// Full 20-row `compoundoption.cpp` `testValues` oracle (Haug/sitmo/mathfinance @ 1e-3).
    #[test]
    fn test_compound_option_values_and_greeks() {
        type Row = (
            OptionType,
            OptionType,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
            Real,
        );
        #[rustfmt::skip]
        let rows: [Row; 20] = [
            // Haug 2007 + sitmo.com:
            (Put,  Call, 50.0, 520.0, 500.0, 0.03, 0.08, 0.25, 0.5, 0.35, 21.1965, -0.1966,  0.0007, -32.1241,  -3.3837),
            (Call, Call, 50.0, 520.0, 500.0, 0.03, 0.08, 0.25, 0.5, 0.35, 17.5945,  0.3219,  0.0038, 106.5185, -65.1614),
            (Call, Put,  50.0, 520.0, 500.0, 0.03, 0.08, 0.25, 0.5, 0.35, 18.7128, -0.2906,  0.0036, 103.3856, -46.6982),
            (Put,  Put,  50.0, 520.0, 500.0, 0.03, 0.08, 0.25, 0.5, 0.35, 15.2601,  0.1760,  0.0005, -35.2570, -10.1126),
            // sitmo.com:
            (Call, Call, 0.05, 1.14,  1.20,  0.00, 0.01, 0.50, 2.0, 0.11, 0.0729,   0.6614,  2.5762,   0.5812,  -0.0297),
            (Call, Put,  0.05, 1.14,  1.20,  0.00, 0.01, 0.50, 2.0, 0.11, 0.0074,  -0.1334,  1.9681,   0.2933,  -0.0155),
            (Put,  Call, 0.05, 1.14,  1.20,  0.00, 0.01, 0.50, 2.0, 0.11, 0.0021,  -0.0426,  0.7252,  -0.0052,  -0.0058),
            (Put,  Put,  0.05, 1.14,  1.20,  0.00, 0.01, 0.50, 2.0, 0.11, 0.0192,   0.1626,  0.1171,  -0.2931,  -0.0028),
            (Call, Call, 10.0, 122.0, 120.0, 0.06, 0.02, 0.10, 0.7, 0.22, 0.4419,   0.1049,  0.0195,  11.3368,  -6.2871),
            (Call, Put,  10.0, 122.0, 120.0, 0.06, 0.02, 0.10, 0.7, 0.22, 2.6112,  -0.3618,  0.0337,  28.4843, -13.4124),
            (Put,  Call, 10.0, 122.0, 120.0, 0.06, 0.02, 0.10, 0.7, 0.22, 4.1616,  -0.3174,  0.0024, -26.6403,  -2.2720),
            (Put,  Put,  10.0, 122.0, 120.0, 0.06, 0.02, 0.10, 0.7, 0.22, 1.0914,   0.1748,  0.0165,  -9.4928,  -4.8995),
            // mathfinance VBA:
            (Call, Call, 0.40, 8.20,  8.00,  0.05, 0.00, 2.00, 3.0, 0.08, 0.0099,   0.0285,  0.0688,   0.7764,  -0.0027),
            (Call, Put,  0.40, 8.20,  8.00,  0.05, 0.00, 2.00, 3.0, 0.08, 0.9826,  -0.7224,  0.2158,   2.7279,  -0.3332),
            (Put,  Call, 0.40, 8.20,  8.00,  0.05, 0.00, 2.00, 3.0, 0.08, 0.3585,  -0.0720, -0.0835,  -1.5633,  -0.0117),
            (Put,  Put,  0.40, 8.20,  8.00,  0.05, 0.00, 2.00, 3.0, 0.08, 0.0168,   0.0378,  0.0635,   0.3882,   0.0021),
            (Call, Call, 0.02, 1.60,  1.60,  0.013, 0.022, 0.45, 0.5, 0.17, 0.0680, 0.4937,  2.1271,   0.4418,  -0.0843),
            (Call, Put,  0.02, 1.60,  1.60,  0.013, 0.022, 0.45, 0.5, 0.17, 0.0605, -0.4169, 2.0836,   0.4330,  -0.0697),
            (Put,  Call, 0.02, 1.60,  1.60,  0.013, 0.022, 0.45, 0.5, 0.17, 0.0081, -0.0417, 0.0761,  -0.0045,  -0.0020),
            (Put,  Put,  0.02, 1.60,  1.60,  0.013, 0.022, 0.45, 0.5, 0.17, 0.0078,  0.0413, 0.0326,  -0.0133,  -0.0016),
        ];

        for (tm, td, km, kd, s, q, r, t_m, t_d, v, exp_npv, exp_d, exp_g, exp_v, exp_th) in rows {
            let (mut opt, _, _, _) = build_option(tm, td, km, kd, s, q, r, t_m, t_d, v);
            let npv = opt.npv().unwrap();
            let delta = opt.delta().unwrap();
            let gamma = opt.gamma().unwrap();
            let vega = opt.vega().unwrap();
            let theta = opt.theta().unwrap();

            assert!(
                (npv - exp_npv).abs() <= 1e-3,
                "{tm:?}/{td:?} NPV: got {npv}, exp {exp_npv}"
            );
            assert!(
                (delta - exp_d).abs() <= 1e-3,
                "{tm:?}/{td:?} delta: got {delta}, exp {exp_d}"
            );
            assert!(
                (gamma - exp_g).abs() <= 1e-3,
                "{tm:?}/{td:?} gamma: got {gamma}, exp {exp_g}"
            );
            assert!(
                (vega - exp_v).abs() <= 1e-3,
                "{tm:?}/{td:?} vega: got {vega}, exp {exp_v}"
            );
            assert!(
                (theta - exp_th).abs() <= 1e-3,
                "{tm:?}/{td:?} theta: got {theta}, exp {exp_th}"
            );
        }
    }

    /// QL `compoundoption.cpp` `testPutCallParity` oracle (Wystup 2002 @ 1e-8).
    ///
    /// Note: QuantLib's `values` table has 11 entries because row 0 and row 1 differ
    /// only by `typeMother` (Put vs Call), but QL's test loop builds both mother Call
    /// and mother Put for every row without inspecting `typeMother`, so rows 0 and 1
    /// test the exact same daughter-market case. We test the 10 unique cases here.
    #[test]
    fn test_compound_option_put_call_parity() {
        type ParityRow = (OptionType, Real, Real, Real, Real, Real, Real, Real, Real);
        #[rustfmt::skip]
        let rows: [ParityRow; 10] = [
            (Call, 50.0, 520.0, 500.0, 0.03,  0.08,  0.25, 0.5, 0.35),
            (Put,  50.0, 520.0, 500.0, 0.03,  0.08,  0.25, 0.5, 0.35),
            (Call, 0.05, 1.14,  1.20,  0.00,  0.01,  0.50, 2.0, 0.11),
            (Put,  0.05, 1.14,  1.20,  0.00,  0.01,  0.50, 2.0, 0.11),
            (Call, 10.0, 122.0, 120.0, 0.06,  0.02,  0.10, 0.7, 0.22),
            (Put,  10.0, 122.0, 120.0, 0.06,  0.02,  0.10, 0.7, 0.22),
            (Call, 0.40, 8.20,  8.00,  0.05,  0.00,  2.00, 3.0, 0.08),
            (Put,  0.40, 8.20,  8.00,  0.05,  0.00,  2.00, 3.0, 0.08),
            (Call, 0.02, 1.60,  1.60,  0.013, 0.022, 0.45, 0.5, 0.17),
            (Put,  0.02, 1.60,  1.60,  0.013, 0.022, 0.45, 0.5, 0.17),
        ];

        for (td, km, kd, s, q, r, t_m, t_d, v) in rows {
            let (mut call_opt, process, today, settings) =
                build_option(Call, td, km, kd, s, q, r, t_m, t_d, v);
            let (mut put_opt, _, _, _) = build_option(Put, td, km, kd, s, q, r, t_m, t_d, v);

            let d_date = today + (t_d * 360.0).round() as i32;
            let m_date = today + (t_m * 360.0).round() as i32;

            let mut vanilla = EuropeanOption::new(
                shared(PlainVanillaPayoff::new(td, kd)),
                shared(EuropeanExercise::new(d_date)),
                settings,
            );
            vanilla
                .base_mut()
                .set_pricing_engine(
                    shared_mut(AnalyticEuropeanEngine::new(Shared::clone(&process)))
                        as SharedMut<dyn PricingEngine>,
                );

            let disc_factor = process
                .risk_free_rate()
                .current_link()
                .unwrap()
                .discount_date(m_date, false)
                .unwrap();
            let disc_strike = km * disc_factor;

            let parity_diff = call_opt.npv().unwrap() + disc_strike
                - put_opt.npv().unwrap()
                - vanilla.npv().unwrap();
            assert!(
                parity_diff.abs() <= 1e-8,
                "Put-call parity failed for {td:?} (km={km}, kd={kd}): diff={parity_diff}"
            );
        }
    }
}
