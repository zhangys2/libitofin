//! Analytic European Margrabe engine (`analyticeuropeanmargrabeengine.{hpp,cpp}`).

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instrument::Instrument;
use crate::instruments::{MargrabeArguments, MargrabeResults};
use crate::math::distributions::normal::{CumulativeNormalDistribution, NormalDistribution};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::daycounters::actual360::Actual360;
use crate::types::Real;

type EngineBase = GenericEngine<MargrabeArguments, MargrabeResults>;

/// Pricing engine for European exchange options.
pub struct AnalyticEuropeanMargrabeEngine {
    base: EngineBase,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
    f: CumulativeNormalDistribution,
}

impl AnalyticEuropeanMargrabeEngine {
    /// `AnalyticEuropeanMargrabeEngine(process1, process2, correlation)`.
    pub fn new(
        process1: Shared<GeneralizedBlackScholesProcess>,
        process2: Shared<GeneralizedBlackScholesProcess>,
        rho: Real,
    ) -> Self {
        let base = EngineBase::new(MargrabeArguments::default(), MargrabeResults::default());
        base.register_with(process1.observable());
        base.register_with(process2.observable());
        Self {
            base,
            process1,
            process2,
            rho,
            f: CumulativeNormalDistribution::standard(),
        }
    }
}

impl AsObservable for AnalyticEuropeanMargrabeEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticEuropeanMargrabeEngine {
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
        let arguments = self.base.arguments();
        let exercise = arguments.exercise.as_ref().expect("validated");
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European Option"
        );
        require!(arguments.payoff.is_some(), "non a Null Payoff type");
        let q1 = Real::from(arguments.q1.expect("validated"));
        let q2 = Real::from(arguments.q2.expect("validated"));
        let maturity = exercise.last_date();

        let s1 = self.process1.x0()?;
        let s2 = self.process2.x0()?;
        require!(s1 > 0.0, "negative or null underlying1");
        require!(s2 > 0.0, "negative or null underlying2");

        let vol1 = self.process1.black_volatility().current_link()?;
        let vol2 = self.process2.black_volatility().current_link()?;
        let variance1 = vol1.black_variance_date(maturity, s1, false)?;
        let variance2 = vol2.black_variance_date(maturity, s2, false)?;

        let rf = self.process1.risk_free_rate().current_link()?;
        let qy1 = self.process1.dividend_yield().current_link()?;
        let qy2 = self.process2.dividend_yield().current_link()?;
        let rf_disc = rf.discount_date(maturity, false)?;
        let q_disc1 = qy1.discount_date(maturity, false)?;
        let q_disc2 = qy2.discount_date(maturity, false)?;
        let forward1 = s1 * q_disc1 / rf_disc;
        let forward2 = s2 * q_disc2 / rf_disc;

        let variance = variance1 + variance2 - 2.0 * self.rho * variance1.sqrt() * variance2.sqrt();
        let std_dev = variance.sqrt();
        let d1 = (((q1 * forward1) / (q2 * forward2)).ln() + 0.5 * variance) / std_dev;
        let d2 = d1 - std_dev;

        let norm = NormalDistribution::standard();
        let nd1_cum = self.f.value(d1);
        let nd2_cum = self.f.value(d2);
        let nd1_pdf = norm.value(d1);
        let nd2_pdf = norm.value(d2);

        let rfdc = rf.day_counter().unwrap_or_else(Actual360::new);
        let t = rfdc.year_fraction(rf.reference_date()?, maturity);
        require!(t > 0.0, "maturity must be in the future");
        let sqt = t.sqrt();
        let q1_rate = -q_disc1.ln() / (sqt * sqt);
        let q2_rate = -q_disc2.ln() / (sqt * sqt);

        let value = rf_disc * (q1 * forward1 * nd1_cum - q2 * forward2 * nd2_cum);
        let delta1 = rf_disc * (q1 * forward1 * nd1_cum) / s1;
        let delta2 = -rf_disc * (q2 * forward2 * nd2_cum) / s2;
        let gamma1 = (rf_disc * (q1 * forward1 * nd1_pdf) / s1) / (q1 * s1 * std_dev);
        let gamma2 = (-rf_disc * (q2 * forward2 * nd2_pdf) / s2) / (-q2 * s2 * std_dev);
        let vega = rf_disc * (q1 * forward1 * nd1_pdf) * sqt;
        let theta = -((std_dev * vega / sqt) / (2.0 * t)
            - (q1_rate * q1 * s1 * delta1)
            - (q2_rate * q2 * s2 * delta2));
        let rho = 0.0;

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        results.delta1 = Some(delta1);
        results.delta2 = Some(delta2);
        results.gamma1 = Some(gamma1);
        results.gamma2 = Some(gamma2);
        results.theta = Some(theta);
        results.rho = Some(rho);
        Ok(())
    }
}

/// Attaches [`AnalyticEuropeanMargrabeEngine`] to `option`.
pub fn set_analytic_european_margrabe_engine(
    option: &mut crate::instruments::MargrabeOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
) {
    let engine = shared_mut(AnalyticEuropeanMargrabeEngine::new(process1, process2, rho))
        as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::MargrabeOption;
    use crate::interestrate::Compounding;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounter::DayCounter;
    use crate::time::frequency::Frequency;
    use crate::types::Integer;

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
    #[rustfmt::skip]
    fn build_test_option(
        s1: Real, s2: Real, q1: Integer, q2: Integer,
        div1: Real, div2: Real, r: Real, t: Real,
        v1: Real, v2: Real, rho: Real,
    ) -> (
        MargrabeOption,
        Date,
        Shared<SimpleQuote>,
        Shared<SimpleQuote>,
        Shared<SimpleQuote>,
        DayCounter,
    ) {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::May, 1998);
        settings.set_evaluation_date(today);
        let spot1 = shared(SimpleQuote::new(s1));
        let spot2 = shared(SimpleQuote::new(s2));
        let r_quote = shared(SimpleQuote::new(r));
        let p1 = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot1),
            flat_rate(today, &shared(SimpleQuote::new(div1))),
            flat_rate(today, &r_quote),
            flat_vol(today, &shared(SimpleQuote::new(v1))),
        ));
        let p2 = shared(BlackScholesMertonProcess::new(
            quote_handle(&spot2),
            flat_rate(today, &shared(SimpleQuote::new(div2))),
            flat_rate(today, &r_quote),
            flat_vol(today, &shared(SimpleQuote::new(v2))),
        ));
        let exercise: Shared<dyn Exercise> =
            shared(EuropeanExercise::new(today + (t * 360.0).round() as i32));
        let mut option = MargrabeOption::new(q1, q2, exercise, settings);
        set_analytic_european_margrabe_engine(&mut option, p1, p2, rho);
        (option, today, spot1, spot2, r_quote, Actual360::new())
    }

    /// Full 21-row `margrabeoption.cpp` `testEuroExchangeTwoAssets` oracle (Haug + quantities @ 1e-3).
    #[test]
    fn test_euro_exchange_two_assets() {
        type Row = (
            Real,    // s1
            Real,    // s2
            Integer, // Q1
            Integer, // Q2
            Real,    // q1
            Real,    // q2
            Real,    // r
            Real,    // t
            Real,    // v1
            Real,    // v2
            Real,    // rho
            Real,    // result
            Real,    // delta1
            Real,    // delta2
            Real,    // gamma1
            Real,    // gamma2
            Real,    // theta
            Real,    // rho_greek
        );
        #[rustfmt::skip]
        let rows: [Row; 21] = [
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.15, -0.50, 2.125, 0.841, -0.818, 0.112, 0.135, -2.043, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.20, -0.50, 2.199, 0.813, -0.784, 0.109, 0.132, -2.723, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.25, -0.50, 2.283, 0.788, -0.753, 0.105, 0.126, -3.419, 0.0),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.15,  0.00, 2.045, 0.883, -0.870, 0.108, 0.131, -1.168, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.20,  0.00, 2.091, 0.857, -0.838, 0.112, 0.135, -1.698, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.25,  0.00, 2.152, 0.830, -0.805, 0.111, 0.134, -2.302, 0.0),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.15,  0.50, 1.974, 0.946, -0.942, 0.079, 0.096, -0.126, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.20,  0.50, 1.989, 0.929, -0.922, 0.092, 0.111, -0.398, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.25,  0.50, 2.019, 0.902, -0.891, 0.104, 0.125, -0.838, 0.0),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.15, -0.50, 2.762, 0.672, -0.602, 0.072, 0.087, -1.207, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20, -0.50, 2.989, 0.661, -0.578, 0.064, 0.078, -1.457, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.25, -0.50, 3.228, 0.653, -0.557, 0.058, 0.070, -1.712, 0.0),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.15,  0.00, 2.479, 0.695, -0.640, 0.085, 0.102, -0.874, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20,  0.00, 2.650, 0.680, -0.616, 0.077, 0.093, -1.078, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.25,  0.00, 2.847, 0.668, -0.592, 0.069, 0.083, -1.302, 0.0),

            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.15,  0.50, 2.138, 0.746, -0.713, 0.106, 0.128, -0.416, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20,  0.50, 2.231, 0.728, -0.689, 0.099, 0.120, -0.550, 0.0),
            (22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.25,  0.50, 2.374, 0.707, -0.659, 0.090, 0.109, -0.741, 0.0),

            // Quantity tests from Excel calculations
            (22.0, 10.0, 1, 2, 0.06, 0.04, 0.10, 0.50, 0.20, 0.15,  0.50, 2.138, 0.746, -1.426, 0.106, 0.255, -0.987, 0.0),
            (11.0, 20.0, 2, 1, 0.06, 0.04, 0.10, 0.50, 0.20, 0.20,  0.50, 2.231, 1.455, -0.689, 0.198, 0.120,  0.410, 0.0),
            (11.0, 10.0, 2, 2, 0.06, 0.04, 0.10, 0.50, 0.20, 0.25,  0.50, 2.374, 1.413, -1.317, 0.181, 0.219, -0.336, 0.0),
        ];

        for (
            s1,
            s2,
            q1,
            q2,
            d1,
            d2,
            r,
            t,
            v1,
            v2,
            rho,
            exp_val,
            exp_d1,
            exp_d2,
            exp_g1,
            exp_g2,
            exp_th,
            exp_rho,
        ) in rows
        {
            let (mut option, _, _, _, _, _) =
                build_test_option(s1, s2, q1, q2, d1, d2, r, t, v1, v2, rho);
            let val = option.npv().unwrap();
            let d1_val = option.delta1().unwrap();
            let d2_val = option.delta2().unwrap();
            let g1_val = option.gamma1().unwrap();
            let g2_val = option.gamma2().unwrap();
            let th_val = option.theta().unwrap();
            let rho_val = option.rho().unwrap();

            assert!(
                (val - exp_val).abs() <= 1e-3,
                "value: got {val}, exp {exp_val}"
            );
            assert!(
                (d1_val - exp_d1).abs() <= 1e-3,
                "delta1: got {d1_val}, exp {exp_d1}"
            );
            assert!(
                (d2_val - exp_d2).abs() <= 1e-3,
                "delta2: got {d2_val}, exp {exp_d2}"
            );
            assert!(
                (g1_val - exp_g1).abs() <= 1e-3,
                "gamma1: got {g1_val}, exp {exp_g1}"
            );
            assert!(
                (g2_val - exp_g2).abs() <= 1e-3,
                "gamma2: got {g2_val}, exp {exp_g2}"
            );
            assert!(
                (th_val - exp_th).abs() <= 1e-3,
                "theta: got {th_val}, exp {exp_th}"
            );
            assert!(
                (rho_val - exp_rho).abs() <= 1e-3,
                "rho: got {rho_val}, exp {exp_rho}"
            );
        }
    }

    /// `testGreeks` finite difference bump consistency from `margrabeoption.cpp`.
    #[test]
    fn test_analytic_european_margrabe_greeks_fd() {
        let (mut option, _today, spot1, spot2, r_quote, _dc) =
            build_test_option(22.0, 20.0, 1, 1, 0.06, 0.04, 0.10, 0.10, 0.20, 0.15, -0.50);

        let delta1 = option.delta1().unwrap();
        let delta2 = option.delta2().unwrap();
        let gamma1 = option.gamma1().unwrap();
        let gamma2 = option.gamma2().unwrap();
        let rho = option.rho().unwrap();

        // Bump spot1
        let u1 = 22.0;
        let du1 = u1 * 1.0e-4;
        spot1.set_value(Some(u1 + du1));
        let val_p1 = option.npv().unwrap();
        let delta_p1 = option.delta1().unwrap();
        spot1.set_value(Some(u1 - du1));
        let val_m1 = option.npv().unwrap();
        let delta_m1 = option.delta1().unwrap();
        spot1.set_value(Some(u1));
        let exp_delta1 = (val_p1 - val_m1) / (2.0 * du1);
        let exp_gamma1 = (delta_p1 - delta_m1) / (2.0 * du1);
        assert!((delta1 - exp_delta1).abs() / u1 <= 1e-4);
        assert!((gamma1 - exp_gamma1).abs() / u1 <= 1e-4);

        // Bump spot2
        let u2 = 20.0;
        let du2 = u2 * 1.0e-4;
        spot2.set_value(Some(u2 + du2));
        let val_p2 = option.npv().unwrap();
        let delta_p2 = option.delta2().unwrap();
        spot2.set_value(Some(u2 - du2));
        let val_m2 = option.npv().unwrap();
        let delta_m2 = option.delta2().unwrap();
        spot2.set_value(Some(u2));
        let exp_delta2 = (val_p2 - val_m2) / (2.0 * du2);
        let exp_gamma2 = (delta_p2 - delta_m2) / (2.0 * du2);
        assert!((delta2 - exp_delta2).abs() / u1 <= 1e-4);
        assert!((gamma2 - exp_gamma2).abs() / u1 <= 1e-4);

        // Bump risk-free rate
        let r = 0.10;
        let dr = r * 1.0e-4;
        r_quote.set_value(Some(r + dr));
        let val_pr = option.npv().unwrap();
        r_quote.set_value(Some(r - dr));
        let val_mr = option.npv().unwrap();
        r_quote.set_value(Some(r));
        let exp_rho = (val_pr - val_mr) / (2.0 * dr);
        assert!((rho - exp_rho).abs() <= 1e-4);
    }
}
