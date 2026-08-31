//! Analytic continuous geometric-average price Asian under Heston.
//!
//! Port of `ql/experimental/asian/analytic_cont_geom_av_price_heston.{hpp,cpp}`
//! (Kim & Wee, *Quantitative Finance* 14:10, 2014).

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, ContinuousAveragingAsianArguments, ContinuousAveragingAsianResults,
    StrikedTypePayoff, TypePayoff,
};
use crate::math::integrals::gaussianquadratures::GaussianQuadrature;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::HestonProcess;
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::{Complex, Real, Size, Time};

type EngineBase = GenericEngine<ContinuousAveragingAsianArguments, ContinuousAveragingAsianResults>;

/// Analytic continuous geometric average-price Asian under Heston.
pub struct AnalyticContinuousGeometricAveragePriceAsianHestonEngine {
    base: EngineBase,
    process: Shared<HestonProcess>,
    v0: Real,
    rho: Real,
    kappa: Real,
    theta: Real,
    sigma: Real,
    dividend_yield: Handle<dyn YieldTermStructure>,
    risk_free_rate: Handle<dyn YieldTermStructure>,
    s0: Handle<dyn Quote>,
    a1: Real,
    a2: Real,
    a3: Real,
    a4: Real,
    a5: Real,
    f_lookup_table: RefCell<HashMap<i32, Complex>>,
    summation_cutoff: Size,
    xi_right_limit: Real,
    integrator: GaussianQuadrature,
}

impl AnalyticContinuousGeometricAveragePriceAsianHestonEngine {
    /// `AnalyticContinuousGeometricAveragePriceAsianHestonEngine(process)`.
    pub fn new(process: Shared<HestonProcess>) -> QlResult<Self> {
        Self::with_cutoffs(process, 50, 100.0)
    }

    /// With summation cutoff (eqs 19–20) and ξ integral right limit (eq 29).
    pub fn with_cutoffs(
        process: Shared<HestonProcess>,
        summation_cutoff: Size,
        xi_right_limit: Real,
    ) -> QlResult<Self> {
        let v0 = process.v0();
        let rho = process.rho();
        let kappa = process.kappa();
        let theta = process.theta();
        let sigma = process.sigma();
        let s0 = process.s0();
        let risk_free_rate = process.risk_free_rate();
        let dividend_yield = process.dividend_yield();
        let a1 = 2.0 * v0 / (sigma * sigma);
        let a2 = 2.0 * kappa * theta / (sigma * sigma);
        let base = EngineBase::new(
            ContinuousAveragingAsianArguments::default(),
            ContinuousAveragingAsianResults::default(),
        );
        base.register_with(process.observable());
        Ok(Self {
            base,
            process,
            v0,
            rho,
            kappa,
            theta,
            sigma,
            dividend_yield,
            risk_free_rate,
            s0,
            a1,
            a2,
            a3: 0.0,
            a4: 0.0,
            a5: 0.0,
            f_lookup_table: RefCell::new(HashMap::new()),
            summation_cutoff,
            xi_right_limit,
            integrator: GaussianQuadrature::legendre(128)?,
        })
    }

    /// Φ(s, w; T, t) — Kim–Wee eq (25).
    pub fn phi(
        &self,
        s: Complex,
        w: Complex,
        t_expiry: Time,
        t: Time,
        cutoff: Size,
    ) -> Complex {
        let tau = t_expiry - t;
        let z1 = self.z1_f(s, w, t_expiry);
        let z2 = self.z2_f(s, w, t_expiry);
        let z3 = self.z3_f(s, w, t_expiry);
        let z4 = self.z4_f(s, w);
        self.f_lookup_table.borrow_mut().clear();
        let (f, f_tilde) = self.f_f_tilde(z1, z2, z3, z4, tau, cutoff);
        (-self.a1 * f_tilde / f - self.a2 * f.ln() + self.a3 * s + self.a4 * w + self.a5).exp()
    }

    fn z1_f(&self, s: Complex, _w: Complex, t_expiry: Time) -> Complex {
        s * s * (1.0 - self.rho * self.rho) / (2.0 * t_expiry * t_expiry)
    }

    fn z2_f(&self, s: Complex, w: Complex, t_expiry: Time) -> Complex {
        s * (2.0 * self.rho * self.kappa - self.sigma) / (2.0 * self.sigma * t_expiry)
            + s * w * (1.0 - self.rho * self.rho) / t_expiry
    }

    fn z3_f(&self, s: Complex, w: Complex, t_expiry: Time) -> Complex {
        s * self.rho / (self.sigma * t_expiry)
            + 0.5 * w * (2.0 * self.rho * self.kappa - self.sigma) / self.sigma
            + 0.5 * w * w * (1.0 - self.rho * self.rho)
    }

    fn z4_f(&self, _s: Complex, w: Complex) -> Complex {
        w * self.rho / self.sigma
    }

    fn f(
        &self,
        z1: Complex,
        z2: Complex,
        z3: Complex,
        z4: Complex,
        n: i32,
        tau: Time,
    ) -> Complex {
        let result = if n < 2 {
            if n < 0 {
                Complex::new(0.0, 0.0)
            } else if n == 0 {
                Complex::new(1.0, 0.0)
            } else {
                0.5 * (self.kappa - z4 * self.sigma * self.sigma) * tau
            }
        } else {
            let prefactor =
                -0.5 * self.sigma * self.sigma * tau * tau / ((n as Real) * ((n - 1) as Real));
            let mut f_minus_n = [Complex::new(0.0, 0.0); 4];
            for offset in 1..5 {
                let location = n - offset;
                let cached = self.f_lookup_table.borrow().get(&location).copied();
                f_minus_n[(offset - 1) as usize] = if let Some(value) = cached {
                    value
                } else {
                    self.f(z1, z2, z3, z4, location, tau)
                };
            }
            prefactor
                * (z1 * tau * tau * f_minus_n[3]
                    + z2 * tau * f_minus_n[2]
                    + (z3 - 0.5 * self.kappa * self.kappa / (self.sigma * self.sigma)) * f_minus_n[1])
        };
        self.f_lookup_table.borrow_mut().insert(n, result);
        result
    }

    fn f_f_tilde(
        &self,
        z1: Complex,
        z2: Complex,
        z3: Complex,
        z4: Complex,
        tau: Time,
        cutoff: Size,
    ) -> (Complex, Complex) {
        let mut running_sum1 = Complex::new(0.0, 0.0);
        let mut running_sum2 = Complex::new(0.0, 0.0);
        for i in 0..cutoff {
            let temp = self.f(z1, z2, z3, z4, i as i32, tau);
            running_sum1 += temp;
            running_sum2 += temp * (i as Real) / tau;
        }
        (running_sum1, running_sum2)
    }
}

impl AsObservable for AnalyticContinuousGeometricAveragePriceAsianHestonEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticContinuousGeometricAveragePriceAsianHestonEngine {
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
        let args = self.base.arguments();
        require!(
            args.average_type == Some(AverageType::Geometric),
            "not a geometric average option"
        );
        let exercise = Shared::clone(args.exercise.as_ref().expect("validated"));
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European Option"
        );
        let payoff = args.payoff.expect("validated");
        let strike = payoff.strike();
        let exercise_date = exercise.last_date();

        let expiry_time = self.process.time(&exercise_date)?;
        require!(expiry_time >= 0.0, "Expiry Date cannot be in the past");

        let r_ts = self.risk_free_rate.current_link()?;
        let q_ts = self.dividend_yield.current_link()?;
        let expiry_dcf = r_ts.discount(expiry_time, false)?;

        // Seasoned options deferred (Kim–Wee).
        let t = 0.0;
        let t_expiry = expiry_time;
        let tau = t_expiry - t;
        let log_s0 = self.s0.current_link()?.value()?.ln();

        let dcf = r_ts.discount(t_expiry, false)? / r_ts.discount(t, false)?;
        let qdcf = q_ts.discount(t_expiry, false)? / q_ts.discount(t, false)?;

        let denom = r_ts.discount(t, false)?.ln() - q_ts.discount(t, false)?.ln();
        let integrated_dcf = self.integrator.integrate(|u| {
            let u_dash = (0.5 + 1e-8 + 0.5 * u) * (t_expiry - t) + t;
            let r_disc = r_ts.discount(u_dash, false).expect("discount");
            let q_disc = q_ts.discount(u_dash, false).expect("discount");
            0.5 * (t_expiry - t) * (-r_disc.ln() + q_disc.ln() + denom)
        });

        self.a3 = (tau * log_s0 + integrated_dcf) / t_expiry
            - self.kappa * self.theta * self.rho * tau * tau / (2.0 * self.sigma * t_expiry)
            - self.rho * tau * self.v0 / (self.sigma * t_expiry);
        self.a4 = log_s0 * qdcf / dcf - self.rho * self.v0 / self.sigma
            + self.rho * self.kappa * self.theta * tau / self.sigma;
        self.a5 = (self.kappa * self.v0 + self.kappa * self.kappa * self.theta * tau)
            / (self.sigma * self.sigma);

        let one = Complex::new(1.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let term1 = 0.5 * (self.phi(one, zero, t_expiry, t, self.summation_cutoff).re - strike);

        let i = Complex::new(0.0, 1.0);
        let log_k = strike.ln();
        let xi_right = self.xi_right_limit;
        let cutoff = self.summation_cutoff;
        let term2 = self.integrator.integrate(|xi| {
            let xi_dash = (0.5 + 1e-8 + 0.5 * xi) * xi_right;
            let inner1 = self.phi(one + xi_dash * i, zero, t_expiry, t, cutoff);
            let inner2 = -strike * self.phi(xi_dash * i, zero, t_expiry, t, cutoff);
            0.5 * xi_right * ((inner1 + inner2) * (-xi_dash * log_k * i).exp() / (xi_dash * i)).re
        }) / PI;

        let value = match payoff.option_type() {
            OptionType::Call => expiry_dcf * (term1 + term2),
            OptionType::Put => expiry_dcf * (-term1 + term2),
        };

        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticContinuousGeometricAveragePriceAsianHestonEngine`] to `option`.
pub fn set_analytic_continuous_geometric_average_price_asian_heston_engine(
    option: &mut crate::instruments::ContinuousAveragingAsianOption,
    process: Shared<HestonProcess>,
) -> QlResult<()> {
    let engine = shared_mut(AnalyticContinuousGeometricAveragePriceAsianHestonEngine::new(
        process,
    )?) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::instruments::{ContinuousAveragingAsianOption, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month, SerialNumber};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::types::Natural;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn flat_rate(reference: Date, rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(
            shared(FlatForward::with_rate(
                reference,
                rate,
                Actual365Fixed::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>,
        )
    }

    /// Kim–Wee / Kim–Kim–Kim–Wee tables via
    /// `asianoptions.cpp` `testAnalyticContinuousGeometricAveragePriceHeston`.
    #[test]
    fn continuous_geometric_asian_heston_matches_kim_wee() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let spot = shared(SimpleQuote::new(100.0));
        let q = flat_rate(today, 0.0);
        let r = flat_rate(today, 0.05);

        let days: [Natural; 15] = [
            73, 73, 73, 73, 73, 548, 548, 548, 548, 548, 1095, 1095, 1095, 1095, 1095,
        ];
        let strikes: [Real; 15] = [
            90.0, 95.0, 100.0, 105.0, 110.0, 90.0, 95.0, 100.0, 105.0, 110.0, 90.0, 95.0, 100.0,
            105.0, 110.0,
        ];
        let prices: [Real; 15] = [
            10.6571, 6.5871, 3.4478, 1.4552, 0.4724, 16.5030, 13.7625, 11.3374, 9.2245, 7.4122,
            20.5102, 18.3060, 16.2895, 14.4531, 12.7882,
        ];
        let prices_2: [Real; 15] = [
            10.6425, 6.4362, 3.1578, 1.1936, 0.3609, 14.9955, 11.6707, 8.7767, 6.3818, 4.5118,
            18.1219, 15.2009, 12.5707, 10.2539, 8.2611,
        ];
        let tolerance = 1.0e-2;

        let process = shared(HestonProcess::new(
            r.clone(),
            q.clone(),
            quote_handle(&spot),
            0.09,
            1.15,
            0.348,
            0.39,
            -0.64,
        ));
        let engine = shared_mut(
            AnalyticContinuousGeometricAveragePriceAsianHestonEngine::new(Shared::clone(&process))
                .unwrap(),
        );

        for i in 0..15 {
            let maturity = today + days[i] as SerialNumber;
            let exercise = shared(EuropeanExercise::new(maturity));
            let payoff = PlainVanillaPayoff::new(OptionType::Call, strikes[i]);
            let mut option = ContinuousAveragingAsianOption::new(
                AverageType::Geometric,
                payoff,
                Shared::clone(&exercise) as Shared<dyn Exercise>,
                Shared::clone(&settings),
            )
            .unwrap();
            option
                .base_mut()
                .set_pricing_engine(SharedMut::clone(&engine) as SharedMut<dyn PricingEngine>);
            let calculated = option.npv().unwrap();
            let error = (calculated - prices[i]).abs();
            assert!(
                error <= tolerance,
                "table1 i={i}: calc={calculated} exp={} err={error}",
                prices[i]
            );
        }

        let process_2 = shared(HestonProcess::new(
            r.clone(),
            q.clone(),
            quote_handle(&spot),
            0.09,
            2.0,
            0.09,
            1.0,
            -0.3,
        ));
        let engine_2 = shared_mut(
            AnalyticContinuousGeometricAveragePriceAsianHestonEngine::new(Shared::clone(
                &process_2,
            ))
            .unwrap(),
        );

        for i in 0..15 {
            let maturity = today + days[i] as SerialNumber;
            let exercise = shared(EuropeanExercise::new(maturity));
            let payoff = PlainVanillaPayoff::new(OptionType::Call, strikes[i]);
            let mut option = ContinuousAveragingAsianOption::new(
                AverageType::Geometric,
                payoff,
                Shared::clone(&exercise) as Shared<dyn Exercise>,
                Shared::clone(&settings),
            )
            .unwrap();
            option
                .base_mut()
                .set_pricing_engine(SharedMut::clone(&engine_2) as SharedMut<dyn PricingEngine>);
            let calculated = option.npv().unwrap();
            let error = (calculated - prices_2[i]).abs();
            assert!(
                error <= tolerance,
                "table4 i={i}: calc={calculated} exp={} err={error}",
                prices_2[i]
            );
        }

        let days_3: [Natural; 18] = [
            30, 91, 182, 365, 730, 1095, 30, 91, 182, 365, 730, 1095, 30, 91, 182, 365, 730, 1095,
        ];
        let strikes_3: [Real; 18] = [
            90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 110.0,
            110.0, 110.0, 110.0, 110.0, 110.0,
        ];
        let tol_3: [Real; 18] = [
            2.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 2.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2,
            1.0e-2, 2.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2,
        ];
        let prices_3: [Real; 18] = [
            10.1513, 10.8175, 11.8664, 13.5931, 16.0988, 17.9475, 2.0472, 3.5735, 5.0588, 7.1132,
            9.9139, 11.9959, 0.0350, 0.4869, 1.3376, 2.8569, 5.2804, 7.2682,
        ];

        let process_3 = shared(HestonProcess::new(
            r,
            q,
            quote_handle(&spot),
            0.09,
            1.15,
            0.0348,
            0.39,
            -0.64,
        ));
        let engine_3 = shared_mut(
            AnalyticContinuousGeometricAveragePriceAsianHestonEngine::new(Shared::clone(
                &process_3,
            ))
            .unwrap(),
        );

        for i in 0..18 {
            let maturity = today + days_3[i] as SerialNumber;
            let exercise = shared(EuropeanExercise::new(maturity));
            let payoff = PlainVanillaPayoff::new(OptionType::Call, strikes_3[i]);
            let mut option = ContinuousAveragingAsianOption::new(
                AverageType::Geometric,
                payoff,
                Shared::clone(&exercise) as Shared<dyn Exercise>,
                Shared::clone(&settings),
            )
            .unwrap();
            option
                .base_mut()
                .set_pricing_engine(SharedMut::clone(&engine_3) as SharedMut<dyn PricingEngine>);
            let calculated = option.npv().unwrap();
            let error = (calculated - prices_3[i]).abs();
            assert!(
                error <= tol_3[i],
                "kkkwee i={i}: calc={calculated} exp={} err={error} tol={}",
                prices_3[i],
                tol_3[i]
            );
        }
    }
}
