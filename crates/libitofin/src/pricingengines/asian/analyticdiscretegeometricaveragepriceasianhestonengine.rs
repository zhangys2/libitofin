//! Analytic discrete geometric-average price Asian under Heston.
//!
//! Port of `ql/experimental/asian/analytic_discr_geom_av_price_heston.{hpp,cpp}`
//! (Kim, Kim, Kim & Wee, *Bull. Korean Math. Soc.* 53, 2016).

use std::cell::RefCell;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{
    AverageType, DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults, StrikedTypePayoff,
    TypePayoff,
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

type EngineBase = GenericEngine<DiscreteAveragingAsianArguments, DiscreteAveragingAsianResults>;

/// Analytic discrete geometric average-price Asian under Heston.
pub struct AnalyticDiscreteGeometricAveragePriceAsianHestonEngine {
    base: EngineBase,
    process: Shared<HestonProcess>,
    v0: Real,
    rho: Real,
    kappa: Real,
    theta: Real,
    sigma: Real,
    log_s0: Real,
    dividend_yield: Handle<dyn YieldTermStructure>,
    risk_free_rate: Handle<dyn YieldTermStructure>,
    s0: Handle<dyn Quote>,
    omega_tilde_lookup: RefCell<HashMap<Size, Complex>>,
    xi_right_limit: Real,
    integrator: GaussianQuadrature,
    tr_t: RefCell<Real>,
    tr_t_expiry: RefCell<Real>,
    tkr_tk: RefCell<Vec<Real>>,
}

impl AnalyticDiscreteGeometricAveragePriceAsianHestonEngine {
    /// `AnalyticDiscreteGeometricAveragePriceAsianHestonEngine(process)`.
    pub fn new(process: Shared<HestonProcess>) -> QlResult<Self> {
        Self::with_xi_right_limit(process, 100.0)
    }

    /// With ξ integral right limit (eqs 23–24).
    pub fn with_xi_right_limit(process: Shared<HestonProcess>, xi_right_limit: Real) -> QlResult<Self> {
        let v0 = process.v0();
        let rho = process.rho();
        let kappa = process.kappa();
        let theta = process.theta();
        let sigma = process.sigma();
        let s0 = process.s0();
        let log_s0 = s0.current_link()?.value()?.ln();
        let risk_free_rate = process.risk_free_rate();
        let dividend_yield = process.dividend_yield();
        let base = EngineBase::new(
            DiscreteAveragingAsianArguments::default(),
            DiscreteAveragingAsianResults::default(),
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
            log_s0,
            dividend_yield,
            risk_free_rate,
            s0,
            omega_tilde_lookup: RefCell::new(HashMap::new()),
            xi_right_limit,
            integrator: GaussianQuadrature::legendre(128)?,
            tr_t: RefCell::new(0.0),
            tr_t_expiry: RefCell::new(0.0),
            tkr_tk: RefCell::new(Vec::new()),
        })
    }

    /// Φ(s, w; t, T) — Kim–Kim–Kim–Wee eq (21).
    #[allow(clippy::too_many_arguments)]
    pub fn phi(
        &self,
        s: Complex,
        w: Complex,
        t: Time,
        t_expiry: Time,
        k_star: Size,
        t_n: &[Time],
        tau_k: &[Time],
    ) -> Complex {
        self.omega_tilde_lookup.borrow_mut().clear();
        let n = t_n.len();
        let a_term = self.a(s, w, t, t_expiry, k_star, t_n);
        let omega_term = self.v0 * self.omega_tilde(s, w, k_star, k_star, n, tau_k);
        let term3 = self.kappa * self.kappa * self.theta * (t_expiry - t) / (self.sigma * self.sigma);

        let mut summation = Complex::new(0.0, 0.0);
        for i in (k_star + 1)..=(n + 1) {
            let d_tau = tau_k[i] - tau_k[i - 1];
            let z_k = self.z(s, w, i, n);
            let omega_tilde_k = self.omega_tilde(s, w, i, k_star, n, tau_k);
            summation += self.f(z_k, omega_tilde_k, d_tau).ln();
        }
        let term4 = 2.0 * self.kappa * self.theta * summation / (self.sigma * self.sigma);
        (a_term + omega_term + term3 - term4).exp()
    }

    fn f(&self, z1: Complex, z2: Complex, tau: Time) -> Complex {
        let temp = (self.kappa * self.kappa - 2.0 * z1 * self.sigma * self.sigma).sqrt();
        if (self.kappa * self.kappa - 2.0 * self.sigma * self.sigma).abs() < 1e-8 {
            Complex::new(1.0, 0.0) + 0.5 * (self.kappa - z2 * self.sigma * self.sigma)
        } else {
            (0.5 * tau * temp).cosh()
                + (self.kappa - z2 * self.sigma * self.sigma) * (0.5 * tau * temp).sinh() / temp
        }
    }

    fn f_tilde(&self, z1: Complex, z2: Complex, tau: Time) -> Complex {
        let temp = (self.kappa * self.kappa - 2.0 * z1 * self.sigma * self.sigma).sqrt();
        0.5 * temp * (0.5 * tau * temp).sinh()
            + 0.5 * (self.kappa - z2 * self.sigma * self.sigma) * (0.5 * tau * temp).cosh()
    }

    fn z(&self, s: Complex, w: Complex, k: Size, n: Size) -> Complex {
        let k_ = k as Real;
        let n_ = n as Real;
        let linear = (n_ - k_ + 1.0) * s + n_ * w;
        let term1 = (2.0 * self.rho * self.kappa - self.sigma) * linear / (2.0 * self.sigma * n_);
        let term2 = (1.0 - self.rho * self.rho) * linear * linear / (2.0 * n_ * n_);
        term1 + term2
    }

    fn omega(&self, s: Complex, w: Complex, k: Size, k_star: Size, n: Size) -> Complex {
        if k == k_star {
            Complex::new(0.0, 0.0)
        } else if k == n + 1 {
            self.rho * w / self.sigma
        } else {
            self.rho * s / (self.sigma * n as Real)
        }
    }

    fn a(
        &self,
        s: Complex,
        w: Complex,
        t: Time,
        t_expiry: Time,
        k_star: Size,
        t_n: &[Time],
    ) -> Complex {
        let k_star_ = k_star as Real;
        let n_ = t_n.len() as Real;
        let temp = -self.rho * self.kappa * self.theta / self.sigma;
        let tr_t = *self.tr_t.borrow();
        let tr_t_expiry = *self.tr_t_expiry.borrow();
        let tkr_tk = self.tkr_tk.borrow();

        let mut summation = 0.0;
        let mut summation2 = 0.0;
        for i in (k_star + 1)..=t_n.len() {
            summation += t_n[i - 1];
            summation2 += tkr_tk[i - 1];
        }
        let term1 = (s * (n_ - k_star_) / n_ + w)
            * (self.log_s0 - self.rho * self.v0 / self.sigma - t * temp - tr_t);
        let term2 = temp * (s * summation / n_ + w * t_expiry) + w * tr_t_expiry + summation2 * s / n_;
        term1 + term2
    }

    fn omega_tilde(
        &self,
        s: Complex,
        w: Complex,
        k: Size,
        k_star: Size,
        n: Size,
        tau_k: &[Time],
    ) -> Complex {
        let omega_k = self.omega(s, w, k, k_star, n);
        if k == n + 1 {
            return omega_k;
        }
        let d_tau_k = tau_k[k + 1] - tau_k[k];
        let z_kp1 = self.z(s, w, k + 1, n);
        let omega_kp1 = if let Some(value) = self.omega_tilde_lookup.borrow().get(&(k + 1)).copied()
        {
            value
        } else {
            self.omega_tilde(s, w, k + 1, k_star, n, tau_k)
        };
        let ratio = self.f_tilde(z_kp1, omega_kp1, d_tau_k) / self.f(z_kp1, omega_kp1, d_tau_k);
        let result =
            omega_k + self.kappa / (self.sigma * self.sigma) - 2.0 * ratio / (self.sigma * self.sigma);
        self.omega_tilde_lookup.borrow_mut().insert(k, result);
        result
    }
}

impl AsObservable for AnalyticDiscreteGeometricAveragePriceAsianHestonEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticDiscreteGeometricAveragePriceAsianHestonEngine {
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
        // Geometric check omitted: also used as CV for arithmetic Asians.
        let args = self.base.arguments();
        let exercise = Shared::clone(args.exercise.as_ref().expect("validated"));
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European Option"
        );

        let average_type = args.average_type.expect("validated");
        let running_accumulator = args.running_accumulator.expect("validated");
        let past_fixings = args.past_fixings.expect("validated");
        let fixing_dates = args.fixing_dates.clone();
        let payoff = args.payoff.expect("validated");

        let (running_log, past_fixings) = if average_type == AverageType::Geometric {
            require!(
                running_accumulator > 0.0,
                "positive running product required: {running_accumulator} not allowed"
            );
            (running_accumulator.ln(), past_fixings)
        } else {
            (0.0, 0)
        };

        let strike = payoff.strike();
        let exercise_date = exercise.last_date();
        let expiry_time = self.process.time(&exercise_date)?;
        require!(expiry_time >= 0.0, "Expiry Date cannot be in the past");

        let r_ts = self.risk_free_rate.current_link()?;
        let q_ts = self.dividend_yield.current_link()?;
        let expiry_dcf = r_ts.discount(expiry_time, false)?;

        let start_time = 0.0;
        let mut fixing_times: Vec<Time> = fixing_dates
            .iter()
            .map(|d| self.process.time(d))
            .collect::<QlResult<_>>()?;
        fixing_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut tau_k = fixing_times.clone();
        tau_k.insert(0, start_time);
        tau_k.push(expiry_time);

        for _ in 0..past_fixings {
            fixing_times.insert(0, -1.0);
            tau_k.insert(0, -1.0);
        }
        let k_star = past_fixings;

        *self.tr_t.borrow_mut() =
            -(r_ts.discount(start_time, false)? / q_ts.discount(start_time, false)?).ln();
        *self.tr_t_expiry.borrow_mut() =
            -(r_ts.discount(expiry_time, false)? / q_ts.discount(expiry_time, false)?).ln();
        let mut tkr = Vec::with_capacity(fixing_times.len());
        for &ft in &fixing_times {
            if ft < 0.0 {
                tkr.push(1.0);
            } else {
                tkr.push(-(r_ts.discount(ft, false)? / q_ts.discount(ft, false)?).ln());
            }
        }
        *self.tkr_tk.borrow_mut() = tkr;

        let prefactor = (running_log / fixing_times.len() as Real).exp();
        let adjusted_strike = strike / prefactor;

        let one = Complex::new(1.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let term1 = 0.5
            * (self
                .phi(one, zero, start_time, expiry_time, k_star, &fixing_times, &tau_k)
                .re
                - adjusted_strike);

        let i = Complex::new(0.0, 1.0);
        let log_k = adjusted_strike.ln();
        let xi_right = self.xi_right_limit;
        let term2 = self.integrator.integrate(|xi| {
            let xi_dash = (0.5 + 1e-8 + 0.5 * xi) * xi_right;
            let inner1 = self.phi(
                one + xi_dash * i,
                zero,
                start_time,
                expiry_time,
                k_star,
                &fixing_times,
                &tau_k,
            );
            let inner2 = -adjusted_strike
                * self.phi(
                    xi_dash * i,
                    zero,
                    start_time,
                    expiry_time,
                    k_star,
                    &fixing_times,
                    &tau_k,
                );
            0.5 * xi_right * ((inner1 + inner2) * (-xi_dash * log_k * i).exp() / (xi_dash * i)).re
        }) / PI;

        let value = match payoff.option_type() {
            OptionType::Call => expiry_dcf * prefactor * (term1 + term2),
            OptionType::Put => expiry_dcf * prefactor * (-term1 + term2),
        };

        let _ = self.s0.current_link()?; // keep quote live / match QL additionalResults
        let results = self.base.results_mut();
        results.instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`AnalyticDiscreteGeometricAveragePriceAsianHestonEngine`] to `option`.
pub fn set_analytic_discrete_geometric_average_price_asian_heston_engine(
    option: &mut crate::instruments::DiscreteAveragingAsianOption,
    process: Shared<HestonProcess>,
) -> QlResult<()> {
    let engine = shared_mut(AnalyticDiscreteGeometricAveragePriceAsianHestonEngine::new(
        process,
    )?) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{EuropeanExercise, Exercise};
    use crate::instruments::{DiscreteAveragingAsianOption, PlainVanillaPayoff};
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

    /// Kim–Kim–Kim–Wee tables via `testAnalyticDiscreteGeometricAveragePriceHeston`.
    #[test]
    fn discrete_geometric_asian_heston_matches_kim() {
        let settings = shared(Settings::new());
        let today = Date::new(15, Month::June, 2026);
        settings.set_evaluation_date(today);

        let tol: [Real; 18] = [
            3.0e-2, 2.0e-2, 2.0e-2, 2.0e-2, 3.0e-2, 4.0e-2, 8.0e-2, 1.0e-2, 2.0e-2, 3.0e-2, 3.0e-2,
            4.0e-2, 2.0e-2, 1.0e-2, 1.0e-2, 2.0e-2, 3.0e-2, 4.0e-2,
        ];
        let days: [Natural; 18] = [
            30, 91, 182, 365, 730, 1095, 30, 91, 182, 365, 730, 1095, 30, 91, 182, 365, 730, 1095,
        ];
        let strikes: [Real; 18] = [
            90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 110.0,
            110.0, 110.0, 110.0, 110.0, 110.0,
        ];
        let prices: [Real; 18] = [
            10.2732, 10.9554, 11.9916, 13.6950, 16.1773, 18.0146, 2.4389, 3.7881, 5.2132, 7.2243,
            9.9948, 12.0639, 0.1012, 0.5949, 1.4444, 2.9479, 5.3531, 7.3315,
        ];

        let spot = shared(SimpleQuote::new(100.0));
        let process = shared(HestonProcess::new(
            flat_rate(today, 0.05),
            flat_rate(today, 0.0),
            quote_handle(&spot),
            0.09,
            1.15,
            0.0348,
            0.39,
            -0.64,
        ));
        let engine = shared_mut(
            AnalyticDiscreteGeometricAveragePriceAsianHestonEngine::new(Shared::clone(&process))
                .unwrap(),
        );

        for i in 0..18 {
            let day = days[i];
            let future_fixings = (day as Real / 7.0).floor() as Size;
            let expiry = today + day as SerialNumber;
            let mut fixing_dates = vec![Date::default(); future_fixings];
            for j in (0..future_fixings).rev() {
                fixing_dates[j] = expiry - (j as SerialNumber) * 7;
            }

            let exercise = shared(EuropeanExercise::new(expiry));
            let payoff = PlainVanillaPayoff::new(OptionType::Call, strikes[i]);
            let mut option = DiscreteAveragingAsianOption::new(
                AverageType::Geometric,
                1.0,
                0,
                fixing_dates,
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
                error <= tol[i],
                "i={i} days={day}: calc={calculated} exp={} err={error} tol={}",
                prices[i],
                tol[i]
            );
        }
    }
}
