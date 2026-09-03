//! Jaehyuk Choi basket pricing engine.
//!
//! Port of `ql/pricingengines/basket/choibasketengine.{hpp,cpp}`.

use std::any::Any;
use std::f64::consts::{PI, SQRT_2};

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instruments::{BasketArguments, BasketOption, BasketResults, TypePayoff};
use crate::math::array::Array;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::math::integrals::gaussianquadratures::{
    GaussianQuadrature, MultiDimGaussianIntegration,
};
use crate::math::matrix::Matrix;
use crate::math::matrixutilities::Svd;
use crate::math::matrixutilities::cholesky_decomposition;
use crate::math::matrixutilities::getcovariance::get_covariance;
use crate::math::matrixutilities::householder::{HouseholderReflection, HouseholderTransformation};
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::basket::singlefactorbsmbasketengine::SingleFactorBsmBasketEngine;
use crate::pricingengines::basket::vectorbsmprocessextractor::VectorBsmProcessExtractor;
use crate::processes::{BlackProcess, GeneralizedBlackScholesProcess};
use crate::quotes::{Quote, SimpleQuote};
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
use crate::time::date::Date;
use crate::types::{Real, Size};

type EngineBase = GenericEngine<BasketArguments, BasketResults>;

fn sign(x: Real) -> Real {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

fn lround_size(x: Real) -> Size {
    x.round() as i64 as Size
}

/// Choi (2018) sum-of-all-BSM basket engine.
pub struct ChoiBasketEngine {
    base: EngineBase,
    n: Size,
    processes: Vec<Shared<GeneralizedBlackScholesProcess>>,
    rho: Matrix,
    lambda: Real,
    max_nr_integration_steps: Size,
    calc_fwd_delta: bool,
    control_variate: bool,
    settings: Shared<Settings<Date>>,
}

impl ChoiBasketEngine {
    pub fn new(
        processes: Vec<Shared<GeneralizedBlackScholesProcess>>,
        rho: Matrix,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        Self::with_params(processes, rho, 10.0, Size::MAX, false, false, settings)
    }

    pub fn with_params(
        processes: Vec<Shared<GeneralizedBlackScholesProcess>>,
        rho: Matrix,
        lambda: Real,
        max_nr_integration_steps: Size,
        calc_fwd_delta: bool,
        control_variate: bool,
        settings: Shared<Settings<Date>>,
    ) -> Self {
        let n = processes.len();
        let calc_fwd_delta = calc_fwd_delta || control_variate;
        let base = EngineBase::new(BasketArguments::default(), BasketResults::default());
        for p in &processes {
            base.register_with(p.observable());
        }
        Self {
            base,
            n,
            processes,
            rho,
            lambda,
            max_nr_integration_steps,
            calc_fwd_delta,
            control_variate,
            settings,
        }
    }
}

impl AsObservable for ChoiBasketEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for ChoiBasketEngine {
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
        require!(self.n > 0, "No Black-Scholes process is given.");
        require!(
            self.n == self.rho.rows() && self.rho.rows() == self.rho.columns(),
            "process and correlation matrix must have the same size."
        );
        require!(self.lambda > 0.0, "lambda must be positive");

        let args = self.base.arguments();
        let exercise = args.exercise.as_ref().expect("validated");
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European exercise"
        );
        let maturity_date = exercise.last_date();

        let extractor = VectorBsmProcessExtractor::new(self.processes.clone());
        let s = extractor.get_spot()?;
        let dq = extractor.get_dividend_yield_df(maturity_date)?;
        let dr0 = extractor.get_interest_rate_df(maturity_date)?;

        let mut std_dev = extractor.get_black_variance(maturity_date)?.sqrt();
        for x in std_dev.iter_mut() {
            *x = x.max(Real::EPSILON * Real::EPSILON);
        }

        let fwd = &(&s * &dq) / dr0;

        let avg_payoff = args.payoff.as_ref().expect("validated");
        let weights = avg_payoff.weights().clone();
        require!(
            self.n == weights.size() && self.n > 1,
            "wrong number of weights arguments in payoff"
        );

        let weighted_fwd = &weights * &fwd;
        let g = &weighted_fwd / weighted_fwd.norm2();

        let sigma = get_covariance(&std_dev, &self.rho, 1e-12)?;
        let mut v_star1: Array = &sigma * &g;
        let norm = (g.dot(&v_star1)).sqrt();
        v_star1 = &v_star1 / norm;

        let c = cholesky_decomposition(&sigma, false);
        let eps = 100.0 * Real::EPSILON.sqrt();
        let tol = 100.0 * Real::EPSILON.sqrt();

        let mut flip = false;
        for i in 0..self.n {
            if sign(g[i]) * v_star1[i] < tol * std_dev[i] {
                flip = true;
                v_star1[i] = eps * sign(g[i]) * std_dev[i];
            }
        }

        let mut q1 = Array::with_size(self.n);
        if flip {
            for i in 0..self.n {
                let partial: Real = (0..i).map(|j| c[(i, j)] * q1[j]).sum();
                q1[i] = (v_star1[i] - partial) / c[(i, i)];
            }
            v_star1 = &v_star1 / q1.norm2();
        } else {
            q1 = &c.transpose() * &g;
        }
        q1 = &q1 / q1.norm2();

        let mut e1 = Array::with_size(self.n);
        e1[0] = 1.0;
        let r =
            HouseholderTransformation::new(HouseholderReflection::new(e1).reflection_vector(&q1)?)
                .matrix();

        let mut r_2_n = Matrix::with_size(self.n, self.n - 1);
        for i in 0..self.n {
            for j in 0..self.n - 1 {
                r_2_n[(i, j)] = r[(i, j + 1)];
            }
        }

        let svd = Svd::new(&(&c * &r_2_n));
        let u = svd.u();
        let sv = svd.singular_values();

        let mut v_mat = Matrix::with_size(self.n, self.n - 1);
        for i in 0..self.n - 1 {
            for row in 0..self.n {
                v_mat[(row, i)] = sv[i] * u[(row, i)];
            }
        }

        let mut n_int_order = vec![0usize; self.n - 1];
        let mut lambda = self.lambda;
        let alpha = 1.0 / g.dot(&v_star1).abs();
        loop {
            let int_scale = lambda * alpha;
            for i in 0..self.n - 1 {
                n_int_order[i] = lround_size(1.0 + int_scale * sv[i]).max(1);
            }
            lambda *= 0.9;
            require!(
                lambda / self.lambda > 1e-10,
                "can not rescale lambda to fit max integration order"
            );
            let product: Size = n_int_order.iter().product();
            if product <= self.max_nr_integration_steps {
                break;
            }
        }

        let mut vq = Array::with_size(self.n);
        for i in 0..self.n {
            vq[i] = 0.5 * v_mat.row(i).iter().map(|x| x * x).sum::<Real>();
        }

        let quotes: Vec<Shared<SimpleQuote>> =
            fwd.iter().map(|&f| shared(SimpleQuote::new(f))).collect();

        let mut inner_processes = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let vol_ts = self.processes[i].black_volatility().current_link()?;
            let vol_ref = vol_ts.reference_date()?;
            let vol_dc = vol_ts.day_counter().expect("vol day counter");
            let t = vol_ts.time_from_reference(maturity_date)?;
            let vol = v_star1[i] / t.sqrt();
            let vol_quote = shared(SimpleQuote::new(vol));
            let r_ts = self.processes[i].risk_free_rate();
            inner_processes.push(shared(BlackProcess::new(
                Handle::new(Shared::clone(&quotes[i]) as Shared<dyn Quote>),
                Handle::clone(&r_ts),
                Handle::clone(&r_ts),
                Handle::new(shared(BlackConstantVol::with_quote(
                    vol_ref,
                    None,
                    Handle::new(Shared::clone(&vol_quote) as Shared<dyn Quote>),
                    vol_dc.clone(),
                )) as Shared<dyn BlackVolTermStructure>),
            )));
        }

        let mut basket = BasketOption::new(
            avg_payoff.clone(),
            Shared::clone(args.exercise.as_ref().expect("validated")),
            Shared::clone(&self.settings),
        );
        basket
            .base_mut()
            .set_pricing_engine_silent(
                shared_mut(SingleFactorBsmBasketEngine::new(inner_processes))
                    as SharedMut<dyn PricingEngine>,
            );

        let ghq = MultiDimGaussianIntegration::new(&n_int_order, |order| {
            GaussianQuadrature::hermite(order, 0.0)
        })?;
        let norm_factor = PI.powf(-0.5 * n_int_order.len() as Real);

        let mut d_store = Vec::with_capacity(ghq.weights().size());
        let mut value = 0.0;
        for (weight, z) in ghq.weights().iter().zip(ghq.abscissas()) {
            let vz = &v_mat * z;
            let f = &((&(&vz * (-SQRT_2)) - &vq).exp()) * &fwd;
            for (quote, &fi) in quotes.iter().zip(f.iter()) {
                quote.set_value(fi);
            }
            basket.recalculate()?;
            let npv = basket.npv()?;
            if self.calc_fwd_delta {
                d_store.push(basket.result::<Real>("d")?);
            }
            value += weight * (-z.dot(z)).exp() * npv;
        }
        value *= norm_factor;

        if self.calc_fwd_delta {
            let payoff = avg_payoff.base_payoff();
            let put_indicator = match payoff.option_type() {
                crate::option::OptionType::Call => 0.0,
                crate::option::OptionType::Put => -1.0,
            };
            let n_cdf = CumulativeNormalDistribution::standard();
            let mut fwd_delta = Array::with_size(self.n);

            for k in 0..self.n {
                let mut d_store_counter = 0usize;
                fwd_delta[k] = {
                    let mut sum = 0.0;
                    for (weight, z) in ghq.weights().iter().zip(ghq.abscissas()) {
                        let d = d_store[d_store_counter];
                        d_store_counter += 1;
                        let vz: Real = (0..self.n - 1).map(|j| v_mat[(k, j)] * z[j]).sum();
                        let f = (-SQRT_2 * vz - vq[k]).exp();
                        sum += weight * (-z.dot(z)).exp() * f * n_cdf.value(d + v_star1[k]);
                    }
                    sum * norm_factor + put_indicator
                };
                fwd_delta[k] *= dr0 * weights[k];

                let delta_name = format!("forwardDelta {k}");
                self.base
                    .results_mut()
                    .instrument
                    .additional_results
                    .insert(delta_name, shared(fwd_delta[k]) as Shared<dyn Any>);
            }

            if self.control_variate {
                let mut f_hat = Array::with_size(self.n);
                for k in 0..self.n {
                    let mut sum = 0.0;
                    for (weight, z) in ghq.weights().iter().zip(ghq.abscissas()) {
                        let vz: Real = (0..self.n - 1).map(|j| v_mat[(k, j)] * z[j]).sum();
                        let f = (-SQRT_2 * vz - vq[k]).exp();
                        sum += weight * (-z.dot(z)).exp() * f;
                    }
                    f_hat[k] = sum * norm_factor;
                }
                let cv = &(&fwd_delta * &fwd) * &(&f_hat - &Array::filled(self.n, 1.0));
                value -= cv.iter().sum::<Real>();
            }
        }

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{AverageBasketPayoff, PlainVanillaPayoff};
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
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
    use crate::types::Volatility;

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn crate::quotes::Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn crate::quotes::Quote>)
    }

    fn flat_rate(reference: Date, rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            reference,
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_vol(reference: Date, vol: Volatility) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(shared(BlackConstantVol::new(
            reference,
            None,
            vol,
            Actual365Fixed::new(),
        )) as Shared<dyn BlackVolTermStructure>)
    }

    /// `basketoption.cpp` `testGoldenChoiBasketEngineExample` (NPV @ 1e-5).
    #[test]
    fn golden_choi_basket_engine_example() {
        let settings = shared(Settings::new());
        let today = Date::new(26, Month::September, 2024);
        settings.set_evaluation_date(today);
        let r_ts = flat_rate(today, 0.05);
        let maturity = today + Period::new(18, TimeUnit::Months);
        let strike = 20.0;

        let spots: Vec<_> = [100.0, 50.0, 75.0, 25.0]
            .into_iter()
            .map(|s| shared(SimpleQuote::new(s)))
            .collect();

        let processes = vec![
            shared(BlackScholesMertonProcess::new(
                quote_handle(&spots[0]),
                flat_rate(today, 0.075),
                Handle::clone(&r_ts),
                flat_vol(today, 0.45),
            )),
            shared(BlackScholesMertonProcess::new(
                quote_handle(&spots[1]),
                flat_rate(today, 0.035),
                Handle::clone(&r_ts),
                flat_vol(today, 0.4),
            )),
            shared(BlackScholesMertonProcess::new(
                quote_handle(&spots[2]),
                flat_rate(today, 0.08),
                Handle::clone(&r_ts),
                flat_vol(today, 0.35),
            )),
            shared(BlackScholesMertonProcess::new(
                quote_handle(&spots[3]),
                flat_rate(today, 0.02),
                Handle::clone(&r_ts),
                flat_vol(today, 0.2),
            )),
        ];

        let rho = Matrix::from([
            [1.0, 0.2, 0.3, 0.0],
            [0.2, 1.0, -0.3, 0.1],
            [0.3, -0.3, 1.0, 0.7],
            [0.0, 0.1, 0.7, 1.0],
        ]);

        let engine = shared_mut(ChoiBasketEngine::with_params(
            processes,
            rho,
            7.0,
            10000,
            true,
            true,
            Shared::clone(&settings),
        )) as SharedMut<dyn PricingEngine>;

        let cases = [
            (OptionType::Put, 15.92008513388834),
            (OptionType::Call, 22.36122704630282),
        ];

        for (option_type, expected) in cases {
            let mut option = BasketOption::new(
                AverageBasketPayoff::new(
                    PlainVanillaPayoff::new(option_type, strike),
                    Array::from([1.0, -2.0, -1.0, 4.0]),
                ),
                shared(EuropeanExercise::new(maturity)),
                Shared::clone(&settings),
            );
            option
                .base_mut()
                .set_pricing_engine(SharedMut::clone(&engine));
            let calculated = option.npv().unwrap();
            assert!(
                (calculated - expected).abs() <= 1e-5,
                "{option_type}: expected={expected}, calculated={calculated}"
            );
        }
    }
}
