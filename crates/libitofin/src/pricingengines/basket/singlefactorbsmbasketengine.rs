//! Single-factor BSM basket engine and sum-of-exponentials root solver.
//!
//! Port of `ql/pricingengines/basket/singlefactorbsmbasketengine.{hpp,cpp}`.

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::instruments::{BasketArguments, BasketResults, StrikedTypePayoff, TypePayoff};
use crate::math::array::Array;
use crate::math::comparison::close_enough;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::math::solver1d::Solver1D;
use crate::math::solvers1d::brent::Brent;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::payoff::Payoff;
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::basket::vectorbsmprocessextractor::VectorBsmProcessExtractor;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, shared};
use crate::types::{Real, Size};

type EngineBase = GenericEngine<BasketArguments, BasketResults>;

/// Root finder for `sum(a_i * exp(sig_i * x)) = K`.
pub struct SumExponentialsRootSolver {
    a: Array,
    sig: Array,
    k: Real,
}

impl SumExponentialsRootSolver {
    pub fn new(a: Array, sig: Array, k: Real) -> QlResult<Self> {
        require!(a.size() == sig.size(), "Arrays must have the same size");
        Ok(Self { a, sig, k })
    }

    fn sum_exp(&self, x: Real) -> Real {
        self.a
            .iter()
            .zip(self.sig.iter())
            .map(|(&a, &s)| a * (s * x).exp())
            .sum()
    }

    pub fn value(&self, x: Real) -> Real {
        self.sum_exp(x) - self.k
    }

    pub fn get_root(&self, x_tol: Real) -> QlResult<Real> {
        let attr = &self.a * &self.sig;
        let log_prob = self.a.iter().all(|&x| x > 0.0);
        require!(
            self.k > 0.0 || !log_prob,
            "non-positive strikes only allowed for spread options"
        );

        let denom: Real = attr.iter().sum();
        let x_init = if denom.abs() > 1000.0 * Real::EPSILON {
            ((self.k - self.a.iter().sum::<Real>()) / denom).clamp(-10.0, 10.0)
        } else {
            0.0
        };

        let mut f = |x: Real| self.value(x);
        Brent::new().solve(&mut f, x_tol, x_init, 1.0)
    }
}

/// Basket engine where all underlyings share one stochastic factor.
pub struct SingleFactorBsmBasketEngine {
    base: EngineBase,
    x_tol: Real,
    n: Size,
    processes: Vec<Shared<GeneralizedBlackScholesProcess>>,
}

impl SingleFactorBsmBasketEngine {
    pub fn new(processes: Vec<Shared<GeneralizedBlackScholesProcess>>) -> Self {
        Self::with_tolerance(processes, 1e4 * Real::EPSILON)
    }

    pub fn with_tolerance(
        processes: Vec<Shared<GeneralizedBlackScholesProcess>>,
        x_tol: Real,
    ) -> Self {
        let n = processes.len();
        let base = EngineBase::new(BasketArguments::default(), BasketResults::default());
        for p in &processes {
            base.register_with(p.observable());
        }
        Self {
            base,
            x_tol,
            n,
            processes,
        }
    }
}

impl AsObservable for SingleFactorBsmBasketEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for SingleFactorBsmBasketEngine {
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
        let avg_payoff = self
            .base
            .arguments()
            .payoff
            .as_ref()
            .expect("validated")
            .clone();
        let payoff = avg_payoff.base_payoff();
        let strike = payoff.strike();
        let option_type = payoff.option_type();
        let weights = avg_payoff.weights();
        require!(
            self.n == weights.size(),
            "wrong number of weights arguments in payoff"
        );

        let exercise = self.base.arguments().exercise.as_ref().expect("validated");
        require!(
            exercise.exercise_type() == ExerciseType::European,
            "not an European exercise"
        );
        let maturity_date = exercise.last_date();

        let extractor = VectorBsmProcessExtractor::new(self.processes.clone());
        let s = extractor.get_spot()?;
        let dq = extractor.get_dividend_yield_df(maturity_date)?;
        let dr0 = extractor.get_interest_rate_df(maturity_date)?;
        let std_dev = extractor.get_black_std_dev(maturity_date)?;
        let v = &std_dev * &std_dev;
        let fwd_basket = &(&(weights * &s) * &dq) / dr0;

        if std_dev.iter().all(|&x| close_enough(x, 0.0)) {
            let sum: Real = fwd_basket.iter().sum();
            let results = self.base.results_mut();
            results.instrument.value = Some(dr0 * payoff.value(sum));
            return Ok(());
        }

        let solver = SumExponentialsRootSolver::new(
            &fwd_basket * &(&v * (-0.5)).exp(),
            std_dev.clone(),
            strike,
        )?;
        // QuantLib: `d = -SumExponentialsRootSolver(...).getRoot(...)`.
        let d = -solver.get_root(self.x_tol)?;
        let n_cdf = CumulativeNormalDistribution::standard();
        let cp = match option_type {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        };

        let mut acc = -strike * n_cdf.value(cp * d);
        for (fwd, sig) in fwd_basket.iter().zip(std_dev.iter()) {
            acc += fwd * n_cdf.value(cp * (d + sig));
        }

        let results = self.base.results_mut();
        results.instrument.value = Some(cp * dr0 * acc);
        results
            .instrument
            .additional_results
            .insert("d".to_string(), shared(d) as Shared<dyn Any>);
        Ok(())
    }
}
