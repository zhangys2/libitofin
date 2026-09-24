//! Chi-Fai Lo (2015) Operator Splitting pricing engine for 2D European spread options.
//!
//! Port of `ql/pricingengines/basket/operatorsplittingspreadengine.{hpp,cpp}` and
//! `ql/pricingengines/basket/spreadblackscholesvanillaengine.{hpp,cpp}`.
//!
//! References:
//! - Chi-Fai Lo, "Pricing Spread Options by the Operator Splitting Method",
//!   SSRN: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2429696>
//! - QuantLib test suite `testOperatorSplittingSpreadEngine`,
//!   `testStrangSplittingSpreadEngineVsMathematica`, and
//!   `testNoDivByZeroOperatorSplitting`.

use std::f64::consts::{PI, SQRT_2};

use crate::errors::{QlError, QlResult};
use crate::exercise::ExerciseType;
use crate::instruments::{
    BasketArguments, BasketOption, BasketPayoff, BasketResults, StrikedTypePayoff, TypePayoff,
};
use crate::math::distributions::normal::{CumulativeNormalDistribution, NormalDistribution};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::Real;

type EngineBase = GenericEngine<BasketArguments, BasketResults>;

/// Approximation order for [`OperatorSplittingSpreadEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OperatorSplittingOrder {
    /// First-order operator splitting correction to Kirk's approximation.
    First,
    /// Second-order operator splitting correction to Kirk's approximation.
    #[default]
    Second,
}

/// Evaluates Chi-Fai Lo's (2015) operator splitting spread option formula on forwards.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn operator_splitting_spread_option_value(
    forward1: Real,
    forward2: Real,
    strike: Real,
    option_type: OptionType,
    variance1: Real,
    variance2: Real,
    discount: Real,
    rho: Real,
    order: OperatorSplittingOrder,
) -> QlResult<Real> {
    require!(
        (-1.0..=1.0).contains(&rho),
        "correlation must be in [-1, 1]"
    );
    require!(forward1 > 0.0, "forward1 must be positive");
    require!(forward2 > 0.0, "forward2 must be positive");
    let denom = forward2 + strike;
    require!(denom > 0.0, "forward2 + strike must be positive");
    require!(variance1 >= 0.0, "variance1 must be non-negative");
    require!(variance2 > 0.0, "variance2 must be positive");
    require!(discount > 0.0, "discount must be positive");

    let call_put_parity_price = |call_price: Real| -> Real {
        match option_type {
            OptionType::Call => call_price,
            OptionType::Put => call_price - discount * (forward1 - forward2 - strike),
        }
    };

    let vol1 = variance1.sqrt();
    let vol2 = variance2.sqrt();
    let sig2 = vol2 * forward2 / denom;
    let sig_m_sq = variance1 + sig2 * (sig2 - 2.0 * rho * vol1);
    require!(sig_m_sq > 0.0, "total volatility must be positive");
    let sig_m = sig_m_sq.sqrt();

    let d1 = (forward1.ln() - denom.ln()) / sig_m + 0.5 * sig_m;
    let d2 = d1 - sig_m;

    let cnd = CumulativeNormalDistribution::standard();
    let norm = NormalDistribution::standard();

    let kirk_call_npv = discount * (forward1 * cnd.value(d1) - denom * cnd.value(d2));

    let vs = vol2 / (sig_m * sig_m);
    let rs = (rho * vol1 - sig2).powi(2);

    let o_plt = -sig2
        * sig2
        * strike
        * discount
        * norm.value(d2)
        * vs
        * (-d2 * rs / sig2
            - 0.5 * vs * sig_m * strike / denom * (rs * d1 * d2 + (1.0 - rho * rho) * variance1));

    if order == OperatorSplittingOrder::First {
        return Ok(call_put_parity_price(kirk_call_npv + 0.5 * o_plt));
    }

    let r2 = denom;
    let r1 = forward1 / r2;
    let vol12 = vol1 * vol1;
    let vol22 = vol2 * vol2;
    let vol23 = vol22 * vol2;

    let eps_threshold = f64::EPSILON.powf(0.625);
    let sqrt_pi = PI.sqrt();
    let sqrt_2 = SQRT_2;

    if rs < eps_threshold {
        let vol24 = vol22 * vol22;
        let vol26 = vol22 * vol24;
        let k2 = strike * strike;
        let r22 = r2 * r2;
        let r24 = r22 * r22;
        let km_r22 = (strike - r2).powi(2);
        let ln_r1 = r1.ln();

        let num = -0.0625
            * (k2
                * km_r22
                * vol26
                * (-8.0 * r22 * r24 * (7.0 * k2 - 7.0 * strike * r2 + r22) * vol12 * vol12
                    + km_r22
                        * r24
                        * vol12
                        * (-112.0 * strike * r2 + 16.0 * r22 + k2 * (124.0 + 3.0 * vol12))
                        * vol22
                    - 2.0
                        * km_r22
                        * km_r22
                        * r22
                        * (-28.0 * strike * r2 + 4.0 * r22 + k2 * (34.0 + 3.0 * vol12))
                        * vol24
                    + 3.0 * k2 * km_r22 * km_r22 * km_r22 * vol26
                    - 4.0
                        * strike
                        * (strike - r2)
                        * r24
                        * ln_r1
                        * (-4.0 * r22 * vol12
                            + 4.0 * km_r22 * vol22
                            + 3.0 * strike * (strike - r2) * vol22 * ln_r1)));

        let exp_arg = (-(r22 * vol12) + km_r22 * vol22 + 2.0 * r22 * ln_r1).powi(2)
            / (8.0 * r24 * vol12 - 8.0 * km_r22 * r22 * vol22);
        let denom_sq = (r22 * vol12 - km_r22 * vol22).powi(2);
        let rad = (vol12 - (km_r22 * vol22) / r22).max(0.0).sqrt();

        let den = exp_arg.exp() * sqrt_pi * sqrt_2 * r22 * r24 * r2 * denom_sq * rad;
        // In QuantLib C++ line 104, `df` was omitted in the Taylor branch, which matches QL test case where r=0 (df=1).
        // Multiplying by `discount` preserves continuity across `rs` threshold for non-zero rates.
        let oo_plt = discount * num / den;

        return Ok(call_put_parity_price(
            kirk_call_npv + 0.5 * o_plt + 0.125 * oo_plt,
        ));
    }

    let f2 = forward2;
    let f22 = f2 * f2;
    let f23 = f22 * f2;
    let f24 = f22 * f22;

    let i_r2 = 1.0 / r2;
    let i_r22 = i_r2 * i_r2;
    let i_r23 = i_r22 * i_r2;
    let i_r24 = i_r22 * i_r22;
    let a = vol12 - 2.0 * f2 * i_r2 * rho * vol1 * vol2 + f22 * i_r22 * vol22;
    let a2 = a * a;
    let b = a / 2.0 + r1.ln();
    let b2 = b * b;
    let c = a.sqrt();
    let d = b / c;
    let e = rho * vol1 - f2 * i_r2 * vol2;
    let e2 = e * e;
    let f = d - c;
    let g = -2.0 * i_r2 * rho * vol1 * vol2
        + 2.0 * f2 * i_r22 * rho * vol1 * vol2
        + 2.0 * f2 * i_r22 * vol22
        - 2.0 * f22 * i_r23 * vol22;
    let h = rho * rho;
    let j = 1.0 - h;
    let iat = 1.0 / c;
    let l = b * iat - c;
    let m = f * (1.0 - (r2 * rho * vol1) / (f2 * vol2))
        - (e * i_r2 * strike * (d * l + (j * vol12) / (e * e)) * vol2) / (2.0 * c);
    let n = (iat * (1.0 - (r2 * rho * vol1) / (f2 * vol2))) / r1
        - (e * i_r2 * strike * ((f * iat) / r1 + b / (a * r1)) * vol2) / (2.0 * c);
    let o = discount * (-0.5 * f * f).exp();
    let p = d * l + (j * vol12) / (e * e);
    let q = (-2.0 * j * vol12 * (-(i_r2 * vol2) + f2 * i_r22 * vol2)) / (e * e * e);
    let s = q - (b2 * g) / (2.0 * a2) - (b * f * g) / (2.0 * a * c) + (f * g) / (2.0 * c);
    let u = f * (-((rho * vol1) / (f2 * vol2)) + (r2 * rho * vol1) / (f22 * vol2));
    let v = -0.5 * (b * g * (1.0 - (r2 * rho * vol1) / (f2 * vol2))) / (a * c);
    let w = (3.0 * g * g) / (4.0 * a2 * c)
        - (4.0 * i_r22 * rho * vol1 * vol2 - 4.0 * f2 * i_r23 * rho * vol1 * vol2
            + 2.0 * i_r22 * vol22
            - 8.0 * f2 * i_r23 * vol22
            + 6.0 * f22 * i_r24 * vol22)
            / (2.0 * a * c);
    let x = u
        + v
        + (e * g * i_r2 * strike * p * vol2) / (4.0 * a * c)
        + (e * i_r22 * strike * p * vol2) / (2.0 * c)
        - (e * i_r2 * strike * s * vol2) / (2.0 * c)
        - (i_r2 * strike * p * vol2 * (-(i_r2 * vol2) + f2 * i_r22 * vol2)) / (2.0 * c);
    let y = (4.0 * i_r22 - 4.0 * f2 * i_r23) * rho * vol1 * vol2
        + (2.0 * i_r22 - 8.0 * f2 * i_r23 + 6.0 * f22 * i_r24) * vol22;
    let z = 4.0 * i_r22 * rho * vol1 * vol2 - 4.0 * f2 * i_r23 * rho * vol1 * vol2
        + 2.0 * i_r22 * vol22
        - 8.0 * f2 * i_r23 * vol22
        + 6.0 * f22 * i_r24 * vol22;

    let oo_plt = (strike
        * o
        * vol23
        * (-2.0 * c * b2 * e2 * e * (-1.0 + f * f) * f23 * f24 * g * g * i_r22 * m * vol23
            + 2.0 * b2 * e2 * e2 * f23 * f24 * g * g * i_r2 * i_r22 * strike * vol22 * vol22
            + 2.0
                * a
                * b
                * e2
                * e
                * f23
                * f22
                * g
                * i_r22
                * vol2
                * (-8.0 * e2 * f2 * i_r2 * strike * vol22 + 7.0 * f * f22 * g * m * vol22)
            - a * c
                * e2
                * e
                * f23
                * f22
                * g
                * i_r22
                * vol2
                * (4.0
                    * e
                    * f2
                    * vol2
                    * (-2.0 * b * (-1.0 + f * f) * m + e * f * i_r2 * strike * vol2)
                    + f22
                        * g
                        * (16.0 * m + e * (2.0 * f + 3.0 * b * iat) * i_r2 * strike * vol2)
                        * vol22)
            - 4.0
                * a2
                * a
                * c
                * e2
                * (e2
                    * f22
                    * vol2
                    * (4.0 * f22 * iat * i_r22 * r2 * rho * vol1
                        + 8.0 * f23 * i_r22 * n * r1 * vol2
                        - 4.0 * f24 * 3.0 * i_r23 * n * r1 * vol2
                        - f23
                            * i_r22
                            * (4.0 * iat * rho * vol1 + f22 * i_r2 * strike * p * vol23 * w))
                    + 4.0
                        * f23
                        * f22
                        * vol22
                        * vol22
                        * (i_r22 * (-2.0 * f2 * i_r2 + 3.0 * f22 * i_r22) * m
                            + f22 * (2.0 * i_r2 - 3.0 * f2 * i_r22) * i_r23 * m
                            + f22 * i_r22 * (-i_r2 + f2 * i_r22) * x)
                    + 2.0
                        * e
                        * f22
                        * (2.0 * f24 * f2 * i_r24 * n * r1 * vol23
                            + 2.0 * f * f2 * f22 * i_r22 * rho * vol1 * vol22
                            - 2.0 * f * f22 * i_r22 * r2 * rho * vol1 * vol22
                            - b * f24 * i_r22 * r2 * rho * vol1 * vol22 * w
                            - 2.0
                                * f24
                                * vol2
                                * (i_r23 * n * r1 * vol22 + 4.0 * i_r23 * m * vol22
                                    - 2.0 * i_r22 * vol22 * x)
                            + f23
                                * (2.0 * i_r22 * m * vol23
                                    + 6.0 * f22 * i_r24 * m * vol23
                                    + b * f22 * i_r22 * vol23 * w
                                    - 4.0 * f22 * i_r23 * vol23 * x)))
            + 2.0
                * a2
                * c
                * e2
                * f23
                * f22
                * vol2
                * (8.0 * f22 * g * i_r22 * (-i_r2 + f2 * i_r22) * m * vol23
                    + e2 * i_r22
                        * vol2
                        * (8.0 * f2 * g * n * r1
                            + b * f22 * iat * i_r2 * strike * vol22 * (y - z))
                    + 4.0
                        * e
                        * vol22
                        * (4.0 * f2 * g * i_r22 * m
                            + f22
                                * (-4.0 * g * i_r23 * m + 2.0 * g * i_r22 * x + i_r22 * m * z)))
            + 2.0
                * a2
                * a
                * f22
                * (-4.0 * e2 * e2 * e * f * f24 * iat * i_r24 * strike * vol23
                    + 8.0
                        * e
                        * f2
                        * f24
                        * i_r23
                        * (-i_r22 + f2 * i_r23)
                        * j
                        * strike
                        * vol12
                        * vol23
                        * vol22
                    + 12.0
                        * f2
                        * f24
                        * i_r23
                        * (i_r2 - f2 * i_r22).powi(2)
                        * j
                        * strike
                        * vol12
                        * vol23
                        * vol23
                    + e2 * e2
                        * f2
                        * vol22
                        * (2.0
                            * f24
                            * i_r22
                            * strike
                            * vol22
                            * (2.0 * (i_r23 * p - i_r22 * s) + b2 * iat * i_r2 * w)
                            + f * (4.0
                                * f22
                                * i_r22
                                * (4.0 * m + f22 * iat * i_r2 * i_r22 * strike * vol22)
                                - 4.0
                                    * f23
                                    * (6.0 * i_r23 * m + iat * i_r24 * strike * vol22
                                        - 2.0 * i_r22 * x)
                                + f24 * i_r23 * strike * vol22 * (2.0 * b * w + iat * y)))
                    - 2.0
                        * e2
                        * e
                        * f22
                        * i_r22
                        * (4.0 * f * f22 * (i_r2 - f2 * i_r22) * m * vol23
                            + f22
                                * vol22
                                * (f2
                                    * vol2
                                    * (2.0
                                        * strike
                                        * (f2 * i_r24 * p + f2 * i_r24 * p + i_r22 * s
                                            - i_r23 * (2.0 * p + f2 * s))
                                        * vol22
                                        + y
                                        - z)
                                    + r2 * rho * vol1 * (-y + z))))
            - 2.0
                * a2
                * e2
                * f23
                * (2.0
                    * e2
                    * e
                    * f23
                    * i_r22
                    * strike
                    * (2.0 * b * i_r22 + g * (-1.0 + f * iat) * i_r2)
                    * vol23
                    + 4.0
                        * b
                        * f
                        * f22
                        * f22
                        * g
                        * i_r22
                        * (-i_r2 + f2 * i_r22)
                        * m
                        * vol22
                        * vol22
                    + 2.0
                        * e2
                        * f22
                        * i_r22
                        * vol2
                        * (2.0 * b * f2 * i_r2 * (i_r2 - f2 * i_r22) * strike * vol23
                            + g * (2.0 * r2 * rho * vol1
                                + 2.0 * f2 * (-1.0 + 3.0 * f * m + b * f * n * r1) * vol2
                                + f22 * strike * (-(i_r22 * p) + i_r2 * s) * vol23))
                    + e * vol22
                        * (f2
                            * f22
                            * g
                            * i_r22
                            * (g * r2 * rho * vol1
                                + f2 * g * (-1.0 + f * m) * vol2
                                + 2.0 * f2 * i_r2 * (-i_r2 + f2 * i_r22) * strike * p * vol23)
                            + 2.0
                                * b
                                * (2.0 * f2 * f22 * g * i_r22 * rho * vol1
                                    - 2.0 * f22 * g * i_r22 * r2 * rho * vol1
                                    + 4.0 * f * f23 * g * i_r22 * m * vol2
                                    + f * f24
                                        * vol2
                                        * (-4.0 * g * i_r23 * m
                                            + 2.0 * g * i_r22 * x
                                            + i_r22 * m * z))))))
        / (16.0 * a2 * a2 * c * e2 * f23 * sqrt_2 * sqrt_pi * vol2);

    Ok(call_put_parity_price(
        kirk_call_npv + 0.5 * o_plt + 0.125 * oo_plt,
    ))
}

/// Pricing engine for spread options using Chi-Fai Lo's (2015) operator splitting method.
pub struct OperatorSplittingSpreadEngine {
    base: EngineBase,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
    order: OperatorSplittingOrder,
}

impl OperatorSplittingSpreadEngine {
    /// Creates a new `OperatorSplittingSpreadEngine` with default second-order approximation.
    pub fn new(
        process1: Shared<GeneralizedBlackScholesProcess>,
        process2: Shared<GeneralizedBlackScholesProcess>,
        rho: Real,
    ) -> QlResult<Self> {
        Self::with_order(process1, process2, rho, OperatorSplittingOrder::Second)
    }

    /// Creates a new `OperatorSplittingSpreadEngine` with explicit approximation order.
    pub fn with_order(
        process1: Shared<GeneralizedBlackScholesProcess>,
        process2: Shared<GeneralizedBlackScholesProcess>,
        rho: Real,
        order: OperatorSplittingOrder,
    ) -> QlResult<Self> {
        require!(
            (-1.0..=1.0).contains(&rho),
            "correlation must be in [-1, 1]"
        );
        let base = EngineBase::new(BasketArguments::default(), BasketResults::default());
        base.register_with(process1.observable());
        base.register_with(process2.observable());
        Ok(Self {
            base,
            process1,
            process2,
            rho,
            order,
        })
    }

    pub fn rho(&self) -> Real {
        self.rho
    }

    pub fn order(&self) -> OperatorSplittingOrder {
        self.order
    }
}

impl AsObservable for OperatorSplittingSpreadEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for OperatorSplittingSpreadEngine {
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
            "not an European exercise"
        );
        let payoff_wrapper = arguments.payoff.as_ref().expect("validated");
        let spread_payoff = match payoff_wrapper {
            BasketPayoff::Spread(p) => p,
            _ => return Err(QlError::new("spread payoff expected", file!(), line!())),
        };
        let vanilla_payoff = spread_payoff.base_payoff();
        let strike = vanilla_payoff.strike();
        let option_type = vanilla_payoff.option_type();

        let maturity = exercise.last_date();
        let s1 = self.process1.x0()?;
        let s2 = self.process2.x0()?;
        require!(s1 > 0.0, "negative or null underlying1");
        require!(s2 > 0.0, "negative or null underlying2");

        let rf1 = self.process1.risk_free_rate().current_link()?;
        let rf2 = self.process2.risk_free_rate().current_link()?;
        let risk_free_discount1 = rf1.discount_date(maturity, false)?;
        let risk_free_discount2 = rf2.discount_date(maturity, false)?;

        let q_disc1 = self
            .process1
            .dividend_yield()
            .current_link()?
            .discount_date(maturity, false)?;
        let q_disc2 = self
            .process2
            .dividend_yield()
            .current_link()?
            .discount_date(maturity, false)?;

        let forward1 = s1 * q_disc1 / risk_free_discount1;
        let forward2 = s2 * q_disc2 / risk_free_discount2;

        let vol1 = self.process1.black_volatility().current_link()?;
        let vol2 = self.process2.black_volatility().current_link()?;
        let variance1 = vol1.black_variance_date(maturity, forward1, false)?;
        let variance2 = vol2.black_variance_date(maturity, forward2, false)?;

        let df = risk_free_discount1;

        let value = operator_splitting_spread_option_value(
            forward1,
            forward2,
            strike,
            option_type,
            variance1,
            variance2,
            df,
            self.rho,
            self.order,
        )?;

        self.base.results_mut().instrument.value = Some(value);
        Ok(())
    }
}

/// Attaches [`OperatorSplittingSpreadEngine`] with default second-order approximation to `option`.
pub fn set_operator_splitting_engine(
    option: &mut BasketOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
) -> QlResult<()> {
    set_operator_splitting_engine_with_order(
        option,
        process1,
        process2,
        rho,
        OperatorSplittingOrder::Second,
    )
}

/// Attaches [`OperatorSplittingSpreadEngine`] with specified order to `option`.
pub fn set_operator_splitting_engine_with_order(
    option: &mut BasketOption,
    process1: Shared<GeneralizedBlackScholesProcess>,
    process2: Shared<GeneralizedBlackScholesProcess>,
    rho: Real,
    order: OperatorSplittingOrder,
) -> QlResult<()> {
    let engine = shared_mut(OperatorSplittingSpreadEngine::with_order(
        process1, process2, rho, order,
    )?) as SharedMut<dyn PricingEngine>;
    option.base_mut().set_pricing_engine(engine);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::{AmericanExercise, EuropeanExercise, Exercise};
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::{MinBasketPayoff, PlainVanillaPayoff, SpreadBasketPayoff};
    use crate::interestrate::Compounding;
    use crate::pricingengines::basket::set_kirk_engine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::settings::Settings;
    use crate::shared::shared;
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::time::period::Period;
    use crate::time::timeunit::TimeUnit;

    fn flat_rate_365(d: Date, r: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            d,
            Handle::new(shared(SimpleQuote::new(r))),
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn make_process_365(
        date: Date,
        spot: Real,
        q: Real,
        r: Real,
        vol: Real,
    ) -> Shared<GeneralizedBlackScholesProcess> {
        let vol_ts: Handle<dyn BlackVolTermStructure> =
            Handle::new(shared(BlackConstantVol::with_quote(
                date,
                None,
                Handle::new(shared(SimpleQuote::new(vol))),
                Actual365Fixed::new(),
            )) as Shared<dyn BlackVolTermStructure>);
        shared(BlackScholesMertonProcess::new(
            Handle::new(shared(SimpleQuote::new(spot))),
            flat_rate_365(date, q),
            flat_rate_365(date, r),
            vol_ts,
        ))
    }

    /// Replicates QuantLib `testOperatorSplittingSpreadEngine` from `basketoption.cpp:1148`:
    /// Tests First and Second order operator splitting against Chi-Fai Lo reference values across rho in [-0.9, 0.9].
    #[test]
    fn test_operator_splitting_spread_engine() {
        let settings = shared(Settings::new());
        let today = Date::new(1, Month::March, 2025);
        settings.set_evaluation_date(today);

        let maturity = today + Period::new(12, TimeUnit::Months);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));

        let s1 = 110.0;
        let s2 = 90.0;
        let r = 0.05;
        let q1 = 0.03;
        let q2 = 0.02;
        let vol1 = 0.3;
        let vol2 = 0.2;
        let strike = 20.0;

        let p1 = make_process_365(today, s1, q1, r, vol1);
        let p2 = make_process_365(today, s2, q2, r, vol2);

        #[rustfmt::skip]
        let test_data: [(Real, Real, Real); 15] = [
            (-0.9, 18.9323, 18.9361),
            (-0.7, 18.0092, 18.012),
            (-0.5, 17.0325, 17.0344),
            (-0.4, 16.5211, 16.5227),
            (-0.3, 15.9925, 15.9937),
            (-0.2, 15.4449, 15.4458),
            (-0.1, 14.8762, 14.8768),
            ( 0.0, 14.284,  14.2843),
            ( 0.1, 13.6651, 13.6654),
            ( 0.2, 13.016,  13.0161),
            ( 0.3, 12.3319, 12.3319),
            ( 0.4, 11.6067, 11.6067),
            ( 0.5, 10.8323, 10.8323),
            ( 0.7,  9.0863,  9.0862),
            ( 0.9,  6.9148,  6.9134),
        ];

        for (rho, exp_first, exp_second) in test_data {
            // First order check
            let payoff = SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike));
            let mut opt_first =
                BasketOption::new(payoff, Shared::clone(&exercise), Shared::clone(&settings));
            set_operator_splitting_engine_with_order(
                &mut opt_first,
                p1.clone(),
                p2.clone(),
                rho,
                OperatorSplittingOrder::First,
            )
            .unwrap();
            let calc_first = opt_first.npv().unwrap();
            let diff_first = (calc_first - exp_first).abs();
            assert!(
                diff_first <= 0.0001,
                "First order failed for rho={rho}: calc={calc_first}, exp={exp_first}, diff={diff_first}"
            );

            // Second order check
            let payoff2 =
                SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike));
            let mut opt_second =
                BasketOption::new(payoff2, Shared::clone(&exercise), Shared::clone(&settings));
            set_operator_splitting_engine_with_order(
                &mut opt_second,
                p1.clone(),
                p2.clone(),
                rho,
                OperatorSplittingOrder::Second,
            )
            .unwrap();
            let calc_second = opt_second.npv().unwrap();
            let diff_second = (calc_second - exp_second).abs();
            assert!(
                diff_second <= 0.0005,
                "Second order failed for rho={rho}: calc={calc_second}, exp={exp_second}, diff={diff_second}"
            );
        }
    }

    /// Replicates QuantLib `testStrangSplittingSpreadEngineVsMathematica` from `basketoption.cpp:1244`:
    /// Tests Kirk, First order (strang1), and Second order (strang2) against Mathematica reference values @ 100*EPSILON.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_strang_splitting_spread_engine_vs_mathematica() {
        type Row = (
            Real, // T
            Real, // K
            Real, // vol1
            Real, // rho
            Real, // kirkNPV
            Real, // strang1
            Real, // strang2
        );

        #[rustfmt::skip]
        let test_cases: [Row; 15] = [
            (5.0, 20.0, 0.1,  0.6, 15.39520956886349, 15.39641179190707, 15.41992212706643),
            (10., 20.0, 0.1,  0.6, 22.91537136258191, 22.89480115264337, 22.95919510928365),
            (20., 20.0, 0.1,  0.6, 33.69859018569740, 33.59697949481467, 33.73582501903848),
            (1.0, 20.0, 0.3,  0.6, 10.97517111578040, 10.97662152028116, 10.97661321814579),
            (2.0, 20.0, 0.3,  0.6, 15.68896063758723, 15.69277461480688, 15.69275497617036),
            (3.0, 20.0, 0.3,  0.6, 19.33110275816226, 19.33760645637910, 19.33758123861756),
            (4.0, 20.0, 0.3,  0.6, 22.40185479100672, 22.41113452093983, 22.41111679131122),
            (5.0, 20.0, 0.3,  0.6, 25.09737848235137, 25.10937922536118, 25.10938819057636),
            (1.0, 10.0, 0.3,  0.6, 16.10447007803242, 16.10494344785443, 16.10494658134660),
            (1.0, 40.0, 0.3,  0.6,  4.657519189575983, 4.657079657030094, 4.656973008981588),
            (1.0, 60.0, 0.3,  0.6,  1.837359067901817, 1.831230481909945, 1.831241843743509),
            (1.0, 20.0, 0.5,  0.6, 18.79838447214884, 18.79674735337080, 18.79654551825391),
            (1.0, 20.0, 0.3, -0.9, 20.17112122874686, 20.14780367419582, 20.15151348149147),
            (1.0, 20.0, 0.3,  0.0, 15.38036208157481, 15.37697052349819, 15.37728179978961),
            (2.0, 20.0, 0.3, -0.5, 25.80847626931109, 25.77323435009942, 25.77810550213640),
        ];

        let s1 = 110.0;
        let s2 = 90.0;
        let r = 0.05;
        let vol2 = 0.20;

        let settings = shared(Settings::new());
        let today = Date::new(27, Month::May, 2024);
        settings.set_evaluation_date(today);

        let tol = 100.0 * f64::EPSILON;

        for (t, strike, vol1, rho, exp_kirk, exp_strang1, exp_strang2) in test_cases {
            let maturity_date = today + (t * 365.0).round() as i32;
            let df = (-r * t).exp();
            let f1 = s1 / df;
            let f2 = s2 / df;

            let p1 = make_process_365(today, f1, r, r, vol1);
            let p2 = make_process_365(today, f2, r, r, vol2);

            let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity_date));

            // Kirk verification
            let mut opt_kirk = BasketOption::new(
                SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike)),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            );
            set_kirk_engine(&mut opt_kirk, p1.clone(), p2.clone(), rho).unwrap();
            let calc_kirk = opt_kirk.npv().unwrap();
            let diff_kirk = (calc_kirk - exp_kirk).abs();
            assert!(
                diff_kirk <= tol * exp_kirk,
                "Kirk failed on T={t}, K={strike}: calc={calc_kirk}, exp={exp_kirk}"
            );

            // First-order Strang splitting
            let mut opt_strang1 = BasketOption::new(
                SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike)),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            );
            set_operator_splitting_engine_with_order(
                &mut opt_strang1,
                p1.clone(),
                p2.clone(),
                rho,
                OperatorSplittingOrder::First,
            )
            .unwrap();
            let calc_strang1 = opt_strang1.npv().unwrap();
            let diff_strang1 = (calc_strang1 - exp_strang1).abs();
            assert!(
                diff_strang1 <= tol * exp_strang1,
                "Strang1 failed on T={t}, K={strike}: calc={calc_strang1}, exp={exp_strang1}, diff={diff_strang1}"
            );

            // Second-order Strang splitting
            let mut opt_strang2 = BasketOption::new(
                SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike)),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            );
            set_operator_splitting_engine_with_order(
                &mut opt_strang2,
                p1,
                p2,
                rho,
                OperatorSplittingOrder::Second,
            )
            .unwrap();
            let calc_strang2 = opt_strang2.npv().unwrap();
            let diff_strang2 = (calc_strang2 - exp_strang2).abs();
            assert!(
                diff_strang2 <= tol * exp_strang2,
                "Strang2 failed on T={t}, K={strike}: calc={calc_strang2}, exp={exp_strang2}, diff={diff_strang2}"
            );
        }
    }

    /// Replicates QuantLib `testNoDivByZeroOperatorSplitting` from `basketoption.cpp:2558`:
    /// Tests Put spread option continuity when rho*vol1 == sig2 (rs -> 0) in second order engine.
    #[test]
    fn test_no_div_by_zero_operator_splitting() {
        let settings = shared(Settings::new());
        let today = Date::new(5, Month::December, 2024);
        settings.set_evaluation_date(today);

        let maturity = today + Period::new(18, TimeUnit::Months);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));

        let strike = 50.0;
        let spot1 = 160.0;
        let spot2 = 100.0;
        let vol1 = 0.25 * 3.0; // 0.75
        let rho = 1.0 / 3.0;
        let r = 0.0; // zeroFlat rate in QL test

        let p1 = make_process_365(today, spot1, 0.0, r, vol1);

        let make_option = |v: Real| -> BasketOption {
            let vol2 = v * 150.0 / 100.0;
            let p2 = make_process_365(today, spot2, 0.0, r, vol2);
            let mut opt = BasketOption::new(
                SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Put, strike)),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            );
            set_operator_splitting_engine_with_order(
                &mut opt,
                p1.clone(),
                p2,
                rho,
                OperatorSplittingOrder::Second,
            )
            .unwrap();
            opt
        };

        let eps = 1e-5;
        let mut l_opt = make_option(0.25 - eps);
        let l_npv = l_opt.npv().unwrap();

        let mut r_opt = make_option(0.25 + eps);
        let r_npv = r_opt.npv().unwrap();

        let expected = 0.5 * (l_npv + r_npv);

        let mut center_opt = make_option(0.25);
        let calculated = center_opt.npv().unwrap();

        let diff = (calculated - expected).abs();
        let tol = 5e-8;
        assert!(
            !calculated.is_nan() && diff <= tol,
            "failed continuity check: calc={calculated}, exp={expected}, diff={diff}, tol={tol}"
        );
    }

    #[test]
    fn test_operator_splitting_validation_rejects_invalid_inputs() {
        let settings = shared(Settings::new());
        let today = Date::new(1, Month::March, 2025);
        settings.set_evaluation_date(today);

        let p1 = make_process_365(today, 100.0, 0.0, 0.05, 0.2);
        let p2 = make_process_365(today, 100.0, 0.0, 0.05, 0.2);

        // Invalid correlation
        assert!(OperatorSplittingSpreadEngine::new(p1.clone(), p2.clone(), 1.5).is_err());
        assert!(OperatorSplittingSpreadEngine::new(p1.clone(), p2.clone(), -1.5).is_err());

        // Non-European exercise
        let american_ex: Shared<dyn Exercise> =
            shared(AmericanExercise::new(today, today + 365, false).unwrap());
        let mut opt_american = BasketOption::new(
            SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 10.0)),
            american_ex,
            Shared::clone(&settings),
        );
        set_operator_splitting_engine(&mut opt_american, p1.clone(), p2.clone(), 0.5).unwrap();
        assert!(opt_american.npv().is_err());

        // Non-spread payoff
        let european_ex: Shared<dyn Exercise> = shared(EuropeanExercise::new(today + 365));
        let mut opt_min = BasketOption::new(
            MinBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, 10.0)),
            european_ex,
            Shared::clone(&settings),
        );
        set_operator_splitting_engine(&mut opt_min, p1, p2, 0.5).unwrap();
        assert!(opt_min.npv().is_err());
    }

    #[test]
    fn test_operator_splitting_put_call_parity_and_dividend() {
        let settings = shared(Settings::new());
        let today = Date::new(1, Month::March, 2025);
        settings.set_evaluation_date(today);

        let maturity = today + Period::new(12, TimeUnit::Months);
        let exercise: Shared<dyn Exercise> = shared(EuropeanExercise::new(maturity));

        let s1 = 110.0;
        let s2 = 90.0;
        let r = 0.05;
        let q1 = 0.04;
        let q2 = 0.02;
        let vol1 = 0.30;
        let vol2 = 0.25;
        let strike = 15.0;
        let rho = 0.40;

        let p1 = make_process_365(today, s1, q1, r, vol1);
        let p2 = make_process_365(today, s2, q2, r, vol2);

        for order in [
            OperatorSplittingOrder::First,
            OperatorSplittingOrder::Second,
        ] {
            let mut call_opt = BasketOption::new(
                SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Call, strike)),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            );
            set_operator_splitting_engine_with_order(
                &mut call_opt,
                p1.clone(),
                p2.clone(),
                rho,
                order,
            )
            .unwrap();
            let call_npv = call_opt.npv().unwrap();

            let mut put_opt = BasketOption::new(
                SpreadBasketPayoff::new(PlainVanillaPayoff::new(OptionType::Put, strike)),
                Shared::clone(&exercise),
                Shared::clone(&settings),
            );
            set_operator_splitting_engine_with_order(
                &mut put_opt,
                p1.clone(),
                p2.clone(),
                rho,
                order,
            )
            .unwrap();
            let put_npv = put_opt.npv().unwrap();

            let df = (-r * 1.0).exp();
            let f1 = s1 * (-q1 * 1.0).exp() / df;
            let f2 = s2 * (-q2 * 1.0).exp() / df;
            let expected_parity = df * (f1 - f2 - strike);
            let parity_diff = (call_npv - put_npv) - expected_parity;
            assert!(
                parity_diff.abs() < 1e-10,
                "Parity failed for order {:?}: C-P={}, exp={}",
                order,
                call_npv - put_npv,
                expected_parity
            );
        }
    }
}
