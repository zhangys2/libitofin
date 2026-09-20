//! Analytic American digital engine (`analyticdigitalamericanengine`).
//!
//! At-hit via `AmericanPayoffAtHit` (fills δ). At-expiry via
//! `AmericanPayoffAtExpiry` (NPV only; QL has no at-expiry greeks). Knock-out
//! is [`AnalyticDigitalAmericanKOEngine`] (`knock_in = false`). γ/ρ stay
//! deferred (D10: `.gamma()` / `.rho()` remain `"not provided"`).

use std::any::Any;

use crate::errors::QlResult;
use crate::exercise::ExerciseType;
use crate::fail;
use crate::instruments::{
    AssetOrNothingPayoff, CashOrNothingPayoff, OneAssetOptionEngine, OneAssetOptionResults,
    OptionArguments, StrikedTypePayoff,
};
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, PricingEngine, Results};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::Shared;
use crate::types::Real;

/// Haug cash/asset-(at-hit)-or-nothing pricer (`americanpayoffathit`).
pub struct AmericanPayoffAtHit {
    spot: Real,
    k: Real,
    std_dev: Real,
    alpha: Real,
    beta: Real,
    dalpha_dd1: Real,
    dbeta_dd2: Real,
    mu_plus_lambda: Real,
    mu_minus_lambda: Real,
    in_the_money: bool,
    forward: Real,
    x: Real,
}

impl AmericanPayoffAtHit {
    pub fn new(
        spot: Real,
        discount: Real,
        q_disc: Real,
        variance: Real,
        payoff: &dyn StrikedTypePayoff,
    ) -> QlResult<Self> {
        require!(spot > 0.0, "positive spot value required");
        require!(discount > 0.0, "positive discount required");
        require!(q_disc > 0.0, "positive dividend discount required");
        require!(variance >= 0.0, "negative variance not allowed");
        let strike = payoff.strike();
        let log_h_s = (strike / spot).ln();
        let std_dev = variance.sqrt();
        let n = CumulativeNormalDistribution::standard();
        let (mu, lambda, cum_d1, cum_d2, n_d1, n_d2) = if variance >= f64::EPSILON {
            require!(discount != 0.0, "null discount not handled yet");
            let mu = (q_disc / discount).ln() / variance - 0.5;
            let lambda = (mu * mu - 2.0 * discount.ln() / variance).sqrt();
            let d1 = log_h_s / std_dev + lambda * std_dev;
            let d2 = d1 - 2.0 * lambda * std_dev;
            (
                mu,
                lambda,
                n.value(d1),
                n.value(d2),
                n.derivative(d1),
                n.derivative(d2),
            )
        } else {
            let mu = (q_disc / discount).ln() / variance - 0.5;
            let lambda = (mu * mu - 2.0 * discount.ln() / variance).sqrt();
            let cum = if log_h_s > 0.0 { 1.0 } else { 0.0 };
            (mu, lambda, cum, cum, 0.0, 0.0)
        };
        let (alpha, dalpha_dd1, beta, dbeta_dd2) = match payoff.option_type() {
            OptionType::Call if strike > spot => (1.0 - cum_d1, -n_d1, 1.0 - cum_d2, -n_d2),
            OptionType::Put if strike < spot => (cum_d1, n_d1, cum_d2, n_d2),
            OptionType::Call | OptionType::Put => (0.5, 0.0, 0.5, 0.0),
        };
        let in_the_money = matches!(
            (payoff.option_type(), strike < spot, strike > spot),
            (OptionType::Call, true, _) | (OptionType::Put, _, true)
        );
        let (forward, x) = if in_the_money {
            (1.0, 1.0)
        } else {
            (
                (strike / spot).powf(mu + lambda),
                (strike / spot).powf(mu - lambda),
            )
        };
        let any = payoff as &dyn Any;
        let k = if let Some(coo) = any.downcast_ref::<CashOrNothingPayoff>() {
            coo.cash_payoff()
        } else if any.downcast_ref::<AssetOrNothingPayoff>().is_some() {
            if in_the_money { spot } else { strike }
        } else {
            fail!("unsupported payoff type");
        };
        Ok(Self {
            spot,
            k,
            std_dev,
            alpha,
            beta,
            dalpha_dd1,
            dbeta_dd2,
            mu_plus_lambda: mu + lambda,
            mu_minus_lambda: mu - lambda,
            in_the_money,
            forward,
            x,
        })
    }

    pub fn value(&self) -> Real {
        self.k * (self.forward * self.alpha + self.x * self.beta)
    }

    /// QL `americanpayoffathit.cpp`: ITM asset-or-nothing uses `K = spot` so
    /// NPV = S, but δ does not take `dK/dS` (result 0; a spot FD is 1).
    /// Cash ITM δ = 0 is the constant cash payoff.
    pub fn delta(&self) -> Real {
        let temp = -self.spot * self.std_dev;
        let da_ds = self.dalpha_dd1 / temp;
        let db_ds = self.dbeta_dd2 / temp;
        let (df_ds, dx_ds) = if self.in_the_money {
            (0.0, 0.0)
        } else {
            (
                -self.mu_plus_lambda * self.forward / self.spot,
                -self.mu_minus_lambda * self.x / self.spot,
            )
        };
        self.k * (da_ds * self.forward + self.alpha * df_ds + db_ds * self.x + self.beta * dx_ds)
    }
}

/// Haug cash/asset-(at-expiry)-or-nothing pricer (`americanpayoffatexpiry`).
pub struct AmericanPayoffAtExpiry {
    discount: Real,
    k: Real,
    x: Real,
    y: Real,
    cum_d1: Real,
    cum_d2: Real,
}

impl AmericanPayoffAtExpiry {
    pub fn new(
        spot: Real,
        discount: Real,
        q_disc: Real,
        variance: Real,
        payoff: &dyn StrikedTypePayoff,
        knock_in: bool,
    ) -> QlResult<Self> {
        require!(spot > 0.0, "positive spot value required");
        require!(discount > 0.0, "positive discount required");
        require!(q_disc > 0.0, "positive dividend discount required");
        require!(variance >= 0.0, "negative variance not allowed");
        let std_dev = variance.sqrt();
        let strike = payoff.strike();
        let ty = payoff.option_type();
        let forward = spot * q_disc / discount;
        let mut mu = (q_disc / discount).ln() / variance - 0.5;
        let any = payoff as &dyn Any;
        let k = if let Some(coo) = any.downcast_ref::<CashOrNothingPayoff>() {
            coo.cash_payoff()
        } else if any.downcast_ref::<AssetOrNothingPayoff>().is_some() {
            mu += 1.0;
            forward
        } else {
            fail!("unsupported payoff type");
        };
        let log_h_s = (strike / spot).ln();
        let log_s_h = (spot / strike).ln();
        let (eta, phi) = match (ty, knock_in) {
            (OptionType::Call, true) => (-1.0, 1.0),
            (OptionType::Call, false) => (-1.0, -1.0),
            (OptionType::Put, true) => (1.0, -1.0),
            (OptionType::Put, false) => (1.0, 1.0),
        };
        let n = CumulativeNormalDistribution::standard();
        let (mut cum_d1, mut cum_d2) = if variance >= f64::EPSILON {
            (
                n.value(phi * (log_s_h / std_dev + mu * std_dev)),
                n.value(eta * (log_h_s / std_dev + mu * std_dev)),
            )
        } else {
            (
                if log_s_h * phi > 0.0 { 1.0 } else { 0.0 },
                if log_h_s * eta > 0.0 { 1.0 } else { 0.0 },
            )
        };
        // QL uses strike_<=spot_ (call) / strike_>=spot_ (put) here, not the
        // strict inTheMoney_ test used for X_/Y_.
        match ty {
            OptionType::Call if strike <= spot => {
                cum_d1 = if knock_in { 0.5 } else { 0.0 };
                cum_d2 = cum_d1;
            }
            OptionType::Put if strike >= spot => {
                cum_d1 = if knock_in { 0.5 } else { 0.0 };
                cum_d2 = cum_d1;
            }
            _ => {}
        }
        let in_the_money = matches!(
            (ty, strike < spot, strike > spot),
            (OptionType::Call, true, _) | (OptionType::Put, _, true)
        );
        let mut y = if in_the_money {
            1.0
        } else if cum_d2 == 0.0 {
            0.0
        } else {
            (strike / spot).powf(2.0 * mu)
        };
        if !knock_in {
            y *= -1.0;
        }
        Ok(Self {
            discount,
            k,
            x: 1.0,
            y,
            cum_d1,
            cum_d2,
        })
    }

    pub fn value(&self) -> Real {
        self.discount * self.k * (self.x * self.cum_d1 + self.y * self.cum_d2)
    }
}

/// Analytic American digital (knock-in / at-hit and at-expiry) engine.
pub struct AnalyticDigitalAmericanEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    knock_in: bool,
}

impl AnalyticDigitalAmericanEngine {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self::with_knock_in(process, true)
    }

    fn with_knock_in(process: Shared<GeneralizedBlackScholesProcess>, knock_in: bool) -> Self {
        let base =
            OneAssetOptionEngine::new(OptionArguments::default(), OneAssetOptionResults::default());
        base.register_with(process.observable());
        Self {
            base,
            process,
            knock_in,
        }
    }
}

/// Analytic American digital knock-out engine (`AnalyticDigitalAmericanKOEngine`).
pub struct AnalyticDigitalAmericanKOEngine {
    inner: AnalyticDigitalAmericanEngine,
}

impl AnalyticDigitalAmericanKOEngine {
    pub fn new(process: Shared<GeneralizedBlackScholesProcess>) -> Self {
        Self {
            inner: AnalyticDigitalAmericanEngine::with_knock_in(process, false),
        }
    }
}

impl AsObservable for AnalyticDigitalAmericanKOEngine {
    fn observable(&self) -> &Observable {
        self.inner.observable()
    }
}

impl PricingEngine for AnalyticDigitalAmericanKOEngine {
    fn arguments_mut(&mut self) -> &mut dyn Arguments {
        self.inner.arguments_mut()
    }
    fn results(&self) -> &dyn Results {
        self.inner.results()
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn calculate(&mut self) -> QlResult<()> {
        self.inner.calculate()
    }
}

impl AsObservable for AnalyticDigitalAmericanEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for AnalyticDigitalAmericanEngine {
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
        let Some(exercise) = args.exercise.as_ref() else {
            fail!("non-American exercise given");
        };
        require!(
            exercise.exercise_type() == ExerciseType::American,
            "non-American exercise given"
        );
        let vol = self.process.black_volatility().current_link()?;
        require!(
            exercise.dates()[0] <= vol.reference_date()?,
            "American option with window exercise not handled yet"
        );
        let Some(payoff) = args.payoff.as_ref() else {
            fail!("non-striked payoff given");
        };
        let spot = self.process.state_variable().current_link()?.value()?;
        require!(spot > 0.0, "negative or null underlying given");
        let last = exercise.last_date();
        let variance = vol.black_variance_date(last, payoff.strike(), false)?;
        let q_disc = self
            .process
            .dividend_yield()
            .current_link()?
            .discount_date(last, false)?;
        let rf_disc = self
            .process
            .risk_free_rate()
            .current_link()?
            .discount_date(last, false)?;
        let at_expiry = exercise.payoff_at_expiry();
        let knock_in = self.knock_in;
        if at_expiry {
            let value =
                AmericanPayoffAtExpiry::new(spot, rf_disc, q_disc, variance, &**payoff, knock_in)?
                    .value();
            self.base.results_mut().instrument.value = Some(value);
        } else {
            let pricer = AmericanPayoffAtHit::new(spot, rf_disc, q_disc, variance, &**payoff)?;
            let results = self.base.results_mut();
            results.instrument.value = Some(pricer.value());
            results.greeks.delta = Some(pricer.delta());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::AmericanExercise;
    use crate::handle::Handle;
    use crate::instrument::Instrument;
    use crate::instruments::VanillaOption;
    use crate::interestrate::Compounding;
    use crate::option::OptionType::{Call, Put};
    use crate::pricingengine::PricingEngine;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::{Quote, SimpleQuote};
    use crate::settings::Settings;
    use crate::shared::{SharedMut, shared, shared_mut};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }
    fn process(
        spot: &Shared<SimpleQuote>,
        q: Real,
        r: Real,
        v: Real,
    ) -> Shared<BlackScholesMertonProcess> {
        let yts = |rate: Real| {
            Handle::new(shared(FlatForward::with_rate(
                today(),
                rate,
                Actual360::new(),
                Compounding::Continuous,
                Frequency::Annual,
            )) as Shared<dyn YieldTermStructure>)
        };
        shared(BlackScholesMertonProcess::new(
            Handle::new(Shared::clone(spot) as Shared<dyn Quote>),
            yts(q),
            yts(r),
            Handle::new(
                shared(BlackConstantVol::new(today(), None, v, Actual360::new()))
                    as Shared<dyn BlackVolTermStructure>,
            ),
        ))
    }

    type Row = (OptionType, Real, Real, Real, Real, Real, Real, Real);
    fn price(row: Row, cash: Real, at_expiry: bool, knock_in: bool) -> Real {
        let (ty, k, s, q, r, t, v, _) = row;
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let payoff: Shared<dyn StrikedTypePayoff> = if cash > 0.0 {
            shared(CashOrNothingPayoff::new(ty, k, cash))
        } else {
            shared(AssetOrNothingPayoff::new(ty, k))
        };
        let mut opt = VanillaOption::new(
            payoff,
            shared(
                AmericanExercise::new(today(), today() + (t * 360.0).round() as i32, at_expiry)
                    .unwrap(),
            ),
            settings,
        );
        let proc = process(&shared(SimpleQuote::new(s)), q, r, v);
        let engine: SharedMut<dyn PricingEngine> = if knock_in {
            shared_mut(AnalyticDigitalAmericanEngine::new(proc)) as SharedMut<dyn PricingEngine>
        } else {
            shared_mut(AnalyticDigitalAmericanKOEngine::new(proc)) as SharedMut<dyn PricingEngine>
        };
        opt.base_mut().set_pricing_engine(engine);
        opt.npv().unwrap()
    }

    /// `digitaloption.cpp` `testCashAtHitOrNothingAmericanValues` (cash=15).
    #[rustfmt::skip]
    const CASH: &[Row] = &[
        (Put,  100.0, 105.0, 0.00, 0.10, 0.5, 0.20,  9.7264),
        (Call, 100.0,  95.0, 0.00, 0.10, 0.5, 0.20, 11.6553),
        (Call, 100.0, 105.0, 0.00, 0.10, 0.5, 0.20, 15.0000),
        (Put,  100.0,  95.0, 0.00, 0.10, 0.5, 0.20, 15.0000),
        (Put,  100.0, 105.0, 0.20, 0.10, 0.5, 0.20, 12.2715),
        (Call, 100.0,  95.0, 0.20, 0.10, 0.5, 0.20,  8.9109),
        (Call, 100.0, 105.0, 0.20, 0.10, 0.5, 0.20, 15.0000),
        (Put,  100.0,  95.0, 0.20, 0.10, 0.5, 0.20, 15.0000),
    ];
    /// `digitaloption.cpp` `testAssetAtHitOrNothingAmericanValues`.
    #[rustfmt::skip]
    const ASSET: &[Row] = &[
        (Put,  100.0, 105.0, 0.00, 0.10, 0.5, 0.20, 64.8426),
        (Call, 100.0,  95.0, 0.00, 0.10, 0.5, 0.20, 77.7017),
        (Put,  100.0, 105.0, 0.01, 0.10, 0.5, 0.20, 65.7811),
        (Call, 100.0,  95.0, 0.01, 0.10, 0.5, 0.20, 76.8858),
        (Call, 100.0, 105.0, 0.00, 0.10, 0.5, 0.20,105.0000),
        (Put,  100.0,  95.0, 0.00, 0.10, 0.5, 0.20, 95.0000),
        (Call, 100.0, 105.0, 0.01, 0.10, 0.5, 0.20,105.0000),
        (Put,  100.0,  95.0, 0.01, 0.10, 0.5, 0.20, 95.0000),
    ];

    fn check(rows: &[Row], cash: Real) {
        for &(ty, k, s, q, r, t, v, expected) in rows {
            let tol = if expected.fract() == 0.0 { 1e-16 } else { 1e-4 };
            let got = price((ty, k, s, q, r, t, v, expected), cash, false, true);
            assert!(
                (got - expected).abs() <= tol,
                "{ty:?} K={k} S={s} q={q} cash={cash}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn cash_at_hit_or_nothing_american_values() {
        check(CASH, 15.0);
    }
    #[test]
    fn asset_at_hit_or_nothing_american_values() {
        check(ASSET, 0.0);
    }

    #[test]
    fn cash_put_delta_matches_spot_fd() {
        let spot = shared(SimpleQuote::new(105.0));
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let mut opt = VanillaOption::new(
            shared(CashOrNothingPayoff::new(Put, 100.0, 15.0)),
            shared(AmericanExercise::over(today(), today() + 180).unwrap()),
            settings,
        );
        opt.base_mut()
            .set_pricing_engine(shared_mut(AnalyticDigitalAmericanEngine::new(process(
                &spot, 0.0, 0.10, 0.20,
            ))) as SharedMut<dyn PricingEngine>);
        let delta = opt.delta().unwrap();
        let h = 105.0 * 1.0e-4;
        spot.set_value(105.0 + h);
        let up = opt.npv().unwrap();
        spot.set_value(105.0 - h);
        let fd = (up - opt.npv().unwrap()) / (2.0 * h);
        assert!((delta - fd).abs() <= 1.0e-4, "δ {delta} vs FD {fd}");
    }

    #[test]
    fn itm_asset_call_delta_is_ql_zero() {
        // QL `K_=spot_` so δ=0; a relative-spot FD of this NPV is 1.
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let mut opt = VanillaOption::new(
            shared(AssetOrNothingPayoff::new(Call, 100.0)),
            shared(AmericanExercise::over(today(), today() + 180).unwrap()),
            settings,
        );
        opt.base_mut()
            .set_pricing_engine(shared_mut(AnalyticDigitalAmericanEngine::new(process(
                &shared(SimpleQuote::new(105.0)),
                0.0,
                0.10,
                0.20,
            ))) as SharedMut<dyn PricingEngine>);
        assert_eq!(opt.npv().unwrap(), 105.0);
        assert_eq!(opt.delta().unwrap(), 0.0);
    }

    /// `digitaloption.cpp` `testCashAtExpiryOrNothingAmericanValues` knock-in (cash=15).
    fn cash_expiry() -> Vec<(Row, Real)> {
        vec![
            ((Put, 100.0, 105.0, 0.00, 0.10, 0.5, 0.20, 9.3604), 1e-4),
            ((Call, 100.0, 95.0, 0.00, 0.10, 0.5, 0.20, 11.2223), 1e-4),
            (
                (
                    Call,
                    100.0,
                    105.0,
                    0.00,
                    0.10,
                    0.5,
                    0.20,
                    15.0 * (-0.05_f64).exp(),
                ),
                1e-12,
            ),
            (
                (
                    Put,
                    100.0,
                    95.0,
                    0.00,
                    0.10,
                    0.5,
                    0.20,
                    15.0 * (-0.05_f64).exp(),
                ),
                1e-12,
            ),
        ]
    }
    /// `digitaloption.cpp` `testAssetAtExpiryOrNothingAmericanValues` knock-in.
    fn asset_expiry() -> Vec<(Row, Real)> {
        vec![
            ((Put, 100.0, 105.0, 0.00, 0.10, 0.5, 0.20, 64.8426), 1e-4),
            ((Call, 100.0, 95.0, 0.00, 0.10, 0.5, 0.20, 77.7017), 1e-4),
            ((Put, 100.0, 105.0, 0.01, 0.10, 0.5, 0.20, 65.5291), 1e-4),
            ((Call, 100.0, 95.0, 0.01, 0.10, 0.5, 0.20, 76.5951), 1e-4),
            ((Call, 100.0, 105.0, 0.00, 0.10, 0.5, 0.20, 105.0000), 1e-12),
            ((Put, 100.0, 95.0, 0.00, 0.10, 0.5, 0.20, 95.0000), 1e-12),
            (
                (
                    Call,
                    100.0,
                    105.0,
                    0.01,
                    0.10,
                    0.5,
                    0.20,
                    105.0 * (-0.005_f64).exp(),
                ),
                1e-12,
            ),
            (
                (
                    Put,
                    100.0,
                    95.0,
                    0.01,
                    0.10,
                    0.5,
                    0.20,
                    95.0 * (-0.005_f64).exp(),
                ),
                1e-12,
            ),
        ]
    }

    fn check_expiry(rows: &[(Row, Real)], cash: Real, knock_in: bool) {
        for &((ty, k, s, q, r, t, v, expected), tol) in rows {
            let got = price((ty, k, s, q, r, t, v, expected), cash, true, knock_in);
            assert!(
                (got - expected).abs() <= tol,
                "{ty:?} K={k} S={s} q={q} cash={cash} knock_in={knock_in}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn cash_at_expiry_or_nothing_american_values() {
        check_expiry(&cash_expiry(), 15.0, true);
    }
    #[test]
    fn asset_at_expiry_or_nothing_american_values() {
        check_expiry(&asset_expiry(), 0.0, true);
    }

    #[test]
    fn cash_at_expiry_knock_out_haug() {
        check_expiry(
            &[
                ((Put, 100.0, 105.0, 0.00, 0.10, 0.5, 0.20, 4.9081), 1e-4),
                ((Call, 100.0, 95.0, 0.00, 0.10, 0.5, 0.20, 3.0461), 1e-4),
                ((Call, 2.37, 2.33, 0.07, 0.43, 0.19, 0.005, 0.0), 1e-4),
            ],
            15.0,
            false,
        );
    }

    #[test]
    fn asset_at_expiry_knock_out_haug() {
        check_expiry(
            &[
                ((Put, 100.0, 105.0, 0.00, 0.10, 0.5, 0.20, 40.1574), 1e-4),
                ((Call, 100.0, 95.0, 0.00, 0.10, 0.5, 0.20, 17.2983), 1e-4),
            ],
            0.0,
            false,
        );
    }

    #[test]
    fn at_expiry_knock_in_plus_out_is_prepaid() {
        let cash_put = (Put, 100.0, 105.0, 0.00, 0.10, 0.5, 0.20, 0.0);
        let ki = price(cash_put, 15.0, true, true);
        let ko = price(cash_put, 15.0, true, false);
        assert!((ki + ko - 15.0 * (-0.05_f64).exp()).abs() <= 1e-4);
        let asset_put = (Put, 100.0, 105.0, 0.00, 0.10, 0.5, 0.20, 0.0);
        let ki = price(asset_put, 0.0, true, true);
        let ko = price(asset_put, 0.0, true, false);
        assert!((ki + ko - 105.0).abs() <= 1e-4);
    }
}
