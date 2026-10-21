//! Carr–Madan default-adjusted local volatility.
//!
//! Working-paper eq. 10, with the factor 2 on the jump term. The working
//! paper's eq. 11 drops that factor; it disagrees with eq. 10 and with the
//! Tanaka–Meyer derivation, so this surface follows eq. 10 (see the spec, §3).
//!
//! `σ²(K, T) = σ²_Dupire(K, T) − 2 f(T) M_p(K, T) / (K C_KK(K, T))`.
//! A zero scale returns the Dupire value unchanged. A non-positive result is
//! an error, not a floor.

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::models::equity::jumptodefault::{
    BlackVolCallSurface, CallSurface, MomentGrid, PowerHazard, truncated_power_price,
};
use crate::patterns::observable::{AsObservable, Observable};
use crate::quotes::Quote;
use crate::shared::Shared;
use crate::termstructures::volatility::{
    BlackVolTermStructure, LocalVolSurface, VolatilityTermStructure,
};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::termstructures::{TermStructure, TermStructureBase};
use crate::time::businessdayconvention::BusinessDayConvention;
use crate::time::date::Date;
use crate::time::daycounter::DayCounter;
use crate::types::{Rate, Real, Time, Volatility};

use super::LocalVolTermStructure;

/// Dupire local volatility with the Carr–Madan jump-to-default correction.
pub struct CarrMadanLocalVolSurface {
    dupire: LocalVolSurface,
    calls: BlackVolCallSurface,
    hazard: Shared<PowerHazard>,
    grid: MomentGrid,
}

impl CarrMadanLocalVolSurface {
    /// Surface on one Black market, with the calibrated scale `hazard`.
    ///
    /// The Dupire piece and the call derivatives read the same Black vol,
    /// yields, and spot. `grid` truncates `M_p`.
    pub fn new(
        black: Handle<dyn BlackVolTermStructure>,
        risk_free: Handle<dyn YieldTermStructure>,
        dividend: Handle<dyn YieldTermStructure>,
        spot: Handle<dyn Quote>,
        hazard: Shared<PowerHazard>,
        grid: MomentGrid,
    ) -> Self {
        let dupire = LocalVolSurface::new(
            black.clone(),
            risk_free.clone(),
            dividend.clone(),
            spot.clone(),
        );
        let calls = BlackVolCallSurface::new(black, risk_free, dividend, spot);
        Self {
            dupire,
            calls,
            hazard,
            grid,
        }
    }
}

impl AsObservable for CarrMadanLocalVolSurface {
    fn observable(&self) -> &Observable {
        self.dupire.observable()
    }
}

impl TermStructure for CarrMadanLocalVolSurface {
    fn base(&self) -> &TermStructureBase {
        self.dupire.base()
    }

    fn reference_date(&self) -> QlResult<Date> {
        self.dupire.reference_date()
    }

    fn day_counter(&self) -> Option<DayCounter> {
        self.dupire.day_counter()
    }

    fn max_date(&self) -> Date {
        self.dupire.max_date()
    }
}

impl VolatilityTermStructure for CarrMadanLocalVolSurface {
    fn business_day_convention(&self) -> BusinessDayConvention {
        self.dupire.business_day_convention()
    }

    fn min_strike(&self) -> Rate {
        self.dupire.min_strike()
    }

    fn max_strike(&self) -> Rate {
        self.dupire.max_strike()
    }
}

impl LocalVolTermStructure for CarrMadanLocalVolSurface {
    fn local_vol_impl(&self, t: Time, strike: Real) -> QlResult<Volatility> {
        let dupire = self.dupire.local_vol(t, strike, true)?;
        // A zero scale is Dupire exactly, including a surface that would make
        // the density ratio noisy.
        if self.hazard.scale(t) == 0.0 {
            return Ok(dupire);
        }
        // Strike-space Dupire, eq. 10. Differentiating the call price keeps
        // `C_T`, `C_K` and `C_KK` on the same surface; the total-variance form
        // inside `LocalVolSurface` loses `w_yy` where its log-moneyness step
        // collapses, which is exactly the smile of a jump-to-default market.
        let variance = self.strike_space_variance(t, strike)?;
        if variance <= 0.0 || !variance.is_finite() {
            fail!(
                "negative Carr-Madan local vol^2 at strike {strike} and time {t}; the black vol surface is not smooth enough"
            );
        }
        Ok(variance.sqrt())
    }
}

impl CarrMadanLocalVolSurface {
    fn strike_space_variance(&self, t: Time, strike: Real) -> QlResult<Real> {
        let dt = (1.0e-4_f64).min(t / 2.0);
        let call = self.calls.call(strike, t)?;
        let call_t = (self.calls.call(strike, t + dt)?
            - self.calls.call(strike, (t - dt).max(dt))?)
            / (t + dt - (t - dt).max(dt));
        let call_k = self.calls.call_dk(strike, t)?;
        let step = (strike * 1.0e-4).max(1.0e-6);
        let density = |step: Real| -> QlResult<Real> {
            Ok((self.calls.call(strike + step, t)? - 2.0 * call
                + self.calls.call(strike - step, t)?)
                / (step * step))
        };
        let fine = density(step)?;
        let coarse = density(2.0 * step)?;
        let call_kk = (4.0 * fine - coarse) / 3.0;
        let discount_up = self.calls.discount(t + dt)?;
        let discount_dn = self.calls.discount((t - dt).max(dt))?;
        let span = t + dt - (t - dt).max(dt);
        let rate = -(discount_up / discount_dn).ln() / span;
        let carry =
            (self.calls.forward(t + dt)? / self.calls.forward((t - dt).max(dt))?).ln() / span;
        let dividend = rate - carry;
        let moment =
            truncated_power_price(&self.calls, self.hazard.exponent(), strike, t, &self.grid)?;
        let numerator = call_t + dividend * call + carry * strike * call_k
            - strike * self.hazard.scale(t) * moment;
        Ok(2.0 * numerator / (strike * strike * call_kk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::interestrate::Compounding;
    use crate::models::equity::jumptodefault::{
        ConstantHazardCallSurface, MomentGrid, PowerHazard,
    };
    use crate::option::OptionType;
    use crate::pricingengines::black_formula_implied_std_dev;
    use crate::quotes::SimpleQuote;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;
    use crate::types::{Natural, Rate};

    fn today() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn flat_yield(rate: Rate) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn flat_black(vol: Volatility) -> Handle<dyn BlackVolTermStructure> {
        Handle::new(
            shared(crate::termstructures::volatility::BlackConstantVol::new(
                today(),
                None,
                vol,
                Actual365Fixed::new(),
            )) as Shared<dyn BlackVolTermStructure>,
        )
    }

    #[test]
    fn zero_scale_matches_dupire() {
        let black = flat_black(0.2);
        let risk_free = flat_yield(0.05);
        let dividend = flat_yield(0.01);
        let spot = Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>);
        let dupire = LocalVolSurface::new(
            black.clone(),
            risk_free.clone(),
            dividend.clone(),
            spot.clone(),
        );
        let surface = CarrMadanLocalVolSurface::new(
            black,
            risk_free,
            dividend,
            spot,
            shared(PowerHazard::constant(0.0, -1.5).unwrap()),
            MomentGrid::standard().unwrap(),
        );
        for (t, strike) in [(0.5, 80.0), (1.0, 100.0), (2.0, 140.0)] {
            let adjusted = surface.local_vol(t, strike, true).unwrap();
            let plain = dupire.local_vol(t, strike, true).unwrap();
            assert_eq!(adjusted, plain, "t={t} strike={strike}");
        }
    }

    /// Black variance of the constant-hazard market, inverted call by call.
    struct ImpliedHazardVol {
        base: TermStructureBase,
        oracle: ConstantHazardCallSurface,
        spot: Real,
        rate: Real,
        dividend: Real,
    }

    impl AsObservable for ImpliedHazardVol {
        fn observable(&self) -> &Observable {
            self.base.observable()
        }
    }

    impl TermStructure for ImpliedHazardVol {
        fn base(&self) -> &TermStructureBase {
            &self.base
        }

        fn max_date(&self) -> Date {
            Date::max_date()
        }
    }

    impl VolatilityTermStructure for ImpliedHazardVol {
        fn business_day_convention(&self) -> BusinessDayConvention {
            BusinessDayConvention::Following
        }

        fn min_strike(&self) -> Rate {
            Rate::MIN
        }

        fn max_strike(&self) -> Rate {
            Rate::MAX
        }
    }

    impl BlackVolTermStructure for ImpliedHazardVol {
        fn black_vol_impl(&self, t: Time, strike: Real) -> QlResult<Volatility> {
            let variance = self.black_variance_impl(t, strike)?;
            let time = if t == 0.0 { 1e-8 } else { t };
            Ok((variance / time).sqrt())
        }

        fn black_variance_impl(&self, t: Time, strike: Real) -> QlResult<Real> {
            let time = if t == 0.0 { 1e-6 } else { t };
            let market_forward = self.spot * ((self.rate - self.dividend) * time).exp();
            let price = self.oracle.call(strike, time)?;
            let discount = self.oracle.discount(time)?;
            let std = black_formula_implied_std_dev(
                OptionType::Call,
                strike,
                market_forward,
                price,
                discount,
                0.0,
                Some(0.2 * time.sqrt()),
                1e-12,
                80 as Natural,
            )?;
            Ok(std * std * t / time)
        }
    }

    #[test]
    fn constant_hazard_local_vol_is_the_flat_diffusion() {
        let spot = 100.0;
        let rate = 0.02;
        let dividend = 0.0;
        let vol = 0.25;
        let hazard = 0.05;
        let oracle = ConstantHazardCallSurface::new(spot, rate, dividend, vol, hazard).unwrap();
        let black = Handle::new(shared(ImpliedHazardVol {
            base: TermStructureBase::with_reference_date(
                today(),
                None,
                Some(Actual365Fixed::new()),
            ),
            oracle,
            spot,
            rate,
            dividend,
        }) as Shared<dyn BlackVolTermStructure>);
        let risk_free = flat_yield(rate);
        let dividend_curve = flat_yield(dividend);
        let spot_quote = Handle::new(shared(SimpleQuote::new(spot)) as Shared<dyn Quote>);
        let dupire = LocalVolSurface::new(
            black.clone(),
            risk_free.clone(),
            dividend_curve.clone(),
            spot_quote.clone(),
        );
        let surface = CarrMadanLocalVolSurface::new(
            black,
            risk_free,
            dividend_curve,
            spot_quote,
            shared(PowerHazard::constant(hazard, 0.0).unwrap()),
            MomentGrid::new(12.0, 96).unwrap(),
        );
        // Finite-difference gamma of the inverted smile loses the far left
        // wing: at 0.5F the absolute vol error is about 2e-5 at T=1 and
        // 3e-3 at T=0.5, because the call is nearly intrinsic. From 0.7F to
        // 2F the corrected vol is within 1e-5, and Dupire itself is not flat.
        let mut dupire_gap: Real = 0.0;
        for (t, moneyness) in [(1.0, 0.7), (1.0, 1.0), (1.0, 1.5), (2.0, 2.0)] {
            let strike = spot * ((rate - dividend) * t).exp() * moneyness;
            let local = surface.local_vol(t, strike, true).unwrap();
            let plain = dupire.local_vol(t, strike, true).unwrap();
            dupire_gap = dupire_gap.max((plain - vol).abs());
            assert!(
                (local - vol).abs() <= 1e-5,
                "t={t} strike={strike} local={local} dupire={plain}"
            );
        }
        assert!(
            dupire_gap > 1e-2,
            "market Dupire should leave the flat diffusion, gap {dupire_gap}"
        );
    }

    #[test]
    fn negative_variance_names_the_strike_and_time() {
        let surface = CarrMadanLocalVolSurface::new(
            flat_black(0.2),
            flat_yield(0.01),
            flat_yield(0.0),
            Handle::new(shared(SimpleQuote::new(100.0)) as Shared<dyn Quote>),
            shared(PowerHazard::constant(2.0, 0.0).unwrap()),
            MomentGrid::new(8.0, 32).unwrap(),
        );
        let strike = 100.0;
        let t = 1.0;
        let error = surface.local_vol(t, strike, true).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("100") && message.contains('1'),
            "error should name strike and time, got {message}"
        );
    }
}
