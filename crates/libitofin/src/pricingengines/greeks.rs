//! Default greek calculations shared across engines.
//!
//! Port of `ql/pricingengines/greeks.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::interestrate::Compounding;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::stochasticprocess::StochasticProcess1D;
use crate::time::frequency::Frequency;
use crate::types::Real;

/// Default theta for Black–Scholes options (`greeks.cpp` `blackScholesTheta`).
pub fn black_scholes_theta(
    process: &GeneralizedBlackScholesProcess,
    value: Real,
    delta: Real,
    gamma: Real,
) -> QlResult<Real> {
    let spot = process.x0()?;
    let r = process
        .risk_free_rate()
        .current_link()?
        .zero_rate(0.0, Compounding::Continuous, Frequency::NoFrequency, false)?
        .rate();
    let q = process
        .dividend_yield()
        .current_link()?
        .zero_rate(0.0, Compounding::Continuous, Frequency::NoFrequency, false)?
        .rate();
    let vol = process
        .local_volatility()?
        .current_link()?
        .local_vol(0.0, spot, true)?;
    Ok(r * value - (r - q) * spot * delta - 0.5 * vol * vol * spot * spot * gamma)
}

/// Default theta-per-day on a 365-day year (`greeks.cpp` `defaultThetaPerDay`).
pub fn default_theta_per_day(theta: Real) -> Real {
    theta / 365.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::Handle;
    use crate::processes::BlackScholesMertonProcess;
    use crate::quotes::SimpleQuote;
    use crate::shared::{Shared, shared};
    use crate::termstructures::volatility::{BlackConstantVol, BlackVolTermStructure};
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;

    #[test]
    fn black_scholes_theta_matches_closed_form_for_vanilla_delta_gamma() {
        let today = Date::new(15, Month::June, 2026);
        let spot = shared(SimpleQuote::new(100.0));
        let q_rate = shared(SimpleQuote::new(0.03));
        let r_rate = shared(SimpleQuote::new(0.06));
        let vol = shared(SimpleQuote::new(0.20));
        let quote_handle = |q: &Shared<SimpleQuote>| {
            Handle::new(Shared::clone(q) as Shared<dyn crate::quotes::Quote>)
        };
        let flat = |q: &Shared<SimpleQuote>| {
            Handle::new(
                shared(FlatForward::new(
                    today,
                    quote_handle(q),
                    Actual360::new(),
                    Compounding::Continuous,
                    Frequency::Annual,
                )) as Shared<dyn crate::termstructures::yieldtermstructure::YieldTermStructure>,
            )
        };
        let flat_vol = Handle::new(
            shared(BlackConstantVol::with_quote(
                today,
                None,
                quote_handle(&vol),
                Actual360::new(),
            )) as Shared<dyn BlackVolTermStructure>,
        );
        let process = BlackScholesMertonProcess::new(
            quote_handle(&spot),
            flat(&q_rate),
            flat(&r_rate),
            flat_vol,
        );

        let value = 5.0;
        let delta = 0.4;
        let gamma = 0.01;
        let theta = black_scholes_theta(&process, value, delta, gamma).unwrap();
        let expected = 0.06 * value - (0.06 - 0.03) * 100.0 * delta - 0.5 * 0.04 * 10_000.0 * gamma;
        // `local_vol` at t = 0 is a finite-difference derivative of variance, so the
        // recovered vol carries ~1e-11 of error; 1e-12 asserts FD accuracy at machine
        // epsilon. A real theta regression would be orders of magnitude larger.
        assert!((theta - expected).abs() <= 1.0e-9, "{theta} vs {expected}");
    }
}
