//! Exchange rate between two currencies.
//!
//! Port of the direct-rate core of `ql/exchangerate.{hpp,cpp}` (derived chaining
//! beyond a single hop lands with the exchange-rate manager).

use crate::currency::Currency;
use crate::errors::QlResult;
use crate::money::Money;
use crate::require;
use crate::types::Real;

/// Whether the rate was supplied directly or derived by chaining.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExchangeRateType {
    Direct,
    Derived,
}

/// An exchange rate: one unit of `source` is worth `rate` units of `target`.
#[derive(Clone, Debug)]
pub struct ExchangeRate {
    source: Currency,
    target: Currency,
    rate: Real,
    rate_type: ExchangeRateType,
}

impl ExchangeRate {
    /// Builds a direct exchange rate.
    pub fn new(source: Currency, target: Currency, rate: Real) -> Self {
        Self {
            source,
            target,
            rate,
            rate_type: ExchangeRateType::Direct,
        }
    }

    /// The source currency.
    pub fn source(&self) -> &Currency {
        &self.source
    }

    /// The target currency.
    pub fn target(&self) -> &Currency {
        &self.target
    }

    /// The rate type.
    pub fn rate_type(&self) -> ExchangeRateType {
        self.rate_type
    }

    /// The numeric rate.
    pub fn rate(&self) -> Real {
        self.rate
    }

    /// Applies the rate to a cash amount.
    pub fn exchange(&self, amount: &Money) -> QlResult<Money> {
        if amount.currency() == &self.source {
            Ok(Money::new(self.target.clone(), amount.value() * self.rate))
        } else if amount.currency() == &self.target {
            Ok(Money::new(self.source.clone(), amount.value() / self.rate))
        } else {
            require!(
                false,
                "exchange rate not applicable: money is {}, rate is {}/{}",
                amount.currency().code(),
                self.source.code(),
                self.target.code()
            );
            unreachable!()
        }
    }

    /// Chains two rates that share a common currency.
    pub fn chain(r1: &ExchangeRate, r2: &ExchangeRate) -> QlResult<ExchangeRate> {
        if r1.target == r2.source {
            Ok(ExchangeRate {
                source: r1.source.clone(),
                target: r2.target.clone(),
                rate: r1.rate * r2.rate,
                rate_type: ExchangeRateType::Derived,
            })
        } else if r1.source == r2.target {
            Ok(ExchangeRate {
                source: r2.source.clone(),
                target: r1.target.clone(),
                rate: r1.rate * r2.rate,
                rate_type: ExchangeRateType::Derived,
            })
        } else if r1.target == r2.target {
            Ok(ExchangeRate {
                source: r1.source.clone(),
                target: r2.source.clone(),
                rate: r1.rate / r2.rate,
                rate_type: ExchangeRateType::Derived,
            })
        } else if r1.source == r2.source {
            Ok(ExchangeRate {
                source: r1.target.clone(),
                target: r2.target.clone(),
                rate: r2.rate / r1.rate,
                rate_type: ExchangeRateType::Derived,
            })
        } else {
            require!(false, "exchange rates not chainable");
            unreachable!()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_exchange_converts_source_to_target() {
        let fx = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10);
        let euros = Money::new(Currency::eur(), 100.0);
        let dollars = fx.exchange(&euros).unwrap();
        assert!((dollars.value() - 110.0).abs() < 1e-12);
        assert_eq!(dollars.currency().code(), "USD");
    }

    #[test]
    fn chain_builds_a_derived_cross_rate() {
        let eur_usd = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10);
        let usd_jpy_source = Currency::usd();
        // Use USD as intermediate; invent a second currency via Currency::new.
        let jpy = Currency::new("Japanese Yen", "JPY", 392, "¥", "", 100);
        let usd_jpy = ExchangeRate::new(usd_jpy_source, jpy.clone(), 150.0);
        let eur_jpy = ExchangeRate::chain(&eur_usd, &usd_jpy).unwrap();
        assert_eq!(eur_jpy.rate_type(), ExchangeRateType::Derived);
        assert!((eur_jpy.rate() - 165.0).abs() < 1e-12);
    }
}
