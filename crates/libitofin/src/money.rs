//! Cash amount in a given currency.
//!
//! Port of the value-type core of `ql/money.{hpp,cpp}` (conversion settings
//! deferred with the exchange-rate manager).

use crate::currency::Currency;
use crate::errors::QlResult;
use crate::require;
use crate::types::Real;
use std::fmt;
use std::ops::{Div, Mul, Neg};

/// An amount of cash denominated in a [`Currency`].
#[derive(Clone, Debug, PartialEq)]
pub struct Money {
    currency: Currency,
    value: Real,
}

impl Money {
    /// Builds a money amount.
    pub fn new(currency: Currency, value: Real) -> Self {
        Self { currency, value }
    }

    /// The currency.
    pub fn currency(&self) -> &Currency {
        &self.currency
    }

    /// The numeric value.
    pub fn value(&self) -> Real {
        self.value
    }

    /// Adds two amounts of the same currency.
    pub fn checked_add(&self, rhs: &Money) -> QlResult<Money> {
        require!(
            self.currency == rhs.currency,
            "currency mismatch: {} vs {}",
            self.currency.code(),
            rhs.currency.code()
        );
        Ok(Money::new(self.currency.clone(), self.value + rhs.value))
    }

    /// Subtracts two amounts of the same currency.
    pub fn checked_sub(&self, rhs: &Money) -> QlResult<Money> {
        self.checked_add(&Money::new(rhs.currency.clone(), -rhs.value))
    }
}

impl Neg for Money {
    type Output = Money;
    fn neg(self) -> Money {
        Money::new(self.currency, -self.value)
    }
}

impl Mul<Real> for Money {
    type Output = Money;
    fn mul(self, rhs: Real) -> Money {
        Money::new(self.currency, self.value * rhs)
    }
}

impl Div<Real> for Money {
    type Output = Money;
    fn div(self, rhs: Real) -> Money {
        Money::new(self.currency, self.value / rhs)
    }
}

impl fmt::Display for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.currency.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_currency_amounts_add() {
        let a = Money::new(Currency::eur(), 10.0);
        let b = Money::new(Currency::eur(), 2.5);
        let sum = a.checked_add(&b).unwrap();
        assert_eq!(sum.value(), 12.5);
        assert_eq!(sum.currency().code(), "EUR");
    }

    #[test]
    fn mismatched_currencies_error() {
        let a = Money::new(Currency::eur(), 1.0);
        let b = Money::new(Currency::usd(), 1.0);
        assert!(a.checked_add(&b).is_err());
    }
}
