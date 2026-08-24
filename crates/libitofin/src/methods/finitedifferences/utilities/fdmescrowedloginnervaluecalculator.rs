//! Inner value for the escrowed cash-dividend model on a log-spot grid.
//!
//! Port of `ql/methods/finitedifferences/utilities/fdmescrowedloginnervaluecalculator.{hpp,cpp}`:
//! the mesh holds the prepaid process `S*`, and the payoff is evaluated at the
//! actual spot `exp(x) − dividendAdjustment(t)`. Cell averaging is not applied
//! (`avgInnerValue` returns `innerValue`, `cpp:47-50`).

use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::methods::finitedifferences::operators::FdmLinearOpIterator;
use crate::payoff::Payoff;
use crate::shared::Shared;
use crate::types::{Real, Size, Time};

use super::{EscrowedDividendAdjustment, FdmInnerValueCalculator};

/// Payoff on the actual spot implied by an escrowed-dividend log-spot mesh.
pub struct FdmEscrowedLogInnerValueCalculator {
    escrowed: Shared<EscrowedDividendAdjustment>,
    payoff: Shared<dyn Payoff>,
    mesher: Shared<dyn FdmMesher>,
    direction: Size,
}

impl FdmEscrowedLogInnerValueCalculator {
    /// `FdmEscrowedLogInnerValueCalculator(escrowedDividendAdj, payoff, mesher, direction)`.
    pub fn new(
        escrowed: Shared<EscrowedDividendAdjustment>,
        payoff: Shared<dyn Payoff>,
        mesher: Shared<dyn FdmMesher>,
        direction: Size,
    ) -> Self {
        Self {
            escrowed,
            payoff,
            mesher,
            direction,
        }
    }
}

impl FdmInnerValueCalculator for FdmEscrowedLogInnerValueCalculator {
    fn inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
        let s_t = self.mesher.location(iter, self.direction).exp();
        let spot = s_t
            - self
                .escrowed
                .dividend_adjustment(t)
                .expect("escrowed dividend adjustment");
        self.payoff.value(spot)
    }

    fn avg_inner_value(&self, iter: &FdmLinearOpIterator, t: Time) -> Real {
        self.inner_value(iter, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflows::FixedDividend;
    use crate::handle::Handle;
    use crate::instruments::PlainVanillaPayoff;
    use crate::interestrate::Compounding;
    use crate::methods::finitedifferences::meshers::UniformGridMesher;
    use crate::methods::finitedifferences::operators::FdmLinearOpLayout;
    use crate::option::OptionType;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::termstructures::yieldtermstructure::YieldTermStructure;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::time::frequency::Frequency;

    fn today() -> Date {
        Date::new(11, Month::November, 2025)
    }

    fn flat(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            today(),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn mesher() -> Shared<dyn FdmMesher> {
        let layout = shared(FdmLinearOpLayout::new(vec![5]));
        shared(UniformGridMesher::new(layout, &[(4.0, 5.0)]).unwrap())
    }

    fn calculator(
        amount: Real,
        div_date: Date,
        maturity: Time,
    ) -> FdmEscrowedLogInnerValueCalculator {
        let r_ts = flat(0.025);
        let q_ts = flat(0.05);
        let schedule =
            vec![shared(FixedDividend::new(amount, div_date))
                as Shared<dyn crate::cashflows::Dividend>];
        let to_time_ts = r_ts.clone();
        let escrowed = shared(EscrowedDividendAdjustment::new(
            schedule,
            r_ts,
            q_ts,
            move |d| to_time_ts.current_link()?.time_from_reference(d),
            maturity,
        ));
        FdmEscrowedLogInnerValueCalculator::new(
            escrowed,
            shared(PlainVanillaPayoff::new(OptionType::Call, 100.0)) as Shared<dyn Payoff>,
            mesher(),
            0,
        )
    }

    #[test]
    fn after_the_dividend_the_payoff_is_plain_exp_x() {
        let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let calc = calculator(5.0, today() + 180, 1.0);
        let mesher = mesher();
        for position in mesher.layout().iter() {
            let x = mesher.location(&position, 0);
            assert!(
                (calc.inner_value(&position, 1.0) - payoff.value(x.exp())).abs() < 1e-12,
                "at maturity remaining dividends vanish"
            );
        }
    }

    #[test]
    fn before_the_dividend_the_payoff_uses_the_actual_spot() {
        let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
        let calc = calculator(5.0, today() + 180, 1.0);
        let mesher = mesher();
        let adj = calc.escrowed.dividend_adjustment(0.0).unwrap();
        assert!(adj < 0.0);
        for position in mesher.layout().iter() {
            let x = mesher.location(&position, 0);
            let expected = payoff.value(x.exp() - adj);
            assert!(
                (calc.inner_value(&position, 0.0) - expected).abs() < 1e-12,
                "spot = exp(x) - adj(t)"
            );
        }
    }

    #[test]
    fn the_average_is_the_grid_point_value() {
        let calc = calculator(5.0, today() + 180, 1.0);
        let mesher = mesher();
        for position in mesher.layout().iter() {
            assert_eq!(
                calc.avg_inner_value(&position, 0.0),
                calc.inner_value(&position, 0.0)
            );
        }
    }
}
