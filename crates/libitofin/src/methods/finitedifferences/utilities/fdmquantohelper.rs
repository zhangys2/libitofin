//! Quanto adjustment for equity finite-difference engines.
//!
//! Port of `ql/methods/finitedifferences/utilities/fdmquantohelper.{hpp,cpp}`:
//! stores the domestic / foreign yield curves, the FX Black vol, the equity–FX
//! correlation and the FX ATM level, and returns
//!
//! ```text
//! r_d − r_f + σ_eq · σ_fx · ρ
//! ```
//!
//! over a time step. The operator subtracts this from the equity drift; the
//! mesher wraps the dividend curve in a
//! [`QuantoTermStructure`](crate::termstructures::yields::QuantoTermStructure).

use crate::errors::QlResult;
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::patterns::observable::{AsObservable, Observable};
use crate::shared::Shared;
use crate::termstructures::volatility::BlackVolTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;
use crate::types::{Rate, Real, Time, Volatility};

/// Market data for the FD quanto adjustment (`fdmquantohelper.hpp:36`).
pub struct FdmQuantoHelper {
    r_ts: Shared<dyn YieldTermStructure>,
    f_ts: Shared<dyn YieldTermStructure>,
    fx_vol_ts: Shared<dyn BlackVolTermStructure>,
    equity_fx_correlation: Real,
    exch_rate_atm_level: Real,
    observable: Observable,
}

impl FdmQuantoHelper {
    /// `FdmQuantoHelper(rTS, fTS, fxVolTS, equityFxCorrelation, exchRateATMlevel)`
    /// (`fdmquantohelper.cpp:32-39`).
    pub fn new(
        r_ts: Shared<dyn YieldTermStructure>,
        f_ts: Shared<dyn YieldTermStructure>,
        fx_vol_ts: Shared<dyn BlackVolTermStructure>,
        equity_fx_correlation: Real,
        exch_rate_atm_level: Real,
    ) -> FdmQuantoHelper {
        FdmQuantoHelper {
            r_ts,
            f_ts,
            fx_vol_ts,
            equity_fx_correlation,
            exch_rate_atm_level,
            observable: Observable::new(),
        }
    }

    /// Domestic (equity-numeraire) yield curve (`rTS_`).
    pub fn r_ts(&self) -> &Shared<dyn YieldTermStructure> {
        &self.r_ts
    }

    /// Foreign yield curve (`fTS_`).
    pub fn f_ts(&self) -> &Shared<dyn YieldTermStructure> {
        &self.f_ts
    }

    /// FX Black volatility (`fxVolTS_`).
    pub fn fx_vol_ts(&self) -> &Shared<dyn BlackVolTermStructure> {
        &self.fx_vol_ts
    }

    /// Equity–FX correlation (`equityFxCorrelation_`).
    pub fn equity_fx_correlation(&self) -> Real {
        self.equity_fx_correlation
    }

    /// FX ATM level used to sample the FX vol (`exchRateATMlevel_`).
    pub fn exch_rate_atm_level(&self) -> Real {
        self.exch_rate_atm_level
    }

    /// Scalar quanto adjustment over `[t1, t2]` (`cpp:42-49`).
    ///
    /// # Errors
    ///
    /// Propagates forward-rate / FX-vol failures on the stored curves.
    pub fn quanto_adjustment(&self, equity_vol: Volatility, t1: Time, t2: Time) -> QlResult<Rate> {
        let (rate_diff, fx_vol) = self.step_inputs(t1, t2)?;
        Ok(rate_diff + equity_vol * fx_vol * self.equity_fx_correlation)
    }

    /// Pointwise quanto adjustment (`cpp:52-66`).
    ///
    /// # Errors
    ///
    /// Propagates forward-rate / FX-vol failures on the stored curves.
    pub fn quanto_adjustment_array(
        &self,
        equity_vol: &Array,
        t1: Time,
        t2: Time,
    ) -> QlResult<Array> {
        let (rate_diff, fx_vol) = self.step_inputs(t1, t2)?;
        let scale = fx_vol * self.equity_fx_correlation;
        Ok(equity_vol
            .iter()
            .map(|&vol| rate_diff + vol * scale)
            .collect())
    }

    fn step_inputs(&self, t1: Time, t2: Time) -> QlResult<(Rate, Volatility)> {
        let r_domestic = self
            .r_ts
            .forward_rate(
                t1,
                t2,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let r_foreign = self
            .f_ts
            .forward_rate(
                t1,
                t2,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let fx_vol = self
            .fx_vol_ts
            .black_forward_vol(t1, t2, self.exch_rate_atm_level, false)?;
        Ok((r_domestic - r_foreign, fx_vol))
    }
}

impl AsObservable for FdmQuantoHelper {
    fn observable(&self) -> &Observable {
        &self.observable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::shared;
    use crate::termstructures::volatility::BlackConstantVol;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;

    fn today() -> Date {
        Date::new(22, Month::April, 2019)
    }

    fn helper() -> FdmQuantoHelper {
        let dc = Actual360::new();
        let r = shared(FlatForward::with_rate(
            today(),
            0.1,
            dc.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>;
        let f = shared(FlatForward::with_rate(
            today(),
            0.2,
            dc.clone(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>;
        let fx = shared(BlackConstantVol::new(today(), None, 0.2, dc))
            as Shared<dyn BlackVolTermStructure>;
        FdmQuantoHelper::new(r, f, fx, -0.75, 1.0)
    }

    /// `quantooption.cpp` `testFDMQuantoHelper`: closed-form drift.
    #[test]
    fn quanto_adjustment_matches_closed_form() {
        let adj = helper().quanto_adjustment(0.3, 0.0, 1.0).unwrap();
        let expected = 0.1 - 0.2 + (-0.75) * 0.3 * 0.2;
        assert!(
            (adj - expected).abs() < 1e-10,
            "adj={adj} expected={expected}"
        );
    }

    #[test]
    fn array_adjustment_matches_scalar() {
        let h = helper();
        let vols = Array::from([0.1, 0.3, 0.5]);
        let arr = h.quanto_adjustment_array(&vols, 0.0, 1.0).unwrap();
        for i in 0..vols.size() {
            let s = h.quanto_adjustment(vols[i], 0.0, 1.0).unwrap();
            assert!((arr[i] - s).abs() < 1e-14, "at {i}: {} vs {s}", arr[i]);
        }
    }
}
