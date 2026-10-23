//! Helper to extract spot, volatilities and discount factors from BSM processes.
//!
//! Port of `ql/pricingengines/basket/vectorbsmprocessextractor.{hpp,cpp}`.

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::comparison::close_enough;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::require;
use crate::shared::Shared;
use crate::time::date::Date;
use crate::types::{DiscountFactor, Real};

/// Extracts common market data from a vector of BSM processes.
pub struct VectorBsmProcessExtractor {
    processes: Vec<Shared<GeneralizedBlackScholesProcess>>,
}

impl VectorBsmProcessExtractor {
    pub fn new(processes: Vec<Shared<GeneralizedBlackScholesProcess>>) -> Self {
        Self { processes }
    }

    pub fn get_spot(&self) -> QlResult<Array> {
        self.extract(|p| p.state_variable().current_link()?.value())
    }

    pub fn get_black_std_dev(&self, maturity_date: Date) -> QlResult<Array> {
        self.extract(|p| {
            let vol_ts = p.black_volatility().current_link()?;
            let spot = p.state_variable().current_link()?.value()?;
            let maturity = vol_ts.time_from_reference(maturity_date)?;
            Ok(vol_ts.black_vol(maturity, spot, false)? * maturity.sqrt())
        })
    }

    pub fn get_black_variance(&self, maturity_date: Date) -> QlResult<Array> {
        self.extract(|p| {
            let vol_ts = p.black_volatility().current_link()?;
            let spot = p.state_variable().current_link()?.value()?;
            let t = vol_ts.time_from_reference(maturity_date)?;
            vol_ts.black_variance(t, spot, false)
        })
    }

    pub fn get_dividend_yield_df(&self, maturity_date: Date) -> QlResult<Array> {
        self.extract(|p| {
            p.dividend_yield()
                .current_link()?
                .discount_date(maturity_date, false)
        })
    }

    pub fn get_interest_rate_df(&self, maturity_date: Date) -> QlResult<DiscountFactor> {
        let dr = self.extract(|p| {
            p.risk_free_rate()
                .current_link()?
                .discount_date(maturity_date, false)
        })?;
        require!(
            dr.iter().skip(1).all(|&x| close_enough(x, dr[0])),
            "interest rates need to be the same for all underlyings"
        );
        Ok(dr[0])
    }

    fn extract<F>(&self, f: F) -> QlResult<Array>
    where
        F: Fn(&GeneralizedBlackScholesProcess) -> QlResult<Real>,
    {
        Ok(Array::from(
            self.processes
                .iter()
                .map(|p| f(p))
                .collect::<QlResult<Vec<_>>>()?,
        ))
    }
}
