//! Vanna/Volga smile interpolation between three strike/vol points.
//!
//! Port of `ql/experimental/barrieroption/vannavolgainterpolation.hpp`.

use crate::errors::QlResult;
use crate::math::distributions::normal::NormalDistribution;
use crate::option::OptionType;
use crate::pricingengines::black_formula;
use crate::pricingengines::black_formula_implied_std_dev;
use crate::types::{DiscountFactor, Natural, Real, Time};

/// Three-point Vanna/Volga volatility interpolation in strike space.
pub struct VannaVolgaInterpolation {
    strikes: [Real; 3],
    vols: [Real; 3],
    spot: Real,
    d_discount: DiscountFactor,
    f_discount: DiscountFactor,
    t: Time,
    atm_vol: Real,
    fwd: Real,
    premia_bs: [Real; 3],
    premia_mkt: [Real; 3],
    vegas: [Real; 3],
    normal: NormalDistribution,
}

impl VannaVolgaInterpolation {
    /// Builds the interpolator; `strikes` must be sorted ascending.
    pub fn new(
        strikes: [Real; 3],
        vols: [Real; 3],
        spot: Real,
        d_discount: DiscountFactor,
        f_discount: DiscountFactor,
        t: Time,
    ) -> QlResult<Self> {
        let mut interp = Self {
            strikes,
            vols,
            spot,
            d_discount,
            f_discount,
            t,
            atm_vol: vols[1],
            fwd: spot * f_discount / d_discount,
            premia_bs: [0.0; 3],
            premia_mkt: [0.0; 3],
            vegas: [0.0; 3],
            normal: NormalDistribution::standard(),
        };
        interp.update()?;
        Ok(interp)
    }

    fn update(&mut self) -> QlResult<()> {
        self.atm_vol = self.vols[1];
        self.fwd = self.spot * self.f_discount / self.d_discount;
        let sqrt_t = self.t.sqrt();
        for i in 0..3 {
            let k = self.strikes[i];
            let std_atm = self.atm_vol * sqrt_t;
            self.premia_bs[i] =
                black_formula(OptionType::Call, k, self.fwd, std_atm, self.d_discount, 0.0)?;
            let std_mkt = self.vols[i] * sqrt_t;
            self.premia_mkt[i] =
                black_formula(OptionType::Call, k, self.fwd, std_mkt, self.d_discount, 0.0)?;
            self.vegas[i] = self.vega_at(k)?;
        }
        Ok(())
    }

    fn vega_at(&self, k: Real) -> QlResult<Real> {
        let sqrt_t = self.t.sqrt();
        let d1 = ((self.fwd / k).ln() + 0.5 * self.atm_vol * self.atm_vol * self.t)
            / (self.atm_vol * sqrt_t);
        Ok(self.spot * self.d_discount * sqrt_t * self.normal.value(d1))
    }

    /// Implied volatility at strike `k` (extrapolation enabled).
    pub fn value(&self, k: Real) -> QlResult<Real> {
        let k0 = self.strikes[0];
        let k1 = self.strikes[1];
        let k2 = self.strikes[2];
        let vega_k = self.vega_at(k)?;
        let x1 = vega_k / self.vegas[0]
            * (k1 / k).ln() * (k2 / k).ln()
            / ((k1 / k0).ln() * (k2 / k0).ln());
        let x2 = vega_k / self.vegas[1]
            * (k / k0).ln() * (k2 / k).ln()
            / ((k1 / k0).ln() * (k2 / k1).ln());
        let x3 = vega_k / self.vegas[2]
            * (k / k0).ln() * (k / k1).ln()
            / ((k2 / k0).ln() * (k2 / k1).ln());

        let sqrt_t = self.t.sqrt();
        let c_bs = black_formula(
            OptionType::Call,
            k,
            self.fwd,
            self.atm_vol * sqrt_t,
            self.d_discount,
            0.0,
        )?;
        let c = c_bs
            + x1 * (self.premia_mkt[0] - self.premia_bs[0])
            + x2 * (self.premia_mkt[1] - self.premia_bs[1])
            + x3 * (self.premia_mkt[2] - self.premia_bs[2]);
        let std = black_formula_implied_std_dev(
            OptionType::Call,
            k,
            self.fwd,
            c,
            self.d_discount,
            0.0,
            self.atm_vol * sqrt_t,
            1e-8,
            100 as Natural,
        )?;
        Ok(std / sqrt_t)
    }
}
