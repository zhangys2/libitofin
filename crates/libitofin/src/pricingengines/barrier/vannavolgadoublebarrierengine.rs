//! Vanna/Volga double-barrier pricing engine.
//!
//! Port of `ql/experimental/barrieroption/vannavolgadoublebarrierengine.hpp`
//! with [`AnalyticDoubleBarrierEngine`] as the inner Black–Scholes pricer.

use crate::errors::QlResult;
use crate::exercise::Exercise;
use crate::handle::Handle;
use crate::instrument::Instrument;
use crate::instrument::InstrumentResults;
use crate::instruments::{
    DoubleBarrierArguments, DoubleBarrierOption, DoubleBarrierType, PlainVanillaPayoff,
    StrikedTypePayoff, TypePayoff,
};
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::math::distributions::normal::{CumulativeNormalDistribution, NormalDistribution};
use crate::math::matrix::{Matrix, inverse_3x3};
use crate::option::OptionType;
use crate::patterns::observable::{AsObservable, Observable};
use crate::pricingengine::{Arguments, GenericEngine, PricingEngine, Results};
use crate::pricingengines::black_formula;
use crate::pricingengines::blackdeltacalculator::BlackDeltaCalculator;
use crate::processes::BlackScholesMertonProcess;
use crate::quotes::{DeltaVolQuote, Quote, SimpleQuote};
use crate::require;
use crate::settings::Settings;
use crate::shared::{Shared, SharedMut, shared, shared_mut};
use crate::termstructures::volatility::BlackConstantVol;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::time::daycounters::actual365fixed::Actual365Fixed;
use crate::time::frequency::Frequency;
use crate::types::{Real, Time};

use super::analyticdoublebarrierengine::AnalyticDoubleBarrierEngine;
use super::vannavolgainterpolation::VannaVolgaInterpolation;

type EngineBase = GenericEngine<DoubleBarrierArguments, InstrumentResults>;

/// Vanna/Volga adjustment on top of a flat-vol double-barrier pricer.
pub struct VannaVolgaDoubleBarrierEngine {
    base: EngineBase,
    atm_vol: Shared<DeltaVolQuote>,
    vol25_put: Shared<DeltaVolQuote>,
    vol25_call: Shared<DeltaVolQuote>,
    t: Time,
    spot_fx: Handle<dyn Quote>,
    domestic_ts: Handle<dyn YieldTermStructure>,
    foreign_ts: Handle<dyn YieldTermStructure>,
    adapt_van_delta: bool,
    bs_price_with_smile: Real,
    series: i32,
    settings: Shared<Settings<Date>>,
    normal: NormalDistribution,
    cnd: CumulativeNormalDistribution,
}

impl VannaVolgaDoubleBarrierEngine {
    /// Builds the engine; inner series truncation defaults to 5.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        atm_vol: Shared<DeltaVolQuote>,
        vol25_put: Shared<DeltaVolQuote>,
        vol25_call: Shared<DeltaVolQuote>,
        spot_fx: Handle<dyn Quote>,
        domestic_ts: Handle<dyn YieldTermStructure>,
        foreign_ts: Handle<dyn YieldTermStructure>,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        Self::with_options(
            atm_vol,
            vol25_put,
            vol25_call,
            spot_fx,
            domestic_ts,
            foreign_ts,
            false,
            0.0,
            5,
            settings,
        )
    }

    /// Full constructor mirroring QuantLib's optional arguments.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        atm_vol: Shared<DeltaVolQuote>,
        vol25_put: Shared<DeltaVolQuote>,
        vol25_call: Shared<DeltaVolQuote>,
        spot_fx: Handle<dyn Quote>,
        domestic_ts: Handle<dyn YieldTermStructure>,
        foreign_ts: Handle<dyn YieldTermStructure>,
        adapt_van_delta: bool,
        bs_price_with_smile: Real,
        series: i32,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        require!(
            vol25_put.delta() == Some(-0.25),
            "25 delta put is required by vanna volga method"
        );
        require!(
            vol25_call.delta() == Some(0.25),
            "25 delta call is required by vanna volga method"
        );
        let t = atm_vol.maturity();
        require!(
            vol25_put.maturity() == vol25_call.maturity() && vol25_put.maturity() == t,
            "maturity of 3 vols are not the same"
        );
        require!(
            !domestic_ts.is_empty(),
            "domestic yield curve is not defined"
        );
        require!(!foreign_ts.is_empty(), "foreign yield curve is not defined");

        let base = EngineBase::new(
            DoubleBarrierArguments::default(),
            InstrumentResults::default(),
        );
        base.register_with(atm_vol.observable());
        base.register_with(vol25_put.observable());
        base.register_with(vol25_call.observable());
        let observer = base.observer();
        spot_fx.register_observer(&observer);
        domestic_ts.register_observer(&observer);
        foreign_ts.register_observer(&observer);

        Ok(Self {
            base,
            atm_vol,
            vol25_put,
            vol25_call,
            t,
            spot_fx,
            domestic_ts,
            foreign_ts,
            adapt_van_delta,
            bs_price_with_smile,
            series,
            settings,
            normal: NormalDistribution::standard(),
            cnd: CumulativeNormalDistribution::standard(),
        })
    }
}

impl AsObservable for VannaVolgaDoubleBarrierEngine {
    fn observable(&self) -> &Observable {
        self.base.observable()
    }
}

impl PricingEngine for VannaVolgaDoubleBarrierEngine {
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
        const SIGMA_SHIFT_VEGA: Real = 0.001;
        const SIGMA_SHIFT_VOLGA: Real = 0.0001;
        const SIGMA_SHIFT_VANNA: Real = 0.0001;

        let (barrier_type, barrier_lo, barrier_hi, rebate, payoff, exercise) = {
            let args = self.base.arguments();
            (
                args.barrier_type.expect("validated"),
                args.barrier_lo.expect("validated"),
                args.barrier_hi.expect("validated"),
                args.rebate.expect("validated"),
                args.payoff.expect("validated"),
                Shared::clone(args.exercise.as_ref().expect("validated")),
            )
        };

        require!(
            matches!(
                barrier_type,
                DoubleBarrierType::KnockIn | DoubleBarrierType::KnockOut
            ),
            "only same type barrier supported"
        );

        let spot0 = self.spot_fx.current_link()?.value()?;
        let atm_vol0 = self.atm_vol.value()?;
        let spot_shift = shared(SimpleQuote::new(spot0));
        let atm_vol_shift = shared(SimpleQuote::new(atm_vol0));
        let spot_shift_delta = 0.0001 * spot0;

        let domestic = self.domestic_ts.current_link()?;
        let foreign = self.foreign_ts.current_link()?;
        let d_disc = domestic.discount(self.t, false)?;
        let f_disc = foreign.discount(self.t, false)?;
        let forward = spot0 * f_disc / d_disc;
        let sqrt_t = self.t.sqrt();

        let ref_date = domestic.reference_date()?;
        let vol_ts = shared(BlackConstantVol::with_quote(
            ref_date,
            None,
            Handle::new(Shared::clone(&atm_vol_shift) as Shared<dyn Quote>),
            Actual365Fixed::new(),
        ));
        let process = shared(BlackScholesMertonProcess::new(
            Handle::new(Shared::clone(&spot_shift) as Shared<dyn Quote>),
            Handle::new(Shared::clone(&foreign) as Shared<dyn YieldTermStructure>),
            Handle::new(Shared::clone(&domestic) as Shared<dyn YieldTermStructure>),
            Handle::new(
                vol_ts as Shared<dyn crate::termstructures::volatility::BlackVolTermStructure>,
            ),
        ));

        let black_atm = BlackDeltaCalculator::new(
            OptionType::Call,
            self.atm_vol.delta_type(),
            spot0,
            d_disc,
            f_disc,
            atm_vol0 * sqrt_t,
        )?;
        let atm_strike = black_atm.atm_strike(self.atm_vol.atm_type())?;

        let call25_vol = self.vol25_call.value()?;
        let put25_vol = self.vol25_put.value()?;
        let put25_strike = BlackDeltaCalculator::new(
            OptionType::Put,
            self.vol25_put.delta_type(),
            spot0,
            d_disc,
            f_disc,
            put25_vol * sqrt_t,
        )?
        .strike_from_delta(-0.25)?;
        let call25_strike = BlackDeltaCalculator::new(
            OptionType::Call,
            self.vol25_call.delta_type(),
            spot0,
            d_disc,
            f_disc,
            call25_vol * sqrt_t,
        )?
        .strike_from_delta(0.25)?;

        let strikes = [put25_strike, atm_strike, call25_strike];
        let vols = [put25_vol, atm_vol0, call25_vol];
        // QuantLib passes foreign discount twice here (vannavolgadoublebarrierengine.hpp).
        let interp = VannaVolgaInterpolation::new(strikes, vols, spot0, f_disc, f_disc, self.t)?;
        let strike_vol = interp.value(payoff.strike())?;
        let vanilla_option = black_formula(
            payoff.option_type(),
            payoff.strike(),
            forward,
            strike_vol * sqrt_t,
            d_disc,
            0.0,
        )?;

        let triggered = spot0 > barrier_hi || spot0 < barrier_lo;
        let results = self.base.results_mut();

        if triggered && barrier_type == DoubleBarrierType::KnockOut {
            results.value = Some(0.0);
            store_additional(
                results,
                self.adapt_van_delta,
                self.bs_price_with_smile,
                vanilla_option,
                vanilla_option,
                0.0,
                None,
            );
            return Ok(());
        }
        if triggered && barrier_type == DoubleBarrierType::KnockIn {
            let van = if self.adapt_van_delta {
                self.bs_price_with_smile
            } else {
                vanilla_option
            };
            results.value = Some(van);
            store_additional(
                results,
                self.adapt_van_delta,
                self.bs_price_with_smile,
                vanilla_option,
                van,
                0.0,
                None,
            );
            return Ok(());
        }

        let mut bump_pricer = BarrierBumpPricer::new(
            Shared::clone(&spot_shift),
            Shared::clone(&atm_vol_shift),
            process,
            barrier_lo,
            barrier_hi,
            rebate,
            payoff,
            exercise,
            self.series,
            Shared::clone(&self.settings),
        )?;

        let price_bs = bump_pricer.npv()?;
        let std_atm = atm_vol0 * sqrt_t;

        let price_atm_call_bs =
            black_formula(OptionType::Call, atm_strike, forward, std_atm, d_disc, 0.0)?;
        let price25_call_bs = black_formula(
            OptionType::Call,
            call25_strike,
            forward,
            std_atm,
            d_disc,
            0.0,
        )?;
        let price25_put_bs =
            black_formula(OptionType::Put, put25_strike, forward, std_atm, d_disc, 0.0)?;

        let price_atm_call_mkt = price_atm_call_bs;
        let price25_call_mkt = black_formula(
            OptionType::Call,
            call25_strike,
            forward,
            call25_vol * sqrt_t,
            d_disc,
            0.0,
        )?;
        let price25_put_mkt = black_formula(
            OptionType::Put,
            put25_strike,
            forward,
            put25_vol * sqrt_t,
            d_disc,
            0.0,
        )?;

        let atm_vol_q = atm_vol_shift.value().unwrap();
        let (vega_atm, vanna_atm, volga_atm) = analytical_greeks(
            forward,
            atm_strike,
            spot0,
            f_disc,
            atm_vol_q,
            self.t,
            &self.normal,
        )?;
        let (vega25_call, vanna25_call, volga25_call) = analytical_greeks(
            forward,
            call25_strike,
            spot0,
            f_disc,
            atm_vol_q,
            self.t,
            &self.normal,
        )?;
        let (vega25_put, vanna25_put, volga25_put) = analytical_greeks(
            forward,
            put25_strike,
            spot0,
            f_disc,
            atm_vol_q,
            self.t,
            &self.normal,
        )?;

        atm_vol_shift.set_value(atm_vol_q + SIGMA_SHIFT_VEGA);
        let vega_bar_bs = (bump_pricer.npv()? - price_bs) / SIGMA_SHIFT_VEGA;
        atm_vol_shift.set_value(atm_vol_q);

        atm_vol_shift.set_value(atm_vol_q + SIGMA_SHIFT_VOLGA);
        let price_bs2 = bump_pricer.npv()?;
        atm_vol_shift.set_value(atm_vol_q + SIGMA_SHIFT_VOLGA + SIGMA_SHIFT_VEGA);
        let vega_bar_bs2 = (bump_pricer.npv()? - price_bs2) / SIGMA_SHIFT_VEGA;
        let volga_bar_bs = (vega_bar_bs2 - vega_bar_bs) / SIGMA_SHIFT_VOLGA;
        atm_vol_shift.set_value(atm_vol_q);

        spot_shift.set_value(spot0 + spot_shift_delta);
        let price_delta1 = bump_pricer.npv()?;
        spot_shift.set_value(spot0 - spot_shift_delta);
        let price_delta2 = bump_pricer.npv()?;
        spot_shift.set_value(spot0);
        let delta_bar1 = (price_delta1 - price_delta2) / (2.0 * spot_shift_delta);

        atm_vol_shift.set_value(atm_vol_q + SIGMA_SHIFT_VANNA);
        spot_shift.set_value(spot0 + spot_shift_delta);
        let price_delta1 = bump_pricer.npv()?;
        spot_shift.set_value(spot0 - spot_shift_delta);
        let price_delta2 = bump_pricer.npv()?;
        spot_shift.set_value(spot0);
        let delta_bar2 = (price_delta1 - price_delta2) / (2.0 * spot_shift_delta);
        let vanna_bar_bs = (delta_bar2 - delta_bar1) / SIGMA_SHIFT_VANNA;
        atm_vol_shift.set_value(atm_vol_q);

        let mut a = Matrix::with_size(3, 3);
        a[(0, 0)] = vega_atm;
        a[(0, 1)] = vega25_call;
        a[(0, 2)] = vega25_put;
        a[(1, 0)] = vanna_atm;
        a[(1, 1)] = vanna25_call;
        a[(1, 2)] = vanna25_put;
        a[(2, 0)] = volga_atm;
        a[(2, 1)] = volga25_call;
        a[(2, 2)] = volga25_put;

        let b = Array::from([vega_bar_bs, vanna_bar_bs, volga_bar_bs]);
        let q = &inverse_3x3(&a) * &b;

        let r_dom = domestic
            .zero_rate(
                self.t,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let r_for = foreign
            .zero_rate(
                self.t,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate();
        let theta_tilt_minus = ((r_dom - r_for) / atm_vol0 - atm_vol0 / 2.0) * sqrt_t;
        let h = (barrier_hi / spot0).ln() / atm_vol0 / sqrt_t;
        let l = (barrier_lo / spot0).ln() / atm_vol0 / sqrt_t;

        let mut double_no_touch = 0.0;
        for j in -self.series..self.series {
            let jf = j as Real;
            let e_minus = 2.0 * jf * (h - l) - theta_tilt_minus;
            double_no_touch += (-2.0 * jf * theta_tilt_minus * (h - l)).exp()
                * (self.cnd.value(h + e_minus) - self.cnd.value(l + e_minus))
                - (-2.0 * jf * theta_tilt_minus * (h - l) + 2.0 * theta_tilt_minus * h).exp()
                    * (self.cnd.value(h - 2.0 * h + e_minus)
                        - self.cnd.value(l - 2.0 * h + e_minus));
        }

        let lambda = double_no_touch;
        let adjust = q[0] * (price_atm_call_mkt - price_atm_call_bs)
            + q[1] * (price25_call_mkt - price25_call_bs)
            + q[2] * (price25_put_mkt - price25_put_bs);
        let mut out_price = price_bs + lambda * adjust;
        let in_price;

        if self.adapt_van_delta {
            out_price += lambda * (self.bs_price_with_smile - vanilla_option);
            out_price = out_price.max(0.0).min(self.bs_price_with_smile);
            in_price = self.bs_price_with_smile - out_price;
        } else {
            out_price = out_price.max(0.0).min(vanilla_option);
            in_price = vanilla_option - out_price;
        }

        results.value = Some(if barrier_type == DoubleBarrierType::KnockOut {
            out_price
        } else {
            in_price
        });
        store_additional(
            results,
            self.adapt_van_delta,
            self.bs_price_with_smile,
            vanilla_option,
            in_price,
            out_price,
            Some(lambda),
        );
        Ok(())
    }
}

fn store_additional(
    results: &mut InstrumentResults,
    adapt_van_delta: bool,
    bs_price_with_smile: Real,
    vanilla_option: Real,
    in_price: Real,
    out_price: Real,
    lambda: Option<Real>,
) {
    use std::any::Any;
    if adapt_van_delta {
        results.additional_results.insert(
            "VanillaPrice".into(),
            shared(bs_price_with_smile) as Shared<dyn Any>,
        );
    } else {
        results.additional_results.insert(
            "VanillaPrice".into(),
            shared(vanilla_option) as Shared<dyn Any>,
        );
    }
    results
        .additional_results
        .insert("BarrierInPrice".into(), shared(in_price) as Shared<dyn Any>);
    results.additional_results.insert(
        "BarrierOutPrice".into(),
        shared(out_price) as Shared<dyn Any>,
    );
    if let Some(lambda) = lambda {
        results
            .additional_results
            .insert("lambda".into(), shared(lambda) as Shared<dyn Any>);
    }
}

fn analytical_greeks(
    forward: Real,
    strike: Real,
    spot: Real,
    f_disc: Real,
    atm_vol: Real,
    t: Time,
    normal: &NormalDistribution,
) -> QlResult<(Real, Real, Real)> {
    let sqrt_t = t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * atm_vol * atm_vol * t) / (atm_vol * sqrt_t);
    let vega = spot * normal.value(d1) * sqrt_t * f_disc;
    let vanna = vega / spot * (1.0 - d1 / (atm_vol * sqrt_t));
    let volga = vega * d1 * (d1 - atm_vol * sqrt_t) / atm_vol;
    Ok((vega, vanna, volga))
}

struct BarrierBumpPricer {
    option: DoubleBarrierOption,
}

impl BarrierBumpPricer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        _spot: Shared<SimpleQuote>,
        _vol: Shared<SimpleQuote>,
        process: Shared<BlackScholesMertonProcess>,
        barrier_lo: Real,
        barrier_hi: Real,
        rebate: Real,
        payoff: PlainVanillaPayoff,
        exercise: Shared<dyn Exercise>,
        series: i32,
        settings: Shared<Settings<Date>>,
    ) -> QlResult<Self> {
        let mut option = DoubleBarrierOption::new(
            DoubleBarrierType::KnockOut,
            barrier_lo,
            barrier_hi,
            rebate,
            payoff,
            exercise,
            settings,
        )?;
        let engine = shared_mut(AnalyticDoubleBarrierEngine::with_series(process, series));
        option
            .base_mut()
            .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
        Ok(Self { option })
    }

    fn npv(&mut self) -> QlResult<Real> {
        self.option.npv()
    }
}

/// Attaches a [`VannaVolgaDoubleBarrierEngine`] to `option`.
pub fn set_vanna_volga_double_barrier_engine(
    option: &mut DoubleBarrierOption,
    engine: SharedMut<VannaVolgaDoubleBarrierEngine>,
) {
    option
        .base_mut()
        .set_pricing_engine(engine as SharedMut<dyn PricingEngine>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exercise::EuropeanExercise;
    use crate::interestrate::Compounding;
    use crate::option::OptionType;
    use crate::quotes::{AtmType, DeltaType};
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;
    use crate::types::{Rate, Time, Volatility};

    struct VvRow {
        barrier_lo: Real,
        barrier_hi: Real,
        option_type: OptionType,
        strike: Real,
        spot: Real,
        q: Rate,
        r: Rate,
        t: Time,
        vol25_put: Volatility,
        vol_atm: Volatility,
        vol25_call: Volatility,
        vol: Volatility,
        result: Real,
    }

    fn today() -> Date {
        Date::new(5, Month::March, 2013)
    }

    fn time_to_days(t: Time) -> i32 {
        (t * 360.0).round() as i32
    }

    fn quote_handle(q: &Shared<SimpleQuote>) -> Handle<dyn Quote> {
        Handle::new(Shared::clone(q) as Shared<dyn Quote>)
    }

    fn flat_rate(reference: Date, quote: &Shared<SimpleQuote>) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::new(
            reference,
            quote_handle(quote),
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn vv_table() -> &'static [VvRow] {
        &[
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Call,
                strike: 1.13321,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.11638,
                result: 0.14413,
            },
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Call,
                strike: 1.22687,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.10088,
                result: 0.07456,
            },
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Call,
                strike: 1.31179,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.08925,
                result: 0.02710,
            },
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Call,
                strike: 1.38843,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.08463,
                result: 0.00569,
            },
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Call,
                strike: 1.46047,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.08412,
                result: 0.00013,
            },
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Put,
                strike: 1.13321,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.11638,
                result: 0.00017,
            },
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Put,
                strike: 1.22687,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.10088,
                result: 0.00353,
            },
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Put,
                strike: 1.31179,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.08925,
                result: 0.02221,
            },
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Put,
                strike: 1.38843,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.08463,
                result: 0.06049,
            },
            VvRow {
                barrier_lo: 1.1,
                barrier_hi: 1.5,
                option_type: OptionType::Put,
                strike: 1.46047,
                spot: 1.30265,
                q: 0.0003541,
                r: 0.0033871,
                t: 1.0,
                vol25_put: 0.10087,
                vol_atm: 0.08925,
                vol25_call: 0.08463,
                vol: 0.08412,
                result: 0.11103,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Call,
                strike: 1.06145,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.12511,
                result: 0.19981,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Call,
                strike: 1.19545,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.10890,
                result: 0.10389,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Call,
                strike: 1.32238,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.09444,
                result: 0.03555,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Call,
                strike: 1.44298,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.09197,
                result: 0.00634,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Call,
                strike: 1.56345,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.09261,
                result: 0.00000,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Put,
                strike: 1.06145,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.12511,
                result: 0.00000,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Put,
                strike: 1.19545,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.10890,
                result: 0.00436,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Put,
                strike: 1.32238,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.09444,
                result: 0.03173,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Put,
                strike: 1.44298,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.09197,
                result: 0.09346,
            },
            VvRow {
                barrier_lo: 1.0,
                barrier_hi: 1.6,
                option_type: OptionType::Put,
                strike: 1.56345,
                spot: 1.30265,
                q: 0.0009418,
                r: 0.0039788,
                t: 2.0,
                vol25_put: 0.10891,
                vol_atm: 0.09525,
                vol25_call: 0.09197,
                vol: 0.09261,
                result: 0.17704,
            },
        ]
    }

    #[test]
    fn vanna_volga_double_barrier_values_match_quantlib() {
        let settings = shared(Settings::new());
        settings.set_evaluation_date(today());
        let spot = shared(SimpleQuote::new(0.0));
        let q_rate = shared(SimpleQuote::new(0.0));
        let r_rate = shared(SimpleQuote::new(0.0));
        let vol25_put_q = shared(SimpleQuote::new(0.0));
        let vol_atm_q = shared(SimpleQuote::new(0.0));
        let vol25_call_q = shared(SimpleQuote::new(0.0));
        let r_ts = flat_rate(today(), &r_rate);
        let q_ts = flat_rate(today(), &q_rate);

        for row in vv_table() {
            for barrier_type in [DoubleBarrierType::KnockOut, DoubleBarrierType::KnockIn] {
                spot.set_value(row.spot);
                q_rate.set_value(row.q);
                r_rate.set_value(row.r);
                vol25_put_q.set_value(row.vol25_put);
                vol_atm_q.set_value(row.vol_atm);
                vol25_call_q.set_value(row.vol25_call);

                let vol_atm_quote = shared(DeltaVolQuote::new_atm(
                    quote_handle(&vol_atm_q),
                    DeltaType::Fwd,
                    row.t,
                    AtmType::DeltaNeutral,
                ));
                let vol25_put_quote = shared(DeltaVolQuote::new(
                    -0.25,
                    quote_handle(&vol25_put_q),
                    row.t,
                    DeltaType::Fwd,
                ));
                let vol25_call_quote = shared(DeltaVolQuote::new(
                    0.25,
                    quote_handle(&vol25_call_q),
                    row.t,
                    DeltaType::Fwd,
                ));

                let payoff = PlainVanillaPayoff::new(row.option_type, row.strike);
                let exercise = shared(EuropeanExercise::new(today() + time_to_days(row.t)));

                let d_disc = r_ts.current_link().unwrap().discount(row.t, false).unwrap();
                let f_disc = q_ts.current_link().unwrap().discount(row.t, false).unwrap();
                let forward = row.spot * f_disc / d_disc;
                let bs_vanilla = black_formula(
                    row.option_type,
                    row.strike,
                    forward,
                    row.vol * row.t.sqrt(),
                    d_disc,
                    0.0,
                )
                .unwrap();

                let expected = if barrier_type == DoubleBarrierType::KnockOut {
                    row.result
                } else {
                    bs_vanilla - row.result
                };

                let mut option = DoubleBarrierOption::new(
                    barrier_type,
                    row.barrier_lo,
                    row.barrier_hi,
                    0.0,
                    payoff,
                    exercise,
                    Shared::clone(&settings),
                )
                .unwrap();

                let engine = shared_mut(
                    VannaVolgaDoubleBarrierEngine::with_options(
                        vol_atm_quote,
                        vol25_put_quote,
                        vol25_call_quote,
                        quote_handle(&spot),
                        r_ts.clone(),
                        q_ts.clone(),
                        true,
                        bs_vanilla,
                        5,
                        Shared::clone(&settings),
                    )
                    .unwrap(),
                );
                set_vanna_volga_double_barrier_engine(&mut option, engine);

                let calculated = option.npv().unwrap();
                assert!(
                    (calculated - expected).abs() <= 5.0e-3,
                    "barrier={barrier_type:?} strike={} expected={expected} got={calculated}",
                    row.strike
                );
            }
        }
    }
}
